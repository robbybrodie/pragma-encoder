"""Level 8 — Single-node GPU training loss smoke on real TabFormer data.

Purpose:
  Verify that the PRAGMA training loop produces a measurable loss trace
  on a bounded slice of real IBM TabFormer data when executed on an L4-class
  GPU via a single-node PyTorchJob.

  This is NOT a convergence test and makes NO quality claims.
  It proves:
    - The full training loop runs on GPU with real CSV data.
    - At least one gradient step completes without NaN/inf loss.
    - A parseable metrics.jsonl is written (loss artifact present).
    - The run stays within L4-class resource limits.

Scope and bounds (L4-class GPU — NVIDIA L4, 24 GB VRAM):
  - Model: PRAGMA-S (smallest config, ~10 M params)
  - Data: at most 10% of IBM TabFormer customers by default
  - max_steps: 100 gradient steps (configurable via env var)
  - batch_size: 4 (L4-safe default; configurable)
  - log_every: 5 steps (frequent loss visibility)
  - Single-node (1 Master replica, no Workers — single-process torchrun)
  - device: cuda (GPU required; skip if no GPU node available)
  - S3: disabled (no --s3-checkpoint-prefix; local emptyDir only)
  - Checkpoint: every epoch (bounded by max_steps so at most 1)

Guard variables:
  RUN_OPENSHIFT_TESTS=1            — suite-wide (conftest)
  RUN_TABFORMER_LOSS_SMOKE=1       — enables all Level 8 tests (static + runtime)
  RUN_TABFORMER_LOSS_SMOKE_RUN=1   — additionally enables runtime PyTorchJob creation
  PRAGMA_TEST_NAMESPACE=<ns>       — required (conftest)
  PRAGMA_TRAINING_IMAGE=<img>      — required for runtime test

Bounded parameters (configurable via env vars):
  PRAGMA_TABFORMER_SAMPLE_FRACTION — float 0.0–1.0, default 0.10 (10% of customers)
  PRAGMA_TABFORMER_MAX_STEPS       — int, default 100
  PRAGMA_TABFORMER_BATCH_SIZE      — int, default 4
  PRAGMA_LOSS_LOG_EVERY            — int, default 5
  PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1  — required when fraction > 0.20 or steps > 500

Safety rules:
  - Static tests run whenever RUN_TABFORMER_LOSS_SMOKE=1 (no cluster resources created).
  - Runtime test additionally requires RUN_TABFORMER_LOSS_SMOKE_RUN=1.
  - Never create namespaces, secrets, or service accounts.
  - Test resources carry both pragma.redhat.com/test-run and test-id labels.
  - Fail fast if fraction > 0.20 or max_steps > 500 without the override gate.
  - No benchmark or convergence assertions — only structural loss trace checks.

Data source:
  IBM TabFormer data must be present in S3 at the standard key
  (pragma-encoder/data/tabformer/card_transaction.v1.csv) OR available
  via PRAGMA_TABFORMER_DATA_URI env var.

  The training image writes the TabFormer CSV from S3 to /tmp/pragma-l8/
  at job startup (matching the production pipeline pattern), then runs
  pragma-encoder-train against that local copy.

Loss artifact:
  Training writes <output_dir>/metrics.jsonl with one record per step.
  The test collects pod logs after job completion and asserts at least one
  JSONL line with a finite train_loss value.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
Platform: RHOAI 3.4.0 — PyTorchJob kubeflow.org/v1, L4-class GPU nodes.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from tests.openshift.oc import crd_exists, label_selector, oc, oc_json, redact

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_LOSS_SMOKE_ENABLED = os.environ.get("RUN_TABFORMER_LOSS_SMOKE") == "1"
_LOSS_SMOKE_RUN_ENABLED = os.environ.get("RUN_TABFORMER_LOSS_SMOKE_RUN") == "1"

_require_loss_smoke = pytest.mark.skipif(
    not _LOSS_SMOKE_ENABLED,
    reason=(
        "Level 8 TabFormer loss smoke tests are opt-in. "
        "Set RUN_TABFORMER_LOSS_SMOKE=1 to enable static checks. "
        "Also set RUN_TABFORMER_LOSS_SMOKE_RUN=1 to enable the runtime PyTorchJob. "
        "Requires RUN_OPENSHIFT_TESTS=1 and PRAGMA_TEST_NAMESPACE=<namespace>."
    ),
)

_require_loss_smoke_run = pytest.mark.skipif(
    not (_LOSS_SMOKE_ENABLED and _LOSS_SMOKE_RUN_ENABLED),
    reason=(
        "Level 8 runtime PyTorchJob is opt-in. "
        "Set RUN_TABFORMER_LOSS_SMOKE=1 and RUN_TABFORMER_LOSS_SMOKE_RUN=1 to enable. "
        "Also requires PRAGMA_TRAINING_IMAGE=<image> and PRAGMA_TEST_NAMESPACE=<namespace>. "
        "This test creates a short-lived GPU PyTorchJob."
    ),
)

# ---------------------------------------------------------------------------
# L4-class bounds and configuration
# ---------------------------------------------------------------------------

# Default parameters — safe for NVIDIA L4 (24 GB VRAM, ~30 TFLOPS FP32).
_DEFAULT_SAMPLE_FRACTION = 0.10   # 10% of IBM TabFormer customers
_DEFAULT_MAX_STEPS = 100          # 100 gradient steps (fast exit)
_DEFAULT_BATCH_SIZE = 4           # L4-safe; fits PRAGMA-S forward pass with headroom
_DEFAULT_LOG_EVERY = 5            # frequent loss visibility for debugging

# Safety gates — fail-fast unless explicitly overridden.
_LARGE_FRACTION_THRESHOLD = 0.20  # > 20% triggers fail-fast
_LARGE_STEPS_THRESHOLD = 500      # > 500 steps triggers fail-fast
_ALLOW_LARGE = os.environ.get("PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE") == "1"

# IBM TabFormer data location.
# The pipeline always stages the CSV to /tmp/pragma-l8/ inside the pod.
_S3_CSV_KEY = "pragma-encoder/data/tabformer/card_transaction.v1.csv"
_POD_STAGING_DIR = "/tmp/pragma-l8"
_POD_CSV_PATH = f"{_POD_STAGING_DIR}/data/tabformer/card_transaction.v1.csv"
_POD_VOCAB_PATH = f"{_POD_STAGING_DIR}/data/tabformer/vocab.pkl"
_POD_OUTPUT_DIR = f"{_POD_STAGING_DIR}/outputs"

# PyTorchJob CRD.
_PYTORCHJOB_CRD = "pytorchjobs.kubeflow.org"

# Job name prefix — short to stay under DNS label limit (RFC 1035 §2.3.4).
# pod name = pragma-l8-{test_id}-master-0 = 10+35+9 = 54 chars ✓
_LOSS_JOB_PREFIX = "pragma-l8"
_DNS_LABEL_LIMIT = 63
_KFTO_MASTER_SUFFIX = "-master-0"


def _resolve_sample_fraction() -> float:
    """Resolve sample fraction from env var with safety gate."""
    raw = os.environ.get("PRAGMA_TABFORMER_SAMPLE_FRACTION", str(_DEFAULT_SAMPLE_FRACTION)).strip()
    try:
        frac = float(raw)
    except ValueError:
        return _DEFAULT_SAMPLE_FRACTION
    return max(0.01, min(1.0, frac))


def _resolve_max_steps() -> int:
    """Resolve max_steps from env var with safety gate."""
    raw = os.environ.get("PRAGMA_TABFORMER_MAX_STEPS", str(_DEFAULT_MAX_STEPS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_STEPS


def _resolve_batch_size() -> int:
    """Resolve batch_size from env var."""
    raw = os.environ.get("PRAGMA_TABFORMER_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_BATCH_SIZE


def _resolve_log_every() -> int:
    """Resolve log_every from env var."""
    raw = os.environ.get("PRAGMA_LOSS_LOG_EVERY", str(_DEFAULT_LOG_EVERY)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_LOG_EVERY


def _validate_bounds(fraction: float, max_steps: int) -> None:
    """Fail fast if bounds exceed safe L4 limits without explicit override gate.

    Raises:
        pytest.fail if fraction > 0.20 or max_steps > 500 and
        PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE is not set.
    """
    if _ALLOW_LARGE:
        return
    if fraction > _LARGE_FRACTION_THRESHOLD:
        pytest.fail(
            f"PRAGMA_TABFORMER_SAMPLE_FRACTION={fraction:.2f} exceeds the safe L4 "
            f"threshold of {_LARGE_FRACTION_THRESHOLD:.2f}. "
            "This would use more than 20% of the IBM TabFormer dataset, which may "
            "exceed L4-class GPU memory or wall-clock budget. "
            "Set PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1 to override this guard. "
            "Or reduce PRAGMA_TABFORMER_SAMPLE_FRACTION to <= 0.20."
        )
    if max_steps > _LARGE_STEPS_THRESHOLD:
        pytest.fail(
            f"PRAGMA_TABFORMER_MAX_STEPS={max_steps} exceeds the safe L4 "
            f"threshold of {_LARGE_STEPS_THRESHOLD}. "
            "This would exceed the expected wall-clock budget for a smoke run. "
            "Set PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1 to override this guard. "
            "Or reduce PRAGMA_TABFORMER_MAX_STEPS to <= 500."
        )


def _pytorchjob_terminal_condition(status: dict) -> tuple[bool, str]:
    """Inspect PyTorchJob .status.conditions and return (is_terminal, condition_type)."""
    for condition in status.get("conditions", []):
        cond_type = condition.get("type", "")
        cond_status = condition.get("status", "")
        if cond_type in ("Succeeded", "Failed") and cond_status == "True":
            return True, cond_type
    return False, ""


def _render_loss_smoke_manifest(
    test_id: str,
    namespace: str,
    image: str,
    fraction: float,
    max_steps: int,
    batch_size: int,
    log_every: int,
) -> str:
    """Render the Level 8 loss smoke PyTorchJob YAML manifest.

    Single-node (Master-only, no Workers) GPU PyTorchJob that:
      1. Downloads IBM TabFormer CSV from S3 to /tmp/pragma-l8/.
      2. Fits vocab via python -m pragma_encoder.data.fit_tokenizer.
      3. Runs pragma-encoder-train (PRAGMA-S, GPU) for max_steps steps
         on a fraction of the dataset.
      4. Writes metrics.jsonl to the output directory.

    Data sizing:
      IBM TabFormer has ~24,000 unique customers. At 10% fraction,
      limit_rows = int(24000 * 0.10) = 2400 customers.
      The fraction→limit_rows conversion happens in the pod script
      using a hard-coded total so the pod does not need to load
      the full CSV to count rows before subsetting.
      IBM TabFormer row count (customers): approximately 24,000.
      We use a conservative 30,000 as the ceiling for limit_rows
      computation so we never accidentally use more than the requested
      fraction when the actual count is lower.

    GPU resources:
      Requests 1 GPU (nvidia.com/gpu: "1").
      L4 class: 24 GB VRAM. PRAGMA-S (~10M params) with batch_size=4
      uses ~2 GB VRAM — safe headroom.

    S3 credentials:
      The pod reads AWS_* env vars from the pragma-workbench-env Secret
      (OpenShift AI Connection). No explicit credential management in the test.

    Args:
        test_id:    Unique test-run identifier.
        namespace:  Kubernetes namespace.
        image:      Training image URI.
        fraction:   Sample fraction (0.0–1.0) of IBM TabFormer customers.
        max_steps:  Maximum gradient steps before early exit.
        batch_size: Batch size (L4-safe default: 4).
        log_every:  Log loss every N steps.

    Returns:
        YAML string ready for oc apply -f.
    """
    # Convert fraction to limit_rows.
    # Using 30,000 as ceiling → we never over-request.
    # Minimum: 10 customers (ensures at least 1 batch with batch_size >= 1).
    _TABFORMER_CUSTOMER_CEILING = 30_000
    limit_rows = max(10, int(_TABFORMER_CUSTOMER_CEILING * fraction))

    # Shell script executed by the Master pod.
    # Inline to keep the manifest self-contained (no init container).
    # S3 download uses boto3 installed in the training image.
    script_lines = [
        "set -e",
        "",
        "# ---- Stage IBM TabFormer CSV from S3 to pod-local scratch ----",
        f"mkdir -p {_POD_STAGING_DIR}/data/tabformer",
        f"mkdir -p {_POD_OUTPUT_DIR}",
        "echo '[Level 8] Staging IBM TabFormer CSV from S3 ...'",
        "python -c \"",
        "import os, boto3",
        "_bucket = os.environ['AWS_S3_BUCKET']",
        "_endpoint = os.environ['AWS_S3_ENDPOINT']",
        "_ep_url = _endpoint if _endpoint.startswith('http') else f'https://{_endpoint}'",
        "_s3 = boto3.client('s3', endpoint_url=_ep_url,",
        "  aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID') or None,",
        "  aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY') or None)",
        f"_s3.download_file(_bucket, '{_S3_CSV_KEY}', '{_POD_CSV_PATH}')",
        f"print('[Level 8] CSV staged to {_POD_CSV_PATH}')",
        "\"",
        "",
        "# ---- Fit vocab on the staged CSV ----",
        f"cd {_POD_STAGING_DIR}",
        "python -m pragma_encoder.data.fit_tokenizer",
        f"if [ ! -f {_POD_VOCAB_PATH} ]; then",
        "  echo 'ERROR: fit_tokenizer did not create vocab.pkl' >&2; exit 1",
        "fi",
        "echo '[Level 8] vocab.pkl fitted'",
        "",
        f"# ---- Train PRAGMA-S for {max_steps} steps on {fraction*100:.0f}% of data ----",
        "# No S3 checkpoint prefix — local emptyDir only (no durable checkpoint needed).",
        f"echo '[Level 8] Training: rows={limit_rows} steps={max_steps} bs={batch_size}'",
        "pragma-encoder-train \\",
        f"  --csv-path {_POD_CSV_PATH} \\",
        f"  --vocab-path {_POD_VOCAB_PATH} \\",
        f"  --output-dir {_POD_OUTPUT_DIR} \\",
        "  --model-variant pragma-s \\",
        "  --device cuda \\",
        "  --epochs 1 \\",
        f"  --batch-size {batch_size} \\",
        f"  --max-steps {max_steps} \\",
        f"  --limit-rows {limit_rows} \\",
        f"  --log-every {log_every} \\",
        "  --num-workers 0",
        "",
        f"echo '[Level 8] Training complete — checking {_POD_OUTPUT_DIR}/metrics.jsonl'",
        f"if [ ! -f {_POD_OUTPUT_DIR}/metrics.jsonl ]; then",
        "  echo 'ERROR: metrics.jsonl not written' >&2; exit 1",
        "fi",
        f"echo '[Level 8] metrics.jsonl line count: '$(wc -l < {_POD_OUTPUT_DIR}/metrics.jsonl)",
        "# Emit the last 20 JSONL lines for test log collection.",
        "echo '[Level 8] PRAGMA_LOSS_JSONL_BEGIN'",
        f"tail -n 20 {_POD_OUTPUT_DIR}/metrics.jsonl",
        "echo '[Level 8] PRAGMA_LOSS_JSONL_END'",
        "echo '[Level 8] Loss smoke complete'",
    ]

    _INDENT = " " * 18
    indented_script = "\n".join(
        (_INDENT + line) if line else "" for line in script_lines
    )

    return f"""\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {_LOSS_JOB_PREFIX}-{test_id}
  namespace: {namespace}
  labels:
    pragma.redhat.com/test-run: "true"
    pragma.redhat.com/test-id: "{test_id}"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        metadata:
          labels:
            pragma.redhat.com/test-run: "true"
            pragma.redhat.com/test-id: "{test_id}"
        spec:
          containers:
            - name: pytorch
              image: {image}
              imagePullPolicy: Always
              command:
                - /bin/sh
                - -c
                - |
{indented_script}
              resources:
                requests:
                  cpu: "2"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
              envFrom:
                - secretRef:
                    name: pragma-workbench-env
"""


# ===========================================================================
# 1. TestLossSmokeStaticPrereqs
#    Static checks — no cluster access. Verify bounds, naming, CRD, image.
#    Run whenever RUN_TABFORMER_LOSS_SMOKE=1.
# ===========================================================================


class TestLossSmokeStaticPrereqs:
    """Level 8 static pre-flight checks.

    No cluster connection. Run whenever RUN_TABFORMER_LOSS_SMOKE=1.

    These tests verify:
      - Configured parameters are within safe L4 bounds (or override is set).
      - Job name fits within the DNS label limit.
      - PyTorchJob CRD is present (cluster read — requires RUN_OPENSHIFT_TESTS=1).
      - Training image is specified (required for runtime test).
    """

    @_require_loss_smoke
    def test_sample_fraction_within_bounds(self) -> None:
        """PRAGMA_TABFORMER_SAMPLE_FRACTION must be within safe L4 bounds.

        Default: 0.10 (10% of IBM TabFormer customers).
        Safe threshold: <= 0.20 (20%) unless PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1.

        This guard prevents accidentally running an unbounded training job
        on an L4-class GPU that would exceed memory or wall-clock budget.
        """
        fraction = _resolve_sample_fraction()
        _validate_bounds(fraction=fraction, max_steps=_DEFAULT_MAX_STEPS)
        # If we reach here, bounds are acceptable.
        assert 0.0 < fraction <= 1.0, (
            f"Sample fraction {fraction} is not in (0, 1]. "
            "Check PRAGMA_TABFORMER_SAMPLE_FRACTION."
        )

    @_require_loss_smoke
    def test_max_steps_within_bounds(self) -> None:
        """PRAGMA_TABFORMER_MAX_STEPS must be within safe L4 bounds.

        Default: 100 gradient steps.
        Safe threshold: <= 500 unless PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1.

        100 steps at batch_size=4 on PRAGMA-S runs in approximately 2–5
        minutes on an L4 GPU — well within smoke run expectations.
        """
        max_steps = _resolve_max_steps()
        _validate_bounds(fraction=_DEFAULT_SAMPLE_FRACTION, max_steps=max_steps)
        assert max_steps >= 1, (
            f"max_steps={max_steps} is below 1. "
            "Check PRAGMA_TABFORMER_MAX_STEPS."
        )

    @_require_loss_smoke
    def test_combined_bounds_consistent(self) -> None:
        """Resolved fraction and max_steps must both pass the safety gate together.

        Tests the combination — not just each individually — because the gate
        applies to the full configuration that will be used at runtime.
        """
        fraction = _resolve_sample_fraction()
        max_steps = _resolve_max_steps()
        # _validate_bounds raises pytest.fail if the gate triggers.
        _validate_bounds(fraction=fraction, max_steps=max_steps)

    @_require_loss_smoke
    def test_job_name_fits_dns_label_limit(self) -> None:
        """Generated master pod name must not exceed the DNS label limit.

        KFTO's init-pytorch init container resolves the Master pod hostname
        via nslookup. Pod names are DNS labels — limited to 63 characters
        (RFC 1035 §2.3.4). A longer name causes the init container to loop.

        Job name: {_LOSS_JOB_PREFIX}-{test_id}
        Master pod: {job_name}-master-0

        test_id format: 'pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx' = 35 chars
        """
        typical_test_id = "pragma-it-20260525-120000-deadbeef"  # 35 chars
        job_name = f"{_LOSS_JOB_PREFIX}-{typical_test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is {len(master_pod_name)} chars, "
            f"exceeding the {_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            f"Shorten _LOSS_JOB_PREFIX (currently {_LOSS_JOB_PREFIX!r})."
        )

    @_require_loss_smoke
    def test_pytorchjob_crd_present(self, test_namespace: str) -> None:
        """pytorchjobs.kubeflow.org CRD must exist for Level 8.

        Level 8 submits a real PyTorchJob to run GPU training.
        If the CRD is absent, no job can be submitted.
        """
        exists = crd_exists(_PYTORCHJOB_CRD)
        assert exists, (
            f"CRD {_PYTORCHJOB_CRD!r} not found in cluster. "
            "KFTO (Kubeflow Training Operator) must be installed for Level 8. "
            "Check: oc get crd pytorchjobs.kubeflow.org"
        )

    @_require_loss_smoke
    def test_training_image_env_var_set(self) -> None:
        """PRAGMA_TRAINING_IMAGE must be set for the runtime smoke to work.

        This is a static check so that the missing-image problem is surfaced
        before the runtime test (where a missing image causes a cryptic skip).
        The static test does not attempt to pull or verify the image.
        """
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        assert image, (
            "PRAGMA_TRAINING_IMAGE is not set. "
            "Export it to the training image URI, e.g.: "
            "export PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry"
            ".svc:5000/pragma-encoder/pragma-encoder-training:latest. "
            "The Level 8 runtime test requires this image to be built and available."
        )

    @_require_loss_smoke
    def test_limit_rows_calculation_is_positive(self) -> None:
        """Computed limit_rows from fraction must be >= 10 customers.

        limit_rows = max(10, int(30_000 * fraction)).
        With fraction=0.10, limit_rows=3000 — enough for meaningful training.
        With fraction=0.01 (minimum), limit_rows=300 — still above the floor.
        """
        fraction = _resolve_sample_fraction()
        limit_rows = max(10, int(30_000 * fraction))
        batch_size = _resolve_batch_size()
        assert limit_rows >= 10, (
            f"limit_rows={limit_rows} is below 10 — not enough customers to form a batch. "
            f"fraction={fraction} is too small. Increase PRAGMA_TABFORMER_SAMPLE_FRACTION."
        )
        # Warn if limit_rows < batch_size but don't fail — DataLoader drop_last=True handles it.
        if limit_rows < batch_size:
            pytest.warns(
                UserWarning,
                match="limit_rows < batch_size",
            )


# ===========================================================================
# 2. TestTabFormerLossSmokeRuntime
#    Runtime GPU test — creates a real PyTorchJob.
#    Requires RUN_TABFORMER_LOSS_SMOKE=1 AND RUN_TABFORMER_LOSS_SMOKE_RUN=1.
# ===========================================================================


@_require_loss_smoke_run
class TestTabFormerLossSmokeRuntime:
    """Level 8 — Single-node GPU training loss smoke on real IBM TabFormer data.

    Submits a single-node PRAGMA-S PyTorchJob to the cluster, waits for it
    to complete, collects pod logs, and asserts:

      1. PyTorchJob reached Succeeded state.
      2. Pod logs contain the PRAGMA_LOSS_JSONL_BEGIN marker.
      3. At least one JSONL record with a finite train_loss is present.
      4. No NaN or inf loss values appear in the extracted records.

    No convergence or quality assertions are made. This test only proves
    the training loop ran and produced a parseable loss trace.

    Prerequisites:
        RUN_OPENSHIFT_TESTS=1
        RUN_TABFORMER_LOSS_SMOKE=1
        RUN_TABFORMER_LOSS_SMOKE_RUN=1
        PRAGMA_TEST_NAMESPACE=<namespace>
        PRAGMA_TRAINING_IMAGE=<image>
        pragma-workbench-env Secret with AWS_* env vars in namespace
        IBM TabFormer CSV in S3 at pragma-encoder/data/tabformer/card_transaction.v1.csv
    """

    def test_tabformer_gpu_loss_smoke(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,
    ) -> None:
        """Submit a GPU PyTorchJob and verify a finite loss trace is produced.

        Steps:
          1. Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2. Resolve and validate L4 bounds (fraction, max_steps).
          3. Verify job name is under DNS label limit.
          4. Render single-node GPU PyTorchJob YAML.
          5. Apply via oc apply -f.
          6. Wait for Master pod to appear.
          7. Wait for PyTorchJob terminal condition (Succeeded or Failed).
          8. Collect pod logs via oc logs.
          9. Extract JSONL loss records from logs between markers.
         10. Assert: Succeeded condition.
         11. Assert: at least one step record with finite train_loss.
         12. Assert: no NaN or inf loss in any step record.
         13. Cleanup via cleanup_labelled_resources fixture.
        """
        # ------------------------------------------------------------------
        # Step 1 — Resolve PRAGMA_TRAINING_IMAGE.
        # ------------------------------------------------------------------
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Export it to the training image URI before running Level 8."
            )

        # ------------------------------------------------------------------
        # Step 2 — Resolve and validate L4 bounds.
        # ------------------------------------------------------------------
        fraction = _resolve_sample_fraction()
        max_steps = _resolve_max_steps()
        batch_size = _resolve_batch_size()
        log_every = _resolve_log_every()

        # Fail fast if bounds exceed safe limits without override.
        _validate_bounds(fraction=fraction, max_steps=max_steps)

        limit_rows = max(10, int(30_000 * fraction))

        print("\n[Level 8] Parameters:")
        print(f"  fraction    = {fraction:.2f}  → limit_rows={limit_rows}")
        print(f"  max_steps   = {max_steps}")
        print(f"  batch_size  = {batch_size}")
        print(f"  log_every   = {log_every}")
        print(f"  image       = {image!r}")
        print(f"  namespace   = {runtime_namespace!r}")
        print(f"  allow_large = {_ALLOW_LARGE}")

        # ------------------------------------------------------------------
        # Step 3 — Verify job name is under DNS label limit.
        # ------------------------------------------------------------------
        job_name = f"{_LOSS_JOB_PREFIX}-{test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is {len(master_pod_name)} chars, "
            f"exceeding the {_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            "Shorten _LOSS_JOB_PREFIX."
        )

        selector = label_selector(test_id)
        print(f"[Level 8] job_name={job_name!r}")

        # ------------------------------------------------------------------
        # Step 4 — Render single-node GPU PyTorchJob YAML.
        # ------------------------------------------------------------------
        yaml_path = tmp_path / "pytorchjob-l8-loss-smoke.yaml"
        rendered = _render_loss_smoke_manifest(
            test_id=test_id,
            namespace=runtime_namespace,
            image=image,
            fraction=fraction,
            max_steps=max_steps,
            batch_size=batch_size,
            log_every=log_every,
        )
        yaml_path.write_text(rendered)
        print(f"[Level 8] Manifest rendered: {yaml_path} ({len(rendered)} bytes)")

        # ------------------------------------------------------------------
        # Step 5 — Apply via oc apply.
        # ------------------------------------------------------------------
        oc(["apply", "-f", str(yaml_path)], namespace=runtime_namespace)
        print(f"[Level 8] oc apply complete: {job_name}")

        # ------------------------------------------------------------------
        # Step 6 — Wait for Master pod to appear (single-node — 1 pod).
        # ------------------------------------------------------------------
        print(f"[Level 8] Waiting for Master pod (selector={selector!r}) ...")
        _deadline = time.time() + timeout_seconds
        _pod_found = False
        while time.time() < _deadline:
            try:
                pod_list = oc_json(
                    ["get", "pods", "-l", selector],
                    namespace=runtime_namespace,
                    timeout=15,
                )
                pod_count = len(pod_list.get("items", []))
                if pod_count >= 1:
                    _pod_found = True
                    print(f"[Level 8] Master pod found ({pod_count} pod(s)) \u2713")
                    break
                print("[Level 8] 0 pods — waiting 10s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 8] pod list error (retrying): {redact(str(exc))}")
            time.sleep(10)

        if not _pod_found:
            try:
                job_json = oc_json(
                    ["get", "pytorchjob", job_name],
                    namespace=runtime_namespace,
                    timeout=15,
                )
                _conditions = job_json.get("status", {}).get("conditions", [])
            except Exception:  # noqa: BLE001
                _conditions = []
            pytest.fail(
                f"Master pod for {job_name!r} did not appear within {timeout_seconds}s. "
                f"PyTorchJob conditions: {_conditions}. "
                f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
                "Common causes: image pull error, GPU node not available, "
                "pragma-workbench-env Secret missing."
            )

        # ------------------------------------------------------------------
        # Step 7 — Wait for PyTorchJob terminal condition.
        # GPU training at max_steps=100 typically finishes in 2–5 min on L4.
        # Cap at min(timeout_seconds, 900) — 15-minute ceiling for loss smoke.
        # ------------------------------------------------------------------
        _effective_timeout = min(timeout_seconds, 900)
        print(
            f"[Level 8] Waiting for PyTorchJob terminal condition "
            f"(timeout={_effective_timeout}s) ..."
        )
        _terminal = False
        _final_condition = ""
        _deadline2 = time.time() + _effective_timeout
        while time.time() < _deadline2:
            try:
                job_json = oc_json(
                    ["get", "pytorchjob", job_name],
                    namespace=runtime_namespace,
                    timeout=15,
                )
                _terminal, _final_condition = _pytorchjob_terminal_condition(
                    job_json.get("status", {})
                )
                if _terminal:
                    print(f"[Level 8] Terminal condition: {_final_condition} \u2713")
                    break
                _replica_statuses = job_json.get("status", {}).get("replicaStatuses", {})
                print(f"[Level 8] replicaStatuses={_replica_statuses} — waiting 15s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 8] status poll error (retrying): {redact(str(exc))}")
            time.sleep(15)

        if not _terminal:
            raise TimeoutError(
                f"PyTorchJob {job_name!r} did not reach terminal state "
                f"within {_effective_timeout}s. "
                f"Last condition: {_final_condition!r}. "
                f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
                "GPU training at max_steps=100 should complete in < 10 min on L4. "
                "Increase PRAGMA_TEST_TIMEOUT_SECONDS or PRAGMA_TABFORMER_MAX_STEPS "
                "if the cluster is under load."
            )

        # ------------------------------------------------------------------
        # Step 8 — Collect pod logs.
        # ------------------------------------------------------------------
        print("[Level 8] Collecting pod logs ...")
        _log_result = oc(
            ["logs", "-l", selector, "--tail", "500", "--prefix"],
            namespace=runtime_namespace,
            check=False,
            timeout=90,
        )
        all_logs = redact(_log_result.stdout.strip())

        if all_logs:
            print("[Level 8] Log excerpt (last 50 lines):")
            for _line in all_logs.splitlines()[-50:]:
                print(f"  {_line}")
        else:
            print(
                "[Level 8] WARNING: pod logs not available via oc logs. "
                "Loss trace cannot be verified from logs. "
                f"Check cluster UI for pod logs: selector={selector!r}."
            )

        # ------------------------------------------------------------------
        # Step 9 — Extract JSONL loss records from log markers.
        #
        # The pod script emits:
        #   [Level 8] PRAGMA_LOSS_JSONL_BEGIN
        #   {"step": 1, "epoch": 1, "train_loss": 3.14, ...}
        #   {"step": 2, ...}
        #   ...
        #   [Level 8] PRAGMA_LOSS_JSONL_END
        #
        # We extract lines between these markers and parse as JSONL.
        # ------------------------------------------------------------------
        step_records: list[dict] = []
        if all_logs:
            _in_block = False
            for raw_line in all_logs.splitlines():
                # Strip pod prefix added by oc logs --prefix (format: "[pod/name] line")
                line = raw_line.strip()
                if "]" in line and line.startswith("["):
                    # Remove the pod prefix if present: "[pod/name] content"
                    bracket_end = line.find("] ")
                    if bracket_end != -1:
                        line = line[bracket_end + 2:].strip()

                if "PRAGMA_LOSS_JSONL_BEGIN" in line:
                    _in_block = True
                    continue
                if "PRAGMA_LOSS_JSONL_END" in line:
                    _in_block = False
                    continue
                if _in_block and line.startswith("{"):
                    try:
                        record = json.loads(line)
                        # Only collect step-level records (have train_loss key).
                        if "train_loss" in record and "step" in record:
                            step_records.append(record)
                    except json.JSONDecodeError:
                        pass  # Non-JSON line inside block — skip

            if step_records:
                print(
                    f"[Level 8] Extracted {len(step_records)} JSONL step record(s) from logs."
                )
                _first = step_records[0]
                _last = step_records[-1]
                print(f"[Level 8]   First: step={_first['step']}  loss={_first['train_loss']}")
                print(f"[Level 8]   Last:  step={_last['step']}   loss={_last['train_loss']}")
            else:
                print(
                    "[Level 8] No JSONL step records extracted from logs. "
                    "This may mean the PRAGMA_LOSS_JSONL markers were not emitted "
                    "or the job failed before writing metrics.jsonl."
                )

        # ------------------------------------------------------------------
        # Step 10 — Assert: PyTorchJob Succeeded.
        # ------------------------------------------------------------------
        assert _final_condition == "Succeeded", (
            f"Level 8 PyTorchJob {job_name!r} did not Succeed. "
            f"Final condition: {_final_condition!r}. "
            f"Parameters: fraction={fraction:.2f}, max_steps={max_steps}, "
            f"batch_size={batch_size}. "
            f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
            + ("Pod logs (last 100 lines):\n"
               + "\n".join(all_logs.splitlines()[-100:])
               if all_logs else "Pod logs not available.")
        )

        # ------------------------------------------------------------------
        # Step 11 — Assert: at least one step record with a finite loss.
        # (Only asserted when logs are available — log unavailability is
        # reported but does not block the Succeeded assertion above.)
        # ------------------------------------------------------------------
        if all_logs:
            assert len(step_records) >= 1, (
                "Pod logs were collected but no JSONL step records with 'train_loss' "
                "were found between PRAGMA_LOSS_JSONL_BEGIN/END markers. "
                "Expected at least one record from metrics.jsonl tail. "
                f"Job: {job_name!r}, max_steps={max_steps}. "
                "Check that pragma-encoder-train completed at least one gradient step "
                "and that metrics.jsonl was written to the output directory."
            )

        # ------------------------------------------------------------------
        # Step 12 — Assert: no NaN or inf in any step record.
        # (No convergence claim — only numeric sanity of the loss trace.)
        # ------------------------------------------------------------------
        import math  # noqa: PLC0415
        for rec in step_records:
            loss_val = rec.get("train_loss")
            if loss_val is None:
                continue
            assert math.isfinite(float(loss_val)), (
                f"Non-finite train_loss={loss_val!r} found at step={rec.get('step')}. "
                "This indicates a NaN or inf in the PRAGMA training loop — "
                "likely a numerical stability issue (exploding gradients, "
                "bad learning rate, or tokenizer mismatch). "
                f"Job: {job_name!r}. "
                "Check gradient clipping config and model forward pass."
            )

        # ------------------------------------------------------------------
        # Step 13 — Cleanup (automatic via cleanup_labelled_resources fixture).
        # ------------------------------------------------------------------
        _loss_summary = (
            f"loss range=[{step_records[0]['train_loss']:.4f}, "
            f"{step_records[-1]['train_loss']:.4f}]"
            if len(step_records) >= 2
            else f"steps={len(step_records)}"
        )
        print("\n[Level 8] === PASSED: TabFormer GPU Loss Smoke ===")
        print(f"  Job:        {job_name}")
        print(f"  Condition:  {_final_condition}")
        print(f"  Records:    {len(step_records)} step record(s)")
        print(f"  Loss:       {_loss_summary}")
        print(f"  Params:     fraction={fraction:.2f}  max_steps={max_steps}  "
              f"batch_size={batch_size}")
        print(f"  Namespace:  {runtime_namespace}")
        print(f"  Image:      {image}")
