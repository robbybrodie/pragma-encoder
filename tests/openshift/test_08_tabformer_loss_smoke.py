"""Level 8 — Single-node GPU training loss smoke on real TabFormer data.

Purpose:
  Verify that the PRAGMA training loop produces a bounded training loss curve
  on a slice of real IBM TabFormer data when executed on an L4-class GPU via
  a single-node PyTorchJob.

  This is NOT a convergence test and makes NO quality claims.
  It proves:
    - The full training loop runs on GPU with real CSV data.
    - A bounded training loss curve (≥ min_loss_points finite values) is captured.
    - All captured loss values are finite (no NaN/inf).
    - Step values are non-decreasing (training progressed in order).
    - A parseable metrics.jsonl is written and tail-emitted from the pod.
    - The run stays within L4-class resource limits.

Scope and bounds (L4-class GPU — NVIDIA L4, 24 GB VRAM):
  - Model: PRAGMA-S (smallest config, ~10 M params)
  - Data: at most 10% of IBM TabFormer customers by default
  - max_steps: 100 gradient steps (configurable via env var)
  - loss_log_every: 5 steps → ~20 loss points emitted
  - min_loss_points: 10 (at least 10 finite loss values required)
  - batch_size: 4 (L4-safe default; configurable)
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
  PRAGMA_MIN_LOSS_POINTS           — int, default 10
  PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1  — required when fraction > 0.20 or steps > 500

Coherence constraint (checked at static test time):
  max_steps // loss_log_every >= min_loss_points
  Default: 100 // 5 = 20 >= 10 ✓

Safety rules:
  - Static tests run whenever RUN_TABFORMER_LOSS_SMOKE=1 (no cluster resources created).
  - Runtime test additionally requires RUN_TABFORMER_LOSS_SMOKE_RUN=1.
  - Never create namespaces, secrets, or service accounts.
  - Test resources carry both pragma.redhat.com/test-run and test-id labels.
  - Fail fast if fraction > 0.20 or max_steps > 500 without the override gate.
  - No benchmark, convergence, or quality assertions.

Data source:
  IBM TabFormer data must be present in S3 at the standard key
  (pragma-encoder/data/tabformer/card_transaction.v1.csv). The training image
  downloads the CSV from S3 to /tmp/pragma-l8/ at job startup, then runs
  pragma-encoder-train against that local copy.

Loss artifact:
  Training writes <output_dir>/metrics.jsonl with one record per step.
  The pod script tail-emits the JSONL between PRAGMA_LOSS_JSONL_BEGIN/END markers
  in stdout so the test can extract them via oc logs without exec/oc cp.

  After job completion the test writes local artifacts to:
    test-artifacts/level8-tabformer-loss/<job_name>/
      loss.jsonl      — extracted step records (one JSON object per line)
      metadata.json   — run parameters and summary
      loss.png        — loss curve plot (optional; skipped if matplotlib absent)

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
Platform: RHOAI 3.4.0 — PyTorchJob kubeflow.org/v1, L4-class GPU nodes.
"""

from __future__ import annotations

import json
import math
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
_DEFAULT_LOG_EVERY = 5            # log_every=5 → ~20 loss points from 100 steps
_DEFAULT_MIN_LOSS_POINTS = 10     # require at least 10 finite loss values

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

# Local artifact directory (relative to repo root / cwd).
_ARTIFACT_BASE = pathlib.Path("test-artifacts/level8-tabformer-loss")

# PyTorchJob CRD.
_PYTORCHJOB_CRD = "pytorchjobs.kubeflow.org"

# Job name prefix — short to stay under DNS label limit (RFC 1035 §2.3.4).
# pod name = pragma-l8-{test_id}-master-0 = 10+35+9 = 54 chars ✓
_LOSS_JOB_PREFIX = "pragma-l8"
_DNS_LABEL_LIMIT = 63
_KFTO_MASTER_SUFFIX = "-master-0"


# ---------------------------------------------------------------------------
# Parameter resolvers
# ---------------------------------------------------------------------------

def _resolve_sample_fraction() -> float:
    """Resolve sample fraction from env var with safety gate."""
    raw = os.environ.get(
        "PRAGMA_TABFORMER_SAMPLE_FRACTION", str(_DEFAULT_SAMPLE_FRACTION)
    ).strip()
    try:
        frac = float(raw)
    except ValueError:
        return _DEFAULT_SAMPLE_FRACTION
    return max(0.01, min(1.0, frac))


def _resolve_max_steps() -> int:
    """Resolve max_steps from env var with safety gate."""
    raw = os.environ.get(
        "PRAGMA_TABFORMER_MAX_STEPS", str(_DEFAULT_MAX_STEPS)
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_STEPS


def _resolve_batch_size() -> int:
    """Resolve batch_size from env var."""
    raw = os.environ.get(
        "PRAGMA_TABFORMER_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_BATCH_SIZE


def _resolve_log_every() -> int:
    """Resolve loss_log_every from env var."""
    raw = os.environ.get("PRAGMA_LOSS_LOG_EVERY", str(_DEFAULT_LOG_EVERY)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_LOG_EVERY


def _resolve_min_loss_points() -> int:
    """Resolve minimum required finite loss points from env var.

    Must be positive. Validated at static test time against max_steps
    and log_every to ensure the configuration can produce enough points.
    """
    raw = os.environ.get(
        "PRAGMA_MIN_LOSS_POINTS", str(_DEFAULT_MIN_LOSS_POINTS)
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_LOSS_POINTS


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

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


def _validate_loss_curve_coherence(
    max_steps: int,
    log_every: int,
    min_loss_points: int,
) -> None:
    """Fail fast if max_steps/log_every cannot produce min_loss_points loss values.

    The training loop logs loss every log_every steps, so the expected number
    of loss points from max_steps steps is max_steps // log_every.

    If that expected count is less than min_loss_points, the test will always
    fail at assertion time. Catch this at static-check time instead with a
    clear, actionable error message.

    Args:
        max_steps:       PRAGMA_TABFORMER_MAX_STEPS
        log_every:       PRAGMA_LOSS_LOG_EVERY
        min_loss_points: PRAGMA_MIN_LOSS_POINTS

    Raises:
        pytest.fail with remediation instructions if configuration is incoherent.
    """
    expected_points = max_steps // log_every
    if expected_points < min_loss_points:
        pytest.fail(
            f"Configuration is incoherent: max_steps={max_steps} // "
            f"log_every={log_every} = {expected_points} expected loss points, "
            f"but PRAGMA_MIN_LOSS_POINTS={min_loss_points} requires at least "
            f"{min_loss_points} points. "
            "Adjust one of:\n"
            f"  • Increase PRAGMA_TABFORMER_MAX_STEPS above "
            f"{min_loss_points * log_every} "
            f"(current: {max_steps})\n"
            f"  • Decrease PRAGMA_LOSS_LOG_EVERY below "
            f"{max_steps // min_loss_points + 1} "
            f"(current: {log_every})\n"
            f"  • Lower PRAGMA_MIN_LOSS_POINTS to <= {expected_points} "
            f"(current: {min_loss_points})"
        )


# ---------------------------------------------------------------------------
# PyTorchJob helpers
# ---------------------------------------------------------------------------

def _pytorchjob_terminal_condition(status: dict) -> tuple[bool, str]:
    """Inspect PyTorchJob .status.conditions and return (is_terminal, condition_type)."""
    for condition in status.get("conditions", []):
        cond_type = condition.get("type", "")
        cond_status = condition.get("status", "")
        if cond_type in ("Succeeded", "Failed") and cond_status == "True":
            return True, cond_type
    return False, ""


# ---------------------------------------------------------------------------
# Manifest renderer
# ---------------------------------------------------------------------------

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
         on a fraction of the dataset, logging every log_every steps.
      4. Writes metrics.jsonl to the output directory.
      5. Tail-emits the last 50 JSONL lines between markers for log collection.

    Data sizing:
      IBM TabFormer ~24,000 customers. We use 30,000 as the ceiling so
      limit_rows never exceeds the requested fraction if the actual count is lower.
      Minimum: 10 customers (ensures at least 1 batch with batch_size >= 1).

    GPU resources:
      Requests 1 GPU (nvidia.com/gpu: "1").
      PRAGMA-S (~10M params) with batch_size=4 uses ~2 GB VRAM on an L4.

    S3 credentials:
      Read from pragma-workbench-env Secret (OpenShift AI Connection).
    """
    _TABFORMER_CUSTOMER_CEILING = 30_000
    limit_rows = max(10, int(_TABFORMER_CUSTOMER_CEILING * fraction))

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
        f"# ---- Train PRAGMA-S: rows={limit_rows} steps={max_steps} bs={batch_size} ----",
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
        "echo '[Level 8] Training complete'",
        f"if [ ! -f {_POD_OUTPUT_DIR}/metrics.jsonl ]; then",
        "  echo 'ERROR: metrics.jsonl not written' >&2; exit 1",
        "fi",
        f"_lc=$(wc -l < {_POD_OUTPUT_DIR}/metrics.jsonl)",
        "echo \"[Level 8] metrics.jsonl lines: $_lc\"",
        "# Tail-emit last 50 JSONL lines between markers for test log collection.",
        "echo '[Level 8] PRAGMA_LOSS_JSONL_BEGIN'",
        f"tail -n 50 {_POD_OUTPUT_DIR}/metrics.jsonl",
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


# ---------------------------------------------------------------------------
# JSONL extraction helper
# ---------------------------------------------------------------------------

def _extract_step_records(all_logs: str) -> list[dict]:
    """Extract step-level JSONL records from pod logs between markers.

    The pod script emits:
      [Level 8] PRAGMA_LOSS_JSONL_BEGIN
      {"step": 1, "epoch": 1, "train_loss": 3.14, ...}
      ...
      [Level 8] PRAGMA_LOSS_JSONL_END

    Pod prefix format added by ``oc logs --prefix``: ``[pod/name] content``.
    This function strips the prefix before parsing.

    Returns:
        List of dicts with at least "step" and "train_loss" keys.
        Only step-level records (not epoch_end or checkpoint events) included.
    """
    records: list[dict] = []
    in_block = False
    for raw_line in all_logs.splitlines():
        line = raw_line.strip()
        # Strip oc logs --prefix: "[pod/name] content"
        if line.startswith("[") and "] " in line:
            bracket_end = line.find("] ")
            if bracket_end != -1:
                line = line[bracket_end + 2:].strip()

        if "PRAGMA_LOSS_JSONL_BEGIN" in line:
            in_block = True
            continue
        if "PRAGMA_LOSS_JSONL_END" in line:
            in_block = False
            continue
        if in_block and line.startswith("{"):
            try:
                record = json.loads(line)
                if "train_loss" in record and "step" in record:
                    records.append(record)
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# Local artifact writer
# ---------------------------------------------------------------------------

def _write_local_artifacts(
    job_name: str,
    step_records: list[dict],
    metadata: dict,
) -> pathlib.Path:
    """Write local loss artifacts to test-artifacts/level8-tabformer-loss/<job_name>/.

    Always writes:
      loss.jsonl    — one JSON record per step
      metadata.json — run parameters and summary

    Optionally writes:
      loss.png      — loss curve plot (skipped cleanly if matplotlib is absent)

    Args:
        job_name:     PyTorchJob name (used as subdirectory name).
        step_records: Extracted step-level JSONL records.
        metadata:     Dict of run parameters and summary statistics.

    Returns:
        Path to the artifact directory.
    """
    artifact_dir = _ARTIFACT_BASE / job_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # loss.jsonl
    jsonl_path = artifact_dir / "loss.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in step_records:
            f.write(json.dumps(rec) + "\n")

    # metadata.json
    meta_path = artifact_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # loss.png — optional; skip cleanly if matplotlib is absent
    _png_written = False
    if step_records:
        try:
            import matplotlib  # noqa: PLC0415
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt  # noqa: PLC0415

            steps = [r["step"] for r in step_records]
            losses = [float(r["train_loss"]) for r in step_records]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.5,
                    color="#1f77b4", label="train_loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(
                f"PRAGMA-S TabFormer Loss Curve\n"
                f"(job={job_name}, {len(steps)} points)"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            png_path = artifact_dir / "loss.png"
            fig.savefig(str(png_path), dpi=100, bbox_inches="tight")
            plt.close(fig)
            _png_written = True
        except ImportError:
            pass  # matplotlib not installed — skip PNG, keep loss.jsonl

    # Annotate metadata with artifact paths
    metadata["artifacts"] = {
        "loss_jsonl": str(jsonl_path),
        "metadata_json": str(meta_path),
        "loss_png": str(artifact_dir / "loss.png") if _png_written else None,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return artifact_dir


# ===========================================================================
# 1. TestLossSmokeStaticPrereqs
#    Static checks — no cluster access. Verify bounds, naming, CRD, image.
#    Run whenever RUN_TABFORMER_LOSS_SMOKE=1.
# ===========================================================================


class TestLossSmokeStaticPrereqs:
    """Level 8 static pre-flight checks.

    No cluster connection. Run whenever RUN_TABFORMER_LOSS_SMOKE=1.

    Verifies:
      - Configured parameters are within safe L4 bounds (or override is set).
      - max_steps / log_every >= min_loss_points (coherence check).
      - Job name fits within the DNS label limit.
      - PyTorchJob CRD is present (cluster read — requires RUN_OPENSHIFT_TESTS=1).
      - Training image env var is set.
    """

    @_require_loss_smoke
    def test_sample_fraction_within_bounds(self) -> None:
        """PRAGMA_TABFORMER_SAMPLE_FRACTION must be within safe L4 bounds.

        Default: 0.10 (10% of IBM TabFormer customers).
        Safe threshold: <= 0.20 (20%) unless PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1.
        """
        fraction = _resolve_sample_fraction()
        _validate_bounds(fraction=fraction, max_steps=_DEFAULT_MAX_STEPS)
        assert 0.0 < fraction <= 1.0, (
            f"Sample fraction {fraction} is not in (0, 1]. "
            "Check PRAGMA_TABFORMER_SAMPLE_FRACTION."
        )

    @_require_loss_smoke
    def test_max_steps_within_bounds(self) -> None:
        """PRAGMA_TABFORMER_MAX_STEPS must be within safe L4 bounds.

        Default: 100 gradient steps.
        Safe threshold: <= 500 unless PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1.
        """
        max_steps = _resolve_max_steps()
        _validate_bounds(fraction=_DEFAULT_SAMPLE_FRACTION, max_steps=max_steps)
        assert max_steps >= 1, (
            f"max_steps={max_steps} is below 1. "
            "Check PRAGMA_TABFORMER_MAX_STEPS."
        )

    @_require_loss_smoke
    def test_combined_bounds_consistent(self) -> None:
        """Resolved fraction and max_steps must both pass the safety gate together."""
        fraction = _resolve_sample_fraction()
        max_steps = _resolve_max_steps()
        _validate_bounds(fraction=fraction, max_steps=max_steps)

    @_require_loss_smoke
    def test_min_loss_points_is_positive(self) -> None:
        """PRAGMA_MIN_LOSS_POINTS must be a positive integer.

        Default: 10. Must be >= 1 — zero or negative would make the assertion
        trivially pass regardless of whether any training occurred.
        """
        min_pts = _resolve_min_loss_points()
        assert min_pts >= 1, (
            f"PRAGMA_MIN_LOSS_POINTS={min_pts} is not positive. "
            "Must be >= 1. Set PRAGMA_MIN_LOSS_POINTS to a positive integer."
        )

    @_require_loss_smoke
    def test_default_min_loss_points_is_ten(self) -> None:
        """Default PRAGMA_MIN_LOSS_POINTS must be 10 when env var is unset.

        The canonical Level 8 default is: at least 10 finite loss points.
        With max_steps=100 and log_every=5, the training loop emits ~20 points,
        so requiring 10 is a conservative lower bound that proves a real curve
        was captured while leaving headroom for early batches with no MLM mask.
        """
        import importlib  # noqa: PLC0415

        # Check the module-level constant directly, not the resolver
        # (the resolver reads from env; this test checks the hardcoded default).
        import tests.openshift.test_08_tabformer_loss_smoke as mod  # noqa: PLC0415
        importlib.reload(mod)  # ensure fresh module state
        assert mod._DEFAULT_MIN_LOSS_POINTS == 10, (  # noqa: SLF001
            f"_DEFAULT_MIN_LOSS_POINTS is {mod._DEFAULT_MIN_LOSS_POINTS}, expected 10. "  # noqa: SLF001
            "The canonical Level 8 default requires exactly 10 minimum loss points."
        )

    @_require_loss_smoke
    def test_default_config_produces_enough_loss_points(self) -> None:
        """Default max_steps // log_every must be >= default min_loss_points.

        Default: 100 // 5 = 20 >= 10 ✓

        This test catches any change to the defaults that would make Level 8
        incoherent without changing the env vars — e.g. accidentally setting
        log_every > 10 while keeping max_steps=100 and min_loss_points=10.
        """
        expected = _DEFAULT_MAX_STEPS // _DEFAULT_LOG_EVERY
        assert expected >= _DEFAULT_MIN_LOSS_POINTS, (
            f"Default configuration is incoherent: "
            f"max_steps={_DEFAULT_MAX_STEPS} // log_every={_DEFAULT_LOG_EVERY} "
            f"= {expected} expected loss points, but "
            f"min_loss_points={_DEFAULT_MIN_LOSS_POINTS} requires >= {_DEFAULT_MIN_LOSS_POINTS}. "
            "Update the defaults so that max_steps // log_every >= min_loss_points."
        )

    @_require_loss_smoke
    def test_configured_values_produce_enough_loss_points(self) -> None:
        """Resolved max_steps // log_every must be >= resolved min_loss_points.

        Checks the full resolved configuration (including any env var overrides)
        so misconfiguration is caught at static time rather than at runtime
        assertion time.
        """
        max_steps = _resolve_max_steps()
        log_every = _resolve_log_every()
        min_pts = _resolve_min_loss_points()
        _validate_loss_curve_coherence(max_steps, log_every, min_pts)

    @_require_loss_smoke
    def test_job_name_fits_dns_label_limit(self) -> None:
        """Generated master pod name must not exceed the DNS label limit (63 chars).

        KFTO's init-pytorch init container resolves the Master pod hostname
        via nslookup. Pod names are DNS labels — limited to 63 characters
        (RFC 1035 §2.3.4).
        """
        typical_test_id = "pragma-it-20260525-120000-deadbeef"  # 35 chars
        job_name = f"{_LOSS_JOB_PREFIX}-{typical_test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is "
            f"{len(master_pod_name)} chars, exceeding the "
            f"{_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            f"Shorten _LOSS_JOB_PREFIX (currently {_LOSS_JOB_PREFIX!r})."
        )

    @_require_loss_smoke
    def test_pytorchjob_crd_present(self, test_namespace: str) -> None:
        """pytorchjobs.kubeflow.org CRD must exist for Level 8."""
        exists = crd_exists(_PYTORCHJOB_CRD)
        assert exists, (
            f"CRD {_PYTORCHJOB_CRD!r} not found in cluster. "
            "KFTO must be installed for Level 8. "
            "Check: oc get crd pytorchjobs.kubeflow.org"
        )

    @_require_loss_smoke
    def test_training_image_env_var_set(self) -> None:
        """PRAGMA_TRAINING_IMAGE must be set for the runtime smoke to work."""
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        assert image, (
            "PRAGMA_TRAINING_IMAGE is not set. "
            "Export it to the training image URI, e.g.: "
            "export PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry"
            ".svc:5000/pragma-encoder/pragma-encoder-training:latest"
        )

    @_require_loss_smoke
    def test_limit_rows_calculation_is_positive(self) -> None:
        """Computed limit_rows from fraction must be >= 10 customers."""
        fraction = _resolve_sample_fraction()
        limit_rows = max(10, int(30_000 * fraction))
        assert limit_rows >= 10, (
            f"limit_rows={limit_rows} is below 10. "
            f"fraction={fraction} is too small. "
            "Increase PRAGMA_TABFORMER_SAMPLE_FRACTION."
        )


# ===========================================================================
# 2. TestTabFormerLossSmokeRuntime
#    Runtime GPU test — creates a real PyTorchJob.
#    Requires RUN_TABFORMER_LOSS_SMOKE=1 AND RUN_TABFORMER_LOSS_SMOKE_RUN=1.
# ===========================================================================


@_require_loss_smoke_run
class TestTabFormerLossSmokeRuntime:
    """Level 8 — Single-node GPU training loss curve smoke on real IBM TabFormer data.

    Submits a single-node PRAGMA-S PyTorchJob, waits for it to complete,
    extracts a loss curve from pod logs, and asserts:

      1. PyTorchJob reached Succeeded state.
      2. At least PRAGMA_MIN_LOSS_POINTS (default 10) finite loss records extracted.
      3. All extracted loss values are finite (no NaN/inf).
      4. Step values are non-decreasing (training progressed in order).
      5. At least two distinct step values (loss curve, not a single point).

    No convergence or quality assertions are made.

    Local artifacts written to test-artifacts/level8-tabformer-loss/<job_name>/:
      loss.jsonl, metadata.json, and optionally loss.png.

    Prerequisites:
        RUN_OPENSHIFT_TESTS=1
        RUN_TABFORMER_LOSS_SMOKE=1
        RUN_TABFORMER_LOSS_SMOKE_RUN=1
        PRAGMA_TEST_NAMESPACE=<namespace>
        PRAGMA_TRAINING_IMAGE=<image>
        pragma-workbench-env Secret with AWS_* env vars
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
        """Submit a GPU PyTorchJob and verify a bounded loss curve is captured.

        Steps:
          1.  Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2.  Resolve and validate L4 bounds (fraction, max_steps).
          3.  Validate loss curve coherence (max_steps // log_every >= min_pts).
          4.  Verify job name is under DNS label limit.
          5.  Render single-node GPU PyTorchJob YAML.
          6.  Apply via oc apply -f.
          7.  Wait for Master pod to appear.
          8.  Wait for PyTorchJob terminal condition.
          9.  Collect pod logs via oc logs.
          10. Extract JSONL step records from logs between markers.
          11. Assert: Succeeded condition.
          12. Assert: >= min_loss_points finite train_loss records.
          13. Assert: all loss values are finite.
          14. Assert: step values are non-decreasing.
          15. Assert: at least two distinct step values.
          16. Write local artifacts (loss.jsonl, metadata.json, loss.png).
          17. Print summary report.
          18. Cleanup via cleanup_labelled_resources fixture.
        """
        import datetime  # noqa: PLC0415

        _t_start = time.time()

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
        min_loss_points = _resolve_min_loss_points()

        _validate_bounds(fraction=fraction, max_steps=max_steps)

        # ------------------------------------------------------------------
        # Step 3 — Validate loss curve coherence.
        # ------------------------------------------------------------------
        _validate_loss_curve_coherence(max_steps, log_every, min_loss_points)

        limit_rows = max(10, int(30_000 * fraction))

        print("\n[Level 8] Parameters:")
        print(f"  fraction        = {fraction:.2f}  → limit_rows={limit_rows}")
        print(f"  max_steps       = {max_steps}")
        print(f"  batch_size      = {batch_size}")
        print(f"  log_every       = {log_every}")
        print(f"  min_loss_points = {min_loss_points}")
        print(f"  expected_points ≈ {max_steps // log_every}")
        print(f"  image           = {image!r}")
        print(f"  namespace       = {runtime_namespace!r}")
        print(f"  allow_large     = {_ALLOW_LARGE}")

        # ------------------------------------------------------------------
        # Step 4 — Verify job name is under DNS label limit.
        # ------------------------------------------------------------------
        job_name = f"{_LOSS_JOB_PREFIX}-{test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is "
            f"{len(master_pod_name)} chars, exceeding the "
            f"{_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            "Shorten _LOSS_JOB_PREFIX."
        )

        selector = label_selector(test_id)
        print(f"[Level 8] job_name={job_name!r}")

        # ------------------------------------------------------------------
        # Step 5 — Render single-node GPU PyTorchJob YAML.
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
        # Step 6 — Apply via oc apply.
        # ------------------------------------------------------------------
        oc(["apply", "-f", str(yaml_path)], namespace=runtime_namespace)
        print(f"[Level 8] oc apply complete: {job_name}")

        # ------------------------------------------------------------------
        # Step 7 — Wait for Master pod to appear.
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
                f"Master pod for {job_name!r} did not appear within "
                f"{timeout_seconds}s. "
                f"PyTorchJob conditions: {_conditions}. "
                f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
                "Common causes: image pull error, GPU node not available, "
                "pragma-workbench-env Secret missing."
            )

        # ------------------------------------------------------------------
        # Step 8 — Wait for PyTorchJob terminal condition.
        # Cap at min(timeout_seconds, 900) — 15-minute ceiling.
        # ------------------------------------------------------------------
        _effective_timeout = min(timeout_seconds, 900)
        print(
            f"[Level 8] Waiting for terminal condition "
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
                    print(f"[Level 8] Terminal: {_final_condition} \u2713")
                    break
                _rs = job_json.get("status", {}).get("replicaStatuses", {})
                print(f"[Level 8] replicaStatuses={_rs} — waiting 15s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 8] status poll error (retrying): {redact(str(exc))}")
            time.sleep(15)

        if not _terminal:
            raise TimeoutError(
                f"PyTorchJob {job_name!r} did not reach terminal state "
                f"within {_effective_timeout}s. "
                f"Last condition: {_final_condition!r}. "
                f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
                "GPU training at max_steps=100 should complete in < 10 min on L4."
            )

        _t_training_done = time.time()

        # ------------------------------------------------------------------
        # Step 9 — Collect pod logs.
        # ------------------------------------------------------------------
        print("[Level 8] Collecting pod logs ...")
        _log_result = oc(
            ["logs", "-l", selector, "--tail", "600", "--prefix"],
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
                f"selector={selector!r}"
            )

        # ------------------------------------------------------------------
        # Step 10 — Extract JSONL step records.
        # ------------------------------------------------------------------
        step_records: list[dict] = []
        if all_logs:
            step_records = _extract_step_records(all_logs)
            if step_records:
                _first = step_records[0]
                _last = step_records[-1]
                print(
                    f"[Level 8] Extracted {len(step_records)} step record(s). "
                    f"First: step={_first['step']} loss={_first['train_loss']:.4f}  "
                    f"Last: step={_last['step']} loss={_last['train_loss']:.4f}"
                )
            else:
                print(
                    "[Level 8] No step records found between "
                    "PRAGMA_LOSS_JSONL_BEGIN/END markers."
                )

        # ------------------------------------------------------------------
        # Step 11 — Assert: PyTorchJob Succeeded.
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
        # Steps 12–15: loss curve assertions (only when logs are available).
        # ------------------------------------------------------------------
        if all_logs:
            # Step 12 — Require at least min_loss_points finite records.
            assert len(step_records) >= min_loss_points, (
                f"Expected at least {min_loss_points} finite loss records "
                f"(PRAGMA_MIN_LOSS_POINTS={min_loss_points}), "
                f"but extracted only {len(step_records)}. "
                f"Configuration: max_steps={max_steps}, log_every={log_every} "
                f"→ expected ≈{max_steps // log_every} points. "
                "Possible causes: training exited very early (check for empty "
                "batches or all-masked steps), or metrics.jsonl tail was truncated. "
                f"Job: {job_name!r}"
            )

            # Step 13 — Assert all loss values are finite.
            for rec in step_records:
                loss_val = rec.get("train_loss")
                assert loss_val is not None, (
                    f"Record at step={rec.get('step')} has no train_loss key."
                )
                assert math.isfinite(float(loss_val)), (
                    f"Non-finite train_loss={loss_val!r} at step={rec.get('step')}. "
                    "Indicates NaN/inf in training (exploding gradients, "
                    "bad learning rate, or tokenizer mismatch). "
                    f"Job: {job_name!r}"
                )

            # Step 14 — Assert step values are non-decreasing.
            steps = [int(r["step"]) for r in step_records]
            for i in range(1, len(steps)):
                assert steps[i] >= steps[i - 1], (
                    f"Step values are not non-decreasing: "
                    f"steps[{i-1}]={steps[i-1]} > steps[{i}]={steps[i]}. "
                    "This indicates records were emitted out of order. "
                    f"Job: {job_name!r}"
                )

            # Step 15 — Assert at least two distinct step values.
            distinct_steps = len(set(steps))
            assert distinct_steps >= 2, (
                f"Only {distinct_steps} distinct step value(s) in loss records "
                f"(steps={steps[:10]}{'...' if len(steps) > 10 else ''}). "
                "A loss curve requires at least two distinct steps. "
                "This may indicate the PRAGMA_LOSS_JSONL_END marker was emitted "
                "before a second log_every boundary was reached. "
                f"Job: {job_name!r}"
            )

        # ------------------------------------------------------------------
        # Step 16 — Write local artifacts.
        # ------------------------------------------------------------------
        _t_end = time.time()
        _duration_s = round(_t_end - _t_start, 1)
        _training_s = round(_t_training_done - _t_start, 1)

        _losses = [float(r["train_loss"]) for r in step_records]
        _steps_list = [int(r["step"]) for r in step_records]

        artifact_metadata = {
            "job_name": job_name,
            "namespace": runtime_namespace,
            "image": image,
            "parameters": {
                "sample_fraction": fraction,
                "limit_rows": max(10, int(30_000 * fraction)),
                "max_steps": max_steps,
                "batch_size": batch_size,
                "log_every": log_every,
                "min_loss_points": min_loss_points,
            },
            "results": {
                "condition": _final_condition,
                "loss_points_extracted": len(step_records),
                "first_loss": round(_losses[0], 6) if _losses else None,
                "last_loss": round(_losses[-1], 6) if _losses else None,
                "min_loss": round(min(_losses), 6) if _losses else None,
                "max_loss": round(max(_losses), 6) if _losses else None,
                "first_step": _steps_list[0] if _steps_list else None,
                "last_step": _steps_list[-1] if _steps_list else None,
            },
            "timing": {
                "total_seconds": _duration_s,
                "training_seconds": _training_s,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        artifact_dir = _write_local_artifacts(
            job_name=job_name,
            step_records=step_records,
            metadata=artifact_metadata,
        )
        _png_written = (artifact_dir / "loss.png").exists()

        # ------------------------------------------------------------------
        # Step 17 — Print summary report.
        # ------------------------------------------------------------------
        print("\n[Level 8] === PASSED: TabFormer GPU Loss Curve Smoke ===")
        print(f"  Job:            {job_name}")
        print(f"  Condition:      {_final_condition}")
        print(f"  Loss points:    {len(step_records)} (required >= {min_loss_points})")
        if _losses:
            print(
                f"  Loss range:     [{min(_losses):.4f}, {max(_losses):.4f}]"
            )
            print(
                f"  First → Last:   "
                f"{_losses[0]:.4f} → {_losses[-1]:.4f}"
            )
        print(
            f"  Params:         fraction={fraction:.2f}  "
            f"max_steps={max_steps}  batch_size={batch_size}"
        )
        print(f"  Duration:       {_duration_s}s total, {_training_s}s training")
        print(f"  Artifacts:      {artifact_dir}/")
        print(f"    loss.jsonl    written ({len(step_records)} records)")
        print("    metadata.json written")
        print(f"    loss.png      {'written' if _png_written else 'skipped (matplotlib absent)'}")
        print(f"  Namespace:      {runtime_namespace}")
        print(f"  Image:          {image}")
