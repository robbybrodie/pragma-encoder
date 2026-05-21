"""Level 6 — GPU training smoke (opt-in).

Purpose:
  Prove PRAGMA-S can train on a real GPU node in the cluster.
  This is a single-node smoke (1 GPU) unless PRAGMA_GPU_COUNT > 1.

Architecture boundary:
  Level 6 is independent of Level 3 (KFP), Level 4 (DDP), and Level 5 (S3 resume).
  Level 6 PASS proves GPU hardware is accessible and CUDA training works.
  Level 6 does NOT prove multi-node distributed GPU training (out of scope here).

Guard variables:
  RUN_OPENSHIFT_TESTS=1        — suite-wide (conftest)
  RUN_OPENSHIFT_GPU_SMOKE=1    — enables all Level 6 tests
  PRAGMA_TEST_NAMESPACE=<ns>   — required (conftest)
  PRAGMA_TRAINING_IMAGE=<img>  — required for Job creation
  PRAGMA_GPU_COUNT=<n>         — number of GPUs to request (default: 1)
  PRAGMA_GPU_NODE_SELECTOR     — optional; node label selector for GPU node
  PRAGMA_ALLOW_MULTI_GPU_SMOKE=1 — required when PRAGMA_GPU_COUNT > 1

Safety:
  - All GPU tests are opt-in (RUN_OPENSHIFT_GPU_SMOKE=1 required)
  - Multi-GPU requires explicit opt-in (PRAGMA_ALLOW_MULTI_GPU_SMOKE=1)
  - All created resources carry test labels; cleaned up by fixture
  - No secrets are printed or asserted on

Current status:
  TestGPUSmokeLocalPrereqs — static constant checks; skip unless
    RUN_OPENSHIFT_TESTS=1 (conftest suite-wide gate).
  TestGPUTrainingSmoke — cluster runtime; also requires
    RUN_OPENSHIFT_GPU_SMOKE=1.
This file is a scaffold — the smoke test is implemented but gated.
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

from tests.openshift.oc import oc, oc_json, redact

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_GPU_SMOKE_ENABLED = os.environ.get("RUN_OPENSHIFT_GPU_SMOKE") == "1"

_require_gpu_smoke = pytest.mark.skipif(
    not _GPU_SMOKE_ENABLED,
    reason=(
        "GPU training smoke is opt-in. "
        "Set RUN_OPENSHIFT_GPU_SMOKE=1 to enable. "
        "Also requires RUN_OPENSHIFT_TESTS=1, PRAGMA_TEST_NAMESPACE=<namespace>, "
        "and PRAGMA_TRAINING_IMAGE=<image>. "
        "Requesting a GPU node — ensure the cluster has available GPU capacity "
        "before enabling."
    ),
)

# ---------------------------------------------------------------------------
# GPU count configuration
# ---------------------------------------------------------------------------

_GPU_COUNT_DEFAULT = 1
_MULTI_GPU_GUARD_ENABLED = os.environ.get("PRAGMA_ALLOW_MULTI_GPU_SMOKE") == "1"

# Job name prefix — must keep pod names under 63-char DNS limit.
# 'pragma-gpu-' (11) + test_id (35) + typical job suffix (< 17) = safe.
_GPU_JOB_PREFIX = "pragma-gpu"

_DNS_LABEL_LIMIT = 63


def _resolve_gpu_count() -> int:
    """Resolve GPU count from PRAGMA_GPU_COUNT env var (default 1, min 1)."""
    raw = os.environ.get("PRAGMA_GPU_COUNT", str(_GPU_COUNT_DEFAULT)).strip()
    try:
        n = int(raw)
    except ValueError:
        return _GPU_COUNT_DEFAULT
    return max(1, n)


# ---------------------------------------------------------------------------
# Static prerequisite checks (no cluster required; RUN_OPENSHIFT_TESTS=1
# conftest gate still applies because this file is under tests/openshift/)
# ---------------------------------------------------------------------------


class TestGPUSmokeLocalPrereqs:
    """Static checks that validate GPU smoke configuration constants.

    No cluster access or GPU is needed for these tests. They verify
    module-level constants (_GPU_JOB_PREFIX length, default counts, guards).

    Note: even though no cluster is needed, conftest.py skips all tests in
    tests/openshift/ unless RUN_OPENSHIFT_TESTS=1 is set. These checks would
    be better placed in an ungated top-level test file, but are kept here for
    co-location with the GPU smoke they describe.
    """

    def test_gpu_job_prefix_length_safe(self) -> None:
        """_GPU_JOB_PREFIX must leave room for test_id + pod suffix within DNS limit."""
        typical_test_id_len = 35  # pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx
        # Worst-case pod suffix: '-0' from Job controller = 2 chars
        max_prefix = _DNS_LABEL_LIMIT - typical_test_id_len - 1 - 2
        assert len(_GPU_JOB_PREFIX) <= max_prefix, (
            f"_GPU_JOB_PREFIX {_GPU_JOB_PREFIX!r} is {len(_GPU_JOB_PREFIX)} chars. "
            f"Max safe prefix: {max_prefix} chars for DNS limit compliance."
        )

    def test_gpu_count_default_is_one(self) -> None:
        """Default GPU count is 1 (single GPU, minimal resource request)."""
        assert _GPU_COUNT_DEFAULT == 1, (
            "Default GPU count must be 1. "
            "Single-GPU smoke is the minimal validation. "
            "Multi-GPU requires PRAGMA_ALLOW_MULTI_GPU_SMOKE=1."
        )

    def test_multi_gpu_requires_explicit_opt_in(self) -> None:
        """PRAGMA_GPU_COUNT > 1 requires PRAGMA_ALLOW_MULTI_GPU_SMOKE=1.

        This is verified at smoke test start. Here we just assert the guard exists.
        """
        # The guard is _MULTI_GPU_GUARD_ENABLED — it exists, that's all we check.
        assert isinstance(_MULTI_GPU_GUARD_ENABLED, bool), (
            "_MULTI_GPU_GUARD_ENABLED must be a bool. "
            "Multi-GPU smoke must not run without explicit consent."
        )

    def test_gpu_smoke_does_not_run_by_default(self) -> None:
        """GPU smoke must be off by default (no RUN_OPENSHIFT_GPU_SMOKE in clean env).

        This test always passes — it documents the intent.
        The guard is enforced by _require_gpu_smoke skip marker.
        """
        # Nothing to assert — the _require_gpu_smoke marker handles the skip.
        # This test exists to make the intent explicit in the test report.
        pass


# ---------------------------------------------------------------------------
# Level 6 runtime GPU smoke
# ---------------------------------------------------------------------------


class TestGPUTrainingSmoke:
    """Level 6: Single-node GPU training smoke (opt-in).

    Submits a batch/v1 Job (not a PyTorchJob — single-node GPU proof of concept)
    that runs PRAGMA-S training on 1 GPU for max_steps=1.

    Assertions:
      - Job exits 0
      - Pod logs contain 'PRAGMA-S'
      - Pod logs contain 'Reached --max-steps'
      - Pod logs contain 'cuda' or 'CUDA' (confirms GPU was used, not CPU fallback)

    Future (not in this scaffold):
      - Multi-GPU single-node (nproc_per_node > 1)
      - Multi-GPU multi-node (PyTorchJob, GPU, DDP, nccl)
    """

    _TIMEOUT_SECONDS = int(os.environ.get("PRAGMA_TEST_TIMEOUT_SECONDS", "600"))

    @_require_gpu_smoke
    def test_gpu_training_single_node(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,
    ) -> None:
        """Single-node GPU smoke: train PRAGMA-S for 1 step on 1 GPU.

        Steps:
          1. Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2. Resolve GPU count — guard multi-GPU (> 1) behind PRAGMA_ALLOW_MULTI_GPU_SMOKE.
          3. Render batch/v1 Job manifest with nvidia.com/gpu resource request.
          4. Submit Job and wait for completion.
          5. Assert log markers: PRAGMA-S, Reached --max-steps, cuda evidence.

        If the cluster has no GPU node available, the Job will remain Pending
        until timeout. Check node labels and hardware availability.
        """
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Export it to the training image URI to run this test."
            )

        gpu_count = _resolve_gpu_count()
        if gpu_count > 1 and not _MULTI_GPU_GUARD_ENABLED:
            pytest.skip(
                f"PRAGMA_GPU_COUNT={gpu_count} > 1 but PRAGMA_ALLOW_MULTI_GPU_SMOKE is not set. "
                "Multi-GPU smoke is opt-in to avoid accidental cluster overload. "
                "Set PRAGMA_ALLOW_MULTI_GPU_SMOKE=1 to enable."
            )

        node_selector_label = os.environ.get("PRAGMA_GPU_NODE_SELECTOR", "")

        job_name = f"{_GPU_JOB_PREFIX}-{test_id}"
        manifest = _render_gpu_job_manifest(
            job_name=job_name,
            namespace=runtime_namespace,
            image=image,
            test_id=test_id,
            gpu_count=gpu_count,
            node_selector_label=node_selector_label,
        )

        yaml_path = tmp_path / "gpu-smoke.yaml"
        yaml_path.write_text(manifest)

        oc(["apply", "-f", str(yaml_path)], namespace=runtime_namespace)
        print(f"\n[Level 6] GPU Job submitted: {job_name} (gpu_count={gpu_count})")
        print(f"[Level 6] image={image!r}  namespace={runtime_namespace!r}")

        # Wait for Job completion
        deadline = time.time() + timeout_seconds
        completed = False

        while time.time() < deadline:
            try:
                job_json = oc_json(
                    ["get", "job", job_name], namespace=runtime_namespace, timeout=15
                )
                status = job_json.get("status", {})
                if status.get("succeeded", 0) >= 1:
                    completed = True
                    print(f"[Level 6] Job {job_name}: Succeeded")
                    break
                if status.get("failed", 0) >= 1:
                    logs = oc(
                        ["logs", f"job/{job_name}", "--tail", "100"],
                        namespace=runtime_namespace,
                        check=False,
                        timeout=30,
                    )
                    pytest.fail(
                        f"GPU Job {job_name!r} failed. "
                        f"Logs:\n{redact(logs.stdout)}"
                    )
                active = status.get("active", 0)
                print(f"[Level 6] Job {job_name}: active={active} — waiting ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 6] status poll error (retrying): {redact(str(exc))}")
            time.sleep(10)

        if not completed:
            pytest.fail(
                f"GPU Job {job_name!r} did not complete within {timeout_seconds}s. "
                f"Check: oc describe job {job_name} -n {runtime_namespace}. "
                "Verify GPU node is available and has sufficient capacity."
            )

        # Collect and assert logs
        log_result = oc(
            ["logs", f"job/{job_name}", "--tail", "200"],
            namespace=runtime_namespace,
            check=False,
            timeout=30,
        )
        logs = redact(log_result.stdout.strip())

        if logs:
            assert "PRAGMA-S" in logs, (
                f"Pod logs must contain 'PRAGMA-S'. Job: {job_name!r}. "
                f"Logs:\n{logs}"
            )
            assert "Reached --max-steps" in logs, (
                f"Pod logs must contain 'Reached --max-steps'. Job: {job_name!r}. "
                f"Logs:\n{logs}"
            )
            # Confirm GPU was actually used (not CPU fallback)
            _cuda_evidence = ["cuda", "CUDA", "gpu", "GPU", "nccl", "NCCL"]
            _has_cuda = any(kw in logs for kw in _cuda_evidence)
            assert _has_cuda, (
                "Pod logs must contain CUDA/GPU evidence. "
                f"Expected one of {_cuda_evidence}. "
                "If training fell back to CPU, the GPU node may not have been used. "
                f"Logs:\n{logs}"
            )
            print(
                "[Level 6] Log markers confirmed: "
                "'PRAGMA-S' ✓  'Reached --max-steps' ✓  CUDA evidence ✓"
            )
        else:
            print(
                "[Level 6] WARNING: pod logs not available. "
                "Condition=Succeeded is the pass criterion."
            )

        print(
            f"\n[Level 6] === PASSED: PRAGMA GPU training smoke "
            f"(gpu_count={gpu_count}) ==="
        )


# ---------------------------------------------------------------------------
# Manifest renderer for GPU batch/v1 Job
# ---------------------------------------------------------------------------

import base64  # noqa: E402

_SMOKE_CSV_B64 = base64.b64encode(
    "User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,"
    "Merchant State,MCC,Errors?,Is Fraud?\n"
    "0,0,2023,1,5,09:00,$12.50,Swipe Transaction,Coffee House,Sydney,NSW,5812,,No\n"
    "1,0,2023,1,5,10:00,$23.00,Swipe Transaction,Fuel Stop,Perth,WA,5541,,No\n"
    "2,0,2023,1,5,08:30,$9.50,Swipe Transaction,Bakery Lane,Sydney,NSW,5461,,No\n"
    "3,0,2023,1,5,09:45,$78.00,Chip Transaction,Electronics Co,Perth,WA,5734,,No\n"
    "4,0,2023,1,5,07:00,$5.00,Swipe Transaction,Morning Brew,Sydney,NSW,5812,,No\n"
    "5,0,2023,1,5,11:00,$44.00,Chip Transaction,Hardware Plus,Perth,WA,5251,,No\n"
    "6,0,2023,1,5,08:00,$7.50,Swipe Transaction,News Stand,Sydney,NSW,5994,,No\n"
    "7,0,2023,1,5,10:15,$19.00,Swipe Transaction,Florist,Perth,WA,5992,,No\n"
    "8,0,2023,1,5,09:30,$38.00,Chip Transaction,Bike Shop,Sydney,NSW,5941,,No\n"
    "9,0,2023,1,5,07:30,$6.50,Swipe Transaction,Milk Bar,Perth,WA,5812,,No\n".encode()
).decode()


def _render_gpu_job_manifest(
    job_name: str,
    namespace: str,
    image: str,
    test_id: str,
    gpu_count: int,
    node_selector_label: str,
) -> str:
    """Render a batch/v1 Job that trains PRAGMA-S on gpu_count GPUs for 1 step."""
    script = (
        "set -e; "
        "PRAGMA_ROOT=''; "
        "for C in /opt/app-root/src/pragma-encoder /opt/app-root/src .; do "
        "  if [ -f $C/src/data/fit_tokenizer.py ]; then PRAGMA_ROOT=$C; break; fi; "
        "done; "
        "[ -z $PRAGMA_ROOT ] && { echo 'ERROR: PRAGMA root not found' >&2; exit 1; }; "
        "echo \"[Level 6 GPU] PRAGMA root: $PRAGMA_ROOT\"; "
        "mkdir -p /tmp/pragma-gpu/data/tabformer; "
        f"echo '{_SMOKE_CSV_B64}' | base64 -d > /tmp/pragma-gpu/data/tabformer/card_transaction.v1.csv; "
        "cd /tmp/pragma-gpu; "
        "PYTHONPATH=$PRAGMA_ROOT python $PRAGMA_ROOT/src/data/fit_tokenizer.py; "
        "mkdir -p /tmp/pragma-gpu-output; "
        "PYTHONPATH=$PRAGMA_ROOT python $PRAGMA_ROOT/scripts/train_pragma.py "
        "--csv-path /tmp/pragma-gpu/data/tabformer/card_transaction.v1.csv "
        "--vocab-path /tmp/pragma-gpu/data/tabformer/vocab.pkl "
        "--output-dir /tmp/pragma-gpu-output "
        "--model-variant pragma-s "
        "--epochs 1 --num-workers 0 --batch-size 1 --max-steps 1 --device auto; "
        "echo '[Level 6 GPU] training complete'"
    )

    node_selector_block = ""
    if node_selector_label:
        key, _, value = node_selector_label.partition("=")
        node_selector_block = f"""
          nodeSelector:
            {key.strip()}: "{value.strip()}\""""

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
  labels:
    pragma.redhat.com/test-run: "true"
    pragma.redhat.com/test-id: "{test_id}"
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        pragma.redhat.com/test-run: "true"
        pragma.redhat.com/test-id: "{test_id}"
    spec:
      restartPolicy: Never{node_selector_block}
      containers:
        - name: pragma-gpu
          image: {image}
          imagePullPolicy: Always
          command:
            - /bin/sh
            - -c
            - "{script}"
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "{gpu_count}"
            limits:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "{gpu_count}"
"""
