"""Level 3b — Training container batch/v1 Job smoke (opt-in).

Purpose:
  Prove the PRAGMA training container can run a minimal PRAGMA-S training
  loop inside the OpenShift cluster. This is a diagnostic stepping stone
  that confirms the training image and command work correctly in-cluster
  before the full DSPA/KFP pipeline runtime is attempted.

  This is NOT the final OpenShift AI Pipeline runtime.
  This is NOT a PyTorchJob (no KFTO, no DDP, no torchrun).
  This is a single-pod batch/v1 Job that runs exactly what the container
  command would run — proving the image boots, the code runs, and PRAGMA-S
  produces a finite loss within one training step.

Level 3b sits between:
  Level 2:  pipeline compile (local, no cluster, no DSPA)
  Level 3:  OpenShift Pipelines runtime smoke (future, DSPA/KFP)

What it does:
  1. Skip unless RUN_OPENSHIFT_TESTS=1, RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1,
     PRAGMA_TEST_NAMESPACE, and PRAGMA_TRAINING_IMAGE are all set.
  2. Create a ConfigMap containing a synthetic 15-row IBM TabFormer-style CSV.
  3. Create a labelled batch/v1 Job using PRAGMA_TRAINING_IMAGE.
     The Job command (bash -c):
       a. Creates scratch directories.
       b. Copies the mounted CSV to the hardcoded path expected by
          src/data/fit_tokenizer.py (data/tabformer/card_transaction.v1.csv).
       c. Runs src/data/fit_tokenizer.py to produce vocab.pkl.
       d. Copies vocab.pkl to /tmp/pragma-smoke/vocab.pkl.
       e. Runs scripts/train_pragma.py with --max-steps 1.
  4. Waits up to PRAGMA_TEST_TIMEOUT_SECONDS for the Job to complete.
  5. On failure: collects redacted oc describe + pod logs for diagnosis.
  6. Asserts pod logs contain the expected training markers.
  7. Cleanup: handled by the cleanup_labelled_resources fixture.
     Only resources with both test labels are deleted.
     Secrets and ServiceAccounts are never deleted.

Required env vars:
  RUN_OPENSHIFT_TESTS=1            — suite-level opt-in (conftest.py)
  RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 — this file's opt-in gate
  PRAGMA_TEST_NAMESPACE=<ns>       — namespace where substrate is deployed
  PRAGMA_TRAINING_IMAGE=<uri>      — image to use for the smoke Job

Optional env vars:
  PRAGMA_TEST_RUNTIME_NAMESPACE    — namespace for ephemeral resources
                                     (defaults to PRAGMA_TEST_NAMESPACE)
  PRAGMA_TEST_TIMEOUT_SECONDS      — wait timeout (default 300)

Required image contract:
  PRAGMA_TRAINING_IMAGE must be built from openshift/training/Dockerfile.training.
  The image must have src/ and scripts/ baked in at WORKDIR so that:
    scripts/train_pragma.py    is present at WORKDIR
    src/data/fit_tokenizer.py  is present at WORKDIR
    python -c "import src.workbench" succeeds from WORKDIR
  A dependency-only image (e.g. pragma-encoder-workbench without code) will
  fail the smoke Job at the image validation step with a clear error message.

Known limitations:
  - No S3: the checkpoint is written to emptyDir /tmp/pragma-smoke only.
  - No PyTorchJob (no KFTO, no distributed training).
  - No DDP: num-workers=0, no torchrun.

Debug fallback:
  PRAGMA_ALLOW_RUNTIME_GIT_CLONE=1 enables a runtime git-clone fallback
  that clones the repo into the Job container before training. This is off
  by default and is a debug tool only. The intended path is a proper
  training image built from openshift/training/Dockerfile.training.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Sections 2.3, 2.4
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Optional

import pytest

from tests.openshift.oc import label_selector, oc, oc_json, redact


# ---------------------------------------------------------------------------
# Suite-level skip guard
# ---------------------------------------------------------------------------

_TRAINING_JOB_SMOKE_ENABLED = (
    os.environ.get("RUN_OPENSHIFT_TRAINING_JOB_SMOKE") == "1"
)

_require_training_job_smoke = pytest.mark.skipif(
    not _TRAINING_JOB_SMOKE_ENABLED,
    reason=(
        "Training container Job smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 to enable. "
        "Also requires: RUN_OPENSHIFT_TESTS=1, PRAGMA_TEST_NAMESPACE, "
        "and PRAGMA_TRAINING_IMAGE=<image-uri>. "
        "Warning: these tests create short-lived batch/v1 Job, ConfigMap, "
        "and Pod resources in PRAGMA_TEST_RUNTIME_NAMESPACE."
    ),
)


# ---------------------------------------------------------------------------
# Tiny IBM TabFormer-style synthetic CSV fixture (15 rows, 5 customers)
#
# Column names match exactly what TabFormerAdapter and _row_to_fields()
# expect (src/data/tabformer_adapter.py).  The Amount column uses the
# '$XX.XX' format that _parse_amount() strips.  Time uses 'HH:MM' format.
#
# 5 customers × 3 transactions gives train split = 4 customers,
# val split = 1 customer.  With batch-size=1 and max-steps=1, this
# is enough to run one forward + backward pass.
# ---------------------------------------------------------------------------

_SMOKE_CSV = """\
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?
0,0,2020,1,1,12:00,$10.00,Chip Transaction,Store A,New York,NY,5411,,No
0,0,2020,1,2,14:30,$25.50,Swipe Transaction,Store B,Los Angeles,CA,5912,,No
0,0,2020,1,3,09:15,$7.99,Online Transaction,Store C,Houston,TX,5734,,No
1,0,2020,1,4,11:00,$50.00,Chip Transaction,Store D,Seattle,WA,5411,,No
1,0,2020,1,5,16:45,$12.00,Swipe Transaction,Store E,Miami,FL,5912,,No
1,0,2020,1,6,08:30,$99.99,Online Transaction,Store A,New York,NY,5734,,No
2,0,2020,1,7,13:00,$35.00,Chip Transaction,Store B,Los Angeles,CA,5411,,No
2,0,2020,1,8,15:30,$8.50,Swipe Transaction,Store C,Houston,TX,5912,,No
2,0,2020,1,9,10:00,$45.00,Online Transaction,Store D,Seattle,WA,5734,,No
3,0,2020,1,10,12:00,$20.00,Chip Transaction,Store E,Miami,FL,5411,,No
3,0,2020,1,11,14:00,$15.00,Swipe Transaction,Store A,New York,NY,5912,,No
3,0,2020,1,12,16:00,$30.00,Online Transaction,Store B,Los Angeles,CA,5734,,No
4,0,2020,1,13,11:00,$55.00,Chip Transaction,Store C,Houston,TX,5411,,No
4,0,2020,1,14,13:00,$22.00,Swipe Transaction,Store D,Seattle,WA,5912,,No
4,0,2020,1,15,15:00,$18.00,Online Transaction,Store E,Miami,FL,5734,,No
"""

# TabFormer columns that TabFormerAdapter._row_to_fields() accesses by name.
_REQUIRED_CSV_COLUMNS = frozenset({
    "User", "Card", "Year", "Month", "Day", "Time",
    "Amount", "Use Chip", "Merchant Name", "Merchant City",
    "Merchant State", "MCC",
})

# ---------------------------------------------------------------------------
# Shell command that runs inside the Job container.
#
# Steps:
#   0. Validate training image — PRAGMA_TRAINING_IMAGE must have src/ and
#      scripts/ baked in at WORKDIR. Fails immediately with a clear error
#      message if the image is dependency-only (no code).
#   0b. (Debug only) If PRAGMA_ALLOW_RUNTIME_GIT_CLONE=1 is injected via
#      the Job env, clone the repo at runtime and cd into it.
#      This is off by default and must never be the primary path.
#   a. Create scratch directories.
#   b. Copy mounted ConfigMap CSV to the hardcoded path expected by
#      src/data/fit_tokenizer.py (data/tabformer/card_transaction.v1.csv).
#      fit_tokenizer.py uses Path("data/tabformer/card_transaction.v1.csv")
#      relative to the container's WORKDIR.
#   c. Fit the tokenizer on the tiny CSV to produce vocab.pkl.
#      Output is hardcoded to data/tabformer/vocab.pkl.
#   d. Copy vocab.pkl to /tmp/pragma-smoke/ for use by train_pragma.py.
#   e. Run train_pragma.py with --max-steps 1 (one forward + backward step).
#
# Assumptions:
#   - PRAGMA_TRAINING_IMAGE is built from openshift/training/Dockerfile.training.
#   - src/ and scripts/ are baked in at WORKDIR (not a dependency-only image).
#   - PYTHONPATH is not required — src/ is importable from WORKDIR directly.
#   - The image has all Python dependencies installed (torch, pandas, etc.).
# ---------------------------------------------------------------------------

_SMOKE_SHELL = """\
set -ex

echo "[smoke] Validating training image contains PRAGMA repo code ..."
for required_path in scripts/train_pragma.py src/data/fit_tokenizer.py; do
  if [ ! -f "$required_path" ]; then
    echo "ERROR: $required_path not found in image WORKDIR."
    echo "PRAGMA_TRAINING_IMAGE is a dependency-only image, not a training image."
    echo "The image must have src/ and scripts/ baked in at WORKDIR."
    echo "Rebuild using: openshift/training/Dockerfile.training"
    exit 1
  fi
done
python -c "import src.workbench" 2>/dev/null || {
  echo "ERROR: import src.workbench failed."
  echo "PRAGMA_TRAINING_IMAGE is a dependency-only image, not a training image."
  echo "The image must have src/ baked in at WORKDIR with PRAGMA code importable."
  echo "Rebuild using: openshift/training/Dockerfile.training"
  exit 1
}
echo "[smoke] Image validation passed."

if [ "${PRAGMA_ALLOW_RUNTIME_GIT_CLONE:-0}" = "1" ]; then
  echo "[smoke] WARNING: PRAGMA_ALLOW_RUNTIME_GIT_CLONE=1 — cloning repo as debug fallback ..."
  git clone https://github.com/robbybrodie/pragma-encoder.git /workspace/repo \
    --branch pragma-implementation --depth 1
  cd /workspace/repo
fi

echo "[smoke] Creating scratch directories ..."
mkdir -p /tmp/pragma-smoke data/tabformer

echo "[smoke] Staging tiny CSV fixture at hardcoded fit_tokenizer.py path ..."
cp /data/smoke.csv data/tabformer/card_transaction.v1.csv

echo "[smoke] Fitting tokenizer on tiny CSV fixture ..."
python src/data/fit_tokenizer.py

echo "[smoke] Copying vocab.pkl to /tmp/pragma-smoke/ ..."
cp data/tabformer/vocab.pkl /tmp/pragma-smoke/vocab.pkl

echo "[smoke] Starting PRAGMA-S training smoke (max-steps=1) ..."
python scripts/train_pragma.py \
  --csv-path data/tabformer/card_transaction.v1.csv \
  --vocab-path /tmp/pragma-smoke/vocab.pkl \
  --output-dir /tmp/pragma-smoke \
  --model-variant pragma-s \
  --epochs 1 \
  --num-workers 0 \
  --batch-size 1 \
  --max-steps 1

echo "[smoke] Training smoke complete."
"""

# Log markers that confirm the Job executed correctly.
#   "Image validation passed" — image contains src/ and scripts/ at WORKDIR
#   "pragma-s"               — model variant logged by train_pragma.py at startup
#   "Reached --max-steps"    — early-stop message written by train_pragma.py
#                              when global_step >= args.max_steps
_EXPECTED_LOG_MARKERS: tuple[str, ...] = (
    "Image validation passed",
    "pragma-s",
    "Reached --max-steps",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_training_image() -> Optional[str]:
    """Return PRAGMA_TRAINING_IMAGE or None if not set / empty."""
    return os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip() or None


def _get_image_pull_secret() -> Optional[str]:
    """Return the image pull secret name for the smoke Job.

    Reads PRAGMA_IMAGE_PULL_SECRET_NAME.
    Defaults to 'pragma-registry' — the pull secret used by PyTorchJob manifests
    in this repository (openshift/training/pytorchjob-pragma-s.yaml).

    Returns None only if PRAGMA_IMAGE_PULL_SECRET_NAME is explicitly set to
    the empty string, which disables imagePullSecrets on the Job pod spec.
    In most cases the default 'pragma-registry' is correct.
    """
    raw = os.environ.get("PRAGMA_IMAGE_PULL_SECRET_NAME", "pragma-registry").strip()
    return raw or None


def _allow_runtime_git_clone() -> bool:
    """Return True if the runtime git-clone debug fallback is explicitly enabled.

    Reads PRAGMA_ALLOW_RUNTIME_GIT_CLONE.
    Off by default (returns False unless the value is exactly '1').

    When True, the smoke Job shell command will git-clone the repo at runtime
    before running training steps. This is a debug fallback for diagnosing
    dependency-only images — it is not the intended primary path.

    The intended path is a training image built from
    openshift/training/Dockerfile.training that has src/ and scripts/ baked
    in at WORKDIR.
    """
    return os.environ.get("PRAGMA_ALLOW_RUNTIME_GIT_CLONE") == "1"


def _build_configmap(
    name: str,
    namespace: str,
    labels: dict[str, str],
    csv_data: str,
) -> dict:
    """Build a ConfigMap manifest dict for the tiny CSV fixture."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
        },
        "data": {
            "smoke.csv": csv_data,
        },
    }


def _build_job(
    name: str,
    namespace: str,
    labels: dict[str, str],
    image: str,
    configmap_name: str,
    image_pull_secret: Optional[str] = None,
    allow_runtime_git_clone: bool = False,
) -> dict:
    """Build a batch/v1 Job manifest dict for the training smoke.

    Design decisions:
      - restartPolicy: Never  — single attempt, clear failure on error
      - backoffLimit: 0       — no retries (smoke must succeed first time)
      - ttlSecondsAfterFinished: 600 — auto-cleanup after 10 min if fixture
                                       cleanup_labelled_resources fails
      - ConfigMap mounted at /data/ — smoke.csv available as /data/smoke.csv
      - resources: 500m/1Gi requests, 2/4Gi limits — fits CPU-only nodes
      - imagePullSecrets: set from image_pull_secret (default: 'pragma-registry')
      - PRAGMA_ALLOW_RUNTIME_GIT_CLONE: injected as container env var;
        controls the debug git-clone fallback in _SMOKE_SHELL (off by default)
    """
    pod_spec: dict = {
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "pragma-smoke",
                "image": image,
                "command": ["bash", "-c"],
                "args": [_SMOKE_SHELL],
                "env": [
                    {
                        "name": "PRAGMA_ALLOW_RUNTIME_GIT_CLONE",
                        "value": "1" if allow_runtime_git_clone else "0",
                    },
                ],
                "volumeMounts": [
                    {
                        "name": "csv-fixture",
                        "mountPath": "/data",
                        "readOnly": True,
                    },
                ],
                "resources": {
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits":   {"cpu": "2",    "memory": "4Gi"},
                },
            },
        ],
        "volumes": [
            {
                "name": "csv-fixture",
                "configMap": {
                    "name": configmap_name,
                },
            },
        ],
    }

    if image_pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": image_pull_secret}]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {
                    "labels": labels,
                },
                "spec": pod_spec,
            },
        },
    }


def _apply_manifest(
    manifest: dict,
    dest: pathlib.Path,
    namespace: str,
) -> None:
    """Write manifest JSON to dest and apply via oc apply -f.

    Uses the safe oc() wrapper from tests.openshift.oc, which:
      - validates that apply is not a forbidden delete command
      - adds the -n namespace flag
      - redacts output on error
    """
    dest.write_text(json.dumps(manifest, indent=2))
    oc(["apply", "-f", str(dest)], namespace=namespace, timeout=30)


def _wait_for_job(
    job_name: str,
    namespace: str,
    timeout_seconds: int,
) -> bool:
    """Poll until the Job completes (succeeded or failed) or times out.

    Returns:
        True if the Job reached Complete status.
        False if the Job reached Failed status or the wait timed out.

    Polls every 10 seconds. Transient API errors are swallowed to avoid
    false failures due to brief API server hiccups.
    """
    deadline = time.monotonic() + timeout_seconds
    poll_interval = 10

    while time.monotonic() < deadline:
        try:
            status = oc_json(
                ["get", "job", job_name],
                namespace=namespace,
                timeout=15,
            )
            conditions = status.get("status", {}).get("conditions", [])
            for cond in conditions:
                if cond.get("type") == "Complete" and cond.get("status") == "True":
                    return True
                if cond.get("type") == "Failed" and cond.get("status") == "True":
                    return False
        except Exception:  # noqa: BLE001
            pass  # transient API error — keep polling

        time.sleep(poll_interval)

    return False  # timed out without a terminal condition


def _collect_diagnostics(
    job_name: str,
    namespace: str,
    test_id: str,
) -> str:
    """Collect and redact diagnostic output for failure analysis.

    Gathers (in order):
      1. oc describe job <job_name>
      2. oc get pods -l <test-labels>
      3. Tail of pod logs (last 100 lines of the first pod)

    All output is passed through redact() to remove obvious secrets.
    Output is truncated to avoid overwhelming test failure messages.

    Never accesses Secret data or ServiceAccount tokens.
    """
    lines: list[str] = []

    # oc describe job
    try:
        r = oc(
            ["describe", "job", job_name],
            namespace=namespace,
            check=False,
            timeout=20,
        )
        lines.append("=== oc describe job ===")
        lines.append(redact(r.stdout[:2000]))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"=== oc describe job failed: {exc} ===")

    # oc get pods with test labels
    selector = label_selector(test_id)
    try:
        r = oc(
            ["get", "pods", "-l", selector],
            namespace=namespace,
            check=False,
            timeout=20,
        )
        lines.append("=== oc get pods ===")
        lines.append(redact(r.stdout[:1000]))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"=== oc get pods failed: {exc} ===")

    # Pod logs (first pod only, tail 100)
    try:
        pods = oc_json(
            ["get", "pods", "-l", selector],
            namespace=namespace,
            timeout=20,
        )
        pod_list = pods.get("items", [])
        if pod_list:
            pod_name = pod_list[0]["metadata"]["name"]
            r = oc(
                ["logs", pod_name, "--tail=100"],
                namespace=namespace,
                check=False,
                timeout=30,
            )
            lines.append(f"=== pod logs: {pod_name} (tail 100) ===")
            lines.append(redact(r.stdout[-2000:]))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"=== pod logs failed: {exc} ===")

    return "\n".join(lines)


def _get_pod_logs(namespace: str, test_id: str) -> str:
    """Return the full stdout of the first Job pod (redacted).

    Used after the Job succeeds to assert expected log markers are present.
    Returns empty string if no pods are found or logs cannot be fetched.
    """
    selector = label_selector(test_id)
    try:
        pods = oc_json(
            ["get", "pods", "-l", selector],
            namespace=namespace,
            timeout=20,
        )
        pod_list = pods.get("items", [])
        if not pod_list:
            return ""
        pod_name = pod_list[0]["metadata"]["name"]
        r = oc(
            ["logs", pod_name],
            namespace=namespace,
            check=False,
            timeout=60,
        )
        return redact(r.stdout)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Pre-execution checks (no cluster needed — local validation only)
# ---------------------------------------------------------------------------


class TestTrainingJobSmokePrereqs:
    """Level 3b: Pre-execution checks for the training container smoke.

    These tests are all locally executable — they do not require a cluster
    connection. They validate that the environment and fixture are correctly
    configured before the execution test creates any cluster resources.

    All tests skip unless RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1.
    """

    @_require_training_job_smoke
    def test_training_image_env_is_set(self) -> None:
        """PRAGMA_TRAINING_IMAGE must be set and look like an image URI.

        This test FAILS (not skips) when PRAGMA_TRAINING_IMAGE is missing
        while RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1. This makes the missing
        configuration clearly visible rather than silently skipping.

        To fix:
            export PRAGMA_TRAINING_IMAGE=<registry>/<image>:<tag>

        Example (OpenShift internal registry):
            export PRAGMA_TRAINING_IMAGE=\\
              image-registry.openshift-image-registry.svc:5000/\\
              pragma-encoder/training:latest
        """
        image = _get_training_image()
        assert image is not None, (
            "PRAGMA_TRAINING_IMAGE is not set. "
            "The training container smoke requires a built PRAGMA image. "
            "Build the image and export: "
            "export PRAGMA_TRAINING_IMAGE=<registry>/<image>:<tag>. "
            "The image must have the PRAGMA code baked in at its WORKDIR "
            "with all Python dependencies installed."
        )
        assert "/" in image, (
            f"PRAGMA_TRAINING_IMAGE={image!r} does not look like an image URI. "
            "Expected format: <registry>/<repo>/<image>:<tag>. "
            "Example: "
            "image-registry.openshift-image-registry.svc:5000/"
            "pragma-encoder/training:latest"
        )

    @_require_training_job_smoke
    def test_smoke_csv_fixture_has_required_columns(self) -> None:
        """The embedded tiny CSV fixture must have all required TabFormer columns.

        Verifies the fixture before it is mounted in a ConfigMap. A malformed
        fixture CSV would cause the Job to fail at the fit_tokenizer.py step
        with a confusing pandas KeyError, not at the training step.

        Required columns (from src/data/tabformer_adapter.py _row_to_fields()):
            User, Card, Year, Month, Day, Time, Amount, Use Chip,
            Merchant Name, Merchant City, Merchant State, MCC
        """
        lines = _SMOKE_CSV.strip().splitlines()
        assert len(lines) >= 2, (
            "Smoke CSV fixture is too short — need at least header + 1 data row."
        )
        header = set(lines[0].split(","))
        missing = _REQUIRED_CSV_COLUMNS - header
        assert not missing, (
            f"Smoke CSV fixture is missing required columns: {sorted(missing)}. "
            "The CSV must have IBM TabFormer-style column names for "
            "TabFormerAdapter._row_to_fields() to parse correctly. "
            "Reference: src/data/tabformer_adapter.py"
        )

    @_require_training_job_smoke
    def test_smoke_csv_fixture_has_enough_users(self) -> None:
        """The CSV fixture must have ≥3 unique User IDs for train/val split.

        fit_tokenizer.py uses an 80/20 customer split (80% for fitting,
        20% for validation). With 5 customers: 4 training, 1 val.

        With batch-size=1 and max-steps=1, 4 training customers is enough
        for one forward + backward pass before --max-steps halts training.

        Fewer than 3 unique users would leave 0 training customers after
        the 80/20 split with only 1 customer total.
        """
        lines = _SMOKE_CSV.strip().splitlines()
        user_ids = {line.split(",")[0] for line in lines[1:] if line.strip()}
        assert len(user_ids) >= 3, (
            f"Smoke CSV fixture has only {len(user_ids)} unique User ID(s). "
            "Need ≥3 for a train/val split that leaves at least 2 training "
            "customers with batch-size=1 and drop_last=True. "
            "Add more synthetic customer rows to the _SMOKE_CSV constant."
        )

    @_require_training_job_smoke
    def test_smoke_shell_command_contains_expected_steps(self) -> None:
        """The smoke shell command must reference each required step.

        Validates that _SMOKE_SHELL has not accidentally dropped a step
        due to editing.

        Required markers:
            Validating training image contains PRAGMA repo code
                        — image validation header line
            dependency-only image
                        — clear error string when image lacks code
            Dockerfile.training
                        — reference to the fix (build the correct image)
            PRAGMA_ALLOW_RUNTIME_GIT_CLONE
                        — debug fallback gate (must be present but off by default)
            fit_tokenizer.py   — tokenizer fit step
            train_pragma.py    — training step
            --max-steps 1      — smoke is bounded (max-steps=1)
            --model-variant pragma-s — model size is PRAGMA-S
        """
        required_markers = {
            "Validating training image contains PRAGMA repo code",
            "dependency-only image",
            "Dockerfile.training",
            "PRAGMA_ALLOW_RUNTIME_GIT_CLONE",
            "fit_tokenizer.py",
            "train_pragma.py",
            "--max-steps 1",
            "--model-variant pragma-s",
        }
        for marker in required_markers:
            assert marker in _SMOKE_SHELL, (
                f"Smoke shell command is missing required step marker: {marker!r}. "
                "The smoke command must validate the image, reference Dockerfile.training "
                "on failure, gate the debug git-clone fallback behind "
                "PRAGMA_ALLOW_RUNTIME_GIT_CLONE, fit the tokenizer, and run "
                "PRAGMA-S with --max-steps 1. Check the _SMOKE_SHELL constant."
            )


# ---------------------------------------------------------------------------
# Execution smoke
# ---------------------------------------------------------------------------


class TestTrainingJobSmoke:
    """Level 3b: batch/v1 Job training container execution smoke.

    Proves the PRAGMA training image and command work correctly inside
    the OpenShift cluster. This is the diagnostic step before DSPA/KFP
    pipeline runtime execution is attempted.

    All tests skip unless RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1.
    The execution test also skips if PRAGMA_TRAINING_IMAGE is not set.
    """

    @_require_training_job_smoke
    def test_training_job_smoke(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,  # triggers cleanup after test
    ) -> None:
        """Create a labelled batch/v1 Job and verify PRAGMA-S training runs.

        End-to-end smoke flow:

        1. Skip if PRAGMA_TRAINING_IMAGE is not set.

        2. Create a ConfigMap (pragma-smoke-csv-<test_id>) containing
           the synthetic 15-row IBM TabFormer-style CSV fixture.

        3. Create a labelled batch/v1 Job (pragma-smoke-job-<test_id>)
           that mounts the ConfigMap and runs _SMOKE_SHELL:
             a. mkdir -p /tmp/pragma-smoke data/tabformer
             b. cp /data/smoke.csv data/tabformer/card_transaction.v1.csv
             c. python src/data/fit_tokenizer.py   → data/tabformer/vocab.pkl
             d. cp data/tabformer/vocab.pkl /tmp/pragma-smoke/vocab.pkl
             e. python scripts/train_pragma.py --max-steps 1

        4. Wait up to PRAGMA_TEST_TIMEOUT_SECONDS for Job completion.

        5. On failure: collect redacted oc describe + pod logs and fail
           with a clear message including the image used.

        6. Assert pod logs contain:
             "pragma-s"            — model variant confirmed at startup
             "Reached --max-steps" — early-stop message from train_pragma.py

        7. Cleanup: handled by cleanup_labelled_resources fixture.
           Deletes only resources with both test labels:
             pragma.redhat.com/test-run=true
             pragma.redhat.com/test-id=<test_id>
           Never deletes Secrets, ServiceAccounts, or namespaces.

        Resources created (all label-scoped to test_id):
          ConfigMap:  pragma-smoke-csv-<test_id>
          Job:        pragma-smoke-job-<test_id>
          Pod:        (auto-created by the Job controller)
        """
        image = _get_training_image()
        if image is None:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Build and push the PRAGMA training image, then set "
                "PRAGMA_TRAINING_IMAGE=<registry>/<image>:<tag>. "
                "The image must have the PRAGMA code baked in at its "
                "WORKDIR with all Python dependencies installed."
            )

        cm_name  = f"pragma-smoke-csv-{test_id}"
        job_name = f"pragma-smoke-job-{test_id}"

        # -- Step 1: Create ConfigMap with tiny CSV fixture ------------------

        cm_manifest = _build_configmap(
            name=cm_name,
            namespace=runtime_namespace,
            labels=test_labels,
            csv_data=_SMOKE_CSV,
        )
        _apply_manifest(cm_manifest, tmp_path / "smoke-configmap.json", runtime_namespace)

        # -- Step 2: Create labelled batch/v1 Job ----------------------------

        image_pull_secret = _get_image_pull_secret()
        allow_git_clone = _allow_runtime_git_clone()

        job_manifest = _build_job(
            name=job_name,
            namespace=runtime_namespace,
            labels=test_labels,
            image=image,
            configmap_name=cm_name,
            image_pull_secret=image_pull_secret,
            allow_runtime_git_clone=allow_git_clone,
        )
        _apply_manifest(job_manifest, tmp_path / "smoke-job.json", runtime_namespace)

        # -- Step 3: Wait for Job completion ---------------------------------

        succeeded = _wait_for_job(
            job_name=job_name,
            namespace=runtime_namespace,
            timeout_seconds=timeout_seconds,
        )

        # -- Step 4: Collect diagnostics on failure --------------------------

        diagnostics = ""
        if not succeeded:
            diagnostics = _collect_diagnostics(
                job_name=job_name,
                namespace=runtime_namespace,
                test_id=test_id,
            )

        assert succeeded, (
            f"batch/v1 Job {job_name!r} did not succeed within "
            f"{timeout_seconds}s in namespace {runtime_namespace!r}.\n"
            f"Image: {image}\n"
            f"imagePullSecret: {image_pull_secret!r}\n"
            f"PRAGMA_ALLOW_RUNTIME_GIT_CLONE: {'1' if allow_git_clone else '0 (off)'}\n"
            f"If the job failed at image validation, the image lacks src/ or scripts/.\n"
            f"Build a training image using: openshift/training/Dockerfile.training\n"
            f"Command: see _SMOKE_SHELL in "
            f"tests/openshift/test_03b_training_job_smoke.py\n"
            f"Diagnostics (redacted):\n{diagnostics}"
        )

        # -- Step 5: Assert expected log markers are present -----------------

        logs = _get_pod_logs(namespace=runtime_namespace, test_id=test_id)
        logs_lower = logs.lower()

        for marker in _EXPECTED_LOG_MARKERS:
            assert marker.lower() in logs_lower, (
                f"Expected log marker {marker!r} not found in pod logs.\n"
                f"This means either:\n"
                f"  - The training script exited before logging this marker\n"
                f"  - The wrong model variant was used (expected pragma-s)\n"
                f"  - --max-steps 1 was not reached (training did not run)\n"
                f"Image: {image}\n"
                f"Pod logs (tail, redacted):\n{redact(logs[-2000:])}"
            )
