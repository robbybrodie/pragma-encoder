"""Level 3b — Training container batch/v1 Job smoke test.

Purpose:
  Prove that the PRAGMA training container image runs correctly in-cluster
  by submitting a labelled batch/v1 Job with --max-steps 1.

  This is NOT a pipeline runtime test. It does not prove:
    - KFP v2 / DSPA pipeline orchestration
    - Argo workflow execution
    - Any pipeline CRD behaviour

  It proves only that the training container:
    - Is pullable from the cluster registry
    - Starts without error and imports PRAGMA source (src/)
    - Runs fit_tokenizer.py on a minimal synthetic CSV
    - Runs the training script with the PRAGMA-S model config
    - Reaches --max-steps 1 and exits 0

Runtime:
  This suite targets batch/v1 Job resources only. Not KFP v2 PipelineRun,
  Tekton PipelineRun, or PyTorchJob.

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>          e.g. pragma-encoder
  - PRAGMA_TRAINING_IMAGE=<image-uri>
    e.g. image-registry.openshift-image-registry.svc:5000/
         pragma-encoder/pragma-encoder-training:latest

Safety rules:
  - Skip unless both RUN_OPENSHIFT_TESTS=1 and RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1.
  - All created resources carry both test labels (test-run + test-id).
  - Cleanup deletes only labelled resources (via cleanup_labelled_resources fixture).
  - Never prints secret data (env vars, registry tokens, S3 credentials).
  - oc exec is never used for submission — diagnostics only if oc logs fails.
  - Never modifies Argo CD-managed resources.
  - Never deletes secrets, serviceaccounts, or namespaces.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import textwrap
import time

import pytest

# ---------------------------------------------------------------------------
# Suite-level skip guard (file-level; conftest handles RUN_OPENSHIFT_TESTS=1)
# ---------------------------------------------------------------------------

_TRAINING_JOB_SMOKE_ENABLED = (
    os.environ.get("RUN_OPENSHIFT_TRAINING_JOB_SMOKE") == "1"
)

_require_training_job_smoke = pytest.mark.skipif(
    not _TRAINING_JOB_SMOKE_ENABLED,
    reason=(
        "Training container Job smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 to enable. "
        "Also requires PRAGMA_TRAINING_IMAGE=<image-uri> and "
        "PRAGMA_TEST_NAMESPACE=<namespace>."
    ),
)

# ---------------------------------------------------------------------------
# Smoke CSV — minimal synthetic IBM TabFormer dataset
#
# 10 users × 3 transactions = 30 rows.
# fit_tokenizer.py splits users 80/20 ascending by User ID:
#   users 0-7 → train (8 users), users 8-9 → val (2 users).
# Amount format: "$XX.XX" (tabformer_adapter._parse_amount strips "$")
# ---------------------------------------------------------------------------

_SMOKE_CSV = textwrap.dedent("""\
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?
0,0,2023,1,5,09:00,$12.50,Swipe Transaction,Coffee House,Sydney,NSW,5812,,No
0,0,2023,1,6,12:30,$45.00,Chip Transaction,Grocery World,Melbourne,VIC,5411,,No
0,0,2023,1,7,18:00,$8.75,Swipe Transaction,Fast Bites,Brisbane,QLD,5812,,No
1,0,2023,1,5,10:00,$23.00,Swipe Transaction,Fuel Stop,Perth,WA,5541,,No
1,0,2023,1,6,14:00,$67.50,Chip Transaction,Supermart,Adelaide,SA,5411,,No
1,0,2023,1,7,19:00,$15.00,Swipe Transaction,Pizza Place,Hobart,TAS,5812,,No
2,0,2023,1,5,08:30,$9.50,Swipe Transaction,Bakery Lane,Sydney,NSW,5461,,No
2,0,2023,1,6,11:00,$34.00,Chip Transaction,Dept Store,Melbourne,VIC,5311,,No
2,0,2023,1,7,17:30,$5.25,Swipe Transaction,Snack Bar,Brisbane,QLD,5812,,No
3,0,2023,1,5,09:45,$78.00,Chip Transaction,Electronics Co,Perth,WA,5734,,No
3,0,2023,1,6,13:30,$22.00,Swipe Transaction,Bookshop,Adelaide,SA,5942,,No
3,0,2023,1,7,20:00,$11.50,Swipe Transaction,Cafe Nero,Hobart,TAS,5812,,No
4,0,2023,1,5,07:00,$5.00,Swipe Transaction,Morning Brew,Sydney,NSW,5812,,No
4,0,2023,1,6,10:30,$150.00,Chip Transaction,Fashion Store,Melbourne,VIC,5621,,No
4,0,2023,1,7,16:00,$28.75,Swipe Transaction,Thai Kitchen,Brisbane,QLD,5812,,No
5,0,2023,1,5,11:00,$44.00,Chip Transaction,Hardware Plus,Perth,WA,5251,,No
5,0,2023,1,6,15:00,$18.50,Swipe Transaction,Juice Bar,Adelaide,SA,5812,,No
5,0,2023,1,7,21:00,$92.00,Chip Transaction,Sports Gear,Hobart,TAS,5941,,No
6,0,2023,1,5,08:00,$7.50,Swipe Transaction,News Stand,Sydney,NSW,5994,,No
6,0,2023,1,6,12:00,$55.00,Chip Transaction,Pharmacy,Melbourne,VIC,5912,,No
6,0,2023,1,7,18:30,$33.00,Swipe Transaction,Italian Rest,Brisbane,QLD,5812,,No
7,0,2023,1,5,10:15,$19.00,Swipe Transaction,Florist,Perth,WA,5992,,No
7,0,2023,1,6,14:30,$62.00,Chip Transaction,Furniture Co,Adelaide,SA,5712,,No
7,0,2023,1,7,19:30,$14.25,Swipe Transaction,Taco Truck,Hobart,TAS,5812,,No
8,0,2023,1,5,09:30,$38.00,Chip Transaction,Bike Shop,Sydney,NSW,5941,,No
8,0,2023,1,6,13:00,$8.00,Swipe Transaction,Hot Dog Stand,Melbourne,VIC,5812,,No
8,0,2023,1,7,17:00,$125.00,Chip Transaction,Jewellery Box,Brisbane,QLD,5944,,No
9,0,2023,1,5,07:30,$6.50,Swipe Transaction,Milk Bar,Perth,WA,5812,,No
9,0,2023,1,6,11:30,$41.00,Chip Transaction,Auto Parts,Adelaide,SA,5533,,No
9,0,2023,1,7,20:30,$16.75,Swipe Transaction,Sushi Bar,Hobart,TAS,5812,,No
""")

_REQUIRED_CSV_COLUMNS = {
    "User", "Card", "Year", "Month", "Day", "Time",
    "Amount", "Use Chip", "Merchant Name", "Merchant City",
    "Merchant State", "MCC", "Errors?", "Is Fraud?",
}

# ---------------------------------------------------------------------------
# Smoke shell script
#
# Runs inside the training container as: bash -c <_SMOKE_SHELL>
#
# Steps:
#   1. Validate that required source files are present in the image WORKDIR.
#   2. Verify that src/ imports work (catches stale images missing src/).
#   3. Set up data directory — copy mounted ConfigMap CSV to the hardcoded
#      fit_tokenizer.py path (data/tabformer/card_transaction.v1.csv).
#   4. Run fit_tokenizer.py to build vocab.pkl.
#   5. Run train_pragma.py with --max-steps 1 on CPU.
#
# fit_tokenizer.py hardcoded paths (relative to CWD):
#   reads:  data/tabformer/card_transaction.v1.csv
#   writes: data/tabformer/vocab.pkl
# ---------------------------------------------------------------------------

_SMOKE_SHELL = textwrap.dedent("""\
    set -ex

    echo "[smoke] === PRAGMA-S Level 3b container smoke ==="
    echo "[smoke] Image: $PRAGMA_TRAINING_IMAGE_REF"
    echo "[smoke] Namespace: $PRAGMA_NAMESPACE"
    echo ""

    echo "[smoke] --- Step 1: Validating required source files ---"
    for required_path in scripts/train_pragma.py src/pragma_encoder/data/fit_tokenizer.py src/pragma_encoder/model/pragma.py; do
        if [ ! -f "$required_path" ]; then
            echo "ERROR: $required_path not found in image WORKDIR $(pwd)."
            echo "       The training image may be stale or missing PRAGMA source."
            exit 1
        fi
        echo "[smoke] OK: $required_path"
    done

    echo "[smoke] --- Step 2: Validating src/ imports ---"
    python -c "import pragma_encoder.model; import pragma_encoder.encoders; import pragma_encoder.tokenizer" || {
        echo "ERROR: Core PRAGMA imports failed. Image may be missing src/."
        exit 1
    }
    python -c "import pragma_encoder.workbench" || {
        echo "ERROR: import pragma_encoder.workbench failed. Image may be stale."
        exit 1
    }
    echo "[smoke] OK: all PRAGMA source imports passed."

    echo "[smoke] --- Step 3: Setting up data directory ---"
    mkdir -p data/tabformer
    cp /data/smoke.csv data/tabformer/card_transaction.v1.csv
    echo "[smoke] CSV rows: $(wc -l < data/tabformer/card_transaction.v1.csv) (including header)"

    echo "[smoke] --- Step 4: Fitting tokenizer ---"
    python src/pragma_encoder/data/fit_tokenizer.py
    if [ ! -f data/tabformer/vocab.pkl ]; then
        echo "ERROR: vocab.pkl was not created by fit_tokenizer.py"
        exit 1
    fi
    echo "[smoke] OK: vocab.pkl created."

    echo "[smoke] --- Step 5: PRAGMA-S training (--max-steps 1) ---"
    python scripts/train_pragma.py \
        --csv-path data/tabformer/card_transaction.v1.csv \
        --vocab-path data/tabformer/vocab.pkl \
        --output-dir /tmp/pragma-smoke-output \
        --model-variant pragma-s \
        --epochs 1 \
        --num-workers 0 \
        --batch-size 1 \
        --max-steps 1 \
        --device cpu

    echo "[smoke] === Level 3b container smoke PASSED ==="
""")


# ---------------------------------------------------------------------------
# Helpers — manifest builders
# ---------------------------------------------------------------------------

def _configmap_manifest(
    name: str,
    namespace: str,
    labels: dict[str, str],
) -> dict:
    """Build a ConfigMap manifest embedding _SMOKE_CSV."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {**labels, "app": "pragma-training-smoke"},
        },
        "data": {
            "smoke.csv": _SMOKE_CSV,
        },
    }


def _job_manifest(
    name: str,
    namespace: str,
    labels: dict[str, str],
    image_uri: str,
    configmap_name: str,
    timeout_seconds: int,
) -> dict:
    """Build a batch/v1 Job manifest for the training container smoke test.

    The Job:
    - Mounts the smoke CSV ConfigMap at /data/smoke.csv via subPath.
    - Runs _SMOKE_SHELL as bash -c inside the training container.
    - Uses serviceAccountName: pragma-encoder-training (anyuid SCC).
    - Sets activeDeadlineSeconds to timeout_seconds - 30s for early abort.
    - backoffLimit: 0 so the Job fails immediately on error.
    """
    active_deadline = max(60, timeout_seconds - 30)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {**labels, "app": "pragma-training-smoke"},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": active_deadline,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {
                    "labels": {**labels, "app": "pragma-training-smoke"},
                },
                "spec": {
                    "serviceAccountName": "pragma-encoder-training",
                    "restartPolicy": "Never",
                    "volumes": [
                        {
                            "name": "smoke-csv",
                            "configMap": {
                                "name": configmap_name,
                                "items": [
                                    {"key": "smoke.csv", "path": "smoke.csv"},
                                ],
                            },
                        },
                    ],
                    "containers": [
                        {
                            "name": "training",
                            "image": image_uri,
                            "imagePullPolicy": "Always",
                            "command": ["bash", "-c", _SMOKE_SHELL],
                            "env": [
                                {
                                    "name": "PRAGMA_TRAINING_IMAGE_REF",
                                    "value": image_uri,
                                },
                                {
                                    "name": "PRAGMA_NAMESPACE",
                                    "value": namespace,
                                },
                            ],
                            "volumeMounts": [
                                {
                                    "name": "smoke-csv",
                                    "mountPath": "/data/smoke.csv",
                                    "subPath": "smoke.csv",
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "2Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"},
                            },
                        },
                    ],
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers — cluster operations
# ---------------------------------------------------------------------------

def _oc(*args: str, namespace: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run oc with explicit namespace. Returns CompletedProcess."""
    cmd = ["oc", *args, "-n", namespace]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )


def _apply_manifest(manifest: dict, namespace: str) -> None:
    """Serialise manifest to JSON and apply via oc apply."""
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as tmp:
        json.dump(manifest, tmp)
        tmp_path = tmp.name
    try:
        result = _oc("apply", "-f", tmp_path, namespace=namespace)
        print(result.stdout.strip())
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def _wait_for_job(
    job_name: str,
    namespace: str,
    timeout_seconds: int,
    poll_interval: int = 10,
) -> tuple[str, dict]:
    """Poll a batch/v1 Job until it reaches a terminal state.

    Returns:
        (terminal_state, status_dict) where terminal_state is one of:
        "Complete", "Failed", or "Timeout".
    """
    deadline = time.monotonic() + timeout_seconds
    last_status: dict = {}

    while time.monotonic() < deadline:
        result = _oc(
            "get", "job", job_name, "-o", "json",
            namespace=namespace,
            check=False,
        )
        if result.returncode != 0:
            print(f"[wait] oc get job returned {result.returncode}: {result.stderr.strip()}")
            time.sleep(poll_interval)
            continue

        try:
            job_obj = json.loads(result.stdout)
        except json.JSONDecodeError:
            time.sleep(poll_interval)
            continue

        last_status = job_obj.get("status", {})
        conditions = last_status.get("conditions", [])

        for cond in conditions:
            if cond.get("type") == "Complete" and cond.get("status") == "True":
                return "Complete", last_status
            if cond.get("type") == "Failed" and cond.get("status") == "True":
                return "Failed", last_status

        # Also check counters (conditions not always set immediately)
        if last_status.get("succeeded", 0) >= 1:
            return "Complete", last_status
        if last_status.get("failed", 0) >= 1:
            return "Failed", last_status

        elapsed = int(time.monotonic() - (deadline - timeout_seconds))
        print(
            f"[wait] Job {job_name}: active={last_status.get('active', 0)} "
            f"succeeded={last_status.get('succeeded', 0)} "
            f"failed={last_status.get('failed', 0)} "
            f"elapsed={elapsed}s"
        )
        time.sleep(poll_interval)

    return "Timeout", last_status


def _collect_pod_logs(
    namespace: str,
    label_selector: str,
    max_lines: int = 200,
) -> str:
    """Collect logs from all pods matching label_selector.

    Returns combined log text. Never raises — returns empty string on error.
    """
    result = _oc(
        "get", "pods",
        "-l", label_selector,
        "-o", "jsonpath={.items[*].metadata.name}",
        namespace=namespace,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "(no pods found)"

    pod_names = result.stdout.strip().split()
    all_logs: list[str] = []

    for pod_name in pod_names:
        log_result = _oc(
            "logs", pod_name,
            "--tail", str(max_lines),
            namespace=namespace,
            check=False,
        )
        pod_logs = log_result.stdout.strip()
        if pod_logs:
            all_logs.append(f"--- logs from pod {pod_name} ---\n{pod_logs}")
        elif log_result.stderr.strip():
            all_logs.append(
                f"--- pod {pod_name}: oc logs error: {log_result.stderr.strip()} ---"
            )

    return "\n".join(all_logs) if all_logs else "(no log output)"


def _diagnostics(
    job_name: str,
    namespace: str,
    label_selector: str,
) -> str:
    """Collect failure diagnostics: job describe, pod list, pod logs."""
    lines: list[str] = ["", "=== Failure diagnostics ==="]

    # oc describe job
    desc = _oc("describe", "job", job_name, namespace=namespace, check=False)
    lines.append(f"\n--- oc describe job {job_name} ---")
    lines.append(desc.stdout.strip() or "(no output)")
    if desc.stderr.strip():
        lines.append(f"stderr: {desc.stderr.strip()}")

    # oc get pods
    pods = _oc(
        "get", "pods", "-l", label_selector,
        "-o", "wide",
        namespace=namespace,
        check=False,
    )
    lines.append(f"\n--- oc get pods -l {label_selector} ---")
    lines.append(pods.stdout.strip() or "(no pods)")

    # pod logs
    logs = _collect_pod_logs(namespace, label_selector)
    lines.append("\n--- pod logs (last 200 lines each) ---")
    lines.append(logs)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TestTrainingJobSmokePrereqs — local tests (no cluster, no network)
# ---------------------------------------------------------------------------


class TestTrainingJobSmokePrereqs:
    """Pre-flight checks that run locally without cluster access.

    These tests verify the test infrastructure and smoke CSV are correct
    before the cluster test runs. All four are non-cluster tests that
    execute even when RUN_OPENSHIFT_TRAINING_JOB_SMOKE=0.
    """

    def test_smoke_csv_has_required_columns(self) -> None:
        """_SMOKE_CSV must have all TabFormer-required column headers.

        tabformer_adapter.py parses: User, Card, Year, Month, Day, Time,
        Amount, Use Chip, Merchant Name, Merchant City, Merchant State,
        MCC, Errors?, Is Fraud?. Missing columns → KeyError at runtime.
        """
        header_line = _SMOKE_CSV.splitlines()[0]
        actual_columns = {col.strip() for col in header_line.split(",")}
        missing = _REQUIRED_CSV_COLUMNS - actual_columns
        assert not missing, (
            f"_SMOKE_CSV is missing required TabFormer columns: {missing}. "
            f"These columns are required by tabformer_adapter.py. "
            f"Header found: {header_line!r}"
        )

    def test_smoke_csv_has_sufficient_users(self) -> None:
        """_SMOKE_CSV must have ≥5 distinct users for a non-empty 80% training split.

        fit_tokenizer.py splits users by ascending User ID: 80% train, 20% val.
        With <5 users the train set may be empty, causing fit() to raise.
        The smoke CSV uses 10 users → 8 train, 2 val.
        """
        rows = _SMOKE_CSV.strip().splitlines()[1:]  # skip header
        users = {row.split(",")[0].strip() for row in rows if row.strip()}
        assert len(users) >= 5, (
            f"_SMOKE_CSV must have ≥5 distinct users for the 80/20 split. "
            f"Found {len(users)} users: {sorted(users)}. "
            "fit_tokenizer.py will produce an empty train set with fewer users."
        )

    def test_smoke_shell_references_required_commands(self) -> None:
        """_SMOKE_SHELL must reference all required training commands.

        The smoke shell must:
          1. Call fit_tokenizer.py (fits tokenizer on the smoke CSV)
          2. Call train_pragma.py (runs training with --max-steps 1)
          3. Reference --max-steps 1 (ensures the training stops early)
          4. Reference --model-variant (ensures PRAGMA-S config is used)
        """
        assert "fit_tokenizer.py" in _SMOKE_SHELL, (
            "_SMOKE_SHELL must call src/pragma_encoder/data/fit_tokenizer.py. "
            "The tokenizer must be fitted before training can start."
        )
        assert "train_pragma.py" in _SMOKE_SHELL, (
            "_SMOKE_SHELL must call scripts/train_pragma.py."
        )
        assert "--max-steps 1" in _SMOKE_SHELL, (
            "_SMOKE_SHELL must pass --max-steps 1 to stop training after 1 step. "
            "Without this, the smoke test would run a full epoch."
        )
        assert "--model-variant" in _SMOKE_SHELL, (
            "_SMOKE_SHELL must pass --model-variant to select PRAGMA-S config. "
            "Without this, the default model variant may be used."
        )

    def test_job_manifest_has_required_fields(self) -> None:
        """_job_manifest() must produce a valid batch/v1 Job with required fields.

        Validates the manifest structure before any cluster interaction.
        A malformed manifest would cause oc apply to fail with an unhelpful error.
        """
        labels = {
            "pragma.redhat.com/test-run": "true",
            "pragma.redhat.com/test-id": "pragma-it-test-001",
        }
        manifest = _job_manifest(
            name="pragma-smoke-test-001",
            namespace="pragma-encoder",
            labels=labels,
            image_uri="image-registry.example.com/pragma-training:latest",
            configmap_name="pragma-smoke-csv-test-001",
            timeout_seconds=300,
        )

        assert manifest["apiVersion"] == "batch/v1", "Job must use batch/v1 API"
        assert manifest["kind"] == "Job", "Kind must be Job"
        assert manifest["spec"]["backoffLimit"] == 0, (
            "backoffLimit must be 0 — smoke test must not retry on failure"
        )

        pod_spec = manifest["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "pragma-encoder-training", (
            "Job pod must use pragma-encoder-training SA for anyuid SCC"
        )
        assert pod_spec["restartPolicy"] == "Never", (
            "restartPolicy must be Never (required for batch/v1 Job)"
        )

        containers = pod_spec["containers"]
        assert len(containers) == 1, "Exactly one container in the smoke Job"
        container = containers[0]
        assert container["command"] == ["bash", "-c", _SMOKE_SHELL], (
            "Container command must be ['bash', '-c', _SMOKE_SHELL]"
        )

        volumes = pod_spec.get("volumes", [])
        assert any(v["name"] == "smoke-csv" for v in volumes), (
            "Job spec must include a 'smoke-csv' volume for the CSV ConfigMap"
        )

        vol_mounts = container.get("volumeMounts", [])
        assert any(
            vm["mountPath"] == "/data/smoke.csv" and vm.get("subPath") == "smoke.csv"
            for vm in vol_mounts
        ), (
            "Container must mount smoke-csv at /data/smoke.csv via subPath='smoke.csv'. "
            "The smoke shell copies /data/smoke.csv to data/tabformer/card_transaction.v1.csv."
        )

        # Labels must propagate to pod template
        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        assert "pragma.redhat.com/test-run" in pod_labels, (
            "Pod template must carry pragma.redhat.com/test-run label for cleanup"
        )
        assert "pragma.redhat.com/test-id" in pod_labels, (
            "Pod template must carry pragma.redhat.com/test-id label for cleanup"
        )


# ---------------------------------------------------------------------------
# TestTrainingJobSmoke — cluster integration test (opt-in)
# ---------------------------------------------------------------------------


class TestTrainingJobSmoke:
    """Level 3b: PRAGMA training container batch/v1 Job smoke.

    Submits a labelled batch/v1 Job to the cluster, waits for completion,
    collects logs, and asserts the training ran successfully.

    Skip guards:
      - conftest.py: skipif RUN_OPENSHIFT_TESTS != "1" (suite-level)
      - _require_training_job_smoke: skipif RUN_OPENSHIFT_TRAINING_JOB_SMOKE != "1"

    Cleanup:
      The cleanup_labelled_resources fixture (conftest.py) deletes all resources
      carrying test labels after the test, regardless of pass/fail.
    """

    @_require_training_job_smoke
    def test_training_job_smoke(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        cleanup_labelled_resources: None,
    ) -> None:
        """Submit a batch/v1 Job with the training image and assert it runs.

        Steps:
          1. Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2. Create a ConfigMap embedding the smoke CSV.
          3. Submit a batch/v1 Job with the CSV mounted at /data/smoke.csv.
          4. Wait for Job completion up to timeout_seconds.
          5. On any failure: collect and print diagnostics (no secrets).
          6. Assert terminal state is Complete (not Failed, not Timeout).
          7. Assert pod logs contain "PRAGMA-S" and "Reached --max-steps".
        """
        # ------------------------------------------------------------------
        # Step 1 — resolve training image
        # ------------------------------------------------------------------
        image_uri = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image_uri:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Set it to the cluster-internal image URI, e.g.: "
                "image-registry.openshift-image-registry.svc:5000/"
                "pragma-encoder/pragma-encoder-training:latest"
            )

        job_name = f"pragma-smoke-{test_id}"
        cm_name = f"pragma-smoke-csv-{test_id}"
        label_selector = (
            f"pragma.redhat.com/test-run=true,"
            f"pragma.redhat.com/test-id={test_id}"
        )

        print(f"\n[Level 3b] test_id     = {test_id}")
        print(f"[Level 3b] namespace   = {runtime_namespace}")
        print(f"[Level 3b] image       = {image_uri}")
        print(f"[Level 3b] job         = {job_name}")
        print(f"[Level 3b] configmap   = {cm_name}")
        print(f"[Level 3b] timeout     = {timeout_seconds}s")

        # ------------------------------------------------------------------
        # Step 2 — create ConfigMap with smoke CSV
        # ------------------------------------------------------------------
        print("\n[Level 3b] Creating smoke CSV ConfigMap ...")
        cm_manifest = _configmap_manifest(
            name=cm_name,
            namespace=runtime_namespace,
            labels=test_labels,
        )
        _apply_manifest(cm_manifest, namespace=runtime_namespace)
        print(f"[Level 3b] ConfigMap {cm_name} created.")

        # ------------------------------------------------------------------
        # Step 3 — submit batch/v1 Job
        # ------------------------------------------------------------------
        print("[Level 3b] Submitting batch/v1 Job ...")
        job_manifest_dict = _job_manifest(
            name=job_name,
            namespace=runtime_namespace,
            labels=test_labels,
            image_uri=image_uri,
            configmap_name=cm_name,
            timeout_seconds=timeout_seconds,
        )
        _apply_manifest(job_manifest_dict, namespace=runtime_namespace)
        print(f"[Level 3b] Job {job_name} submitted.")

        # ------------------------------------------------------------------
        # Step 4 — wait for terminal state
        # ------------------------------------------------------------------
        print(f"[Level 3b] Waiting up to {timeout_seconds}s for Job to complete ...")
        terminal_state, job_status = _wait_for_job(
            job_name=job_name,
            namespace=runtime_namespace,
            timeout_seconds=timeout_seconds,
            poll_interval=15,
        )
        print(f"[Level 3b] Job terminal state: {terminal_state}")

        # ------------------------------------------------------------------
        # Step 5 — collect logs (always; needed for pass confirmation too)
        # ------------------------------------------------------------------
        print("[Level 3b] Collecting pod logs ...")
        all_logs = _collect_pod_logs(
            namespace=runtime_namespace,
            label_selector=label_selector,
        )
        print("[Level 3b] Log excerpt (last ~50 lines):")
        log_lines = all_logs.splitlines()
        for line in log_lines[-50:]:
            print(f"  {line}")

        # ------------------------------------------------------------------
        # Step 6 — assert terminal state is Complete
        # ------------------------------------------------------------------
        if terminal_state != "Complete":
            diag = _diagnostics(job_name, runtime_namespace, label_selector)
            pytest.fail(
                f"Training Job {job_name!r} did not complete successfully.\n"
                f"Terminal state: {terminal_state}\n"
                f"Job status: {job_status}\n"
                f"{diag}\n"
                f"\nFull logs:\n{all_logs}"
            )

        # ------------------------------------------------------------------
        # Step 7 — assert log markers
        # ------------------------------------------------------------------
        assert "PRAGMA-S" in all_logs, (
            "Pod logs must contain 'PRAGMA-S' — the model size confirmation.\n"
            "Expected from: logger.info(f\"{args.model_variant.upper()}: {n_params:,} parameters\")\n"
            f"Full logs:\n{all_logs}"
        )
        assert "Reached --max-steps" in all_logs, (
            "Pod logs must contain 'Reached --max-steps' — the early-stop marker.\n"
            "Expected from: logger.info(f\"Reached --max-steps {args.max_steps}; stopping early.\")\n"
            f"Full logs:\n{all_logs}"
        )

        print("\n[Level 3b] === PASSED: PRAGMA training container smoke complete ===")
        print(f"  Job:    {job_name}  →  {terminal_state}")
        print(f"  Image:  {image_uri}")
        print("  Markers confirmed: 'PRAGMA-S' ✓  'Reached --max-steps' ✓")
