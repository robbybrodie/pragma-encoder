"""Level 4 — PyTorchJob two-node distributed training smoke (opt-in).

Purpose:
  Prove the Level 4 distributed training path end-to-end:
    - PyTorchJob CRD is available (KFTO operator installed)
    - Master (rank 0) and Worker (rank 1) pods are created
    - Both ranks initialise the DDP process group via torchrun
    - WORLD_SIZE=2 / RANK=0 / RANK=1 visible via KFTO env injection
    - Tiny PRAGMA-S training reaches --max-steps 1 on both ranks
    - Rank 0 logs completion; rank 1 skips upload (no S3 in smoke)
    - Test resources carry both test labels; cleanup is label-scoped

Architecture boundary:
  Level 4 is independent of Level 3 (DSPA/KFP v2 pipeline smoke).
  Level 4 PASS proves distributed PyTorchJob execution.
  Level 4 does NOT prove KFP v2 pipeline orchestration (Level 3).

  The runtime smoke (test 7) uses a purpose-built inline manifest —
  NOT the production pytorchjob-pragma-s-2node.yaml — because the
  production manifest requires S3 credentials, runtime git clone, and
  GPU hardware that are not available or appropriate for smoke testing.

  The smoke manifest follows the same KFTO pattern as the production manifest:
    - apiVersion: kubeflow.org/v1 / kind: PyTorchJob
    - Master (rank 0) + Worker (rank 1) replicas
    - torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$RANK
    - KFTO injects: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    - Image: PRAGMA_TRAINING_IMAGE (source + deps baked in, no git clone)
    - Data: inline 30-row TabFormer CSV (no S3, no init container)
    - CPU only (--device cpu, no GPU resource request)
    - --max-steps 1 for fast exit

Known limitation (TD-006):
  --resume does not work with per-pod emptyDir on restart.
  This smoke uses --max-steps 1 and omits --resume so no checkpoint
  is created. TD-006 remains open; its resolution is Level 5.

Guard variables:
  RUN_OPENSHIFT_TESTS=1       — suite-wide (conftest)
  RUN_PYTORCHJOB_TESTS=1      — enables all Level 4 tests (1-7)
  RUN_PYTORCHJOB_SMOKE=1      — additionally enables runtime job creation (test 7)
  PRAGMA_TEST_NAMESPACE=<ns>  — required (conftest)
  PRAGMA_TRAINING_IMAGE=<img> — required for test 7 runtime smoke

Tests 1-6 require only RUN_PYTORCHJOB_TESTS=1 (read-only, no cluster resources created).
Test 7 additionally requires RUN_PYTORCHJOB_SMOKE=1 (creates a short-lived PyTorchJob).
"""

from __future__ import annotations

import base64
import os
import pathlib
import time

import pytest

from tests.openshift.oc import crd_exists, label_selector, oc, oc_json, redact

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_PYTORCHJOB_ENABLED = os.environ.get("RUN_PYTORCHJOB_TESTS") == "1"
_PYTORCHJOB_SMOKE_ENABLED = os.environ.get("RUN_PYTORCHJOB_SMOKE") == "1"

_require_pytorchjob = pytest.mark.skipif(
    not _PYTORCHJOB_ENABLED,
    reason=(
        "PyTorchJob tests are opt-in. "
        "Set RUN_PYTORCHJOB_TESTS=1 to enable static manifest checks and "
        "the CRD presence test. Also requires RUN_OPENSHIFT_TESTS=1 and "
        "PRAGMA_TEST_NAMESPACE=<namespace>."
    ),
)

_require_pytorchjob_smoke = pytest.mark.skipif(
    not (_PYTORCHJOB_ENABLED and _PYTORCHJOB_SMOKE_ENABLED),
    reason=(
        "PyTorchJob runtime smoke is opt-in. "
        "Set RUN_PYTORCHJOB_TESTS=1 and RUN_PYTORCHJOB_SMOKE=1 to enable. "
        "Also requires PRAGMA_TRAINING_IMAGE=<image> and "
        "PRAGMA_TEST_NAMESPACE=<namespace>. "
        "This test creates a short-lived PyTorchJob in "
        "PRAGMA_TEST_RUNTIME_NAMESPACE."
    ),
)

# Path to the two-node manifest committed to the repo.
_TWO_NODE_MANIFEST = pathlib.Path("openshift/training/pytorchjob-pragma-s-2node.yaml")

# PyTorchJob CRD installed by KFTO.
_PYTORCHJOB_CRD = "pytorchjobs.kubeflow.org"


# ---------------------------------------------------------------------------
# Smoke manifest helpers
# ---------------------------------------------------------------------------

# Inline TabFormer CSV — 30 rows, 10 users × 3 transactions.
# Same dataset as Level 3b (test_03b_training_job_smoke.py) and Level 3
# (pipeline/pragma_smoke_pipeline.py).  Single-quoted heredoc in the shell
# script prevents dollar-sign expansion on Amount values like $12.50.
_SMOKE_CSV_ROWS = """\
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
9,0,2023,1,7,20:30,$16.75,Swipe Transaction,Sushi Bar,Hobart,TAS,5812,,No"""

# Base64-encoded CSV — avoids heredoc indentation issues in YAML block scalars.
# A heredoc terminator must appear at the start of a line; when a shell script is
# embedded in a YAML block scalar at N-space indent, the terminator would also be
# indented N spaces and the shell would never see it as a terminator.
# echo "<b64>" | base64 -d is a single line — no indentation dependency.
_SMOKE_CSV_B64: str = base64.b64encode(_SMOKE_CSV_ROWS.encode()).decode()


def _render_smoke_manifest(test_id: str, namespace: str, image: str) -> str:
    """Render the smoke PyTorchJob YAML manifest with test-specific values.

    Both Master and Worker replicas run an identical shell script that:
      1. Locates the PRAGMA project root in the training image.
      2. Writes the inline 30-row TabFormer CSV to /tmp/pragma-smoke/.
      3. Runs fit_tokenizer.py (cwd=/tmp/pragma-smoke/) to build vocab.pkl.
      4. Runs torchrun with KFTO-injected RANK / MASTER_ADDR / MASTER_PORT.

    The manifest is structurally equivalent to the production two-node
    manifest but uses:
      - PRAGMA_TRAINING_IMAGE instead of the workbench image
      - Inline CSV instead of S3 init container
      - CPU only (no GPU resource request or node selector)
      - --max-steps 1 for fast exit
      - No --resume (TD-006 avoidance)

    Args:
        test_id:   Unique test-run identifier (used in resource name and labels).
        namespace: Kubernetes namespace to deploy the PyTorchJob into.
        image:     Training image URI (must have PRAGMA source baked in).

    Returns:
        YAML string ready for ``oc apply -f``.
    """
    # Shell script run by both Master and Worker.
    # Line indentation: 18 spaces for YAML block scalar under "- |".
    # Shell variables use $VAR (not ${VAR}) to avoid Python f-string confusion.
    # Single-quoted heredoc 'CSVEOF' prevents shell from expanding $12.50 etc.
    script_lines = [
        "set -e",
        "",
        "# Locate PRAGMA project root baked into the training image.",
        "PRAGMA_ROOT=''",
        "for CANDIDATE in /opt/app-root/src/pragma-encoder /opt/app-root/src /pragma-encoder .; do",
        "  if [ -f $CANDIDATE/src/data/fit_tokenizer.py ]; then",
        "    PRAGMA_ROOT=$CANDIDATE",
        "    break",
        "  fi",
        "done",
        "if [ -z $PRAGMA_ROOT ]; then",
        "  echo 'ERROR: Cannot find PRAGMA project root in training image' >&2; exit 1",
        "fi",
        "echo \"[Level 4 smoke] PRAGMA root: $PRAGMA_ROOT\"",
        "",
        "# Write inline 30-row TabFormer CSV — base64 avoids heredoc/YAML indent issues.",
        "mkdir -p /tmp/pragma-smoke/data/tabformer",
        f"echo '{_SMOKE_CSV_B64}' | base64 -d > /tmp/pragma-smoke/data/tabformer/card_transaction.v1.csv",
        "echo \"[Level 4 smoke] CSV written ($(wc -l < /tmp/pragma-smoke/data/tabformer/card_transaction.v1.csv) lines)\"",
        "",
        "# Fit tokenizer — must run from /tmp/pragma-smoke/ (hardcoded relative paths).",
        "cd /tmp/pragma-smoke",
        "PYTHONPATH=$PRAGMA_ROOT python $PRAGMA_ROOT/src/data/fit_tokenizer.py",
        "if [ ! -f /tmp/pragma-smoke/data/tabformer/vocab.pkl ]; then",
        "  echo 'ERROR: fit_tokenizer.py did not create vocab.pkl' >&2; exit 1",
        "fi",
        "echo '[Level 4 smoke] vocab.pkl created'",
        "",
        "# Run PRAGMA-S distributed training via torchrun.",
        "# KFTO injects: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.",
        "echo \"[Level 4 smoke] torchrun RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR\"",
        "mkdir -p /tmp/pragma-smoke-output",
        "PYTHONPATH=$PRAGMA_ROOT torchrun \\",
        "  --nnodes=2 \\",
        "  --nproc_per_node=1 \\",
        "  --node_rank=$RANK \\",
        "  --master_addr=$MASTER_ADDR \\",
        "  --master_port=$MASTER_PORT \\",
        "  $PRAGMA_ROOT/scripts/train_pragma.py \\",
        "  --csv-path /tmp/pragma-smoke/data/tabformer/card_transaction.v1.csv \\",
        "  --vocab-path /tmp/pragma-smoke/data/tabformer/vocab.pkl \\",
        "  --output-dir /tmp/pragma-smoke-output \\",
        "  --model-variant pragma-s \\",
        "  --epochs 1 \\",
        "  --num-workers 0 \\",
        "  --batch-size 1 \\",
        "  --max-steps 1 \\",
        "  --device cpu",
        "echo '[Level 4 smoke] torchrun complete'",
    ]

    # Indent each non-empty script line by 18 spaces for the YAML block scalar.
    _INDENT = " " * 18
    indented_script = "\n".join(
        (_INDENT + line) if line else "" for line in script_lines
    )

    return f"""\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pragma-pytorchjob-smoke-{test_id}
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
                  cpu: "500m"
                  memory: "2Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi"
    Worker:
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
                  cpu: "500m"
                  memory: "2Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi"
"""


def _pytorchjob_terminal_condition(status: dict) -> tuple[bool, str]:
    """Inspect PyTorchJob .status.conditions and return (is_terminal, condition_type).

    A PyTorchJob is terminal when any condition has type=Succeeded or type=Failed
    with status=True.

    Args:
        status: The .status dict from oc_json output.

    Returns:
        (True, "Succeeded") — job succeeded.
        (True, "Failed")    — job failed.
        (False, "")         — job still running or conditions not yet set.
    """
    for condition in status.get("conditions", []):
        cond_type = condition.get("type", "")
        cond_status = condition.get("status", "")
        if cond_type in ("Succeeded", "Failed") and cond_status == "True":
            return True, cond_type
    return False, ""


# ===========================================================================
# 1. TestPyTorchJobCRD
#    Cluster read-only check — confirms KFTO is installed.
#    Runs when RUN_PYTORCHJOB_TESTS=1. No resources created.
# ===========================================================================


class TestPyTorchJobCRD:
    """Level 4 prereq: verify PyTorchJob CRD is installed (KFTO required).

    1 test — cluster read-only.
    Run when RUN_PYTORCHJOB_TESTS=1.

    Purpose: fail early with a clear message if KFTO is not installed,
    before any resource creation is attempted.
    """

    @_require_pytorchjob
    def test_pytorchjob_crd_exists_when_enabled(
        self, test_namespace: str
    ) -> None:
        """pytorchjobs.kubeflow.org CRD must exist when RUN_PYTORCHJOB_TESTS=1.

        KFTO (Kubeflow Training Operator) installs the PyTorchJob CRD
        (kubeflow.org/v1). Without this CRD, no PyTorchJob can be submitted
        and Level 4 is impossible.

        This test runs before any resource creation so a missing CRD
        produces a clear, early failure with actionable steps rather
        than a cryptic apply error later.

        RHOAI 3.x (OpenShift AI) includes KFTO as a component of the
        Training Operator. Verify it is enabled in the RHOAI DataScienceCluster.

        If this test fails:
          1. Check: oc get crd pytorchjobs.kubeflow.org
          2. Verify KFTO is enabled in the RHOAI DataScienceCluster spec.
          3. Confirm RHOAI version >= 2.x (KFTO requires a recent release).
          4. Contact the cluster admin if KFTO is not installed or enabled.
        """
        exists = crd_exists(_PYTORCHJOB_CRD)

        assert exists, (
            f"CRD {_PYTORCHJOB_CRD!r} not found in cluster. "
            "KFTO (Kubeflow Training Operator) must be installed for Level 4. "
            "Check: oc get crd pytorchjobs.kubeflow.org. "
            "Verify KFTO is enabled in the RHOAI DataScienceCluster spec."
        )

        print(f"\n[Level 4] CRD {_PYTORCHJOB_CRD!r}: present \u2713")


# ===========================================================================
# 2-6. TestTwoNodeManifest
#    Static manifest structure checks — read-only, no cluster access.
#    5 tests. Run when RUN_PYTORCHJOB_TESTS=1.
# ===========================================================================


class TestTwoNodeManifest:
    """Level 4: PyTorchJob two-node manifest structural correctness (read-only).

    5 tests — no cluster access needed.
    Run when RUN_PYTORCHJOB_TESTS=1.

    Purpose: verify the committed production manifest is structurally
    correct before any cluster interaction. These tests catch manifest
    regressions (removed replicas, added PVCs, lost TD-006 documentation)
    without requiring a cluster connection.
    """

    @_require_pytorchjob
    def test_two_node_manifest_exists(self) -> None:
        """openshift/training/pytorchjob-pragma-s-2node.yaml must exist.

        This manifest is the canonical two-node DDP training definition.
        If it is missing, the distributed training path cannot be verified
        and the static structure tests cannot run.
        """
        assert _TWO_NODE_MANIFEST.exists(), (
            f"Two-node manifest {_TWO_NODE_MANIFEST} not found. "
            "This file must exist in the repository for Level 4 tests."
        )

    @_require_pytorchjob
    def test_two_node_manifest_has_master_and_worker(self) -> None:
        """The manifest must define both a Master replica and a Worker replica.

        KFTO PyTorchJob two-node topology:
          pytorchReplicaSpecs.Master (rank 0) — rendezvous initiator
          pytorchReplicaSpecs.Worker (rank 1) — connects to Master

        KFTO injects MASTER_ADDR and MASTER_PORT into every replica pod.
        If either replica type is missing, the DDP ring cannot form and
        distributed training silently degrades to single-node.

        Expected manifest structure:
          spec.pytorchReplicaSpecs.Master.replicas: 1
          spec.pytorchReplicaSpecs.Worker.replicas: 1
        """
        content = _TWO_NODE_MANIFEST.read_text()

        assert "Master" in content, (
            "Two-node manifest does not define a Master replica. "
            "Expected: spec.pytorchReplicaSpecs.Master with replicas: 1. "
            "Master (rank 0) is required for the DDP rendezvous."
        )
        assert "Worker" in content, (
            "Two-node manifest does not define a Worker replica. "
            "Expected: spec.pytorchReplicaSpecs.Worker with replicas: 1. "
            "Worker (rank 1) connects to Master for DDP coordination."
        )

    @_require_pytorchjob
    def test_two_node_manifest_has_no_canonical_pvc(self) -> None:
        """The manifest must not use PVC as canonical persistent storage.

        The two-node manifest uses per-pod emptyDir for local scratch space
        and S3 as the durable artifact store. PVCs are avoided because:
          - ReadWriteOnce PVCs cannot be mounted by two pods simultaneously.
          - RWX PVCs require shared storage infrastructure not assumed here.
          - S3 is the correct canonical checkpoint location for multi-node DDP.

        A PVC definition in the manifest would indicate a regression to the
        single-node storage model and would cause scheduling conflicts when
        two pods attempt to bind the same RWO volume on different nodes.

        emptyDir is expected and allowed for /workspace (local scratch) and
        /dev/shm (shared memory for DDP communication).

        Reference: docs/tech-debt.md TD-006.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        assert "kind: PersistentVolumeClaim" not in content, (
            "Two-node manifest defines a PersistentVolumeClaim resource. "
            "Multi-node PyTorchJobs must use per-pod emptyDir + S3 storage. "
            "PVCs with ReadWriteOnce cause scheduling conflicts on multi-node runs. "
            "See docs/tech-debt.md TD-006."
        )
        assert "volumeClaimTemplates" not in content, (
            "Two-node manifest uses volumeClaimTemplates (StatefulSet pattern). "
            "Multi-node PyTorchJobs must use per-pod emptyDir + S3 storage. "
            "See docs/tech-debt.md TD-006."
        )

    @_require_pytorchjob
    def test_two_node_manifest_mentions_torchrun_or_distributed(self) -> None:
        """The manifest must reference distributed training configuration.

        A valid two-node DDP manifest must mention at least one of:
          - torchrun (the PyTorch distributed process launcher)
          - WORLD_SIZE (environment variable set by the distributed framework)
          - MASTER_ADDR (rendezvous address injected by KFTO into every pod)

        If none are present, the manifest does not configure DDP correctly
        and both pods would train independently (no parameter synchronisation),
        silently producing incorrect results rather than true distributed training.

        torchrun is the recommended launcher for KFTO PyTorchJobs.
        KFTO injects MASTER_ADDR and MASTER_PORT into every replica pod.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        has_torchrun = "torchrun" in content
        has_world_size = "WORLD_SIZE" in content
        has_master_addr = "MASTER_ADDR" in content

        assert has_torchrun or has_world_size or has_master_addr, (
            "Two-node manifest does not reference torchrun, WORLD_SIZE, or MASTER_ADDR. "
            "A valid DDP manifest must configure the distributed launcher. "
            "Expected at least one of: "
            "torchrun (launcher), WORLD_SIZE (env), MASTER_ADDR (rendezvous). "
            "Without these, pods train independently — no DDP synchronisation."
        )

    @_require_pytorchjob
    def test_two_node_manifest_warns_about_td006(self) -> None:
        """The manifest must document the TD-006 --resume limitation.

        TD-006 (docs/tech-debt.md): --resume does not work correctly with
        per-pod emptyDir on restart. When a pod restarts after failure:
          - emptyDir is destroyed (all local workspace lost)
          - Rank 0 resumes from the S3 checkpoint
          - Other ranks have no local checkpoint and start from scratch
          - The ranks diverge, producing incorrect results

        The manifest must acknowledge this limitation so that operators
        are not surprised by unexpected restart behaviour. A comment or
        reference to docs/tech-debt.md is sufficient — the limitation
        does not need to be fixed here (that is Level 5 work).

        This test does NOT require the limitation to be resolved.
        It only requires honest documentation.

        If this test fails: add a comment to the manifest explaining the
        --resume + emptyDir restart limitation. Reference TD-006.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        has_td006_ref = "TD-006" in content
        has_resume_limitation_comment = (
            "--resume" in content
            and (
                "limitation" in content.lower()
                or "does not work" in content.lower()
                or "emptyDir" in content
            )
        )

        assert has_td006_ref or has_resume_limitation_comment, (
            "Two-node manifest does not document the TD-006 --resume limitation. "
            "The manifest must mention TD-006 or explain that --resume does not "
            "work correctly with per-pod emptyDir on pod restart. "
            "Add a comment referencing docs/tech-debt.md TD-006."
        )


# ===========================================================================
# 7. TestPyTorchJobSmoke
#    Runtime distributed training smoke.
#    Requires RUN_PYTORCHJOB_TESTS=1 AND RUN_PYTORCHJOB_SMOKE=1.
#    Creates a short-lived PyTorchJob in runtime_namespace.
# ===========================================================================


class TestPyTorchJobSmoke:
    """Level 4: PyTorchJob two-node runtime distributed training smoke.

    1 test — requires RUN_PYTORCHJOB_TESTS=1 and RUN_PYTORCHJOB_SMOKE=1.

    Applies a purpose-built smoke PyTorchJob (NOT the production manifest),
    waits for both Master and Worker pods to complete, verifies DDP log
    markers from both ranks, and cleans up via the cleanup fixture.

    The smoke manifest is rendered by _render_smoke_manifest().
    It follows the same KFTO topology as the production manifest but uses:
      - PRAGMA_TRAINING_IMAGE (no git clone, source baked in)
      - Inline 30-row CSV (no S3 credentials)
      - CPU only (no GPU resource request)
      - --max-steps 1 (fast exit, no checkpoint)
    """

    @_require_pytorchjob_smoke
    def test_pytorchjob_two_node_smoke(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,
    ) -> None:
        """Apply a two-node PyTorchJob and verify DDP training completes.

        Steps:
          1. Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2. Render smoke PyTorchJob YAML with test_id, namespace, image.
          3. Apply via oc apply -f (idempotent, label-safe).
          4. Wait for both Master and Worker pods to appear (poll every 10s).
          5. Wait for PyTorchJob terminal condition (Succeeded or Failed).
          6. Collect pod logs via oc logs -l <test-id-selector> (best-effort).
          7. Assert log markers: PRAGMA-S, Reached --max-steps, DDP evidence,
             both rank=0 and rank=1 present.
          8. Assert PyTorchJob Succeeded.
          9. Cleanup via cleanup_labelled_resources fixture (automatic).

        Known limitation:
          TD-006 remains open. --max-steps 1 ensures no checkpoint is
          created and --resume is not needed. TD-006 is Level 5 work.
        """
        # ------------------------------------------------------------------
        # Step 1 — Resolve PRAGMA_TRAINING_IMAGE.
        # ------------------------------------------------------------------
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Export it to the training image URI, e.g.: "
                "export PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry"
                ".svc:5000/pragma-encoder/pragma-encoder-training:latest"
            )

        job_name = f"pragma-pytorchjob-smoke-{test_id}"
        selector = label_selector(test_id)

        print(f"\n[Level 4] job_name={job_name!r}")
        print(f"[Level 4] image={image!r}")
        print(f"[Level 4] namespace={runtime_namespace!r}")

        # ------------------------------------------------------------------
        # Step 2 — Render smoke PyTorchJob YAML.
        # ------------------------------------------------------------------
        yaml_path = tmp_path / "pytorchjob-smoke.yaml"
        rendered = _render_smoke_manifest(
            test_id=test_id,
            namespace=runtime_namespace,
            image=image,
        )
        yaml_path.write_text(rendered)
        print(f"[Level 4] Rendered manifest: {yaml_path} ({len(rendered)} bytes)")

        # ------------------------------------------------------------------
        # Step 3 — Apply via oc apply.
        # oc apply is idempotent and does not touch Argo-managed resources.
        # ------------------------------------------------------------------
        oc(["apply", "-f", str(yaml_path)], namespace=runtime_namespace)
        print(f"[Level 4] oc apply complete: {job_name}")

        # ------------------------------------------------------------------
        # Step 4 — Wait for both pods to appear.
        # Poll every 10 seconds up to timeout_seconds.
        # ------------------------------------------------------------------
        print(f"[Level 4] Waiting for Master + Worker pods (selector={selector!r}) ...")
        _deadline = time.time() + timeout_seconds
        _pods_found = False
        while time.time() < _deadline:
            try:
                pod_list = oc_json(
                    ["get", "pods", "-l", selector],
                    namespace=runtime_namespace,
                    timeout=15,
                )
                pod_count = len(pod_list.get("items", []))
                if pod_count >= 2:
                    _pods_found = True
                    print(f"[Level 4] {pod_count} pods found \u2713")
                    break
                print(f"[Level 4] {pod_count}/2 pods found — waiting 10s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 4] pod list error (retrying): {redact(str(exc))}")
            time.sleep(10)

        if not _pods_found:
            # Collect job status for diagnostics before failing.
            try:
                job_json = oc_json(
                    ["get", "pytorchjob", job_name],
                    namespace=runtime_namespace,
                    timeout=15,
                )
                _status_conditions = job_json.get("status", {}).get("conditions", [])
            except Exception:  # noqa: BLE001
                _status_conditions = []
            pytest.fail(
                f"Master and Worker pods did not appear within {timeout_seconds}s. "
                f"Selector: {selector!r}. "
                f"PyTorchJob conditions: {_status_conditions}. "
                "Check KFTO operator logs and pod events: "
                f"oc describe pytorchjob {job_name} -n {runtime_namespace}"
            )

        # ------------------------------------------------------------------
        # Step 5 — Wait for PyTorchJob terminal condition (Succeeded/Failed).
        # Poll every 10 seconds.
        # ------------------------------------------------------------------
        print(f"[Level 4] Waiting for PyTorchJob terminal condition (timeout={timeout_seconds}s) ...")
        _terminal = False
        _final_condition = ""
        _deadline2 = time.time() + timeout_seconds
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
                    print(f"[Level 4] PyTorchJob reached terminal condition: {_final_condition}")
                    break
                # Print current replica status for progress visibility.
                _replica_statuses = job_json.get("status", {}).get("replicaStatuses", {})
                print(f"[Level 4] replicaStatuses={_replica_statuses} — waiting 10s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 4] status poll error (retrying): {redact(str(exc))}")
            time.sleep(10)

        if not _terminal:
            raise TimeoutError(
                f"PyTorchJob {job_name!r} did not reach Succeeded or Failed "
                f"within {timeout_seconds}s. "
                f"Last condition: {_final_condition!r}. "
                f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}"
            )

        # ------------------------------------------------------------------
        # Step 6 — Collect pod logs (best-effort).
        # Redact before any assertion or printing.
        # ------------------------------------------------------------------
        print("[Level 4] Collecting pod logs ...")
        _log_result = oc(
            ["logs", "-l", selector, "--tail", "200", "--prefix"],
            namespace=runtime_namespace,
            check=False,
            timeout=60,
        )
        all_logs = redact(_log_result.stdout.strip())

        if all_logs:
            print("[Level 4] Log excerpt (last 40 lines):")
            for _line in all_logs.splitlines()[-40:]:
                print(f"  {_line}")
        else:
            print(
                "[Level 4] WARNING: pod logs not available via oc logs "
                f"(selector={selector!r}). "
                "Check the cluster UI for pod-level logs."
            )

        # ------------------------------------------------------------------
        # Step 7 — Assert log markers (only when logs are available).
        # ------------------------------------------------------------------
        if all_logs:
            # Model variant confirmation — printed by train_pragma.py.
            assert "PRAGMA-S" in all_logs, (
                "Pod logs must contain 'PRAGMA-S' — the model variant confirmation. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )

            # Early-stop marker — printed by train_pragma.py.
            assert "Reached --max-steps" in all_logs, (
                "Pod logs must contain 'Reached --max-steps' — the early-stop marker. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )

            # DDP initialisation evidence — at least one marker must appear.
            _ddp_keywords = [
                "WORLD_SIZE", "MASTER_ADDR", "gloo", "nccl",
                "rendezvous", "process group", "dist.init_process_group",
            ]
            _has_ddp_evidence = any(kw in all_logs for kw in _ddp_keywords)
            assert _has_ddp_evidence, (
                "Pod logs must contain DDP initialisation evidence. "
                f"Expected at least one of: {_ddp_keywords}. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )

            # Both ranks must appear in the combined logs.
            _rank0_patterns = ["rank=0", "RANK=0", "rank 0", "[rank0]"]
            _rank1_patterns = ["rank=1", "RANK=1", "rank 1", "[rank1]"]
            _has_rank0 = any(p in all_logs for p in _rank0_patterns)
            _has_rank1 = any(p in all_logs for p in _rank1_patterns)

            assert _has_rank0, (
                "Pod logs must show output from rank 0 (Master). "
                f"Expected one of: {_rank0_patterns}. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )
            assert _has_rank1, (
                "Pod logs must show output from rank 1 (Worker). "
                f"Expected one of: {_rank1_patterns}. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )

            print(
                "[Level 4] Log markers confirmed: "
                "'PRAGMA-S' \u2713  'Reached --max-steps' \u2713  "
                "DDP evidence \u2713  rank=0 \u2713  rank=1 \u2713"
            )
        else:
            # Succeeded state is the definitive pass criterion when logs
            # are unavailable (e.g. running the test from outside the cluster).
            print(
                "[Level 4] Log markers not verified (logs unavailable). "
                f"Condition={_final_condition!r} is the pass criterion."
            )

        # ------------------------------------------------------------------
        # Step 8 — Assert PyTorchJob Succeeded.
        # ------------------------------------------------------------------
        assert _final_condition == "Succeeded", (
            f"PyTorchJob {job_name!r} did not Succeed. "
            f"Final condition: {_final_condition!r}. "
            f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
            + (f"Pod logs (redacted):\n{all_logs}" if all_logs else "Pod logs not available.")
        )

        # ------------------------------------------------------------------
        # Step 9 — Cleanup (automatic via cleanup_labelled_resources fixture).
        # The fixture yields before the test body and runs teardown after.
        # Resources with both test labels are deleted after this return.
        # ------------------------------------------------------------------
        print("\n[Level 4] === PASSED: PRAGMA PyTorchJob two-node distributed training smoke ===")
        print(f"  Job:         {job_name}")
        print(f"  Condition:   {_final_condition}")
        print(f"  Namespace:   {runtime_namespace}")
        print(f"  Image:       {image}")
