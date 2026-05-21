"""Level 4 — N-node distributed training smoke (default: nnodes=2).

Purpose:
  Prove the PRAGMA architecture is N-node capable with a 2-node smoke:
    - PyTorchJob CRD is available (KFTO operator installed)
    - Master (rank 0) and Worker (rank 1..nnodes-1) pods are created
    - All ranks initialise the DDP process group via torchrun
    - WORLD_SIZE=nnodes / RANK=0..nnodes-1 visible via KFTO env injection
    - Tiny PRAGMA-S training reaches --max-steps 1 on all ranks
    - Rank 0 logs completion; other ranks skip upload (no S3 in smoke)
    - Test resources carry both test labels; cleanup is label-scoped

Architecture — N-node capable, 2-node smoke:
  The PRAGMA training architecture is N-node capable through:
    - torchrun --nnodes=N --nproc_per_node=P
    - KFTO PyTorchJob: 1 Master + (N-1) Workers
    - gloo (CPU) or nccl (GPU) process group backend
    - DistributedSampler for disjoint data sharding across ranks

  The Level 4 smoke validates the minimal distributed case: nnodes=2.
  Two nodes is the smallest configuration that exercises the full DDP path
  (rendezvous, barrier, gradient averaging). Single-node is Level 3b.

  Larger N (N>2) is a future scale-testing concern (Level 6). The
  architecture supports it through KFTO Worker replicas = nnodes-1;
  the default smoke uses nnodes=2 to keep CI fast and cluster-load low.

Architecture boundary:
  Level 4 is independent of Level 3 (DSPA/KFP v2 pipeline smoke).
  Level 4 PASS proves distributed PyTorchJob execution.
  Level 4 does NOT prove KFP v2 pipeline orchestration (Level 3).

Smoke manifest design:
  The runtime smoke (test 7) uses a purpose-built inline manifest —
  NOT the production pytorchjob-pragma-s-2node.yaml — because the
  production manifest requires S3 credentials, runtime git clone, and
  GPU hardware that are not available or appropriate for smoke testing.

  The smoke manifest follows the same KFTO pattern as the production manifest:
    - apiVersion: kubeflow.org/v1 / kind: PyTorchJob
    - Master (rank 0) + Worker (rank 1..nnodes-1) replicas
    - torchrun --nnodes=<nnodes> --nproc_per_node=1 --node_rank=$RANK
    - KFTO injects: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    - Image: PRAGMA_TRAINING_IMAGE (source + deps baked in, no git clone)
    - Data: inline 30-row TabFormer CSV (no S3, no init container)
    - CPU only (--device cpu, no GPU resource request)
    - --max-steps 1 for fast exit

DNS label length constraint (RFC 1035 §2.3.4):
  KFTO's init-pytorch init container resolves the Master pod hostname
  via nslookup before starting Worker main containers. Pod names are
  used as DNS labels — the limit is 63 characters.
  Job name prefix is 'pragma-smoke-' to keep pod names under the limit.
  A static test verifies the generated name length remains <= 63 chars.

Known limitation (TD-006):
  --resume does not work with per-pod emptyDir on restart.
  This smoke uses --max-steps 1 and omits --resume so no checkpoint
  is created. TD-006 remains open; its resolution is Level 5.

Guard variables:
  RUN_OPENSHIFT_TESTS=1            — suite-wide (conftest)
  RUN_PYTORCHJOB_TESTS=1           — enables all Level 4 tests (1-8)
  RUN_PYTORCHJOB_SMOKE=1           — additionally enables runtime job creation (test 8)
  PRAGMA_TEST_NAMESPACE=<ns>       — required (conftest)
  PRAGMA_TRAINING_IMAGE=<img>      — required for test 8 runtime smoke
  PRAGMA_PYTORCHJOB_NNODES=<n>    — optional, default 2, minimum 2
  PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1 — required when nnodes > 2

Tests 1-7 require only RUN_PYTORCHJOB_TESTS=1 (read-only, no cluster resources created).
Test 8 additionally requires RUN_PYTORCHJOB_SMOKE=1 (creates a short-lived PyTorchJob).
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

# ---------------------------------------------------------------------------
# N-node configuration
# ---------------------------------------------------------------------------

_SMOKE_NNODES_DEFAULT = 2
_SMOKE_NNODES_MINIMUM = 2  # nnodes=1 is non-distributed — use Level 3b instead
_LARGE_NNODE_SMOKE_ENABLED = os.environ.get("PRAGMA_ALLOW_LARGE_NNODE_SMOKE") == "1"

# Path to the committed production 2-node manifest (the canonical example).
_TWO_NODE_MANIFEST = pathlib.Path("openshift/training/pytorchjob-pragma-s-2node.yaml")

# PyTorchJob CRD installed by KFTO.
_PYTORCHJOB_CRD = "pytorchjobs.kubeflow.org"

# Smoke job name prefix.  Must be short: KFTO's init-pytorch init container
# resolves Master pod hostname via nslookup (RFC 1035 §2.3.4 — 63-char limit).
# Pod name = pragma-smoke-{test_id}-master-0 = 13+35+9 = 57 chars ✓
_SMOKE_JOB_PREFIX = "pragma-smoke"

# DNS label hard limit (RFC 1035 §2.3.4).
_DNS_LABEL_LIMIT = 63

# Suffix added by KFTO for the master pod: "-master-0" (9 chars).
_KFTO_MASTER_SUFFIX = "-master-0"


def _resolve_nnodes() -> int:
    """Resolve nnodes from PRAGMA_PYTORCHJOB_NNODES env var.

    Rules:
      - Default: 2 (the minimal distributed case — exercises full DDP path)
      - Minimum: 2 (nnodes=1 is non-distributed; use Level 3b batch/v1 Job instead)
      - nnodes > 2 requires PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1

    Returns:
        Integer nnodes >= 2.
    """
    raw = os.environ.get("PRAGMA_PYTORCHJOB_NNODES", str(_SMOKE_NNODES_DEFAULT)).strip()
    try:
        n = int(raw)
    except ValueError:
        return _SMOKE_NNODES_DEFAULT
    return max(_SMOKE_NNODES_MINIMUM, n)


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


def _render_smoke_manifest(test_id: str, namespace: str, image: str, nnodes: int = 2) -> str:
    """Render the smoke PyTorchJob YAML manifest with test-specific values.

    Both Master and each Worker replica run an identical shell script that:
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
      - Worker replicas = nnodes - 1 (N-node capable pattern)

    Args:
        test_id:   Unique test-run identifier (used in resource name and labels).
        namespace: Kubernetes namespace to deploy the PyTorchJob into.
        image:     Training image URI (must have PRAGMA source baked in).
        nnodes:    Total node count. Default 2 (minimal distributed smoke).
                   Worker replicas = nnodes - 1.

    Returns:
        YAML string ready for ``oc apply -f``.
    """
    # Shell script run by all replicas (identical on Master and every Worker).
    # Line indentation: 18 spaces for YAML block scalar under "- |".
    # Shell variables use $VAR (not ${VAR}) to avoid Python f-string confusion.
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
        f"# Run PRAGMA-S N-node distributed training via torchrun (nnodes={nnodes}).",
        "# KFTO injects: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.",
        "echo \"[Level 4 smoke] torchrun RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR\"",
        "mkdir -p /tmp/pragma-smoke-output",
        "PYTHONPATH=$PRAGMA_ROOT torchrun \\",
        f"  --nnodes={nnodes} \\",
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

    # Resource spec shared by all replicas.
    _resources = """\
              resources:
                requests:
                  cpu: "500m"
                  memory: "2Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi\""""

    # Build the container spec block (identical for all replicas).
    def _container_spec() -> str:
        return f"""\
            - name: pytorch
              image: {image}
              imagePullPolicy: Always
              command:
                - /bin/sh
                - -c
                - |
{indented_script}
{_resources}"""

    # Build Worker spec if nnodes > 1 (always true since minimum is 2).
    worker_replicas = nnodes - 1
    _worker_block = f"""    Worker:
      replicas: {worker_replicas}
      restartPolicy: Never
      template:
        metadata:
          labels:
            pragma.redhat.com/test-run: "true"
            pragma.redhat.com/test-id: "{test_id}"
        spec:
          containers:
{_container_spec()}
"""

    return f"""\
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {_SMOKE_JOB_PREFIX}-{test_id}
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
{_container_spec()}
{_worker_block}"""


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
        and Level 4 N-node smoke is impossible.

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
# 2-6. TestNNodeManifest
#    Static structure checks on the committed production 2-node manifest.
#    Read-only, no cluster access. 5 tests.
#    Run when RUN_PYTORCHJOB_TESTS=1.
# ===========================================================================


class TestNNodeManifest:
    """Level 4: PyTorchJob manifest structural correctness (read-only).

    5 tests — no cluster access needed.
    Run when RUN_PYTORCHJOB_TESTS=1.

    These checks target the committed production manifest
    (openshift/training/pytorchjob-pragma-s-2node.yaml), which is the
    canonical N-node example using the 2-node configuration. The manifest
    demonstrates the pattern; N-node training uses Worker replicas = nnodes-1.

    Purpose: catch manifest regressions (removed replicas, added PVCs,
    lost TD-006 documentation) without requiring a cluster connection.
    """

    @_require_pytorchjob
    def test_production_manifest_exists(self) -> None:
        """openshift/training/pytorchjob-pragma-s-2node.yaml must exist.

        This manifest is the canonical N-node DDP training definition
        (committed at 2-node scale as the minimal distributed example).
        If it is missing, the distributed training pattern cannot be verified.
        """
        assert _TWO_NODE_MANIFEST.exists(), (
            f"Production manifest {_TWO_NODE_MANIFEST} not found. "
            "This file must exist in the repository for Level 4 tests."
        )

    @_require_pytorchjob
    def test_production_manifest_has_master_and_worker(self) -> None:
        """The manifest must define both a Master replica and at least one Worker.

        KFTO N-node topology:
          pytorchReplicaSpecs.Master (rank 0) — rendezvous initiator
          pytorchReplicaSpecs.Worker (rank 1..N-1) — connect to Master

        KFTO injects MASTER_ADDR and MASTER_PORT into every replica pod.
        If either replica type is missing, the DDP ring cannot form and
        distributed training silently degrades to single-node.

        Expected manifest structure:
          spec.pytorchReplicaSpecs.Master.replicas: 1
          spec.pytorchReplicaSpecs.Worker.replicas: nnodes-1
        """
        content = _TWO_NODE_MANIFEST.read_text()

        assert "Master" in content, (
            "Production manifest does not define a Master replica. "
            "Expected: spec.pytorchReplicaSpecs.Master with replicas: 1. "
            "Master (rank 0) is required for the DDP rendezvous."
        )
        assert "Worker" in content, (
            "Production manifest does not define a Worker replica. "
            "Expected: spec.pytorchReplicaSpecs.Worker with replicas >= 1. "
            "Worker (rank 1..N-1) connects to Master for DDP coordination."
        )

    @_require_pytorchjob
    def test_production_manifest_has_no_canonical_pvc(self) -> None:
        """The manifest must not use PVC as canonical persistent storage.

        The N-node manifest uses per-pod emptyDir for local scratch space
        and S3 as the durable artifact store. PVCs are avoided because:
          - ReadWriteOnce PVCs cannot be mounted by two pods simultaneously.
          - RWX PVCs require shared storage infrastructure not assumed here.
          - S3 is the correct canonical checkpoint location for multi-node DDP.

        emptyDir is expected and allowed for /workspace and /dev/shm.

        Reference: docs/tech-debt.md TD-006.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        assert "kind: PersistentVolumeClaim" not in content, (
            "Production manifest defines a PersistentVolumeClaim resource. "
            "Multi-node PyTorchJobs must use per-pod emptyDir + S3 storage. "
            "PVCs with ReadWriteOnce cause scheduling conflicts on multi-node runs. "
            "See docs/tech-debt.md TD-006."
        )
        assert "volumeClaimTemplates" not in content, (
            "Production manifest uses volumeClaimTemplates (StatefulSet pattern). "
            "Multi-node PyTorchJobs must use per-pod emptyDir + S3 storage. "
            "See docs/tech-debt.md TD-006."
        )

    @_require_pytorchjob
    def test_production_manifest_mentions_torchrun_or_distributed(self) -> None:
        """The manifest must reference distributed training configuration.

        A valid N-node DDP manifest must mention at least one of:
          - torchrun (the PyTorch distributed process launcher)
          - WORLD_SIZE (environment variable set by the distributed framework)
          - MASTER_ADDR (rendezvous address injected by KFTO into every pod)

        If none are present, the manifest does not configure DDP correctly
        and all pods would train independently (no parameter synchronisation).

        torchrun is the recommended launcher for KFTO PyTorchJobs.
        KFTO injects MASTER_ADDR and MASTER_PORT into every replica pod.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        has_torchrun = "torchrun" in content
        has_world_size = "WORLD_SIZE" in content
        has_master_addr = "MASTER_ADDR" in content

        assert has_torchrun or has_world_size or has_master_addr, (
            "Production manifest does not reference torchrun, WORLD_SIZE, or MASTER_ADDR. "
            "A valid DDP manifest must configure the distributed launcher. "
            "Expected at least one of: "
            "torchrun (launcher), WORLD_SIZE (env), MASTER_ADDR (rendezvous). "
            "Without these, pods train independently — no DDP synchronisation."
        )

    @_require_pytorchjob
    def test_production_manifest_warns_about_td006(self) -> None:
        """The manifest must document the TD-006 --resume limitation.

        TD-006 (docs/tech-debt.md): --resume does not work correctly with
        per-pod emptyDir on restart in multi-node training.

        This test does NOT require the limitation to be resolved.
        It only requires honest documentation in the manifest.

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
            "Production manifest does not document the TD-006 --resume limitation. "
            "The manifest must mention TD-006 or explain that --resume does not "
            "work correctly with per-pod emptyDir on pod restart. "
            "Add a comment referencing docs/tech-debt.md TD-006."
        )


# ===========================================================================
# 7. TestSmokeManifestNaming
#    Static checks on generated smoke manifest name length.
#    Read-only, no cluster access. Run when RUN_PYTORCHJOB_TESTS=1.
# ===========================================================================


class TestSmokeManifestNaming:
    """Level 4: Verify smoke manifest job names respect DNS label length limit.

    3 tests — no cluster access needed.
    Run when RUN_PYTORCHJOB_TESTS=1.

    Background:
      KFTO's init-pytorch init container resolves the Master pod hostname
      via nslookup before starting Worker main containers. DNS labels are
      limited to 63 characters (RFC 1035 §2.3.4). A name > 63 chars causes
      the init container to loop and error, with the job never starting.

      Previous failure: pragma-pytorchjob-smoke-{test_id}-master-0 = 67 chars.
      Fixed by using prefix 'pragma-smoke-': pragma-smoke-{test_id}-master-0 = 56 chars.

    These tests are purely static — they verify the naming constants and
    a representative test_id produce pod names under the DNS limit.
    """

    @_require_pytorchjob
    def test_smoke_job_prefix_length_is_safe(self) -> None:
        """_SMOKE_JOB_PREFIX must leave room for test_id + KFTO master suffix.

        Constraint:
          len(prefix) + 1 + len(test_id) + len('-master-0') <= 63

        conftest test_id format: 'pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx'
        = 35 characters (fixed).

        Maximum safe prefix length: 63 - 35 - 1 - 9 = 18 chars.
        Current prefix 'pragma-smoke': 12 chars.
        """
        _TYPICAL_TEST_ID_LEN = 35  # 'pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx'
        max_prefix = _DNS_LABEL_LIMIT - _TYPICAL_TEST_ID_LEN - 1 - len(_KFTO_MASTER_SUFFIX)
        assert len(_SMOKE_JOB_PREFIX) <= max_prefix, (
            f"_SMOKE_JOB_PREFIX {_SMOKE_JOB_PREFIX!r} is {len(_SMOKE_JOB_PREFIX)} chars. "
            f"Maximum safe prefix length is {max_prefix} chars "
            f"(DNS limit {_DNS_LABEL_LIMIT} - test_id {_TYPICAL_TEST_ID_LEN} - "
            f"separator 1 - KFTO suffix {len(_KFTO_MASTER_SUFFIX)})."
        )

    @_require_pytorchjob
    def test_generated_master_pod_name_under_dns_limit(self) -> None:
        """A representative generated master pod name must be <= 63 chars.

        Uses a fixed typical test_id to verify the full pod name length.
        """
        typical_test_id = "pragma-it-20260520-233933-cdf02e3e"  # 35 chars
        job_name = f"{_SMOKE_JOB_PREFIX}-{typical_test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is {len(master_pod_name)} chars, "
            f"exceeding the {_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            "Shorten _SMOKE_JOB_PREFIX to fix."
        )
        print(
            f"\n[Level 4] master pod name: {master_pod_name!r} "
            f"({len(master_pod_name)} chars \u2264 {_DNS_LABEL_LIMIT}) \u2713"
        )

    @_require_pytorchjob
    def test_generated_worker_pod_name_under_dns_limit(self) -> None:
        """A representative generated worker pod name must be <= 63 chars.

        Worker pod name = {job_name}-worker-{n}.  Suffix '-worker-0' = 9 chars,
        same length as '-master-0'.  Larger N still fits since only the digit
        changes (single char for N < 10).
        """
        typical_test_id = "pragma-it-20260520-233933-cdf02e3e"  # 35 chars
        job_name = f"{_SMOKE_JOB_PREFIX}-{typical_test_id}"
        worker_pod_name = f"{job_name}-worker-0"
        assert len(worker_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated worker pod name {worker_pod_name!r} is {len(worker_pod_name)} chars, "
            f"exceeding the {_DNS_LABEL_LIMIT}-char DNS label limit (RFC 1035 §2.3.4). "
            "Shorten _SMOKE_JOB_PREFIX to fix."
        )
        print(
            f"\n[Level 4] worker pod name: {worker_pod_name!r} "
            f"({len(worker_pod_name)} chars \u2264 {_DNS_LABEL_LIMIT}) \u2713"
        )


# ===========================================================================
# 8. TestPyTorchJobSmoke
#    Runtime N-node distributed training smoke.
#    Requires RUN_PYTORCHJOB_TESTS=1 AND RUN_PYTORCHJOB_SMOKE=1.
#    Creates a short-lived PyTorchJob in runtime_namespace.
# ===========================================================================


class TestPyTorchJobSmoke:
    """Level 4: PyTorchJob N-node distributed training runtime smoke.

    1 test — requires RUN_PYTORCHJOB_TESTS=1 and RUN_PYTORCHJOB_SMOKE=1.

    Default: nnodes=2 (minimal distributed case — exercises full DDP path).
    Override: PRAGMA_PYTORCHJOB_NNODES=<n> (n >= 2).
    Large N:  requires PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1 (n > 2).

    Applies a purpose-built smoke PyTorchJob (NOT the production manifest),
    waits for Master + all Worker pods to start, waits for terminal condition,
    verifies DDP log markers from all ranks, and cleans up via cleanup fixture.

    The smoke manifest is rendered by _render_smoke_manifest().
    It follows the same KFTO topology as the production manifest but uses:
      - PRAGMA_TRAINING_IMAGE (no git clone, source baked in)
      - Inline 30-row CSV (no S3 credentials)
      - CPU only (no GPU resource request)
      - --max-steps 1 (fast exit, no checkpoint)
      - Worker replicas = nnodes - 1 (N-node pattern)
    """

    @_require_pytorchjob_smoke
    def test_pytorchjob_nnode_smoke(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,
    ) -> None:
        """Apply an N-node PyTorchJob and verify DDP training completes.

        Default nnodes=2 (PRAGMA_PYTORCHJOB_NNODES env var, min 2).
        nnodes > 2 requires PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1.

        Steps:
          1. Resolve PRAGMA_TRAINING_IMAGE — skip if not set.
          2. Resolve nnodes from PRAGMA_PYTORCHJOB_NNODES (default 2).
             Reject nnodes=1 (non-distributed). Guard nnodes > 2.
          3. Verify job name is under DNS label limit (63 chars).
          4. Render smoke PyTorchJob YAML with test_id, namespace, image, nnodes.
          5. Apply via oc apply -f (idempotent, label-safe).
          6. Wait for all nnodes pods to appear (poll every 10s).
          7. Wait for PyTorchJob terminal condition (Succeeded or Failed).
          8. Collect pod logs via oc logs -l <test-id-selector> (best-effort).
          9. Assert log markers: PRAGMA-S, Reached --max-steps, DDP evidence,
             rank=0 present (rank=1 present for nnodes >= 2).
         10. Assert PyTorchJob Succeeded.
         11. Cleanup via cleanup_labelled_resources fixture (automatic).

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

        # ------------------------------------------------------------------
        # Step 2 — Resolve nnodes with validation.
        # ------------------------------------------------------------------
        nnodes = _resolve_nnodes()

        if nnodes < _SMOKE_NNODES_MINIMUM:
            # Should not reach here because _resolve_nnodes() clamps to minimum,
            # but guard explicitly for clarity.
            pytest.fail(
                f"PRAGMA_PYTORCHJOB_NNODES={nnodes} is below the minimum of "
                f"{_SMOKE_NNODES_MINIMUM}. This is the distributed smoke — "
                "nnodes=1 is non-distributed (use Level 3b batch/v1 Job instead). "
                f"Set PRAGMA_PYTORCHJOB_NNODES >= {_SMOKE_NNODES_MINIMUM}."
            )

        if nnodes > _SMOKE_NNODES_DEFAULT and not _LARGE_NNODE_SMOKE_ENABLED:
            pytest.skip(
                f"PRAGMA_PYTORCHJOB_NNODES={nnodes} > {_SMOKE_NNODES_DEFAULT} but "
                "PRAGMA_ALLOW_LARGE_NNODE_SMOKE is not set. "
                "Large N-node smoke is opt-in to avoid cluster overload. "
                "Set PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1 to enable."
            )

        worker_replicas = nnodes - 1

        # ------------------------------------------------------------------
        # Step 3 — Verify job name is under DNS label limit.
        # ------------------------------------------------------------------
        job_name = f"{_SMOKE_JOB_PREFIX}-{test_id}"
        master_pod_name = f"{job_name}{_KFTO_MASTER_SUFFIX}"
        assert len(master_pod_name) <= _DNS_LABEL_LIMIT, (
            f"Generated master pod name {master_pod_name!r} is {len(master_pod_name)} chars, "
            f"which exceeds the {_DNS_LABEL_LIMIT}-char DNS label limit. "
            "KFTO's init-pytorch would fail to resolve it via nslookup. "
            "Shorten _SMOKE_JOB_PREFIX or reduce test_id length."
        )

        selector = label_selector(test_id)

        print(f"\n[Level 4] job_name={job_name!r}")
        print(f"[Level 4] nnodes={nnodes}  (master=1 + workers={worker_replicas})")
        print(f"[Level 4] image={image!r}")
        print(f"[Level 4] namespace={runtime_namespace!r}")

        # ------------------------------------------------------------------
        # Step 4 — Render smoke PyTorchJob YAML.
        # ------------------------------------------------------------------
        yaml_path = tmp_path / "pytorchjob-smoke.yaml"
        rendered = _render_smoke_manifest(
            test_id=test_id,
            namespace=runtime_namespace,
            image=image,
            nnodes=nnodes,
        )
        yaml_path.write_text(rendered)
        print(f"[Level 4] Rendered manifest: {yaml_path} ({len(rendered)} bytes)")

        # ------------------------------------------------------------------
        # Step 5 — Apply via oc apply.
        # oc apply is idempotent and does not touch Argo-managed resources.
        # ------------------------------------------------------------------
        oc(["apply", "-f", str(yaml_path)], namespace=runtime_namespace)
        print(f"[Level 4] oc apply complete: {job_name}")

        # ------------------------------------------------------------------
        # Step 6 — Wait for all nnodes pods to appear.
        # Poll every 10 seconds up to timeout_seconds.
        # ------------------------------------------------------------------
        print(
            f"[Level 4] Waiting for {nnodes} pods "
            f"(selector={selector!r}) ..."
        )
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
                if pod_count >= nnodes:
                    _pods_found = True
                    print(f"[Level 4] {pod_count}/{nnodes} pods found \u2713")
                    break
                print(f"[Level 4] {pod_count}/{nnodes} pods found — waiting 10s ...")
            except Exception as exc:  # noqa: BLE001
                print(f"[Level 4] pod list error (retrying): {redact(str(exc))}")
            time.sleep(10)

        if not _pods_found:
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
                f"Expected {nnodes} pods but they did not appear within {timeout_seconds}s. "
                f"Selector: {selector!r}. "
                f"PyTorchJob conditions: {_status_conditions}. "
                "Check KFTO operator logs and pod events: "
                f"oc describe pytorchjob {job_name} -n {runtime_namespace}"
            )

        # ------------------------------------------------------------------
        # Step 7 — Wait for PyTorchJob terminal condition (Succeeded/Failed).
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
        # Step 8 — Collect pod logs (best-effort).
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
        # Step 9 — Assert log markers (only when logs are available).
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

            # Rank 0 (Master) must always appear in logs.
            _rank0_patterns = ["rank=0", "RANK=0", "rank 0", "[rank0]"]
            _has_rank0 = any(p in all_logs for p in _rank0_patterns)
            assert _has_rank0, (
                "Pod logs must show output from rank 0 (Master). "
                f"Expected one of: {_rank0_patterns}. "
                f"Job: {job_name!r}. "
                f"Full logs (redacted):\n{all_logs}"
            )

            # Rank 1 (first Worker) must appear for nnodes >= 2.
            if nnodes >= 2:
                _rank1_patterns = ["rank=1", "RANK=1", "rank 1", "[rank1]"]
                _has_rank1 = any(p in all_logs for p in _rank1_patterns)
                assert _has_rank1, (
                    "Pod logs must show output from rank 1 (first Worker). "
                    f"Expected one of: {_rank1_patterns}. "
                    f"Job: {job_name!r}. "
                    f"Full logs (redacted):\n{all_logs}"
                )

            print(
                "[Level 4] Log markers confirmed: "
                "'PRAGMA-S' \u2713  'Reached --max-steps' \u2713  "
                "DDP evidence \u2713  rank=0 \u2713"
                + ("  rank=1 \u2713" if nnodes >= 2 else "")
            )
        else:
            print(
                "[Level 4] Log markers not verified (logs unavailable). "
                f"Condition={_final_condition!r} is the pass criterion."
            )

        # ------------------------------------------------------------------
        # Step 10 — Assert PyTorchJob Succeeded.
        # ------------------------------------------------------------------
        assert _final_condition == "Succeeded", (
            f"PyTorchJob {job_name!r} did not Succeed. "
            f"Final condition: {_final_condition!r}. "
            f"Check: oc describe pytorchjob {job_name} -n {runtime_namespace}. "
            + (f"Pod logs (redacted):\n{all_logs}" if all_logs else "Pod logs not available.")
        )

        # ------------------------------------------------------------------
        # Step 11 — Cleanup (automatic via cleanup_labelled_resources fixture).
        # ------------------------------------------------------------------
        print(f"\n[Level 4] === PASSED: PRAGMA PyTorchJob {nnodes}-node distributed training smoke ===")
        print(f"  Job:         {job_name}")
        print(f"  nnodes:      {nnodes}  (master=1 + workers={worker_replicas})")
        print(f"  Condition:   {_final_condition}")
        print(f"  Namespace:   {runtime_namespace}")
        print(f"  Image:       {image}")
