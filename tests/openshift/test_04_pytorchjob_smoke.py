"""Level 4 — PyTorchJob two-node smoke (opt-in / future xfail).

Purpose:
  Verify the two-node PyTorchJob manifest is structurally correct, and
  (future) apply it with test labels and verify distributed training runs.

Current status:
  - Manifest structure tests: IMPLEMENTED (read-only, no cluster access)
  - Execution smoke test: FUTURE / XFAIL

All tests in this file skip unless RUN_PYTORCHJOB_TESTS=1.

Manifest tests (read-only):
  test_two_node_manifest_exists
  test_two_node_manifest_has_no_pvc_canonical_storage
  test_two_node_manifest_mentions_world_size_or_torchrun
  test_two_node_manifest_has_master_and_worker

Execution smoke (future/xfail):
  test_pytorchjob_two_node_smoke_future

Safety rules for the future execution test:
  - Copy manifest to temp file; patch name to include test_id.
  - Add test labels before apply.
  - Submit via oc apply (not oc create) to allow idempotent re-runs.
  - Wait for Master and Worker pods to reach Running or Completed state.
  - Cleanup deletes only resources with both test labels.
  - Never apply to Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.

Known limitations (TD-006 in docs/tech-debt.md):
  --resume does not work correctly on restart with per-pod emptyDir.
  The execution smoke uses max_steps=1 (no checkpoint created).

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_PYTORCHJOB_TESTS=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - pytorchjobs.kubeflow.org CRD present (test_01 verified)
"""

from __future__ import annotations

import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Suite-level skip guard
# ---------------------------------------------------------------------------

_PYTORCHJOB_ENABLED = os.environ.get("RUN_PYTORCHJOB_TESTS") == "1"

_require_pytorchjob = pytest.mark.skipif(
    not _PYTORCHJOB_ENABLED,
    reason=(
        "PyTorchJob tests are opt-in. "
        "Set RUN_PYTORCHJOB_TESTS=1 to enable. "
        "Warning: the execution smoke test creates a short-lived PyTorchJob "
        "in PRAGMA_TEST_RUNTIME_NAMESPACE."
    ),
)

# Path to the two-node manifest committed to the repo.
_TWO_NODE_MANIFEST = pathlib.Path(
    "openshift/training/pytorchjob-pragma-s-2node.yaml"
)


# ---------------------------------------------------------------------------
# Read-only manifest structure tests
# ---------------------------------------------------------------------------

class TestTwoNodeManifest:
    """Level 4: PyTorchJob two-node manifest correctness (read-only)."""

    @_require_pytorchjob
    def test_two_node_manifest_exists(self) -> None:
        """openshift/training/pytorchjob-pragma-s-2node.yaml must exist.

        This manifest is the canonical two-node DDP training definition.
        If it is missing, the distributed training path cannot be tested.
        """
        assert _TWO_NODE_MANIFEST.exists(), (
            f"Two-node manifest {_TWO_NODE_MANIFEST} not found. "
            "This file must exist in the repository for distributed training tests."
        )

    @_require_pytorchjob
    def test_two_node_manifest_has_no_pvc_canonical_storage(self) -> None:
        """The two-node manifest must not use PVC as canonical persistent storage.

        Per the architecture decision (docs/tech-debt.md TD-006), the
        two-node manifest uses per-pod emptyDir for local scratch space and
        S3 as the durable artifact store. PVCs are avoided because:
          - ReadWriteOnce PVCs cannot be mounted by two pods simultaneously.
          - S3 is the canonical checkpoint location for multi-node training.

        A PVC in the manifest may indicate a regression to the single-node
        storage model, which would cause scheduling conflicts on multi-node runs.

        Note: an init-container reading FROM S3 via a Job or init pod may
        reference a PVC; this test checks that no 'kind: PersistentVolumeClaim'
        resource definition appears in the manifest itself.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        # A PVC definition in the manifest would be "kind: PersistentVolumeClaim"
        # We also check that volumeClaimTemplates is absent (StatefulSet pattern).
        assert "kind: PersistentVolumeClaim" not in content, (
            "Two-node manifest defines a PersistentVolumeClaim. "
            "The canonical storage model uses per-pod emptyDir + S3. "
            "PVCs are not appropriate for multi-pod PyTorchJobs (scheduling conflict). "
            "See docs/tech-debt.md TD-006."
        )
        assert "volumeClaimTemplates" not in content, (
            "Two-node manifest uses volumeClaimTemplates. "
            "The canonical storage model uses per-pod emptyDir + S3. "
            "See docs/tech-debt.md TD-006."
        )

    @_require_pytorchjob
    def test_two_node_manifest_mentions_world_size_or_torchrun(self) -> None:
        """The manifest must reference distributed training configuration.

        A valid two-node DDP manifest must mention either:
          - WORLD_SIZE (environment variable for distributed setup)
          - torchrun (the PyTorch distributed launcher)

        If neither is present, the manifest does not configure DDP correctly
        and both pods would train independently rather than in coordination.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        has_world_size = "WORLD_SIZE" in content
        has_torchrun = "torchrun" in content

        assert has_world_size or has_torchrun, (
            "Two-node manifest does not mention WORLD_SIZE or torchrun. "
            "A valid DDP manifest must configure the distributed launcher. "
            "Expected at least one of: WORLD_SIZE env var, torchrun entrypoint."
        )

    @_require_pytorchjob
    def test_two_node_manifest_has_master_and_worker(self) -> None:
        """The manifest must define both a Master replica and a Worker replica.

        KFTO PyTorchJob two-node topology:
          pytorchReplicaSpecs.Master (rank 0) — rendezvous initiator
          pytorchReplicaSpecs.Worker (rank 1) — connects to Master

        KFTO injects MASTER_ADDR and MASTER_PORT into every replica pod.
        If either replica type is missing, the DDP ring cannot form.
        """
        content = _TWO_NODE_MANIFEST.read_text()

        assert "Master" in content, (
            "Two-node manifest does not define a Master replica. "
            "Expected: pytorchReplicaSpecs.Master with replicas: 1."
        )
        assert "Worker" in content, (
            "Two-node manifest does not define a Worker replica. "
            "Expected: pytorchReplicaSpecs.Worker with replicas: 1."
        )


# ---------------------------------------------------------------------------
# Execution smoke — future / xfail
# ---------------------------------------------------------------------------

class TestPyTorchJobSmoke:
    """Level 4: PyTorchJob two-node execution smoke (future)."""

    @_require_pytorchjob
    @pytest.mark.xfail(
        reason=(
            "PyTorchJob execution smoke is opt-in future work. "
            "Implement the safe apply + wait + cleanup path before enabling. "
            "Expected behaviour when implemented: copy manifest to temp file, "
            "patch name to include test_id, add test labels, apply via oc apply, "
            "wait for Master and Worker pods to complete, verify logs include "
            "'WORLD_SIZE=2', 'DDP initialized', and '--max-steps' reached, "
            "cleanup only labelled resources."
        ),
        strict=False,  # allow xpass when implementation lands
    )
    def test_pytorchjob_two_node_smoke_future(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        tmp_path: pathlib.Path,
        cleanup_labelled_resources: None,
    ) -> None:
        """Apply the two-node PyTorchJob manifest and verify DDP execution.

        Future implementation steps (do not implement until KFTO is verified):

        1. Read openshift/training/pytorchjob-pragma-s-2node.yaml.

        2. Parse YAML and patch:
             metadata.name: pragma-s-smoke-<test_id>  (unique, no collision)
             metadata.labels: add test labels (both pragma.redhat.com/* keys)

        3. Write patched manifest to tmp_path / "pytorchjob-smoke.yaml".

        4. Apply via oc apply (not oc create):
               oc(["apply", "-f", str(patched_yaml)], namespace=runtime_namespace)
           This is safe to re-apply and does not modify any Argo-managed resource.

        5. Wait for Master and Worker pods to reach Running state,
           polling every 10 seconds up to timeout_seconds.
           Identify pods by label: job-name=pragma-s-smoke-<test_id>.

        6. Wait for job completion (status.completionTime set).

        7. Fetch pod logs and assert:
               "WORLD_SIZE=2" in logs (distributed config)
               "DDP initialized" or "master addr" in logs (rendezvous)
               "--max-steps" or "max_steps" in logs (training ran)

        8. Assert PyTorchJob reached Succeeded condition.

        9. Cleanup is handled by the cleanup_labelled_resources fixture.
           Only resources with both test labels are deleted.

        Known limitation (TD-006): --resume does not work with emptyDir.
        Use max_steps=1 or max_steps=20 only (no checkpoint left behind).
        """
        pytest.xfail(
            "PyTorchJob execution smoke is opt-in future work. "
            "This test will xpass once the safe apply + wait + cleanup path "
            "is implemented."
        )
