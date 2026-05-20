"""Level 3 — DSPA / KFP v2 pipeline runtime smoke (future / xfail).

Purpose:
  Verify end-to-end PRAGMA pipeline execution via OpenShift AI Data Science
  Pipelines (DSPA) / KFP v2. A minimal PRAGMA-S pipeline run is submitted
  with max_steps=1, allowed to complete (or timeout), and its logs are
  inspected.

Pipeline runtime context:
  This environment uses OpenShift AI Data Science Pipelines (DSPA) / KFP v2.
  Red Hat OpenShift AI Data Science Pipelines 2.0 does NOT use kfp-tekton.
  The submission path is:
    1. Compile KFP v2 pipeline to YAML.
    2. Upload/import pipeline to DSPA via the KFP v2 API.
    3. Create a pipeline run via the DSPA / KFP v2 API.
    4. Wait for the run to complete.
    5. Verify logs. Clean up labelled resources.

  Tekton PipelineRun/TaskRun are NOT part of this execution path.
  Use RUN_TEKTON_TESTS=1 only for alternate-runtime cluster validation.

Current status: FUTURE / XFAIL
  DSPA / KFP v2 pipeline submission is not yet implemented in src/workbench/.
  All tests in this file are either:
    - Skipped unless RUN_OPENSHIFT_PIPELINE_SMOKE=1
    - Marked xfail because the submission path is not implemented

  See test_03_dspa_runtime_discovery.py for read-only discovery of the
  DSPA substrate (already implemented, does not create runs).

Safety rules:
  - Skip unless RUN_OPENSHIFT_PIPELINE_SMOKE=1.
  - Create only resources labelled with test-run=true, test-id=<test_id>.
  - Cleanup deletes only labelled resources.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.
  - Do not use oc exec as a pipeline execution path.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - Level 0, 1, and 3d tests passing
  - DSPA instance healthy in PRAGMA_TEST_NAMESPACE
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Suite-level skip guard
# ---------------------------------------------------------------------------

_PIPELINE_SMOKE_ENABLED = os.environ.get("RUN_OPENSHIFT_PIPELINE_SMOKE") == "1"

_require_smoke = pytest.mark.skipif(
    not _PIPELINE_SMOKE_ENABLED,
    reason=(
        "DSPA / KFP v2 pipeline smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_PIPELINE_SMOKE=1 to enable. "
        "Warning: these tests upload a pipeline to DSPA and create a run "
        "in PRAGMA_TEST_RUNTIME_NAMESPACE."
    ),
)


# ---------------------------------------------------------------------------
# Skeleton smoke test — xfail until DSPA submission is implemented
# ---------------------------------------------------------------------------

class TestOpenShiftPipelineSmoke:
    """Level 3: DSPA / KFP v2 pipeline runtime smoke (future).

    This test verifies end-to-end pipeline execution via OpenShift AI
    Data Science Pipelines (DSPA) / KFP v2. It is NOT a Tekton PipelineRun.

    The submission path (when implemented):
      1. Compile decorated pipeline to KFP v2 YAML.
      2. Upload/import pipeline to DSPA via the KFP v2 API.
      3. Create a pipeline run via the DSPA / KFP v2 API.
      4. Wait for completion. Verify logs. Clean up.

    See test_03_dspa_runtime_discovery.py for the implemented read-only
    discovery layer that maps the DSPA substrate.
    """

    @_require_smoke
    @pytest.mark.xfail(
        reason=(
            "DSPA / KFP v2 pipeline submission is not implemented yet. "
            "Implement pipeline upload + run creation in src/workbench/ "
            "(ADR 004) targeting the DSPA API before enabling this test. "
            "Expected behaviour when implemented: "
            "upload compiled KFP v2 YAML to DSPA, create a pipeline run "
            "with max_steps=1, wait for completion, "
            "assert logs contain 'PRAGMA-S' and 'Reached --max-steps 1', "
            "cleanup all labelled resources."
        ),
        strict=False,  # allow xpass when implementation lands
    )
    def test_pipeline_smoke_run_future(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        cleanup_labelled_resources: None,  # fixture — triggers cleanup after test
    ) -> None:
        """Submit a tiny PRAGMA-S PipelineRun and verify it completes.

        Future implementation steps (do not implement until submission path exists):

        1. Compile the decorated pipeline to a temp YAML:
               @pragma_pipeline(name=f"pragma-s-smoke-{test_id}")
               def _smoke():
                   ds = dataset("ibm-tabformer", prepare_if_missing=True)
                   train(dataset=ds, model_size="S", epochs=1, max_steps=1)
               _smoke.compile(str(tmp_path / "smoke.yaml"))

        2. Patch the compiled YAML to inject test labels into the PipelineRun
           metadata (pragma.redhat.com/test-run=true, test-id=<test_id>).

        3. Submit via oc apply:
               oc(["apply", "-f", str(patched_yaml)], namespace=runtime_namespace)

        4. Wait for PipelineRun to reach Succeeded or Failed state,
           polling every 10 seconds up to timeout_seconds.

        5. On failure: fetch describe and pod logs for diagnosis.
           Never print secret data.

        6. Assert final condition is Succeeded.

        7. Assert pod logs contain:
               "PRAGMA-S" (model size confirmation)
               "Reached --max-steps 1" (training completed minimal step)

        8. Cleanup is handled by the cleanup_labelled_resources fixture.
           Only resources with both test labels are deleted.

        This test currently xfails because step 3 (submit) is not implemented.
        Do not fake success — an xpass indicates the implementation has landed.
        """
        pytest.xfail(
            "DSPA / KFP v2 pipeline submission is not implemented yet. "
            "This test will xpass once DSPA pipeline upload + run creation "
            "is implemented in src/workbench/."
        )
