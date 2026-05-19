"""Level 3 — OpenShift Pipelines runtime smoke (future / xfail).

Purpose:
  Verify end-to-end PRAGMA pipeline execution on OpenShift Pipelines
  (Tekton). A minimal PRAGMA-S PipelineRun is submitted with max_steps=1,
  allowed to complete (or timeout), and its logs are inspected.

Current status: FUTURE / XFAIL
  OpenShift Pipeline runtime submission is not yet implemented.
  All tests in this file are either:
    - Skipped unless RUN_OPENSHIFT_PIPELINE_SMOKE=1
    - Marked xfail because the submission path is not implemented

When implemented, this suite will:
  1. Compile the decorated pipeline to YAML.
  2. Submit a PipelineRun via oc apply with test labels.
  3. Wait for completion (up to PRAGMA_TEST_TIMEOUT_SECONDS).
  4. Fetch and inspect Pod logs on failure.
  5. Assert the run succeeded and logged expected output.
  6. Clean up all labelled PipelineRun/TaskRun/Pod resources.

Safety rules:
  - Skip unless RUN_OPENSHIFT_PIPELINE_SMOKE=1.
  - Create only resources labelled with test-run=true, test-id=<test_id>.
  - Cleanup deletes only labelled resources.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - Level 0 and Level 1 tests passing
  - OpenShift Pipelines CRDs present (test_01 verified)
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
        "OpenShift Pipeline smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_PIPELINE_SMOKE=1 to enable. "
        "Warning: these tests create short-lived PipelineRun resources in "
        "PRAGMA_TEST_RUNTIME_NAMESPACE."
    ),
)


# ---------------------------------------------------------------------------
# Skeleton smoke test — xfail until submission is implemented
# ---------------------------------------------------------------------------

class TestOpenShiftPipelineSmoke:
    """Level 3: OpenShift Pipelines runtime execution smoke (future)."""

    @_require_smoke
    @pytest.mark.xfail(
        reason=(
            "OpenShift Pipeline runtime submission is not implemented yet. "
            "Implement pipeline submission in src/workbench/ (ADR 004) before "
            "enabling this test. Expected behaviour when implemented: "
            "submit PRAGMA-S PipelineRun with max_steps=1, wait for completion, "
            "assert logs contain 'PRAGMA-S' and 'Reached --max-steps 1', "
            "cleanup all labelled PipelineRun/TaskRun/Pod resources."
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
            "OpenShift Pipeline runtime submission is not implemented yet. "
            "This test will xpass once pipeline submission is implemented."
        )
