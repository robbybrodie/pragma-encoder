"""Level 3 — OpenShift AI KFP v2 pipeline smoke (future / xfail).

Purpose:
  Verify end-to-end PRAGMA pipeline execution via OpenShift AI Data Science
  Pipelines (DSPA / KFP v2). A minimal PRAGMA-S pipeline is compiled,
  uploaded to the KFP v2 API server, a Run is created with max_steps=1,
  and its completion and logs are verified.

Runtime:
  This suite targets OpenShift AI Pipelines backed by KFP v2 via
  DataSciencePipelinesApplication (DSPA), NOT Tekton PipelineRun/TaskRun.
  The pipeline backend is the ds-pipeline-* pods running in the namespace.

Current status: FUTURE / XFAIL
  KFP v2 run submission via the DSPA API is not yet implemented.
  All tests in this file are xfail because the submission path
  (upload pipeline YAML + create Run via KFP SDK or Pipeline/Run CR)
  has not been approved and implemented.

When implemented, this suite will:
  1. Compile the decorated pipeline to a KFP v2 YAML.
  2. Upload the pipeline YAML to the DSPA KFP v2 API server.
  3. Create a KFP v2 Run with max_steps=1 and test labels.
  4. Poll run status until Succeeded or Failed (or timeout).
  5. On failure: collect pod describe and logs for diagnosis.
  6. Assert the run status is Succeeded.
  7. Assert pod logs contain 'PRAGMA-S' and max_steps completion marker.
  8. Cleanup: delete all resources carrying both test labels.

Submission mechanism (to be decided before implementation):
  Option A: KFP SDK v2 client pointed at the DSPA route with bearer token.
  Option B: Create Pipeline/PipelineVersion/Run CRs via oc apply.
  Do not implement until the mechanism is explicitly approved.

Safety rules:
  - Skip unless RUN_OPENSHIFT_PIPELINE_SMOKE=1.
  - Create only resources labelled with test-run=true, test-id=<test_id>.
  - Cleanup deletes only labelled resources.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.
  - oc exec is for diagnostics only — never the pipeline submission path.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - Level 0 and Level 1 (DSPA/KFP v2) tests passing
  - DSPA pods Running in namespace (test_01 verified)
  - Training container image accessible in the cluster registry
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
        "OpenShift AI KFP v2 pipeline smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_PIPELINE_SMOKE=1 to enable. "
        "Warning: these tests will create short-lived pipeline Run resources "
        "in PRAGMA_TEST_RUNTIME_NAMESPACE when implemented."
    ),
)


# ---------------------------------------------------------------------------
# Skeleton smoke test — xfail until KFP v2 run submission is implemented
# ---------------------------------------------------------------------------


class TestKFPPipelineSmoke:
    """Level 3: OpenShift AI KFP v2 pipeline runtime smoke (future).

    Targets the DataSciencePipelinesApplication (DSPA) / KFP v2 runtime,
    NOT Tekton PipelineRun. The DSPA API server (ds-pipeline-*) is the
    submission endpoint.
    """

    @_require_smoke
    @pytest.mark.xfail(
        reason=(
            "KFP v2 pipeline run submission via OpenShift AI DSPA is not implemented yet. "
            "The submission mechanism (KFP SDK client or Pipeline/Run CR) has not been "
            "approved. This test will xpass once the submission path is implemented "
            "and approved in src/workbench/. "
            "Expected behaviour when implemented: "
            "compile PRAGMA-S decorated pipeline to KFP v2 YAML, upload to DSPA API, "
            "create a Run with max_steps=1, poll until Succeeded, "
            "assert logs contain 'PRAGMA-S' and max_steps completion marker, "
            "cleanup all labelled pipeline Run resources."
        ),
        strict=False,  # allow xpass when implementation lands
    )
    def test_kfp_pipeline_smoke_run_future(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        cleanup_labelled_resources: None,  # fixture — triggers cleanup after test
    ) -> None:
        """Compile, upload, and run a tiny PRAGMA-S KFP v2 pipeline.

        Future implementation steps (do not implement until submission path
        is approved — see module docstring):

        1. Compile the decorated pipeline to a temp KFP v2 YAML:
               @pragma_pipeline(name=f"pragma-s-smoke-{test_id}")
               def _smoke():
                   ds = dataset("ibm-tabformer", prepare_if_missing=True)
                   train(dataset=ds, model_size="S", epochs=1, max_steps=1)
               _smoke.compile(str(tmp_path / "smoke.yaml"))

        2. Discover DSPA endpoint in runtime_namespace:
               dspa_host = <ds-pipeline-* service route or in-cluster URL>

        3. Upload the compiled pipeline YAML to the DSPA KFP v2 API:
               Method to be approved: KFP SDK client or Pipeline CR.
               All uploaded resources must carry both test labels.

        4. Create a KFP v2 Run with max_steps=1:
               Pass test labels as pipeline run annotations/labels.
               Record the run ID for status polling.

        5. Poll run status every 10 seconds up to timeout_seconds:
               GET /apis/v2beta1/runs/<run_id>
               Accept: Succeeded or Failed (not just RUNNING).

        6. On failure: collect logs from the workflow executor pods.
               Use oc logs (for diagnostics only — not the product path).
               Never print secret values.

        7. Assert final run status is Succeeded.

        8. Assert pod logs contain:
               'PRAGMA-S'             (model size confirmation)
               'Reached --max-steps 1' (training completed minimal step)

        9. Cleanup is handled by the cleanup_labelled_resources fixture.
               Only resources with both test labels are deleted.

        This test currently xfails because the DSPA run submission path
        is not yet implemented. Do not fake success — an xpass indicates
        the implementation has landed and been approved.
        """
        pytest.xfail(
            "KFP v2 pipeline run submission via OpenShift AI DSPA is not implemented yet. "
            "Approve and implement the submission mechanism before enabling this test."
        )
