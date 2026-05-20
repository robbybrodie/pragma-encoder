"""Level 3b — Training container batch/v1 Job smoke (opt-in / future xfail).

Purpose:
  Prove that the PRAGMA training container image runs correctly in-cluster
  by submitting a labelled batch/v1 Job with --max-steps 1.

  This is NOT a pipeline runtime test. It does not prove:
    - KFP v2 / DSPA pipeline orchestration
    - Argo workflow execution
    - Any pipeline CRD behaviour

  It proves only that the training container:
    - Is pullable from the cluster registry
    - Starts without error
    - Runs the training script with the PRAGMA-S model config
    - Reaches --max-steps 1 and exits 0

Runtime:
  This suite targets batch/v1 Job resources, NOT KFP v2 PipelineRun,
  Tekton PipelineRun, or PyTorchJob. It is a minimal sanity check for the
  training image independently of the pipeline orchestration layer.

Current status: FUTURE / XFAIL
  The training container image is not yet built or pushed to the cluster
  registry. The Job submission path requires:
    1. A built and pushed training image (PRAGMA_TRAINING_IMAGE env var).
    2. An S3-accessible dataset shard (or --dry-run stub in the image).
  All tests in this file are xfail because the training image is not
  yet available. Do not implement until the image is confirmed accessible.

When implemented, this suite will:
  1. Derive the training image URI from PRAGMA_TRAINING_IMAGE.
  2. Build a minimal batch/v1 Job spec with:
       image: <PRAGMA_TRAINING_IMAGE>
       args: ["--model-size", "S", "--max-steps", "1"]
       labels: test labels (both pragma.redhat.com/* keys)
       imagePullPolicy: Always
       restartPolicy: Never
  3. Apply the Job via oc apply (not oc create) for idempotency.
  4. Poll Job status until Complete or Failed (or timeout).
  5. On failure: collect pod logs for diagnosis (oc logs, NOT oc exec).
  6. Assert Job status.conditions contains type=Complete.
  7. Assert pod logs contain:
       'PRAGMA-S'          (model size confirmation)
       'Reached --max-steps 1' or 'max_steps' (training ran)
  8. Cleanup: delete only resources carrying both test labels.

Submission mechanism:
  batch/v1 Job applied via oc apply with a JSON or YAML manifest.
  No KFP SDK, no Tekton, no oc exec for submission.
  oc exec is for diagnostics/log collection only if oc logs is insufficient.

Safety rules:
  - Skip unless RUN_OPENSHIFT_TESTS=1 (suite-level guard in conftest.py).
  - Skip unless RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 (file-level guard).
  - Create only resources labelled with both test labels.
  - Cleanup deletes only labelled resources.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.
  - oc exec is for diagnostics only — never the job submission path.
  - Secret data (S3 credentials, registry tokens) is never printed.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - PRAGMA_TRAINING_IMAGE=<image-uri>
    e.g. image-registry.openshift-image-registry.svc:5000/
         pragma-encoder/pragma-encoder-training:latest
  - Level 0 and Level 1 (DSPA/KFP v2) tests passing
  - Training container image accessible in the cluster registry
"""

from __future__ import annotations

import os

import pytest

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
        "Warning: these tests will create a short-lived batch/v1 Job "
        "in PRAGMA_TEST_RUNTIME_NAMESPACE when implemented. "
        "Also requires PRAGMA_TRAINING_IMAGE=<image-uri>."
    ),
)


# ---------------------------------------------------------------------------
# Skeleton smoke test — xfail until training image is available
# ---------------------------------------------------------------------------


class TestTrainingJobSmoke:
    """Level 3b: PRAGMA training container batch/v1 Job smoke (future).

    Targets batch/v1 Job resources only. This is not a pipeline runtime
    test — it proves the training image runs in-cluster independently of
    KFP v2 / DSPA / Tekton.
    """

    @_require_training_job_smoke
    @pytest.mark.xfail(
        reason=(
            "Training container Job smoke is not implemented yet. "
            "The PRAGMA training image (PRAGMA_TRAINING_IMAGE) has not been built "
            "and pushed to the cluster registry. "
            "This test will xpass once: "
            "(1) the training image is built and accessible in-cluster, "
            "(2) the batch/v1 Job apply + wait + log assertion path is implemented here. "
            "Expected behaviour when implemented: "
            "submit a labelled batch/v1 Job running the training container with "
            "--max-steps 1, wait for Job completion, collect pod logs, "
            "assert logs contain 'PRAGMA-S' and '--max-steps' completion marker, "
            "cleanup all labelled Job resources."
        ),
        strict=False,  # allow xpass when implementation lands
    )
    def test_training_job_smoke_future(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        cleanup_labelled_resources: None,  # fixture — triggers cleanup after test
    ) -> None:
        """Submit a labelled batch/v1 Job and verify the training container runs.

        Future implementation steps (do not implement until training image is
        confirmed accessible in the cluster registry — see module docstring):

        1. Read PRAGMA_TRAINING_IMAGE from the environment.
           Skip (not xfail) if the variable is not set:
               image_uri = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
               if not image_uri:
                   pytest.skip("PRAGMA_TRAINING_IMAGE is not set; skipping smoke.")

        2. Build a minimal batch/v1 Job manifest as a Python dict:
               job_manifest = {
                   "apiVersion": "batch/v1",
                   "kind": "Job",
                   "metadata": {
                       "name": f"pragma-smoke-{test_id}",
                       "namespace": runtime_namespace,
                       "labels": {
                           **test_labels,
                           "app": "pragma-training-smoke",
                       },
                   },
                   "spec": {
                       "backoffLimit": 0,
                       "template": {
                           "metadata": {"labels": test_labels},
                           "spec": {
                               "restartPolicy": "Never",
                               "containers": [{
                                   "name": "training",
                                   "image": image_uri,
                                   "imagePullPolicy": "Always",
                                   "args": [
                                       "--model-size", "S",
                                       "--max-steps", "1",
                                   ],
                               }],
                           },
                       },
                   },
               }

        3. Serialise to a temp YAML file and apply via oc apply (not oc create):
               import json, tempfile, pathlib
               manifest_path = tmp_path / "pragma-smoke-job.json"
               manifest_path.write_text(json.dumps(job_manifest))
               oc(["apply", "-f", str(manifest_path)], namespace=runtime_namespace)

        4. Poll Job status every 10 seconds up to timeout_seconds:
               GET .status.conditions[] where type in (Complete, Failed).

        5. On failure: collect pod logs via oc logs (not oc exec).
               Identify pods by label selector from test_labels.
               Do not print secret values from env or volumes.

        6. Assert Job status.conditions contains type=Complete (not Failed).

        7. Assert pod logs contain:
               'PRAGMA-S'   — model size confirmed
               '--max-steps' or 'max_steps' — training loop ran

        8. Cleanup is handled by the cleanup_labelled_resources fixture.
           Only resources with both test labels are deleted.

        This test currently xfails because the training image is not yet
        available. Do not fake success — an xpass indicates the implementation
        has landed and the image is accessible in the cluster registry.
        """
        pytest.xfail(
            "Training container Job smoke is not implemented yet. "
            "Build and push the PRAGMA training image, then set "
            "PRAGMA_TRAINING_IMAGE=<image-uri> before enabling this test."
        )
