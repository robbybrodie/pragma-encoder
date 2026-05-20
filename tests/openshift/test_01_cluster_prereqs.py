"""Level 1 — Argo-managed platform substrate verification.

Verifies that the long-lived substrate resources deployed by Argo CD are
present and correctly configured. All tests are read-only — they verify
existence and status only, and never read secret data.

This environment uses OpenShift AI Data Science Pipelines (DSPA) backed by
KFP v2. Tekton PipelineRun/TaskRun CRDs are NOT required here; Tekton checks
are opt-in via RUN_TEKTON_TESTS=1.

Substrate resources verified:

  OpenShift AI / KFP v2 (required):
    - DataSciencePipelinesApplication CRD
    - KFP v2 Pipeline and PipelineVersion CRDs
    - At least one DSPA instance in PRAGMA_TEST_NAMESPACE
    - DSPA runtime pods Running in PRAGMA_TEST_NAMESPACE

  Tekton/OpenShift Pipelines CRDs (opt-in, gated by RUN_TEKTON_TESTS=1):
    - pipelineruns.tekton.dev
    - taskruns.tekton.dev

  KFTO PyTorchJob CRD (opt-in, gated by RUN_PYTORCHJOB_TESTS=1):
    - pytorchjobs.kubeflow.org

  PRAGMA Argo-managed substrate resources:
    - PRAGMA training ServiceAccount
    - S3 secret existence (opt-in, gated by PRAGMA_S3_SECRET_NAME)
    - Image pull secret existence (opt-in, gated by PRAGMA_IMAGE_PULL_SECRET_NAME)

All tests are read-only. No resources are created or modified.
Secret data is never accessed or printed.

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - Level 0 tests passing (oc access)
"""

from __future__ import annotations

import os

import pytest

from tests.openshift.oc import crd_exists, oc, resource_exists

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SA = "pragma-encoder-training"

# Exact CRD names as registered on the cluster.
_DSPA_CRD = (
    "datasciencepipelinesapplications"
    ".datasciencepipelinesapplications.opendatahub.io"
)
_KFP_PIPELINE_CRD = "pipelines.pipelines.kubeflow.org"
_KFP_PIPELINEVERSION_CRD = "pipelineversions.pipelines.kubeflow.org"

# Label selector that identifies all DSPA-managed runtime pods.
_DSPA_COMPONENT_LABEL = "component=data-science-pipelines"


# ---------------------------------------------------------------------------
# TestDSPACRDs — OpenShift AI / KFP v2 CRD presence (required)
# ---------------------------------------------------------------------------


class TestDSPACRDs:
    """Level 1: OpenShift AI / KFP v2 CRD presence.

    These CRDs are registered by the OpenShift AI (RHOAI) operator when
    Data Science Pipelines support is enabled. They are required for DSPA
    management and KFP v2 pipeline submission.

    Failure means the OpenShift AI operator is not installed or has not
    registered its CRDs. Tekton CRDs are NOT checked here.
    """

    def test_dspa_crd_exists(self) -> None:
        """DataSciencePipelinesApplication CRD must be present on the cluster.

        CRD: datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io

        Registered by the OpenShift AI operator. Required for DSPA lifecycle
        management. If missing: verify the OpenShift AI (RHOAI) operator is
        installed via Operator Hub.
        """
        assert crd_exists(_DSPA_CRD), (
            f"CRD {_DSPA_CRD!r} not found on the cluster. "
            "Install the OpenShift AI (RHOAI) operator before running this suite. "
            "Operator Hub → Red Hat OpenShift AI."
        )

    def test_kfp_pipeline_crd_exists(self) -> None:
        """KFP v2 Pipeline CRD must be present.

        CRD: pipelines.pipelines.kubeflow.org

        Registered when a DataSciencePipelinesApplication is deployed.
        Required for pipeline upload and run submission via the KFP v2 API.
        """
        assert crd_exists(_KFP_PIPELINE_CRD), (
            f"CRD {_KFP_PIPELINE_CRD!r} not found. "
            "This CRD is registered when a DataSciencePipelinesApplication (DSPA) "
            "is deployed. Verify DSPA is present in the namespace."
        )

    def test_kfp_pipelineversion_crd_exists(self) -> None:
        """KFP v2 PipelineVersion CRD must be present.

        CRD: pipelineversions.pipelines.kubeflow.org

        Co-installed with pipelines.pipelines.kubeflow.org by the DSPA controller.
        If this is missing but pipelines.pipelines.kubeflow.org exists, the
        operator installation may be incomplete.
        """
        assert crd_exists(_KFP_PIPELINEVERSION_CRD), (
            f"CRD {_KFP_PIPELINEVERSION_CRD!r} not found. "
            "Co-installed with pipelines.pipelines.kubeflow.org. "
            "Verify the DSPA controller is fully initialised."
        )


# ---------------------------------------------------------------------------
# TestDSPASubstrate — DSPA instance and pods in PRAGMA_TEST_NAMESPACE (required)
# ---------------------------------------------------------------------------


class TestDSPASubstrate:
    """Level 1: DSPA instance and runtime pod health in PRAGMA_TEST_NAMESPACE.

    Verifies that the Argo-managed DataSciencePipelinesApplication and its
    runtime pods are present and Running. These are prerequisites for any
    pipeline submission or run-status check.
    """

    def test_dspa_instance_exists_in_namespace(self, test_namespace: str) -> None:
        """At least one DataSciencePipelinesApplication must exist in the namespace.

        Deployed by Argo CD as part of the PRAGMA platform substrate.
        Verifies existence only — no DSPA configuration is read.

        If missing: check that Argo CD has synced the DSPA resource to
        PRAGMA_TEST_NAMESPACE and that the DSPA manifest is committed to
        openshift/gitops/.
        """
        result = oc(
            ["get", "datasciencepipelinesapplication", "-o", "name"],
            namespace=test_namespace,
            check=False,
        )
        has_dspa = result.returncode == 0 and bool(result.stdout.strip())
        assert has_dspa, (
            f"No DataSciencePipelinesApplication found in namespace {test_namespace!r}. "
            "Verify Argo CD has synced the PRAGMA substrate. "
            "The DSPA resource should be committed under openshift/gitops/."
        )

    def test_dspa_pods_running_in_namespace(self, test_namespace: str) -> None:
        """DSPA runtime pods must be Running in PRAGMA_TEST_NAMESPACE.

        Checks for pods labelled component=data-science-pipelines that are
        in Running phase. A running DSPA backend is required before any
        pipeline upload or run submission can succeed.

        Expected pods (names vary by DSPA instance name):
          ds-pipeline-<dspa-name>-*         — KFP v2 API server
          ds-pipeline-workflow-controller-*  — Argo Workflow controller
          ds-pipeline-persistenceagent-*     — persistence agent

        If missing: check the DataSciencePipelinesApplication status and the
        OpenShift AI operator logs.
        """
        result = oc(
            ["get", "pods", "-l", _DSPA_COMPONENT_LABEL, "-o", "name"],
            namespace=test_namespace,
            check=False,
        )
        pod_names = [
            line.strip()
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        assert pod_names, (
            f"No pods with label {_DSPA_COMPONENT_LABEL!r} found "
            f"in namespace {test_namespace!r}. "
            "DSPA runtime pods must exist before pipeline submission tests can run. "
            "Check the DataSciencePipelinesApplication resource status."
        )
        # Confirm at least one pod is Running by checking field-selector.
        # oc --field-selector=status.phase=Running requires server support;
        # we do a second pass to confirm Running pods exist.
        result_running = oc(
            [
                "get", "pods",
                "-l", _DSPA_COMPONENT_LABEL,
                "--field-selector=status.phase=Running",
                "-o", "name",
            ],
            namespace=test_namespace,
            check=False,
        )
        running_pods = [
            line.strip()
            for line in result_running.stdout.strip().splitlines()
            if line.strip()
        ]
        assert running_pods, (
            f"DSPA pods exist in {test_namespace!r} but none are in Running phase. "
            f"Found pods: {pod_names}. "
            "Check the DSPA instance and OpenShift AI operator for errors."
        )


# ---------------------------------------------------------------------------
# TestTektonCRDs — opt-in (RUN_TEKTON_TESTS=1)
# ---------------------------------------------------------------------------


class TestTektonCRDs:
    """Level 1: Tekton/OpenShift Pipelines CRD presence (opt-in).

    Gated by RUN_TEKTON_TESTS=1. Skipped by default.

    Tekton PipelineRun/TaskRun CRDs are NOT required when the cluster uses
    OpenShift AI KFP v2 / DSPA as the pipeline runtime. Enable these checks
    only if the cluster explicitly uses the Tekton pipeline runtime.
    """

    def test_tekton_pipelinerun_crd_exists_if_enabled(self) -> None:
        """CRD pipelineruns.tekton.dev must exist when RUN_TEKTON_TESTS=1.

        Installed by the Red Hat OpenShift Pipelines operator.
        Skipped by default — Tekton is not required for OpenShift AI KFP v2.
        """
        if os.environ.get("RUN_TEKTON_TESTS") != "1":
            pytest.skip(
                "Tekton CRD check is opt-in. "
                "Set RUN_TEKTON_TESTS=1 to enable. "
                "Tekton PipelineRun/TaskRun is not required for OpenShift AI KFP v2 pipelines."
            )

        assert crd_exists("pipelineruns.tekton.dev"), (
            "CRD pipelineruns.tekton.dev not found. "
            "Install the OpenShift Pipelines operator (Operator Hub → "
            "Red Hat OpenShift Pipelines) or disable RUN_TEKTON_TESTS."
        )

    def test_tekton_taskrun_crd_exists_if_enabled(self) -> None:
        """CRD taskruns.tekton.dev must exist when RUN_TEKTON_TESTS=1.

        Co-installed with pipelineruns.tekton.dev by the OpenShift Pipelines
        operator. Skipped by default.
        """
        if os.environ.get("RUN_TEKTON_TESTS") != "1":
            pytest.skip(
                "Tekton CRD check is opt-in. "
                "Set RUN_TEKTON_TESTS=1 to enable. "
                "Tekton PipelineRun/TaskRun is not required for OpenShift AI KFP v2 pipelines."
            )

        assert crd_exists("taskruns.tekton.dev"), (
            "CRD taskruns.tekton.dev not found. "
            "The OpenShift Pipelines operator may be partially installed."
        )


# ---------------------------------------------------------------------------
# TestKFTOCRDs — opt-in (RUN_PYTORCHJOB_TESTS=1)
# ---------------------------------------------------------------------------


class TestKFTOCRDs:
    """Level 1: KFTO PyTorchJob CRD presence (opt-in).

    Gated by RUN_PYTORCHJOB_TESTS=1. Skipped by default.
    """

    def test_pytorchjob_crd_exists_if_enabled(self) -> None:
        """CRD pytorchjobs.kubeflow.org must exist when RUN_PYTORCHJOB_TESTS=1.

        Installed by the OpenShift AI Training Operator (KFTO). Skipped by
        default — enable only when running Level 4 PyTorchJob smoke tests.

        If enabled and missing: verify the Training Operator is installed via
        Operator Hub → Red Hat OpenShift AI.
        """
        if os.environ.get("RUN_PYTORCHJOB_TESTS") != "1":
            pytest.skip(
                "PyTorchJob CRD check is opt-in. "
                "Set RUN_PYTORCHJOB_TESTS=1 to enable."
            )

        assert crd_exists("pytorchjobs.kubeflow.org"), (
            "CRD pytorchjobs.kubeflow.org not found. "
            "Install the Training Operator (KFTO) via Operator Hub → "
            "Red Hat OpenShift AI, or disable RUN_PYTORCHJOB_TESTS."
        )


# ---------------------------------------------------------------------------
# TestSubstrateResources — Argo-managed runtime resources
# ---------------------------------------------------------------------------


class TestSubstrateResources:
    """Level 1: Argo-managed runtime substrate resources.

    Verifies that resources deployed by Argo CD are present.
    All checks are existence-only — no resource data is read.
    """

    def test_training_service_account_exists(self, test_namespace: str) -> None:
        """The PRAGMA training ServiceAccount must exist in PRAGMA_TEST_NAMESPACE.

        Deployed by Argo CD as part of the PRAGMA platform substrate.
        Name is read from PRAGMA_TRAINING_SERVICE_ACCOUNT
        (default: ``pragma-encoder-training``).

        Verifies existence only. Does not read or modify the SA.
        If missing: check that Argo CD has synced the platform substrate.
        """
        sa_name = (
            os.environ.get("PRAGMA_TRAINING_SERVICE_ACCOUNT", _DEFAULT_SA).strip()
            or _DEFAULT_SA
        )
        assert resource_exists("serviceaccount", sa_name, namespace=test_namespace), (
            f"ServiceAccount {sa_name!r} not found in namespace {test_namespace!r}. "
            "This SA is expected to be deployed by Argo CD. "
            "Verify Argo CD has synced the PRAGMA substrate and that "
            "PRAGMA_TRAINING_SERVICE_ACCOUNT is set correctly if the SA name differs."
        )

    def test_s3_secret_exists_if_configured(self, test_namespace: str) -> None:
        """If PRAGMA_S3_SECRET_NAME is set, the named Secret must exist.

        Gated by PRAGMA_S3_SECRET_NAME. Skipped when unset.
        Verifies existence only — secret data is never read or printed.
        """
        secret_name = os.environ.get("PRAGMA_S3_SECRET_NAME", "").strip()
        if not secret_name:
            pytest.skip(
                "PRAGMA_S3_SECRET_NAME is not set. "
                "Set it to the name of the S3 credentials secret to verify its existence."
            )

        assert resource_exists("secret", secret_name, namespace=test_namespace), (
            f"S3 secret {secret_name!r} not found in namespace {test_namespace!r}. "
            "Verify the secret exists and PRAGMA_S3_SECRET_NAME is correct. "
            "Secret data is never read by this test."
        )

    def test_image_pull_secret_exists_if_configured(self, test_namespace: str) -> None:
        """If PRAGMA_IMAGE_PULL_SECRET_NAME is set, the named Secret must exist.

        Gated by PRAGMA_IMAGE_PULL_SECRET_NAME. Skipped when unset.
        Verifies existence only — registry credentials are never read or printed.
        """
        secret_name = os.environ.get("PRAGMA_IMAGE_PULL_SECRET_NAME", "").strip()
        if not secret_name:
            pytest.skip(
                "PRAGMA_IMAGE_PULL_SECRET_NAME is not set. "
                "Set it to the name of the image pull secret to verify its existence."
            )

        assert resource_exists("secret", secret_name, namespace=test_namespace), (
            f"Image pull secret {secret_name!r} not found in namespace {test_namespace!r}. "
            "Verify the secret exists and PRAGMA_IMAGE_PULL_SECRET_NAME is correct. "
            "Secret data is never read by this test."
        )
