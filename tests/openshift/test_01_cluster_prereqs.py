"""Level 1 — Argo-managed platform substrate verification.

Verifies that the long-lived substrate resources deployed by Argo CD are
present and correctly configured. These tests are read-only — they verify
existence only and never read secret data.

Pipeline runtime context:
  This environment uses OpenShift AI Data Science Pipelines (DSPA) / KFP v2.
  Red Hat OpenShift AI Data Science Pipelines 2.0 does not use kfp-tekton.
  Tekton CRDs (pipelineruns.tekton.dev, taskruns.tekton.dev) are NOT installed
  by default and are not required for PRAGMA pipeline execution.

Substrate resources verified by default:
  - DSPA/KFP v2 CRDs (DataSciencePipelinesApplication, pipelines.kubeflow.org)
  - KFTO PyTorchJob CRD (optional, gated by RUN_PYTORCHJOB_TESTS=1)
  - PRAGMA training ServiceAccount
  - S3 secret existence (optional, gated by PRAGMA_S3_SECRET_NAME)
  - Image pull secret existence (optional, gated by PRAGMA_IMAGE_PULL_SECRET_NAME)

Optional/legacy Tekton substrate (gated by RUN_TEKTON_TESTS=1):
  - pipelineruns.tekton.dev
  - taskruns.tekton.dev

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

from tests.openshift.oc import crd_exists, resource_exists

# Default service account name — overridden by PRAGMA_TRAINING_SERVICE_ACCOUNT.
_DEFAULT_SA = "pragma-encoder-training"

# Skip guard for Tekton-specific tests.
_require_tekton = pytest.mark.skipif(
    os.environ.get("RUN_TEKTON_TESTS") != "1",
    reason=(
        "Tekton CRD checks are opt-in. "
        "This environment uses OpenShift AI DSPA / KFP v2, not Tekton. "
        "Set RUN_TEKTON_TESTS=1 to enable Tekton substrate checks."
    ),
)


class TestDSPACRDs:
    """Level 1: OpenShift AI Data Science Pipelines (DSPA) / KFP v2 CRD presence.

    This is the default pipeline substrate for PRAGMA.
    OpenShift AI Data Science Pipelines 2.0 uses KFP v2 and does NOT use
    kfp-tekton. These CRDs are installed by the RHOAI operator.
    """

    def test_dspa_crd_exists(self) -> None:
        """CRD datasciencepipelinesapplications must exist on the cluster.

        The DataSciencePipelinesApplication CRD is installed by the Red Hat
        OpenShift AI (RHOAI) operator. It manages the DSPA instance that
        provides the KFP v2 API server for pipeline submission.

        If this fails: verify the RHOAI operator is installed and the
        DataSciencePipelines component is enabled in the DataScienceCluster.
        CRD: datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io
        """
        assert crd_exists(
            "datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io"
        ), (
            "CRD datasciencepipelinesapplications...opendatahub.io not found. "
            "Verify the Red Hat OpenShift AI operator is installed and the "
            "DataSciencePipelines component is enabled in the DataScienceCluster CR."
        )

    def test_kfp_pipeline_crd_exists(self) -> None:
        """CRD pipelines.pipelines.kubeflow.org must exist on the cluster.

        This CRD is installed by the DSPA operator alongside the KFP v2 API
        server. Its presence confirms KFP v2 is installed and functional.

        If this fails: verify the DSPA instance is healthy in PRAGMA_TEST_NAMESPACE
        and that the RHOAI operator has finished reconciling.
        """
        assert crd_exists("pipelines.pipelines.kubeflow.org"), (
            "CRD pipelines.pipelines.kubeflow.org not found. "
            "The DSPA / KFP v2 API server may not be fully installed. "
            "Check: oc get dspa -n <namespace> and oc get pods -n <namespace>."
        )


class TestTektonCRDs:
    """Level 1: Tekton/OpenShift Pipelines CRD presence (opt-in, legacy/alternate runtime).

    Tekton is NOT the default pipeline runtime for PRAGMA.
    This environment uses OpenShift AI DSPA / KFP v2.

    These tests are gated by RUN_TEKTON_TESTS=1. They should only be run
    when validating a cluster that has the OpenShift Pipelines operator installed
    as an alternate or legacy runtime (e.g. a different cluster configuration).
    """

    @_require_tekton
    def test_pipelinerun_crd_exists(self) -> None:
        """CRD pipelineruns.tekton.dev must exist when RUN_TEKTON_TESTS=1.

        OpenShift Pipelines (Tekton) must be installed and the operator must
        have registered its CRDs. This check is for alternate/legacy runtime
        validation only — PRAGMA pipelines use DSPA / KFP v2 by default.

        If this fails: verify OpenShift Pipelines operator is installed
        in the cluster (Operator Hub → Red Hat OpenShift Pipelines).
        """
        assert crd_exists("pipelineruns.tekton.dev"), (
            "CRD pipelineruns.tekton.dev not found. "
            "Install the OpenShift Pipelines operator before running Tekton tests."
        )

    @_require_tekton
    def test_taskrun_crd_exists(self) -> None:
        """CRD taskruns.tekton.dev must exist when RUN_TEKTON_TESTS=1.

        Co-installed with pipelineruns.tekton.dev by the OpenShift Pipelines operator.
        If pipelineruns.tekton.dev passes but this fails, the operator installation
        is incomplete.
        """
        assert crd_exists("taskruns.tekton.dev"), (
            "CRD taskruns.tekton.dev not found. "
            "The OpenShift Pipelines operator may be partially installed."
        )


class TestKFTOCRDs:
    """Level 1: KFTO PyTorchJob CRD presence (opt-in)."""

    def test_pytorchjob_crd_exists_if_enabled(self) -> None:
        """CRD pytorchjobs.kubeflow.org must exist when RUN_PYTORCHJOB_TESTS=1.

        Gated by RUN_PYTORCHJOB_TESTS=1. Skipped by default.

        If enabled and the CRD is missing, the Training Operator (KFTO) is
        not installed. Install via: Operator Hub → OpenShift AI → Training Operator.
        """
        if os.environ.get("RUN_PYTORCHJOB_TESTS") != "1":
            pytest.skip(
                "PyTorchJob CRD check is opt-in. "
                "Set RUN_PYTORCHJOB_TESTS=1 to enable."
            )

        assert crd_exists("pytorchjobs.kubeflow.org"), (
            "CRD pytorchjobs.kubeflow.org not found. "
            "Install the Training Operator (KFTO) before running PyTorchJob tests. "
            "On RHOAI: Operator Hub → OpenShift AI."
        )


class TestSubstrateResources:
    """Level 1: Argo-managed runtime substrate resources."""

    def test_training_service_account_exists(self, test_namespace: str) -> None:
        """The PRAGMA training ServiceAccount must exist in PRAGMA_TEST_NAMESPACE.

        The service account is created by Argo CD as part of the PRAGMA platform
        substrate. Its name is read from PRAGMA_TRAINING_SERVICE_ACCOUNT
        (default: ``pragma-encoder-training``).

        This test verifies existence only. It does not read or modify the SA.
        If missing: check that Argo CD has synced the platform substrate to
        ``PRAGMA_TEST_NAMESPACE``.
        """
        sa_name = os.environ.get(
            "PRAGMA_TRAINING_SERVICE_ACCOUNT", _DEFAULT_SA
        ).strip() or _DEFAULT_SA

        assert resource_exists("serviceaccount", sa_name, namespace=test_namespace), (
            f"ServiceAccount {sa_name!r} not found in namespace {test_namespace!r}. "
            "This service account is expected to be deployed by Argo CD. "
            "Verify Argo CD has synced the PRAGMA substrate and that "
            "PRAGMA_TRAINING_SERVICE_ACCOUNT is set correctly if the SA name differs."
        )

    def test_s3_secret_exists_if_configured(self, test_namespace: str) -> None:
        """If PRAGMA_S3_SECRET_NAME is set, the named Secret must exist.

        Gated by PRAGMA_S3_SECRET_NAME. Skipped when unset.

        Verifies existence only. Secret data (keys, credentials) is never
        read or printed by this test.
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

        Verifies existence only. Secret data (registry credentials) is never
        read or printed by this test.
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
