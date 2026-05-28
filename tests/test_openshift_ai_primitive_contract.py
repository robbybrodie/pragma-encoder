"""Static primitive contract tests for the OpenShift AI pathway.

These tests verify that platform-facing YAML fixtures, manifests, and resource
definitions conform to the expected OpenShift AI/OpenShift primitive contracts.

All tests are:
  - Static (no cluster, no network, no S3, no GPU)
  - Read-only (no cluster state mutated)
  - Pure Python (no OpenShift SDK required)
  - Runnable in any Python environment with PyYAML

The contract being tested:
  - Data Science Project / namespace
  - ArgoCD Application (syncs declared platform state; not runtime objects)
  - Connection / data connection (Secret with RHOAI annotations)
  - HardwareProfile (infrastructure.opendatahub.io/v1; GA in RHOAI 3.3)
  - DSPA / DataSciencePipelinesApplication
  - Workbench (Notebook CR)
  - PyTorchJob manifests (wheel-based entrypoints, no git clone)
  - TrainJob (clearly marked Tech Preview; example only)
  - InferenceService / ServingRuntime (platform resources, not wheel)
  - Boundary: no platform YAML inside src/pragma_encoder
  - Boundary: pragma_encoder.workbench not importable from wheel
  - Boundary: ArgoCD does not sync runtime-only objects

Reference: docs/openshift-ai-primitives.md
Reference: docs/openshift-ai-3.3-alignment.md
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import yaml

# ---------------------------------------------------------------------------
# Repository root and canonical paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent

_FIXTURES = _REPO_ROOT / "tests" / "openshift" / "fixtures"
_GITOPS = _REPO_ROOT / "openshift" / "gitops"
_TRAINING = _REPO_ROOT / "openshift" / "training"
_SERVING = _REPO_ROOT / "openshift" / "serving"
_ARGOCD = _REPO_ROOT / "openshift" / "argocd"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: pathlib.Path) -> dict:
    """Load a single-document YAML file (or first document of a multi-doc file)."""
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[return-value]


def _load_yaml_all(path: pathlib.Path) -> list[dict]:
    """Load all documents from a multi-document YAML file (--- separated)."""
    with open(path) as f:
        return [doc for doc in yaml.safe_load_all(f) if doc is not None]


# ===========================================================================
# 1. Data Science Project / Namespace
# ===========================================================================


class TestNamespaceDataScienceProject:
    """Namespace carries required labels for RHOAI Data Science Project visibility.

    The OpenShift AI dashboard lists a namespace as a Data Science Project when
    it has the label opendatahub.io/dashboard: 'true'.

    The namespace is managed by ArgoCD (wave -1) so it exists before all other
    resources are applied. CreateNamespace=false in the ArgoCD Application prevents
    conflicts with existing namespaces.
    """

    def test_namespace_file_exists(self) -> None:
        assert (_GITOPS / "namespace.yaml").exists(), (
            "openshift/gitops/namespace.yaml must exist. "
            "This is the Data Science Project definition."
        )

    def test_namespace_kind(self) -> None:
        doc = _load_yaml(_GITOPS / "namespace.yaml")
        assert doc["kind"] == "Namespace", (
            "Data Science Project is represented as a Kubernetes Namespace. "
            f"Found kind: {doc.get('kind')!r}"
        )

    def test_namespace_has_dashboard_label(self) -> None:
        """RHOAI dashboard requires opendatahub.io/dashboard: 'true' to show the project."""
        doc = _load_yaml(_GITOPS / "namespace.yaml")
        labels = doc.get("metadata", {}).get("labels", {})
        assert labels.get("opendatahub.io/dashboard") == "true", (
            "Namespace must carry label opendatahub.io/dashboard: 'true'. "
            "Without this label the RHOAI dashboard does not show the project. "
            f"Current labels: {labels}"
        )

    def test_namespace_name(self) -> None:
        doc = _load_yaml(_GITOPS / "namespace.yaml")
        assert doc["metadata"]["name"] == "pragma-encoder"

    def test_namespace_has_argocd_sync_wave(self) -> None:
        """Namespace must be wave -1 so it exists before all other resources."""
        doc = _load_yaml(_GITOPS / "namespace.yaml")
        annotations = doc.get("metadata", {}).get("annotations", {})
        wave = annotations.get("argocd.argoproj.io/sync-wave")
        assert wave == "-1", (
            "Namespace must have argocd.argoproj.io/sync-wave: '-1'. "
            "This ensures the namespace exists before all other sync-wave resources. "
            f"Current wave: {wave!r}"
        )


# ===========================================================================
# 2. ArgoCD Application
# ===========================================================================


class TestArgoCDApplication:
    """ArgoCD Application syncs declared platform state — not runtime objects.

    ArgoCD is a promotion surface over OpenShift AI/OpenShift primitives.
    It must sync openshift/gitops/ (the declared platform state) and must
    not sync src/, pipeline/ Python source, or runtime-generated objects.
    """

    def test_application_file_exists(self) -> None:
        assert (_ARGOCD / "application.yaml").exists()

    def test_application_api_version_and_kind(self) -> None:
        doc = _load_yaml(_ARGOCD / "application.yaml")
        assert doc["apiVersion"] == "argoproj.io/v1alpha1"
        assert doc["kind"] == "Application"

    def test_application_syncs_gitops_path(self) -> None:
        """ArgoCD must sync openshift/gitops/ — the declared platform state directory.

        openshift/gitops/ contains: namespace, RBAC, SealedSecrets, DSPA,
        workbench Notebook, BuildConfig, ImageStream.

        It must NOT be set to sync src/ (wheel source) or pipeline/ (Python).
        """
        doc = _load_yaml(_ARGOCD / "application.yaml")
        path = doc["spec"]["source"]["path"]
        assert path == "openshift/gitops", (
            f"ArgoCD Application source.path is {path!r}. "
            "Expected 'openshift/gitops' — the declared platform state directory. "
            "ArgoCD must not sync runtime objects or src/ (wheel source). "
            "Reference: docs/openshift-ai-primitives.md §ArgoCD Ownership"
        )

    def test_application_does_not_sync_src(self) -> None:
        """ArgoCD must not sync src/ — the wheel source is not a cluster resource."""
        doc = _load_yaml(_ARGOCD / "application.yaml")
        path = doc["spec"]["source"]["path"]
        assert not path.startswith("src"), (
            f"ArgoCD syncs {path!r} which starts with 'src'. "
            "The wheel source (src/pragma_encoder/) is not a Kubernetes resource. "
            "Reference: docs/openshift-ai-primitives.md"
        )

    def test_application_has_automated_sync(self) -> None:
        """ArgoCD sync policy must configure automated prune+selfHeal."""
        doc = _load_yaml(_ARGOCD / "application.yaml")
        automated = doc["spec"]["syncPolicy"].get("automated", {})
        assert automated.get("prune") is True, (
            "ArgoCD Application must have syncPolicy.automated.prune: true "
            "to remove deleted resources from the cluster."
        )
        assert automated.get("selfHeal") is True, (
            "ArgoCD Application must have syncPolicy.automated.selfHeal: true "
            "to correct cluster drift automatically."
        )

    def test_application_target_namespace(self) -> None:
        doc = _load_yaml(_ARGOCD / "application.yaml")
        ns = doc["spec"]["destination"]["namespace"]
        assert ns == "pragma-encoder", (
            f"ArgoCD Application must target namespace 'pragma-encoder'. Got: {ns!r}"
        )


# ===========================================================================
# 3. Connection / Data Connection (object storage)
# ===========================================================================


class TestConnectionFixture:
    """Connection fixture matches the RHOAI object-storage Connection primitive.

    On OpenShift AI, object storage is supplied through a Connection.
    The Connection materializes as a Kubernetes Secret annotated with
    RHOAI-specific labels/annotations. It is injected into pods as env vars.

    pragma_encoder reads these env vars from the process environment.
    It does not own or create Connections, Secrets, or credential lifecycle.

    Reference: docs/openshift-ai-primitives.md §Object Storage — Connection Contract
    """

    _CONN = _FIXTURES / "object-storage-connection.yaml.template"

    def test_connection_fixture_exists(self) -> None:
        assert self._CONN.exists(), (
            "tests/openshift/fixtures/object-storage-connection.yaml.template must exist. "
            "This is the RHOAI-aligned Connection fixture example."
        )

    def test_connection_kind_is_secret(self) -> None:
        """RHOAI Connection manifests as a Kubernetes Secret."""
        doc = _load_yaml(self._CONN)
        assert doc["kind"] == "Secret", (
            "RHOAI Connection is a Kubernetes Secret. "
            f"Found kind: {doc.get('kind')!r}"
        )

    def test_connection_has_managed_annotation(self) -> None:
        """opendatahub.io/managed: 'true' is required for RHOAI to manage the Connection."""
        doc = _load_yaml(self._CONN)
        annotations = doc.get("metadata", {}).get("annotations", {})
        assert annotations.get("opendatahub.io/managed") == "true", (
            "RHOAI Connection must have annotation opendatahub.io/managed: 'true'. "
            f"Current annotations: {annotations}"
        )

    def test_connection_has_s3_type_annotation(self) -> None:
        """opendatahub.io/connection-type: s3 identifies this as an S3 Connection."""
        doc = _load_yaml(self._CONN)
        annotations = doc.get("metadata", {}).get("annotations", {})
        assert annotations.get("opendatahub.io/connection-type") == "s3", (
            "RHOAI object-storage Connection must have annotation "
            "opendatahub.io/connection-type: s3. "
            f"Current annotations: {annotations}"
        )

    def test_connection_has_required_storage_keys(self) -> None:
        """Connection fixture must carry the native OpenShift AI S3 Connection env var names.

        Source of truth: redhat-ods-applications/s3 ConfigMap, RHOAI 2.25.6.
        Required fields (injected into pods as env vars by the RHOAI dashboard):
            AWS_ACCESS_KEY_ID      required
            AWS_SECRET_ACCESS_KEY  required
            AWS_S3_ENDPOINT        required
        """
        doc = _load_yaml(self._CONN)
        string_data = doc.get("stringData", {})
        required_keys = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT",
        }
        missing = required_keys - set(string_data.keys())
        assert not missing, (
            f"Connection fixture is missing required native S3 keys: {missing}. "
            "Source of truth: oc get cm s3 -n redhat-ods-applications -o yaml. "
            "Reference: docs/openshift-storage-pattern.md §Credentials"
        )

    def test_connection_credentials_are_placeholder(self) -> None:
        """Credential values must be REPLACE_ME — never real credentials."""
        doc = _load_yaml(self._CONN)
        string_data = doc.get("stringData", {})
        cred_keys = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
        for key in cred_keys:
            if key in string_data:
                assert string_data[key] == "REPLACE_ME", (
                    f"Connection fixture key {key!r} must be 'REPLACE_ME', not a real credential. "
                    "Never commit real S3 access keys to this file."
                )

    def test_connection_not_in_wheel_source(self) -> None:
        """The Secret name pragma-workbench-env must not appear in src/pragma_encoder/."""
        wheel_src = _REPO_ROOT / "src" / "pragma_encoder"
        for py_file in wheel_src.rglob("*.py"):
            content = py_file.read_text()
            assert "pragma-workbench-env" not in content, (
                f"{py_file.relative_to(_REPO_ROOT)}: contains 'pragma-workbench-env'. "
                "The Secret name is a platform fixture — it must not appear in the wheel. "
                "pragma_encoder reads env vars from the process; it must not name Secrets. "
                "Reference: docs/openshift-ai-3.3-alignment.md §Storage Adapter Boundary"
            )


# ===========================================================================
# 3b. SealedSecret Connection metadata — regression guard
# ===========================================================================


class TestSealedSecretConnectionMetadata:
    """The SealedSecret template.metadata must include RHOAI Connection annotations/labels.

    A SealedSecret's spec.template.metadata is plain (unencrypted) YAML. The
    Sealed Secrets controller copies it verbatim to the resulting Secret's metadata.
    Without the RHOAI annotations/labels the Secret decrypts successfully but does
    NOT appear as a Connection in the OpenShift AI dashboard.

    Root-cause history: workbench-runtime-secret.sealed.yaml was committed
    without opendatahub.io/managed, opendatahub.io/connection-type, or
    opendatahub.io/dashboard labels. The Secret existed on the cluster but the
    RHOAI dashboard showed no Connections. Fixed 2026-05-22.

    This test prevents that regression without re-sealing the credential data.
    """

    _SEALED = (
        _REPO_ROOT
        / "openshift"
        / "gitops"
        / "secrets"
        / "workbench-runtime-secret.sealed.yaml"
    )

    def test_sealed_secret_file_exists(self) -> None:
        assert self._SEALED.exists(), (
            "openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml must exist. "
            "This is the ArgoCD-synced encrypted form of the workbench Connection secret."
        )

    def test_sealed_secret_kind(self) -> None:
        doc = _load_yaml(self._SEALED)
        assert doc["kind"] == "SealedSecret", (
            "openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml "
            f"must be kind: SealedSecret. Found: {doc.get('kind')!r}"
        )

    def test_sealed_secret_template_has_managed_annotation(self) -> None:
        """spec.template.metadata.annotations must include opendatahub.io/managed: 'true'.

        Without this annotation the resulting Secret is not shown as a Connection
        in the RHOAI dashboard. The template section is plain YAML — no re-sealing needed.
        """
        doc = _load_yaml(self._SEALED)
        annotations = (
            doc.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("annotations", {})
        )
        assert annotations.get("opendatahub.io/managed") == "true", (
            "SealedSecret spec.template.metadata.annotations must include "
            "opendatahub.io/managed: 'true'. "
            "Without this the resulting Secret is not shown as an RHOAI Connection. "
            f"Current annotations: {annotations}"
        )

    def test_sealed_secret_template_has_connection_type_annotation(self) -> None:
        """spec.template.metadata.annotations must include opendatahub.io/connection-type: s3."""
        doc = _load_yaml(self._SEALED)
        annotations = (
            doc.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("annotations", {})
        )
        assert annotations.get("opendatahub.io/connection-type") == "s3", (
            "SealedSecret spec.template.metadata.annotations must include "
            "opendatahub.io/connection-type: s3. "
            "Without this the RHOAI dashboard does not classify the Connection as S3. "
            f"Current annotations: {annotations}"
        )

    def test_sealed_secret_template_has_dashboard_label(self) -> None:
        """spec.template.metadata.labels must include opendatahub.io/dashboard: 'true'."""
        doc = _load_yaml(self._SEALED)
        labels = (
            doc.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("labels", {})
        )
        assert labels.get("opendatahub.io/dashboard") == "true", (
            "SealedSecret spec.template.metadata.labels must include "
            "opendatahub.io/dashboard: 'true'. "
            "Without this the resulting Secret is not visible in the RHOAI project view. "
            f"Current labels: {labels}"
        )


# ===========================================================================
# 4. HardwareProfile
# ===========================================================================


class TestHardwareProfiles:
    """HardwareProfile fixtures match the RHOAI 3.3 GA primitive API.

    HardwareProfile (infrastructure.opendatahub.io/v1) is the GA primitive
    for CPU/GPU/resource shape selection in RHOAI 3.3. It replaces deprecated
    Accelerator Profiles and Container Size selector.

    pragma_encoder does not select hardware. Training code uses whatever
    device it is given. Hardware selection is the platform's responsibility.
    """

    _CPU = _FIXTURES / "hardware-profile-cpu-smoke.yaml"
    _GPU = _FIXTURES / "hardware-profile-gpu-pragma-s.yaml"

    def test_cpu_profile_exists(self) -> None:
        assert self._CPU.exists()

    def test_gpu_profile_exists(self) -> None:
        assert self._GPU.exists()

    def test_cpu_profile_api_version(self) -> None:
        doc = _load_yaml(self._CPU)
        assert doc["apiVersion"] == "infrastructure.opendatahub.io/v1", (
            "HardwareProfile must use apiVersion: infrastructure.opendatahub.io/v1. "
            "This is the GA API in RHOAI 3.3. "
            f"Found: {doc.get('apiVersion')!r}"
        )

    def test_cpu_profile_kind(self) -> None:
        doc = _load_yaml(self._CPU)
        assert doc["kind"] == "HardwareProfile"

    def test_gpu_profile_api_version(self) -> None:
        doc = _load_yaml(self._GPU)
        assert doc["apiVersion"] == "infrastructure.opendatahub.io/v1"

    def test_gpu_profile_kind(self) -> None:
        doc = _load_yaml(self._GPU)
        assert doc["kind"] == "HardwareProfile"

    def test_cpu_profile_has_no_gpu_identifier(self) -> None:
        """CPU smoke profile must not request nvidia.com/gpu or any GPU resource."""
        doc = _load_yaml(self._CPU)
        identifiers = doc.get("spec", {}).get("identifiers", [])
        gpu_identifiers = [
            ident for ident in identifiers
            if "gpu" in ident.get("identifier", "").lower()
            or "nvidia" in ident.get("identifier", "").lower()
        ]
        assert len(gpu_identifiers) == 0, (
            "CPU smoke HardwareProfile must not include GPU identifiers. "
            f"Found GPU identifiers: {gpu_identifiers}. "
            "CPU profiles are for test workloads only — no GPU resource is appropriate."
        )

    def test_gpu_profile_has_nvidia_gpu_identifier(self) -> None:
        """GPU production profile must include nvidia.com/gpu identifier."""
        doc = _load_yaml(self._GPU)
        identifiers = doc.get("spec", {}).get("identifiers", [])
        gpu_identifiers = [
            ident for ident in identifiers
            if "nvidia.com/gpu" == ident.get("identifier", "")
        ]
        assert len(gpu_identifiers) >= 1, (
            "GPU HardwareProfile must include nvidia.com/gpu identifier. "
            "This maps to NVIDIA GPU scheduling in OpenShift. "
            f"Current identifiers: {[i.get('identifier') for i in identifiers]}"
        )

    def test_cpu_profile_has_display_name(self) -> None:
        doc = _load_yaml(self._CPU)
        assert doc.get("spec", {}).get("displayName"), (
            "HardwareProfile must have spec.displayName for RHOAI dashboard display."
        )

    def test_gpu_profile_has_display_name(self) -> None:
        doc = _load_yaml(self._GPU)
        assert doc.get("spec", {}).get("displayName"), (
            "HardwareProfile must have spec.displayName for RHOAI dashboard display."
        )

    def test_hardware_profiles_not_in_wheel(self) -> None:
        """HardwareProfile YAMLs must not be inside src/pragma_encoder/."""
        wheel_src = _REPO_ROOT / "src" / "pragma_encoder"
        yaml_files = list(wheel_src.rglob("*.yaml")) + list(wheel_src.rglob("*.yml"))
        hw_profile_files = [
            f for f in yaml_files
            if "HardwareProfile" in f.read_text()
        ]
        assert len(hw_profile_files) == 0, (
            f"HardwareProfile YAML found inside wheel: {hw_profile_files}. "
            "Platform resources must not be packaged into pragma_encoder."
        )


# ===========================================================================
# 5. Data Science Pipeline (DSPA)
# ===========================================================================


class TestDSPA:
    """DataSciencePipelinesApplication is a platform resource, not package config.

    The DSPA CR is managed by ArgoCD and lives outside the pragma_encoder wheel.
    pragma_encoder owns the pipeline component logic (pipeline/components_pragma.py).
    OpenShift AI owns the DSPA runtime (KFP v2 server).
    """

    _DSPA = _GITOPS / "pipeline" / "dspa.yaml"

    def test_dspa_file_exists(self) -> None:
        assert self._DSPA.exists()

    def test_dspa_kind(self) -> None:
        doc = _load_yaml(self._DSPA)
        assert doc["kind"] == "DataSciencePipelinesApplication", (
            f"Expected kind DataSciencePipelinesApplication. Found: {doc.get('kind')!r}"
        )

    def test_dspa_not_in_wheel(self) -> None:
        """DSPA is a platform resource — must not exist under src/pragma_encoder/."""
        wheel_src = _REPO_ROOT / "src" / "pragma_encoder"
        yaml_files = list(wheel_src.rglob("*.yaml")) + list(wheel_src.rglob("*.yml"))
        assert len(yaml_files) == 0, (
            f"Found YAML inside the wheel: {[str(f) for f in yaml_files]}. "
            "Platform resources (DSPA, manifests) must not be packaged into pragma_encoder."
        )

    def test_dspa_has_external_object_storage(self) -> None:
        """DSPA must configure externalStorage — S3 is the Connection-backed store."""
        doc = _load_yaml(self._DSPA)
        ext = doc.get("spec", {}).get("objectStorage", {}).get("externalStorage", {})
        assert ext, (
            "DSPA must configure spec.objectStorage.externalStorage. "
            "S3 is the canonical object store for pipeline artifacts."
        )

    def test_dspa_references_connection_secret(self) -> None:
        """DSPA must reference an S3 credentials Secret — the Connection backing store."""
        doc = _load_yaml(self._DSPA)
        s3_secret = (
            doc.get("spec", {})
            .get("objectStorage", {})
            .get("externalStorage", {})
            .get("s3CredentialsSecret", {})
        )
        assert s3_secret.get("secretName"), (
            "DSPA must reference an S3 credentials Secret via "
            "spec.objectStorage.externalStorage.s3CredentialsSecret.secretName. "
            "This Secret is the OpenShift AI Connection backing pipeline artifact storage."
        )


# ===========================================================================
# 6. Workbench / Notebook
# ===========================================================================


class TestWorkbenchPrimitive:
    """Workbench Notebook CR is the correct RHOAI primitive.

    The Notebook CR (notebooks.kubeflow.org/v1) creates the JupyterLab
    workbench pod. It is managed by ArgoCD and uses the custom workbench image.

    The workbench is the authoring environment. The pragma_encoder wheel is
    NOT installed in the workbench image — it is installed in the training image.
    Workbench users access pragma_encoder via PYTHONPATH=. (source tree).
    """

    _NOTEBOOK = _GITOPS / "workbench" / "notebook.yaml"

    def test_notebook_file_exists(self) -> None:
        assert self._NOTEBOOK.exists()

    def test_notebook_contains_notebook_kind(self) -> None:
        docs = _load_yaml_all(self._NOTEBOOK)
        kinds = {d.get("kind") for d in docs if d}
        assert "Notebook" in kinds, (
            f"Workbench manifest must contain kind: Notebook. Found kinds: {kinds}. "
            "The Notebook CR is the RHOAI primitive for workbench environments."
        )

    def test_notebook_has_inject_auth_annotation(self) -> None:
        """Notebook must use inject-auth (RHOAI 3.4), not inject-oauth (RHOAI 3.3).

        RHOAI 3.4 replaced the oauth-proxy sidecar with kube-rbac-proxy.
        The controller injects kube-rbac-proxy only when
        notebooks.opendatahub.io/inject-auth='true' is set.
        inject-oauth='true' on a 3.4 cluster does not inject any auth sidecar,
        which permanently disables the dashboard Open button.
        """
        docs = _load_yaml_all(self._NOTEBOOK)
        notebook_docs = [d for d in docs if d and d.get("kind") == "Notebook"]
        assert notebook_docs, "No Notebook document found in notebook.yaml"
        annotations = notebook_docs[0].get("metadata", {}).get("annotations", {})
        assert annotations.get("notebooks.opendatahub.io/inject-auth") == "true", (
            "Workbench Notebook must have annotation "
            "notebooks.opendatahub.io/inject-auth: 'true'. "
            "RHOAI 3.4 requires inject-auth=true to inject the kube-rbac-proxy sidecar. "
            "The old inject-oauth annotation is a 3.3 pattern and disables the Open button on 3.4. "
            f"Current annotations: {list(annotations.keys())}"
        )
        assert "notebooks.opendatahub.io/inject-oauth" not in annotations, (
            "Workbench Notebook must not have the deprecated inject-oauth annotation. "
            "Remove it — it must not coexist with inject-auth on RHOAI 3.4. "
            f"Current annotations: {list(annotations.keys())}"
        )

    def test_notebook_references_workbench_image(self) -> None:
        """Notebook must reference the workbench image, not the training image."""
        content = self._NOTEBOOK.read_text()
        assert "pragma-encoder-workbench" in content, (
            "Workbench Notebook must reference pragma-encoder-workbench image. "
            "The workbench image provides the JupyterLab authoring environment."
        )

    def test_notebook_does_not_install_wheel_at_startup(self) -> None:
        """Workbench manifest must not install the pragma_encoder wheel at pod startup.

        The wheel is installed in the training image only.
        Workbench accesses pragma_encoder via PYTHONPATH=. (source tree).
        """
        content = self._NOTEBOOK.read_text()
        bad_phrases = [
            "pip install pragma_encoder",
            "pip install pragma-encoder",
            "pip install dist/pragma_encoder",
        ]
        for phrase in bad_phrases:
            assert phrase not in content, (
                f"Workbench manifest contains '{phrase}'. "
                "The wheel must NOT be installed in the workbench image. "
                "The training image installs the wheel. "
                "See docs/openshift-image-contract.md §Two-image model."
            )


# ===========================================================================
# 7. PyTorchJob — wheel-based entrypoints
# ===========================================================================


class TestPyTorchJobPrimitives:
    """PyTorchJob manifests use wheel-based entrypoints — no git clone, no src/ PYTHONPATH.

    All production PyTorchJob manifests must:
    - Use kubeflow.org/v1 (the current project API — not trainer.kubeflow.org/v1alpha1)
    - Reference pragma-encoder-training image (wheel installed)
    - Not contain git clone steps
    - Not use scripts/train_pragma.py as a non-comment execution command
    - Not set PYTHONPATH pointing at a source tree (PYTHONPATH=$PRAGMA_ROOT/src etc.)

    PyTorchJob is the current proven project path. Verify availability and support
    status on the target RHOAI 3.4 cluster before running Level 4/5 cluster tests.
    Upstream Kubeflow Training Operator v1 source was removed Feb 2025; CRD presence
    does not equal production support.

    Reference: docs/openshift-image-contract.md
    Reference: docs/openshift-ai-primitives.md §Distributed Training
    """

    def _pytorchjob_files(self) -> list[pathlib.Path]:
        return sorted(_TRAINING.glob("pytorchjob-*.yaml"))

    def test_pytorchjob_files_exist(self) -> None:
        files = self._pytorchjob_files()
        assert len(files) >= 1, (
            f"Expected at least one pytorchjob-*.yaml in {_TRAINING}. "
            "PyTorchJob (kubeflow.org/v1) is the current proven project path "
            "for distributed training."
        )

    def test_pytorchjob_api_version_is_ga(self) -> None:
        """All PyTorchJob manifests must use the kubeflow.org/v1 API (current project path)."""
        for path in self._pytorchjob_files():
            doc = _load_yaml(path)
            assert doc.get("apiVersion") == "kubeflow.org/v1", (
                f"{path.name}: PyTorchJob must use apiVersion: kubeflow.org/v1 "
                "(the current project API). "
                "trainer.kubeflow.org/v1alpha1 (TrainJob) is Tech Preview — see ADR 005. "
                f"Found: {doc.get('apiVersion')!r}"
            )
            assert doc.get("kind") == "PyTorchJob", (
                f"{path.name}: expected kind: PyTorchJob. Found: {doc.get('kind')!r}"
            )

    def test_pytorchjob_no_git_clone(self) -> None:
        """No PyTorchJob manifest may contain a git clone step.

        The training image has the pragma_encoder wheel installed at build time.
        No runtime git clone is needed or permitted.
        """
        for path in self._pytorchjob_files():
            content = path.read_text()
            non_comment_lines = [
                line for line in content.splitlines()
                if "git clone" in line
                and not line.lstrip().startswith("#")
            ]
            assert len(non_comment_lines) == 0, (
                f"{path.name}: contains 'git clone' in non-comment line. "
                "PyTorchJob manifests must use the wheel-based training image. "
                "No runtime git clone is permitted. "
                "Reference: docs/openshift-image-contract.md"
            )

    def test_pytorchjob_no_scripts_train_pragma_as_command(self) -> None:
        """scripts/train_pragma.py must not be the primary execution command.

        scripts/train_pragma.py is a compatibility wrapper only.
        The canonical entrypoint is python -m pragma_encoder.training.train.
        """
        for path in self._pytorchjob_files():
            content = path.read_text()
            non_comment_lines = [
                line for line in content.splitlines()
                if "scripts/train_pragma.py" in line
                and not line.lstrip().startswith("#")
            ]
            assert len(non_comment_lines) == 0, (
                f"{path.name}: scripts/train_pragma.py appears as a non-comment command. "
                "The canonical entrypoint is: python -m pragma_encoder.training.train. "
                "scripts/train_pragma.py is a compatibility wrapper. "
                f"Non-comment lines: {non_comment_lines}"
            )

    def test_pytorchjob_uses_training_image(self) -> None:
        """All production PyTorchJob manifests must reference pragma-encoder-training image."""
        for path in self._pytorchjob_files():
            content = path.read_text()
            assert "pragma-encoder-training" in content, (
                f"{path.name}: does not reference pragma-encoder-training image. "
                "The training image has the pragma_encoder wheel installed. "
                "Reference: docs/openshift-image-contract.md §Training image"
            )

    def test_pytorchjob_no_source_tree_pythonpath(self) -> None:
        """No PyTorchJob manifest may set PYTHONPATH pointing to a source tree.

        Wheel-based images do not need PYTHONPATH set to src/ or a repo clone path.
        """
        bad_patterns = [
            "PYTHONPATH=$PRAGMA_ROOT",
            "PYTHONPATH=/workspace/repo/src",
            "PYTHONPATH=./src",
        ]
        for path in self._pytorchjob_files():
            content = path.read_text()
            for pattern in bad_patterns:
                assert pattern not in content, (
                    f"{path.name}: contains source-tree PYTHONPATH pattern '{pattern}'. "
                    "Wheel-based images do not need PYTHONPATH to src/. "
                    "Reference: docs/openshift-image-contract.md"
                )


# ===========================================================================
# 8. TrainJob — Tech Preview, example only
# ===========================================================================


class TestTrainJobTechPreview:
    """TrainJob is clearly marked Technology Preview and must not appear in production manifests.

    Kubeflow Trainer v2 / TrainJob is Technology Preview in RHOAI 3.3/3.4.
    It must not be used in production manifests or CI tests.
    The fixture exists as a documented evaluation reference.

    Reference: docs/openshift-ai-primitives.md §Distributed Training
    Reference: docs/openshift-ai-3.3-alignment.md §Kubeflow Trainer v2 / TrainJob
    Reference: ADR 005 (docs/decisions/005-training-orchestration.md)
    """

    _TRAINJOB = _FIXTURES / "trainjob-example.yaml"

    def test_trainjob_fixture_exists(self) -> None:
        assert self._TRAINJOB.exists(), (
            "tests/openshift/fixtures/trainjob-example.yaml must exist "
            "as the documented future/evaluation reference for TrainJob."
        )

    def test_trainjob_api_version(self) -> None:
        """TrainJob must use trainer.kubeflow.org/v1alpha1 (the Tech Preview API)."""
        doc = _load_yaml(self._TRAINJOB)
        assert doc.get("apiVersion") == "trainer.kubeflow.org/v1alpha1", (
            f"TrainJob fixture must use apiVersion: trainer.kubeflow.org/v1alpha1. "
            f"Found: {doc.get('apiVersion')!r}"
        )
        assert doc.get("kind") == "TrainJob"

    def test_trainjob_marked_tech_preview(self) -> None:
        """TrainJob fixture must be clearly marked as Tech Preview / not for production."""
        content = self._TRAINJOB.read_text()
        has_marker = (
            "TECHNOLOGY PREVIEW" in content.upper()
            or "Tech Preview" in content
            or "technology preview" in content.lower()
        )
        assert has_marker, (
            "trainjob-example.yaml must contain 'TECHNOLOGY PREVIEW' or 'Tech Preview'. "
            "TrainJob is Technology Preview in RHOAI 3.3/3.4 and must not be used in production. "
            "Reference: docs/openshift-ai-3.3-alignment.md §Kubeflow Trainer v2"
        )

    def test_trainjob_not_in_production_manifests(self) -> None:
        """trainer.kubeflow.org must not appear in openshift/training/ production manifests."""
        for path in sorted(_TRAINING.glob("*.yaml")):
            content = path.read_text()
            non_comment_lines = [
                line for line in content.splitlines()
                if "trainer.kubeflow.org" in line
                and not line.lstrip().startswith("#")
            ]
            assert len(non_comment_lines) == 0, (
                f"{path.name}: references trainer.kubeflow.org (Kubeflow Trainer v2). "
                "Production manifests must use kubeflow.org/v1 (PyTorchJob — current proven path). "
                "TrainJob is Technology Preview in RHOAI 3.4 unless GA is confirmed. "
                "Reference: ADR 005, docs/openshift-ai-primitives.md §Distributed Training"
            )

    def test_trainjob_uses_wheel_entrypoint(self) -> None:
        """TrainJob example must use wheel-based entrypoint — no scripts/train_pragma.py."""
        content = self._TRAINJOB.read_text()
        command_lines = [
            line for line in content.splitlines()
            if "scripts/train_pragma.py" in line
            and not line.lstrip().startswith("#")
        ]
        assert len(command_lines) == 0, (
            "trainjob-example.yaml uses scripts/train_pragma.py as an execution command. "
            "The canonical entrypoint is: -m pragma_encoder.training.train. "
            f"Lines: {command_lines}"
        )


# ===========================================================================
# 8b. TrainJob evaluation context — TD-012 static guards
# ===========================================================================


class TestTrainJobEvaluationContext:
    """Static guards for the RHOAI 3.4 TrainJob checkpoint evaluation artefacts.

    These tests verify that the TD-012 evaluation context is present in the
    relevant fixture and docs files. They guard against silent regression if
    someone removes the evaluation commentary without closing TD-012 properly.

    Category: structural
    Reference: docs/rhoai-3.4-trainjob-checkpointing.md
    Reference: docs/tech-debt.md §TD-012
    Reference: tests/openshift/fixtures/trainjob-example.yaml
    Reference: docs/openshift-ai-primitives.md §Distributed Training
    """

    _TRAINJOB = _FIXTURES / "trainjob-example.yaml"
    _PRIMITIVES = _REPO_ROOT / "docs" / "openshift-ai-primitives.md"
    _EVAL_DOC = _REPO_ROOT / "docs" / "rhoai-3.4-trainjob-checkpointing.md"
    _TECH_DEBT = _REPO_ROOT / "docs" / "tech-debt.md"

    def test_rhoai_34_evaluation_doc_exists(self) -> None:
        """TD-012: docs/rhoai-3.4-trainjob-checkpointing.md must exist.

        This is the authoritative evaluation doc for the RHOAI 3.4 TrainJob +
        Kubeflow Trainer v2 checkpointing assessment. It contains the full
        responsibility comparison table, gap analysis, and decision matrix.
        """
        assert self._EVAL_DOC.exists(), (
            "docs/rhoai-3.4-trainjob-checkpointing.md must exist. "
            "This is the TD-012 evaluation document for RHOAI 3.4 TrainJob "
            "checkpointing. It must not be deleted until TD-012 is resolved. "
            "Reference: docs/tech-debt.md §TD-012"
        )

    def test_td012_registered_in_tech_debt(self) -> None:
        """TD-012 must be registered in docs/tech-debt.md.

        The evaluation of RHOAI 3.4 TrainJob checkpointing is an open tech debt
        item. It must remain registered until the evaluation is complete and one
        of the resolution options (A/B/C) is implemented.
        """
        content = self._TECH_DEBT.read_text()
        assert "TD-012" in content, (
            "docs/tech-debt.md must contain TD-012. "
            "TD-012 tracks the RHOAI 3.4 TrainJob + Kubeflow Trainer v2 "
            "checkpointing evaluation. Do not remove until evaluation is resolved. "
            "Reference: docs/rhoai-3.4-trainjob-checkpointing.md"
        )

    def test_trainjob_fixture_references_evaluation_doc(self) -> None:
        """TrainJob fixture must reference the TD-012 evaluation doc.

        The fixture header must point to docs/rhoai-3.4-trainjob-checkpointing.md
        so readers understand this is an evaluation artefact, not a production
        template. Removing this reference while TD-012 is open is a regression.
        """
        content = self._TRAINJOB.read_text()
        assert "rhoai-3.4-trainjob-checkpointing.md" in content, (
            "tests/openshift/fixtures/trainjob-example.yaml must reference "
            "docs/rhoai-3.4-trainjob-checkpointing.md in its header. "
            "This ties the fixture to the TD-012 evaluation context. "
            "Reference: docs/tech-debt.md §TD-012"
        )

    def test_trainjob_fixture_notes_checkpoint_gaps(self) -> None:
        """TrainJob fixture must document the checkpoint evaluation gaps.

        The fixture must note that PRAGMA's custom PyTorch loop has gaps
        relative to the SDK-native checkpointing path. This prevents readers
        from assuming the RHOAI 3.4 checkpoint SDK works automatically with
        the PRAGMA training loop.
        """
        content = self._TRAINJOB.read_text()
        has_gap_notation = "GAP-" in content or "checkpoint" in content.lower()
        assert has_gap_notation, (
            "tests/openshift/fixtures/trainjob-example.yaml must document "
            "the checkpoint evaluation gaps (GAP-1 through GAP-5) for the "
            "PRAGMA custom training loop. "
            "Reference: docs/rhoai-3.4-trainjob-checkpointing.md §4 Gap Analysis"
        )

    def test_primitives_doc_notes_pytorchjob_upstream_deprecation(self) -> None:
        """openshift-ai-primitives.md must note the PyTorchJob upstream deprecation.

        Kubeflow Training Operator v1 source code was removed from the upstream
        kubeflow/trainer repo (Feb 2025). The primitives doc must carry this
        notice so developers know to verify CRD availability before running
        Level 5 cluster tests on a RHOAI 3.4 cluster.

        This is a critical operational guard: missing CRDs cause Level 5 failures.
        """
        content = self._PRIMITIVES.read_text()
        has_deprecation_notice = (
            "deprecated" in content.lower()
            or "deprecation" in content.lower()
            or "removed" in content.lower()
        ) and "kubeflow" in content.lower()
        assert has_deprecation_notice, (
            "docs/openshift-ai-primitives.md must note that PyTorchJob "
            "(kubeflow.org/v1) is deprecated upstream. "
            "The upstream Kubeflow Training Operator v1 source was removed "
            "from kubeflow/trainer in Feb 2025. Without this notice, developers "
            "may run Level 5 cluster tests without first verifying CRD availability. "
            "Reference: docs/rhoai-3.4-trainjob-checkpointing.md §6"
        )

    def test_primitives_doc_has_cluster_verification_command(self) -> None:
        """openshift-ai-primitives.md must document the oc api-resources verification step.

        Before running Level 5 tests on a RHOAI 3.4 cluster, developers must
        verify that kubeflow.org/v1 CRDs are present. The verification command
        must appear in the primitives doc so it is not overlooked.
        """
        content = self._PRIMITIVES.read_text()
        assert "oc api-resources" in content, (
            "docs/openshift-ai-primitives.md must include the 'oc api-resources' "
            "command for verifying PyTorchJob CRD availability on a RHOAI 3.4 cluster. "
            "Without this, Level 5 cluster tests may fail with confusing errors. "
            "Reference: docs/rhoai-3.4-trainjob-checkpointing.md §6"
        )


# ===========================================================================
# 9. Serving primitives
# ===========================================================================


class TestServingPrimitives:
    """Serving resources are platform objects — they are not part of the wheel.

    InferenceService (KServe) and ServingRuntime are RHOAI primitives for
    model serving. pragma_encoder does not own serving runtime configuration.
    """

    _IS = _SERVING / "inference-service.yaml"
    _SR = _SERVING / "serving-runtime.yaml"

    def test_inference_service_exists(self) -> None:
        assert self._IS.exists()

    def test_serving_runtime_exists(self) -> None:
        assert self._SR.exists()

    def test_inference_service_kind(self) -> None:
        doc = _load_yaml(self._IS)
        assert doc["kind"] == "InferenceService", (
            f"Expected kind: InferenceService. Found: {doc.get('kind')!r}"
        )

    def test_serving_runtime_kind(self) -> None:
        doc = _load_yaml(self._SR)
        assert doc["kind"] == "ServingRuntime", (
            f"Expected kind: ServingRuntime. Found: {doc.get('kind')!r}"
        )

    def test_serving_resources_not_in_wheel(self) -> None:
        """Serving resource YAML must not be inside src/pragma_encoder/."""
        wheel_src = _REPO_ROOT / "src" / "pragma_encoder"
        yaml_files = list(wheel_src.rglob("*.yaml")) + list(wheel_src.rglob("*.yml"))
        serving_yaml = [
            f for f in yaml_files
            if "InferenceService" in f.read_text() or "ServingRuntime" in f.read_text()
        ]
        assert len(serving_yaml) == 0, (
            f"Serving resource YAML found inside the wheel: {serving_yaml}. "
            "Platform resources must not be packaged into pragma_encoder."
        )


# ===========================================================================
# 10. Boundary contracts
# ===========================================================================


class TestBoundaryContracts:
    """Platform resources must stay outside the wheel; wheel must stay platform-neutral.

    Reference: docs/openshift-ai-primitives.md §Platform Neutrality — Wheel Boundary
    Reference: docs/openshift-ai-3.3-alignment.md §Platform Neutrality
    Reference: tests/test_platform_neutral_wheel.py (complementary boundary tests)
    """

    def test_no_yaml_manifests_in_wheel(self) -> None:
        """No YAML/YML files under src/pragma_encoder/ — platform manifests are not wheel code."""
        wheel_src = _REPO_ROOT / "src" / "pragma_encoder"
        yaml_files = list(wheel_src.rglob("*.yaml")) + list(wheel_src.rglob("*.yml"))
        rel_paths = [str(f.relative_to(_REPO_ROOT)) for f in yaml_files]
        assert len(yaml_files) == 0, (
            f"Found YAML files inside the wheel: {rel_paths}. "
            "OpenShift/RHOAI platform manifests must not be packaged into pragma_encoder."
        )

    def test_pragma_encoder_workbench_not_in_wheel(self) -> None:
        """pragma_encoder.workbench must raise ModuleNotFoundError — it is not in the wheel.

        The workbench tooling lives in tools/workbench/ and is
        explicitly excluded from the package by pyproject.toml ([tool.setuptools.packages.find]
        where = ['src']). This is TD-009 resolved.
        """
        try:
            import pragma_encoder.workbench  # noqa: F401
            pytest.fail(
                "pragma_encoder.workbench was importable. "
                "The workbench module must live in tools/workbench/, "
                "not in the pragma_encoder package. "
                "See docs/openshift-ai-3.3-alignment.md §Platform Neutrality."
            )
        except ModuleNotFoundError:
            pass  # Expected — pragma_encoder.workbench is not in the wheel

    def test_tools_workbench_importable(self) -> None:
        """tools.workbench must be importable when PYTHONPATH=repo root."""
        spec = importlib.util.find_spec("tools.workbench")
        assert spec is not None, (
            "tools.workbench is not importable. "
            "Ensure PYTHONPATH includes the repository root (PYTHONPATH=.). "
            "The workbench tooling lives in tools/workbench/."
        )

    def test_argocd_does_not_sync_src(self) -> None:
        """ArgoCD must not sync src/ — the wheel source is not a cluster resource."""
        doc = _load_yaml(_ARGOCD / "application.yaml")
        source_path = doc["spec"]["source"]["path"]
        assert not source_path.startswith("src"), (
            f"ArgoCD Application syncs {source_path!r}. "
            "src/pragma_encoder/ is Python source, not Kubernetes resources. "
            "Reference: docs/openshift-ai-primitives.md §ArgoCD Ownership"
        )

    def test_argocd_does_not_sync_pipeline_python(self) -> None:
        """ArgoCD must not sync pipeline/ Python source directly as cluster resources."""
        doc = _load_yaml(_ARGOCD / "application.yaml")
        source_path = doc["spec"]["source"]["path"]
        assert source_path != "pipeline", (
            "ArgoCD syncs 'pipeline' directly. "
            "pipeline/*.py files are Python source, not Kubernetes resources. "
            "If promoting compiled pipeline artifacts, use openshift/gitops/pipelines/. "
            "Reference: docs/openshift-ai-primitives.md §ArgoCD Ownership"
        )

    def test_gitops_resources_not_in_generic_tests(self) -> None:
        """Cluster manifests must not live in tests/ root — use tests/openshift/."""
        tests_root = _REPO_ROOT / "tests"
        cluster_kinds = [
            "PyTorchJob",
            "DataSciencePipelinesApplication",
            "HardwareProfile",
            "InferenceService",
            "ServingRuntime",
            "TrainJob",
        ]
        for yaml_file in tests_root.glob("*.yaml"):
            content = yaml_file.read_text()
            for kind in cluster_kinds:
                assert f"kind: {kind}" not in content, (
                    f"tests/{yaml_file.name}: contains cluster resource kind: {kind}. "
                    "Cluster manifests must live in tests/openshift/ or openshift/. "
                    "Generic tests/ must remain cluster-free."
                )

