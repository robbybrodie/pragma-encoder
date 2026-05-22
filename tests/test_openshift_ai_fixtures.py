"""Static contract tests for OpenShift AI 3.3 fixture YAML files.

These tests parse the fixture files under tests/openshift/fixtures/ and assert
that each file conforms to its documented RHOAI 3.3 primitive contract.

No cluster access, S3 credentials, GPU, or registry required.
All tests run by default in CI:

  pytest tests/test_openshift_ai_fixtures.py

Fixtures tested:
  tests/openshift/fixtures/object-storage-connection.yaml.template
  tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml
  tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml
  tests/openshift/fixtures/trainjob-example.yaml

Reference: docs/openshift-ai-3.3-alignment.md
"""

from __future__ import annotations

import pathlib

import yaml

# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = (
    pathlib.Path(__file__).parent / "openshift" / "fixtures"
)

_CONNECTION_TEMPLATE = _FIXTURES_DIR / "object-storage-connection.yaml.template"
_CPU_PROFILE = _FIXTURES_DIR / "hardware-profile-cpu-smoke.yaml"
_GPU_PROFILE = _FIXTURES_DIR / "hardware-profile-gpu-pragma-s.yaml"
_TRAINJOB_EXAMPLE = _FIXTURES_DIR / "trainjob-example.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> dict:
    """Load first YAML document from file. Raises on parse error."""
    with path.open() as f:
        return yaml.safe_load(f)


def _load_text(path: pathlib.Path) -> str:
    """Read fixture file as text (for comment/annotation checks)."""
    return path.read_text()


# ---------------------------------------------------------------------------
# 1. Object-storage Connection fixture
# ---------------------------------------------------------------------------

class TestObjectStorageConnectionFixture:
    """Contract tests for the object-storage Connection fixture.

    RHOAI 3.3 primitive: Connection / Data Connection
    API: Kubernetes Secret with opendatahub.io/ annotations
    Status: GA

    The fixture is a test stand-in for what an RHOAI object-storage Connection
    would inject into workbench and training pods.
    """

    def test_fixture_file_exists(self) -> None:
        """object-storage-connection.yaml.template must exist in fixtures dir."""
        assert _CONNECTION_TEMPLATE.exists(), (
            f"Fixture not found: {_CONNECTION_TEMPLATE}\n"
            "Create it from docs/openshift-ai-3.3-alignment.md §Connection."
        )

    def test_kind_is_secret(self) -> None:
        """Connection fixture kind must be Secret (RHOAI Connection is a Secret)."""
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        assert doc.get("kind") == "Secret", (
            f"Connection fixture must have kind=Secret. Got: {doc.get('kind')!r}\n"
            "RHOAI Connections are Kubernetes Secrets with opendatahub.io/ annotations."
        )

    def test_has_s3_connection_type_annotation(self) -> None:
        """Fixture must have opendatahub.io/connection-type: s3 annotation."""
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        annotations = doc.get("metadata", {}).get("annotations", {}) or {}
        assert annotations.get("opendatahub.io/connection-type") == "s3", (
            "Connection fixture must have annotation "
            "'opendatahub.io/connection-type: s3'. "
            f"Annotations found: {annotations}\n"
            "This annotation marks it as an RHOAI object-storage Connection."
        )

    def test_has_managed_annotation(self) -> None:
        """Fixture must have opendatahub.io/managed: 'true' annotation."""
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        annotations = doc.get("metadata", {}).get("annotations", {}) or {}
        assert annotations.get("opendatahub.io/managed") == "true", (
            "Connection fixture must have annotation "
            "'opendatahub.io/managed: true'. "
            f"Annotations found: {annotations}\n"
            "This annotation is applied by RHOAI to Connections it manages."
        )

    def test_has_native_s3_env_vars(self) -> None:
        """Fixture must carry the native OpenShift AI S3 Connection env var names.

        Source of truth: redhat-ods-applications/s3 ConfigMap (RHOAI 2.25.6):
          AWS_ACCESS_KEY_ID      required
          AWS_SECRET_ACCESS_KEY  required
          AWS_S3_ENDPOINT        required
          AWS_DEFAULT_REGION     optional
          AWS_S3_BUCKET          optional

        The Connection fixture must use these native names so the RHOAI dashboard
        can populate the mandatory fields from the Secret.
        """
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        string_data = doc.get("stringData", {}) or {}
        required_native_keys = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT",
        }
        missing = required_native_keys - set(string_data.keys())
        assert not missing, (
            f"Connection fixture is missing native S3 keys: {sorted(missing)}\n"
            "Source of truth: oc get cm s3 -n redhat-ods-applications -o yaml\n"
            "See: docs/openshift-storage-pattern.md §Credentials"
        )

    def test_no_model_registry_aliases_in_connection_fixture(self) -> None:
        """S3 Connection fixture must not contain MODEL_REGISTRY_* aliases.

        MODEL_REGISTRY_* are custom names that do not match the native RHOAI
        Connection schema. Including them alongside native AWS_* names would
        silently duplicate credentials and create inconsistency risk.
        """
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        string_data = doc.get("stringData", {}) or {}
        alias_keys = {k for k in string_data if k.startswith("MODEL_REGISTRY_")}
        assert not alias_keys, (
            f"Connection fixture contains MODEL_REGISTRY_* alias keys: {sorted(alias_keys)}\n"
            "Use only native OpenShift AI S3 Connection keys (AWS_*).\n"
            "The training code reads AWS_* with MODEL_REGISTRY_* as a deprecated fallback."
        )

    def test_no_ngc_api_key_in_connection_fixture(self) -> None:
        """S3 Connection fixture must not contain NGC_API_KEY.

        NGC_API_KEY is an NVIDIA registry credential — it does not belong
        in an S3-type Connection. Move it to a separate registry-credentials Secret.
        """
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        string_data = doc.get("stringData", {}) or {}
        assert "NGC_API_KEY" not in string_data, (
            "Connection fixture contains NGC_API_KEY. "
            "NGC credentials belong in a separate registry-credentials Secret, "
            "not in the S3 Connection."
        )

    def test_no_ai_platform_api_url_in_connection_fixture(self) -> None:
        """S3 Connection fixture must not contain AI_PLATFORM_API_URL.

        AI_PLATFORM_API_URL is an RHOAI dashboard URL — it does not belong
        in an S3-type Connection. Move it to a platform ConfigMap or workbench env.
        """
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        string_data = doc.get("stringData", {}) or {}
        assert "AI_PLATFORM_API_URL" not in string_data, (
            "Connection fixture contains AI_PLATFORM_API_URL. "
            "This is an RHOAI platform URL, not an S3 credential. "
            "It does not belong in the S3 Connection."
        )

    def test_no_real_credentials(self) -> None:
        """Fixture must not contain real credentials — only REPLACE_ME placeholders."""
        doc = _load_yaml(_CONNECTION_TEMPLATE)
        string_data = doc.get("stringData", {}) or {}
        for key, value in string_data.items():
            assert value == "REPLACE_ME", (
                f"Connection fixture key {key!r} has value {value!r}. "
                "All credential fields must be 'REPLACE_ME' in the template. "
                "Never commit real credentials. "
                "Use kubeseal and openshift/gitops/secrets/ for sealed production secrets."
            )

    def test_is_documented_as_template(self) -> None:
        """Fixture file text must document it is a fixture/template (not real credentials)."""
        text = _load_text(_CONNECTION_TEMPLATE)
        assert "REPLACE_ME" in text, (
            "Connection fixture must contain REPLACE_ME placeholders. "
            "This confirms it is a template, not a populated secret."
        )
        # The file comment should mention it is a fixture/template
        assert "template" in text.lower() or "fixture" in text.lower(), (
            "Connection fixture header comment should describe it as a "
            "fixture or template. "
            "Add a comment explaining this is NOT a live credential file."
        )


# ---------------------------------------------------------------------------
# 2. CPU Hardware Profile fixture
# ---------------------------------------------------------------------------

class TestCPUHardwareProfileFixture:
    """Contract tests for the CPU-only Hardware Profile fixture.

    RHOAI 3.3 primitive: Hardware Profile
    API: infrastructure.opendatahub.io/v1, kind: HardwareProfile
    Status: GA in RHOAI 3.3 (replaces deprecated Accelerator Profiles)

    This profile defines the lightweight CPU-only shape for smoke tests.
    """

    def test_fixture_file_exists(self) -> None:
        """hardware-profile-cpu-smoke.yaml must exist in fixtures dir."""
        assert _CPU_PROFILE.exists(), (
            f"Fixture not found: {_CPU_PROFILE}\n"
            "Create it from docs/openshift-ai-3.3-alignment.md §Hardware Profile."
        )

    def test_api_version(self) -> None:
        """CPU HardwareProfile apiVersion must be infrastructure.opendatahub.io/v1."""
        doc = _load_yaml(_CPU_PROFILE)
        assert doc.get("apiVersion") == "infrastructure.opendatahub.io/v1", (
            "CPU HardwareProfile apiVersion must be 'infrastructure.opendatahub.io/v1'. "
            f"Got: {doc.get('apiVersion')!r}\n"
            "This is the GA RHOAI 3.3 Hardware Profile API."
        )

    def test_kind_is_hardware_profile(self) -> None:
        """CPU fixture kind must be HardwareProfile."""
        doc = _load_yaml(_CPU_PROFILE)
        assert doc.get("kind") == "HardwareProfile", (
            f"CPU fixture must have kind=HardwareProfile. Got: {doc.get('kind')!r}"
        )

    def test_has_display_name(self) -> None:
        """CPU HardwareProfile must have a displayName in spec."""
        doc = _load_yaml(_CPU_PROFILE)
        assert doc.get("spec", {}).get("displayName"), (
            "CPU HardwareProfile must have spec.displayName. "
            "This is displayed in the RHOAI dashboard when users select a profile."
        )

    def test_no_gpu_identifier(self) -> None:
        """CPU HardwareProfile must not include nvidia.com/gpu identifier.

        This profile is for CPU-only smoke/test workloads.
        GPU requests must come from a GPU Hardware Profile.
        """
        doc = _load_yaml(_CPU_PROFILE)
        identifiers = doc.get("spec", {}).get("identifiers", []) or []
        gpu_identifiers = [
            i for i in identifiers
            if "gpu" in str(i.get("identifier", "")).lower()
            or "nvidia.com" in str(i.get("identifier", ""))
        ]
        assert not gpu_identifiers, (
            "CPU HardwareProfile must not request GPU resources. "
            f"Found GPU identifiers: {gpu_identifiers}\n"
            "GPU selection belongs in hardware-profile-gpu-pragma-s.yaml."
        )

    def test_has_cpu_identifier(self) -> None:
        """CPU HardwareProfile must define a CPU identifier."""
        doc = _load_yaml(_CPU_PROFILE)
        identifiers = doc.get("spec", {}).get("identifiers", []) or []
        cpu_ids = [
            i for i in identifiers
            if i.get("identifier") == "cpu"
        ]
        assert cpu_ids, (
            "CPU HardwareProfile must define a 'cpu' identifier in spec.identifiers. "
            f"Identifiers found: {[i.get('identifier') for i in identifiers]}"
        )


# ---------------------------------------------------------------------------
# 3. GPU Hardware Profile fixture
# ---------------------------------------------------------------------------

class TestGPUHardwareProfileFixture:
    """Contract tests for the GPU Hardware Profile fixture.

    RHOAI 3.3 primitive: Hardware Profile
    API: infrastructure.opendatahub.io/v1, kind: HardwareProfile
    Status: GA in RHOAI 3.3

    This profile defines the GPU shape for PRAGMA-S production training.
    """

    def test_fixture_file_exists(self) -> None:
        """hardware-profile-gpu-pragma-s.yaml must exist in fixtures dir."""
        assert _GPU_PROFILE.exists(), (
            f"Fixture not found: {_GPU_PROFILE}\n"
            "Create it from docs/openshift-ai-3.3-alignment.md §Hardware Profile."
        )

    def test_api_version(self) -> None:
        """GPU HardwareProfile apiVersion must be infrastructure.opendatahub.io/v1."""
        doc = _load_yaml(_GPU_PROFILE)
        assert doc.get("apiVersion") == "infrastructure.opendatahub.io/v1", (
            "GPU HardwareProfile apiVersion must be 'infrastructure.opendatahub.io/v1'. "
            f"Got: {doc.get('apiVersion')!r}\n"
            "This is the GA RHOAI 3.3 Hardware Profile API."
        )

    def test_kind_is_hardware_profile(self) -> None:
        """GPU fixture kind must be HardwareProfile."""
        doc = _load_yaml(_GPU_PROFILE)
        assert doc.get("kind") == "HardwareProfile", (
            f"GPU fixture must have kind=HardwareProfile. Got: {doc.get('kind')!r}"
        )

    def test_has_gpu_identifier(self) -> None:
        """GPU HardwareProfile must include an nvidia.com/gpu identifier.

        This proves the fixture correctly represents a GPU-backed training shape.
        The pragma_encoder training code is GPU-agnostic (reads RANK/WORLD_SIZE);
        GPU allocation is an OpenShift AI / Hardware Profile concern.
        """
        doc = _load_yaml(_GPU_PROFILE)
        identifiers = doc.get("spec", {}).get("identifiers", []) or []
        gpu_ids = [
            i for i in identifiers
            if "nvidia.com/gpu" in str(i.get("identifier", ""))
        ]
        assert gpu_ids, (
            "GPU HardwareProfile must include nvidia.com/gpu in spec.identifiers. "
            f"Identifiers found: {[i.get('identifier') for i in identifiers]}\n"
            "GPU resource selection is a Hardware Profile concern, "
            "not a pragma_encoder concern."
        )

    def test_has_node_selector_for_gpu_nodes(self) -> None:
        """GPU HardwareProfile should target GPU nodes via nodeSelector."""
        doc = _load_yaml(_GPU_PROFILE)
        node_selector = doc.get("spec", {}).get("nodeSelector", {}) or {}
        assert node_selector, (
            "GPU HardwareProfile should define spec.nodeSelector to target GPU nodes. "
            "Without this, training pods may be scheduled on non-GPU nodes."
        )

    def test_gpu_profile_does_not_conflict_with_cpu_profile(self) -> None:
        """GPU and CPU profiles must have different names.

        Prevents accidental profile collision on the cluster.
        """
        cpu_doc = _load_yaml(_CPU_PROFILE)
        gpu_doc = _load_yaml(_GPU_PROFILE)
        cpu_name = cpu_doc.get("metadata", {}).get("name")
        gpu_name = gpu_doc.get("metadata", {}).get("name")
        assert cpu_name != gpu_name, (
            f"CPU and GPU Hardware Profiles must have different names. "
            f"Both are named: {cpu_name!r}"
        )


# ---------------------------------------------------------------------------
# 4. TrainJob Tech Preview example
# ---------------------------------------------------------------------------

class TestTrainJobExampleFixture:
    """Contract tests for the Kubeflow Trainer v2 TrainJob example.

    RHOAI 3.3 primitive: TrainJob (Kubeflow Trainer v2)
    API: trainer.kubeflow.org/v1alpha1
    Status: TECHNOLOGY PREVIEW in RHOAI 3.3

    This fixture is a reference example only. It must not be used in
    production or referenced by runtime tests. Its presence documents the
    forward-looking TrainJob direction without replacing the current
    PyTorchJob (kubeflow.org/v1) GA tests.
    """

    def test_fixture_file_exists(self) -> None:
        """trainjob-example.yaml must exist in fixtures dir."""
        assert _TRAINJOB_EXAMPLE.exists(), (
            f"Fixture not found: {_TRAINJOB_EXAMPLE}\n"
            "Create it from docs/openshift-ai-3.3-alignment.md "
            "§Kubeflow Trainer v2 / TrainJob (Tech Preview)."
        )

    def test_api_version_is_trainer_v1alpha1(self) -> None:
        """TrainJob apiVersion must be trainer.kubeflow.org/v1alpha1."""
        doc = _load_yaml(_TRAINJOB_EXAMPLE)
        assert doc.get("apiVersion") == "trainer.kubeflow.org/v1alpha1", (
            "TrainJob fixture apiVersion must be 'trainer.kubeflow.org/v1alpha1'. "
            f"Got: {doc.get('apiVersion')!r}\n"
            "This is the Tech Preview Kubeflow Trainer v2 API in RHOAI 3.3."
        )

    def test_kind_is_trainjob(self) -> None:
        """TrainJob fixture kind must be TrainJob."""
        doc = _load_yaml(_TRAINJOB_EXAMPLE)
        assert doc.get("kind") == "TrainJob", (
            f"TrainJob fixture must have kind=TrainJob. Got: {doc.get('kind')!r}"
        )

    def test_file_text_documents_tech_preview_status(self) -> None:
        """TrainJob fixture file must explicitly state it is Tech Preview.

        This prevents the file from being mistaken for a production manifest.
        The file comment must use the words 'Technology Preview' or 'Tech Preview'
        and 'NOT for production use' or equivalent.
        """
        text = _load_text(_TRAINJOB_EXAMPLE)
        has_tp_label = (
            "Technology Preview" in text
            or "Tech Preview" in text
            or "TECHNOLOGY PREVIEW" in text
        )
        assert has_tp_label, (
            "TrainJob fixture file must explicitly state its Technology Preview status. "
            "Add a comment: '# Status: TECHNOLOGY PREVIEW in RHOAI 3.3'. "
            "This prevents it from being applied to production clusters."
        )

    def test_file_text_warns_not_for_production(self) -> None:
        """TrainJob fixture must warn it is not for production use."""
        text = _load_text(_TRAINJOB_EXAMPLE)
        has_prod_warning = (
            "NOT for production" in text
            or "not for production" in text.lower()
            or "Do NOT" in text
            or "do not use in production" in text.lower()
        )
        assert has_prod_warning, (
            "TrainJob fixture must warn it is not for production use. "
            "Add a comment: '# Do NOT use in production.' "
            "TrainJob is Tech Preview; GA primitive is PyTorchJob (kubeflow.org/v1)."
        )

    def test_trainjob_does_not_replace_pytorchjob_tests(self) -> None:
        """TrainJob fixture must not be referenced by any runtime test file.

        Runtime tests must use PyTorchJob (kubeflow.org/v1) — the current GA API.
        TrainJob is documented here as the forward-looking direction only.
        """
        tests_dir = pathlib.Path(__file__).parent
        runtime_test_files = [
            tests_dir / "openshift" / "test_04_pytorchjob_smoke.py",
            tests_dir / "openshift" / "test_05_s3_checkpoint_resume.py",
            tests_dir / "openshift" / "test_06_gpu_training_smoke.py",
        ]
        for test_file in runtime_test_files:
            if not test_file.exists():
                continue
            text = test_file.read_text()
            # If the test file loads the TrainJob fixture, that is a violation
            assert "trainjob-example" not in text, (
                f"{test_file.name} must not reference the TrainJob fixture. "
                "Runtime tests must use PyTorchJob (kubeflow.org/v1), the GA API. "
                "TrainJob fixture is for documentation purposes only."
            )
            assert "trainer.kubeflow.org/v1alpha1" not in text, (
                f"{test_file.name} must not use trainer.kubeflow.org/v1alpha1. "
                "This is a Tech Preview API. Use kubeflow.org/v1 PyTorchJob instead."
            )


# ---------------------------------------------------------------------------
# 5. Fixtures directory integrity
# ---------------------------------------------------------------------------

class TestFixturesDirectoryIntegrity:
    """Structural checks on the tests/openshift/fixtures/ directory itself.

    Ensures the fixtures directory is properly structured and no fixture
    files have leaked into inappropriate locations.
    """

    def test_fixtures_dir_exists(self) -> None:
        """tests/openshift/fixtures/ must exist."""
        assert _FIXTURES_DIR.is_dir(), (
            f"fixtures directory not found: {_FIXTURES_DIR}\n"
            "Create tests/openshift/fixtures/ per the repo structure rules."
        )

    def test_fixtures_readme_exists(self) -> None:
        """tests/openshift/fixtures/README.md must document the directory purpose."""
        readme = _FIXTURES_DIR / "README.md"
        assert readme.exists(), (
            f"README.md not found in: {_FIXTURES_DIR}\n"
            "Add a README documenting what the fixtures are and the boundary rules."
        )

    def test_no_fixture_yaml_in_wheel_src(self) -> None:
        """No HardwareProfile or Connection YAML must exist inside src/pragma_encoder/.

        OpenShift AI platform constructs must not be owned by the wheel.
        """
        src_dir = pathlib.Path(__file__).parent.parent / "src" / "pragma_encoder"
        rhoai_yaml_files = list(src_dir.rglob("hardware-profile*.yaml")) + \
                           list(src_dir.rglob("*connection*.yaml")) + \
                           list(src_dir.rglob("*trainjob*.yaml"))
        assert not rhoai_yaml_files, (
            "RHOAI platform YAML found inside src/pragma_encoder/. "
            "OpenShift AI constructs must not be owned by the wheel. "
            f"Found: {[str(f) for f in rhoai_yaml_files]}\n"
            "Move them to tests/openshift/fixtures/ or openshift/."
        )

    def test_all_expected_fixtures_present(self) -> None:
        """All four expected fixture files must be present."""
        expected = [
            "object-storage-connection.yaml.template",
            "hardware-profile-cpu-smoke.yaml",
            "hardware-profile-gpu-pragma-s.yaml",
            "trainjob-example.yaml",
            "README.md",
        ]
        for name in expected:
            path = _FIXTURES_DIR / name
            assert path.exists(), (
                f"Expected fixture file not found: {path}\n"
                "Each fixture documents an RHOAI 3.3 platform primitive. "
                f"See docs/openshift-ai-3.3-alignment.md."
            )
