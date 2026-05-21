"""Tests for pipeline/pragma_smoke_pipeline.py.

Verifies the minimal KFP v2 smoke pipeline component used by the Level 3
DSPA/KFP v2 integration test (tests/openshift/test_03_pipeline_smoke_run.py).

This module contains no cluster tests. All tests run locally without
a KFP server, DSPA endpoint, or OpenShift cluster.

Tests:
    Contract tests:   module imports cleanly; image env var pattern correct
    Source tests:     component body references required training commands
    Safety tests:     no PyTorchJob / Tekton / S3 references in smoke module
    Compile test:     pipeline compiles to KFP v2 YAML (skips if kfp absent)
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pathlib
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_smoke_pipeline():
    """Import pipeline.pragma_smoke_pipeline, handling optional kfp.

    Returns the module object. Does not raise on import even when kfp is
    absent (that is what test_smoke_module_importable_without_kfp verifies).
    """
    # Ensure a fresh import so monkeypatch effects on sys.modules are visible.
    if "pipeline.pragma_smoke_pipeline" in sys.modules:
        del sys.modules["pipeline.pragma_smoke_pipeline"]
    return importlib.import_module("pipeline.pragma_smoke_pipeline")


def _smoke_pipeline_source() -> str:
    """Return the full source text of pragma_smoke_pipeline.py."""
    import pipeline.pragma_smoke_pipeline as mod  # noqa: PLC0415 — intentional
    src_file = inspect.getfile(mod)
    return pathlib.Path(src_file).read_text()


# ===========================================================================
# 1. Contract tests — module-level properties
# ===========================================================================


class TestSmokePipelineContract:
    """Contract tests: importability, image env var, pipeline symbols."""

    def test_smoke_module_importable_without_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pipeline.pragma_smoke_pipeline must import cleanly without kfp.

        kfp is an optional dependency (pyproject.toml [workbench] extra).
        The module must guard the kfp import exactly as components_pragma.py
        does — try/except at module level, identity decorator fallback.
        """
        # Simulate kfp absent.
        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]
        if "pipeline.pragma_smoke_pipeline" in sys.modules:
            monkeypatch.delitem(sys.modules, "pipeline.pragma_smoke_pipeline")

        try:
            mod = importlib.import_module("pipeline.pragma_smoke_pipeline")
            assert mod is not None, (
                "pipeline.pragma_smoke_pipeline must be importable without kfp. "
                "Guard the kfp import with try/except ImportError."
            )
        except ImportError as exc:
            pytest.fail(
                f"pipeline.pragma_smoke_pipeline raised ImportError without kfp: {exc}. "
                "Use the same try/except guard as components_pragma.py."
            )

    def test_smoke_image_reads_pragma_training_image_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_SMOKE_IMAGE must be read from PRAGMA_TRAINING_IMAGE env var.

        KFP @dsl.component(base_image=...) captures the image at decoration
        time (module import). The env var must be read at module-load time
        so that the workbench can set it before compiling the pipeline.

        Confirms _SMOKE_IMAGE == PRAGMA_TRAINING_IMAGE when that var is set.
        """
        sentinel = "image-registry.example.com/test/pragma-training:sentinel"
        monkeypatch.setenv("PRAGMA_TRAINING_IMAGE", sentinel)
        # Clear any PRAGMA_KFP_COMPONENT_IMAGE that might shadow it.
        monkeypatch.delenv("PRAGMA_KFP_COMPONENT_IMAGE", raising=False)

        # Force re-import so the new env var is read at module load time.
        if "pipeline.pragma_smoke_pipeline" in sys.modules:
            del sys.modules["pipeline.pragma_smoke_pipeline"]

        mod = importlib.import_module("pipeline.pragma_smoke_pipeline")

        assert mod._SMOKE_IMAGE == sentinel, (
            f"_SMOKE_IMAGE should equal PRAGMA_TRAINING_IMAGE={sentinel!r} "
            f"when that var is set. Got: {mod._SMOKE_IMAGE!r}. "
            "Read the env var at module load time (same pattern as _BASE_IMAGE "
            "in components_pragma.py)."
        )

    def test_smoke_image_has_non_empty_default_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_SMOKE_IMAGE must have a non-empty default when env vars are unset.

        The fallback is the cluster-internal training image URI. It allows
        the module to be imported and inspected without the env var being set
        (e.g. in local development and CI where PRAGMA_TRAINING_IMAGE is absent).
        """
        monkeypatch.delenv("PRAGMA_TRAINING_IMAGE", raising=False)
        monkeypatch.delenv("PRAGMA_KFP_COMPONENT_IMAGE", raising=False)

        if "pipeline.pragma_smoke_pipeline" in sys.modules:
            del sys.modules["pipeline.pragma_smoke_pipeline"]

        mod = importlib.import_module("pipeline.pragma_smoke_pipeline")

        assert mod._SMOKE_IMAGE, (
            "_SMOKE_IMAGE must not be empty when PRAGMA_TRAINING_IMAGE is unset. "
            "Provide a non-empty default fallback (the cluster-internal image URI)."
        )
        assert isinstance(mod._SMOKE_IMAGE, str), (
            "_SMOKE_IMAGE must be a str."
        )

    def test_pragma_smoke_training_pipeline_symbol_exists(self) -> None:
        """pipeline.pragma_smoke_pipeline must export pragma_smoke_training_pipeline.

        This symbol is imported by test_03_pipeline_smoke_run.py to compile
        and submit the KFP v2 Run. It must be defined at module level.
        """
        mod = _import_smoke_pipeline()
        assert hasattr(mod, "pragma_smoke_training_pipeline"), (
            "pipeline.pragma_smoke_pipeline must define pragma_smoke_training_pipeline. "
            "This is the @dsl.pipeline function submitted to the DSPA KFP v2 API."
        )

    def test_pragma_smoke_training_component_symbol_exists(self) -> None:
        """pipeline.pragma_smoke_pipeline must export pragma_smoke_training.

        The component function is decorated with @dsl.component. It is
        importable and callable without kfp (identity decorator fallback).
        """
        mod = _import_smoke_pipeline()
        assert hasattr(mod, "pragma_smoke_training"), (
            "pipeline.pragma_smoke_pipeline must define pragma_smoke_training. "
            "This is the @dsl.component function that runs the training scripts."
        )


# ===========================================================================
# 2. Source-inspection tests — component body content
# ===========================================================================


class TestSmokePipelineSource:
    """Source inspection: verify the component body calls the right scripts."""

    def test_component_source_references_fit_tokenizer(self) -> None:
        """pragma_smoke_training component must invoke fit_tokenizer.

        fit_tokenizer builds vocab.pkl from the inline CSV before training
        starts. Without this, train_pragma.py would fail with a missing vocab.
        The wheel-based image provides it as python -m pragma_encoder.data.fit_tokenizer.
        Verified by inspecting the source text (not execution — no cluster needed).
        """
        src = _smoke_pipeline_source()
        assert "fit_tokenizer" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference fit_tokenizer. "
            "The component must invoke pragma_encoder.data.fit_tokenizer before training."
        )

    def test_component_source_references_train_pragma_max_steps(self) -> None:
        """pragma_smoke_training component must invoke pragma_encoder.training.train
        with --max-steps.

        pragma_encoder.training.train is the canonical wheel-based training entrypoint.
        --max-steps 1 (or the max_steps parameter) ensures the smoke test terminates quickly.
        Verified by source inspection — no cluster needed.
        """
        src = _smoke_pipeline_source()
        assert "pragma_encoder.training.train" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference "
            "pragma_encoder.training.train. "
            "The component must invoke python -m pragma_encoder.training.train "
            "(wheel-based module invocation, not scripts/train_pragma.py)."
        )
        assert "max-steps" in src or "max_steps" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference max-steps or max_steps. "
            "The component must pass --max-steps to pragma_encoder.training.train to stop early."
        )

    def test_component_uses_module_invocation_for_fit_tokenizer(self) -> None:
        """pragma_smoke_training must invoke fit_tokenizer as a Python module.

        The wheel-based training image installs pragma_encoder into site-packages
        but does not contain a src/ source tree. The component must invoke
        fit_tokenizer via module execution:
            [sys.executable, "-m", "pragma_encoder.data.fit_tokenizer"]

        Direct file-path invocation (str(project_root / "src" / ...)) fails in
        the wheel image because the src/ directory is not present.
        """
        src = _smoke_pipeline_source()
        assert "pragma_encoder.data.fit_tokenizer" in src, (
            "pragma_smoke_pipeline.py must invoke fit_tokenizer as a module: "
            "[sys.executable, '-m', 'pragma_encoder.data.fit_tokenizer']. "
            "Source-tree file-path invocation is not compatible with the wheel image."
        )
        assert '"-m"' in src or "'-m'" in src, (
            "pragma_smoke_pipeline.py must use the '-m' flag to invoke fit_tokenizer. "
            "Expected: [sys.executable, '-m', 'pragma_encoder.data.fit_tokenizer']. "
            "The wheel image has no src/ directory — module invocation is required."
        )

    def test_component_source_references_pragma_s_variant(self) -> None:
        """pragma_smoke_training component must use the pragma-s model variant.

        The smoke test must use PRAGMA-S (10M params) so the component pod
        can run on CPU within the smoke test timeout. Larger variants risk OOM
        or timeout. Verified by source inspection.
        """
        src = _smoke_pipeline_source()
        assert "pragma-s" in src or "pragma_s" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference pragma-s (PRAGMA-S). "
            "The smoke component must use the small model variant for fast pod execution."
        )

    def test_component_source_prints_completion_marker(self) -> None:
        """pragma_smoke_training component must print a completion marker.

        The Level 3 smoke test asserts 'PRAGMA smoke training completed' in
        the KFP pod logs. The component must print this exact string to stdout
        on successful completion. Verified by source inspection.
        """
        src = _smoke_pipeline_source()
        assert "PRAGMA smoke training completed" in src, (
            "pipeline/pragma_smoke_pipeline.py must print 'PRAGMA smoke training completed'. "
            "test_03_pipeline_smoke_run.py asserts this marker in the KFP pod logs."
        )


# ===========================================================================
# 3. Safety tests — no forbidden references
# ===========================================================================


class TestSmokePipelineSafety:
    """Safety: smoke component must not reference forbidden infrastructure."""

    def test_smoke_module_has_no_pytorchjob_reference(self) -> None:
        """pragma_smoke_pipeline.py must not reference PyTorchJob.

        Level 3 smoke runs in a single KFP component pod (no DDP, no KFTO).
        A PyTorchJob reference would indicate the wrong infrastructure is being
        used — PyTorchJob is Level 4 (distributed training), not Level 3.
        """
        src = _smoke_pipeline_source()
        assert "PyTorchJob" not in src and "pytorchjob" not in src.lower(), (
            "pipeline/pragma_smoke_pipeline.py must not reference PyTorchJob. "
            "Level 3 smoke uses a single component pod — PyTorchJob is Level 4."
        )

    def test_smoke_module_has_no_s3_upload_reference(self) -> None:
        """pragma_smoke_pipeline.py must not upload to S3.

        Level 3 smoke proves the KFP v2 pipeline path works. It must not
        depend on S3 credentials or object storage configuration. S3 upload
        is a production pipeline concern (Stage 2/5), not a smoke concern.
        """
        src = _smoke_pipeline_source()
        forbidden = ["boto3", "s3_upload", "upload_artifacts", "MODEL_REGISTRY_BUCKET"]
        for term in forbidden:
            assert term not in src, (
                f"pipeline/pragma_smoke_pipeline.py must not reference {term!r}. "
                "Level 3 smoke must not depend on S3 — no credentials required."
            )

    def test_component_has_no_src_tree_assumption(self) -> None:
        """pragma_smoke_training must not reference src/pragma_encoder source-tree paths.

        The wheel-based training image (openshift/training/Dockerfile.training)
        installs pragma_encoder into site-packages. There is no src/ directory
        in the image. Any reference to a source-tree path for fit_tokenizer
        causes RuntimeError at pod startup when the path is not found.

        The component must use module invocation:
            [sys.executable, "-m", "pragma_encoder.data.fit_tokenizer"]
        Not direct file execution:
            [sys.executable, str(project_root / "src" / "pragma_encoder" / ...)]
        """
        src = _smoke_pipeline_source()
        # The old broken invocation searched for the source file
        assert '"src" / "pragma_encoder"' not in src, (
            "pragma_smoke_pipeline.py must not construct source-tree paths via "
            "'src' / 'pragma_encoder'. The wheel image has no src/ directory. "
            "Use: [sys.executable, '-m', 'pragma_encoder.data.fit_tokenizer']"
        )
        # No direct file execution of fit_tokenizer via path
        assert "/src/pragma_encoder/data/fit_tokenizer" not in src, (
            "pragma_smoke_pipeline.py must not reference "
            "'/src/pragma_encoder/data/fit_tokenizer' as a file path. "
            "Use module invocation: python -m pragma_encoder.data.fit_tokenizer"
        )

    def test_smoke_module_has_no_pvc_reference(self) -> None:
        """pragma_smoke_pipeline.py must not reference PVC or persistent volumes.

        The smoke component writes transient data to /tmp inside the pod.
        No PVC or shared storage is required — the pod is ephemeral.
        """
        src = _smoke_pipeline_source()
        assert "PersistentVolumeClaim" not in src and "pvc" not in src.lower(), (
            "pipeline/pragma_smoke_pipeline.py must not reference PVC. "
            "Smoke writes to /tmp — no persistent storage needed."
        )


# ===========================================================================
# 4. Compile test — produces valid KFP v2 YAML
# ===========================================================================


class TestSmokePipelineCompile:
    """Compile: pipeline.compile() produces a non-empty KFP v2 YAML."""

    def test_smoke_pipeline_compiles_to_yaml(self) -> None:
        """pragma_smoke_training_pipeline.compile() produces valid KFP v2 YAML.

        kfp.compiler.Compiler().compile() converts the decorated pipeline to
        a YAML specification suitable for upload to the DSPA KFP v2 API.
        Skips if kfp is not installed (optional dependency).

        This test does NOT connect to any cluster — it is a local compile check.
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip(
                "kfp is not installed. Install with: pip install kfp. "
                "kfp is available in the workbench image."
            )

        import kfp  # noqa: PLC0415 — guarded by find_spec above

        from pipeline.pragma_smoke_pipeline import (  # noqa: PLC0415
            pragma_smoke_training_pipeline,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = pathlib.Path(tmp_dir) / "smoke_pipeline.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_smoke_training_pipeline,
                package_path=str(yaml_path),
            )

            assert yaml_path.exists(), (
                "kfp.compiler.Compiler().compile() did not create a YAML file. "
                "Check that pragma_smoke_training_pipeline is a valid @dsl.pipeline."
            )

            content = yaml_path.read_text()
            assert len(content) > 100, (
                f"Compiled YAML is suspiciously small ({len(content)} bytes). "
                "The pipeline definition may be empty or malformed."
            )
            assert "pragma-smoke-training" in content or "pragma_smoke" in content, (
                "Compiled YAML does not reference the smoke pipeline name. "
                "Check the @dsl.pipeline name= argument."
            )

        print(
            f"\n[Compile] pragma_smoke_training_pipeline compiled to "
            f"{len(content)} bytes of KFP v2 YAML."
        )

    def test_compiled_yaml_has_no_src_tree_paths(self) -> None:
        """Compiled pipeline YAML must not embed source-tree path invocations.

        KFP serialises the component function body into the compiled YAML.
        The compiled YAML must not contain old source-tree path patterns
        (e.g. 'src/pragma_encoder/data/fit_tokenizer') because those paths
        do not exist in the wheel-based training image.

        Skips if kfp is not installed (optional dependency).
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip("kfp is not installed. Install with: pip install kfp.")

        import kfp  # noqa: PLC0415

        from pipeline.pragma_smoke_pipeline import (  # noqa: PLC0415
            pragma_smoke_training_pipeline,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = pathlib.Path(tmp_dir) / "smoke_pipeline.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_smoke_training_pipeline,
                package_path=str(yaml_path),
            )
            content = yaml_path.read_text()

        # Old source-tree path — must not appear in compiled YAML
        assert "/src/pragma_encoder/data/fit_tokenizer" not in content, (
            "Compiled pipeline YAML embeds the old source-tree path "
            "'/src/pragma_encoder/data/fit_tokenizer'. "
            "Update the component body to use module invocation: "
            "python -m pragma_encoder.data.fit_tokenizer"
        )
        # Module invocation must be present in compiled YAML
        assert "pragma_encoder.data.fit_tokenizer" in content, (
            "Compiled pipeline YAML must embed 'pragma_encoder.data.fit_tokenizer' "
            "from the module invocation in the component body. "
            "The component should call: [sys.executable, '-m', 'pragma_encoder.data.fit_tokenizer']"
        )
        # Module invocation for training must be present (not file-path script search)
        assert "pragma_encoder.training.train" in content, (
            "Compiled pipeline YAML must embed 'pragma_encoder.training.train' "
            "from the module invocation in the component body. "
            "The component should call: [sys.executable, '-m', 'pragma_encoder.training.train']. "
            "The wheel image has no scripts/ source dependency — module invocation is required."
        )


# ===========================================================================
# 5. Image contract tests — compiled YAML bakes training image from env var
# ===========================================================================


class TestSmokePipelineImageContract:
    """Image contract: compiled YAML embeds base_image from PRAGMA_TRAINING_IMAGE.

    KFP @dsl.component(base_image=...) captures the image at decoration time
    (module import). The env var must be set before the module is imported —
    i.e., before compile time in the workbench — for the compiled YAML to
    reference the correct training image.

    These tests confirm the env-var-to-compiled-YAML pipeline works end-to-end.
    """

    def test_compiled_yaml_embeds_training_image_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compiled YAML must embed the image URI set via PRAGMA_TRAINING_IMAGE.

        Workflow that this test validates:
          1. Workbench sets PRAGMA_TRAINING_IMAGE=<image-uri> before importing the module.
          2. pragma_smoke_pipeline.py reads it at module import time into _SMOKE_IMAGE.
          3. @dsl.component(base_image=_SMOKE_IMAGE) captures _SMOKE_IMAGE at decoration.
          4. kfp.compiler.Compiler().compile() bakes the image URI into the YAML.
          5. Compiled YAML contains the image URI — confirmed here.

        If this test fails: the env var was set AFTER module import, so the old
        (default) image was captured. Set PRAGMA_TRAINING_IMAGE before importing
        pipeline.pragma_smoke_pipeline.

        Skips if kfp is not installed.
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip("kfp not installed. Install with: pip install kfp.")

        import kfp  # noqa: PLC0415

        sentinel_image = "image-registry.example.com/test/pragma-training:image-contract-test"
        monkeypatch.setenv("PRAGMA_TRAINING_IMAGE", sentinel_image)
        monkeypatch.delenv("PRAGMA_KFP_COMPONENT_IMAGE", raising=False)

        # Force re-import so the sentinel image is captured by @dsl.component at decoration.
        if "pipeline.pragma_smoke_pipeline" in sys.modules:
            del sys.modules["pipeline.pragma_smoke_pipeline"]

        from pipeline.pragma_smoke_pipeline import (  # noqa: PLC0415
            pragma_smoke_training_pipeline,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = pathlib.Path(tmp_dir) / "smoke_image_contract.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_smoke_training_pipeline,
                package_path=str(yaml_path),
            )
            content = yaml_path.read_text()

        assert sentinel_image in content, (
            f"Compiled pipeline YAML must embed the training image URI "
            f"({sentinel_image!r}) set via PRAGMA_TRAINING_IMAGE. "
            "The env var must be set BEFORE importing pipeline.pragma_smoke_pipeline. "
            "KFP captures base_image at decoration time (module import), not at compile time. "
            f"Compiled YAML excerpt (first 500 chars):\n{content[:500]}"
        )

    def test_compiled_yaml_pragma_kfp_component_image_takes_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PRAGMA_KFP_COMPONENT_IMAGE must shadow PRAGMA_TRAINING_IMAGE in compiled YAML.

        Priority order (highest to lowest):
          1. PRAGMA_KFP_COMPONENT_IMAGE — explicit KFP component override
          2. PRAGMA_TRAINING_IMAGE       — training image
          3. Default cluster-internal URI

        When PRAGMA_KFP_COMPONENT_IMAGE is set, it must appear in the compiled YAML
        instead of PRAGMA_TRAINING_IMAGE. Confirms the priority chain works.

        Skips if kfp is not installed.
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip("kfp not installed. Install with: pip install kfp.")

        import kfp  # noqa: PLC0415

        override_image = "image-registry.example.com/test/pragma-training:kfp-override"
        shadow_image = "image-registry.example.com/test/pragma-training:should-not-appear"
        monkeypatch.setenv("PRAGMA_KFP_COMPONENT_IMAGE", override_image)
        monkeypatch.setenv("PRAGMA_TRAINING_IMAGE", shadow_image)

        if "pipeline.pragma_smoke_pipeline" in sys.modules:
            del sys.modules["pipeline.pragma_smoke_pipeline"]

        from pipeline.pragma_smoke_pipeline import (  # noqa: PLC0415
            pragma_smoke_training_pipeline,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = pathlib.Path(tmp_dir) / "smoke_priority.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_smoke_training_pipeline,
                package_path=str(yaml_path),
            )
            content = yaml_path.read_text()

        assert override_image in content, (
            f"Compiled YAML must embed PRAGMA_KFP_COMPONENT_IMAGE={override_image!r} "
            "when that env var is set. "
            "PRAGMA_KFP_COMPONENT_IMAGE has higher priority than PRAGMA_TRAINING_IMAGE."
        )
        assert shadow_image not in content, (
            f"Compiled YAML must NOT embed PRAGMA_TRAINING_IMAGE={shadow_image!r} "
            f"when PRAGMA_KFP_COMPONENT_IMAGE={override_image!r} is set. "
            "PRAGMA_KFP_COMPONENT_IMAGE must shadow PRAGMA_TRAINING_IMAGE."
        )


# ===========================================================================
# 6. Hardware profile contract
# ===========================================================================


class TestSmokePipelineHardwareProfile:
    """Hardware profile: smoke pipeline correctly omits HardwareProfile selection.

    In RHOAI 3.3, KFP pipeline run pods cannot select a HardwareProfile via
    pipeline parameters. HardwareProfile.spec.identifiers applies to workbench
    notebooks (Notebook CR), not to KFP component pods.

    The smoke pipeline does NOT attempt HardwareProfile selection. This class
    verifies that constraint is preserved and documents the limitation for operators.

    Reference: docs/openshift-ai-3.3-alignment.md §Hardware Profile
    Fixtures:   tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml
                tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml
    """

    def test_smoke_module_has_no_hardware_profile_selector(self) -> None:
        """pragma_smoke_pipeline.py must not attempt HardwareProfile selection.

        HardwareProfile is an RHOAI 3.3 primitive for workbench notebooks.
        KFP component pods inherit resource requests from the @dsl.component
        decorator or default cluster limits — not from HardwareProfile.

        Attempting to wire a HardwareProfile to a KFP pipeline run would fail
        silently or error. The smoke pipeline must not include such a reference.

        Hardware profile fixtures exist in tests/openshift/fixtures/ for
        operators to apply to the cluster (requires RHOAI admin). They are
        reference fixtures documenting the expected CPU/GPU resource shapes —
        not inputs to the smoke pipeline.
        """
        src = _smoke_pipeline_source()
        assert "HardwareProfile" not in src, (
            "pragma_smoke_pipeline.py must not reference HardwareProfile. "
            "KFP component pod resources are set via @dsl.component resource limits "
            "or cluster defaults, not via OpenShift AI HardwareProfile. "
            "HardwareProfile applies to workbench notebooks (RHOAI 3.3 limitation). "
            "See docs/openshift-ai-3.3-alignment.md §Hardware Profile."
        )
        assert "hardware_profile" not in src.lower(), (
            "pragma_smoke_pipeline.py must not reference hardware_profile (case-insensitive). "
            "See test_smoke_module_has_no_hardware_profile_selector for context."
        )

    def test_hardware_profile_cpu_smoke_fixture_exists(self) -> None:
        """CPU smoke HardwareProfile fixture must exist in tests/openshift/fixtures/.

        The fixture documents the expected resource shape for smoke tests:
        cpu=500m, memory=2Gi. Operators apply it with:
          oc apply -f tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml \\
            -n redhat-ods-applications

        The fixture is a reference document for the platform configuration —
        it is not consumed by the smoke pipeline itself.
        """
        fixture_path = (
            pathlib.Path(__file__).parent
            / "openshift"
            / "fixtures"
            / "hardware-profile-cpu-smoke.yaml"
        )
        assert fixture_path.exists(), (
            f"CPU smoke HardwareProfile fixture not found: {fixture_path}. "
            "Create tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml "
            "documenting the expected CPU/memory shape for smoke workloads."
        )
        content = fixture_path.read_text()
        assert "HardwareProfile" in content, (
            "hardware-profile-cpu-smoke.yaml must define a HardwareProfile resource. "
            "Check the apiVersion and kind fields."
        )
        assert "pragma-cpu-smoke" in content, (
            "hardware-profile-cpu-smoke.yaml must define the 'pragma-cpu-smoke' profile. "
            "This name is documented in tests/openshift/README.md."
        )

    def test_hardware_profile_gpu_pragma_s_fixture_exists(self) -> None:
        """GPU PRAGMA-S HardwareProfile fixture must exist in tests/openshift/fixtures/.

        The fixture documents the expected resource shape for PRAGMA-S GPU training:
        1×NVIDIA GPU, 8 CPU, 64Gi RAM. Operators apply it with:
          oc apply -f tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml \\
            -n redhat-ods-applications

        The fixture is a reference document for the platform configuration —
        it is not consumed by the smoke pipeline itself.
        """
        fixture_path = (
            pathlib.Path(__file__).parent
            / "openshift"
            / "fixtures"
            / "hardware-profile-gpu-pragma-s.yaml"
        )
        assert fixture_path.exists(), (
            f"GPU PRAGMA-S HardwareProfile fixture not found: {fixture_path}. "
            "Create tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml "
            "documenting the expected GPU/CPU/memory shape for PRAGMA-S training."
        )
        content = fixture_path.read_text()
        assert "HardwareProfile" in content, (
            "hardware-profile-gpu-pragma-s.yaml must define a HardwareProfile resource."
        )
        assert "pragma-gpu-pragma-s" in content, (
            "hardware-profile-gpu-pragma-s.yaml must define the 'pragma-gpu-pragma-s' profile."
        )
        assert "nvidia.com/gpu" in content, (
            "hardware-profile-gpu-pragma-s.yaml must reference nvidia.com/gpu. "
            "PRAGMA-S GPU training requires NVIDIA GPU scheduling."
        )
