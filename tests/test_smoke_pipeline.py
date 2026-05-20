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
        """pragma_smoke_training component must call fit_tokenizer.py.

        fit_tokenizer.py builds vocab.pkl from the inline CSV before training
        starts. Without this, train_pragma.py would fail with a missing vocab.
        Verified by inspecting the source text (not execution — no cluster needed).
        """
        src = _smoke_pipeline_source()
        assert "fit_tokenizer" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference fit_tokenizer. "
            "The component must run src/data/fit_tokenizer.py before training."
        )

    def test_component_source_references_train_pragma_max_steps(self) -> None:
        """pragma_smoke_training component must call train_pragma.py with --max-steps.

        train_pragma.py is the training entrypoint. --max-steps 1 (or the
        max_steps parameter) ensures the smoke test terminates quickly.
        Verified by source inspection — no cluster needed.
        """
        src = _smoke_pipeline_source()
        assert "train_pragma" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference train_pragma. "
            "The component must call scripts/train_pragma.py."
        )
        assert "max-steps" in src or "max_steps" in src, (
            "pipeline/pragma_smoke_pipeline.py source must reference max-steps or max_steps. "
            "The component must pass --max-steps to train_pragma.py to stop early."
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
                "kfp is not installed. Install with: pip install 'pragma-encoder[workbench]'. "
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
