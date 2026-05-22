"""Level 2 — Decorated pipeline compilation verification.

Verifies that the PRAGMA workbench pipeline can be compiled to KFP YAML
without submitting anything to the cluster. This is a pure local operation.

These tests are included in tests/openshift/ because they are part of the
cluster-facing development gate — a failing compile means the pipeline
cannot be submitted to OpenShift Pipelines even if the cluster is ready.

No cluster resources are created by any test in this file.
KFP must be installed for the compile tests; they skip if kfp is absent.

Tests:
  - test_decorated_pipeline_example_exists
  - test_decorated_pipeline_compile_succeeds_if_kfp_installed
  - test_generated_pipeline_yaml_exists_after_compile
  - test_generated_pipeline_yaml_contains_expected_stages
  - test_compile_does_not_require_oc_or_cluster

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - PRAGMA_TEST_NAMESPACE=<namespace> (required by conftest; not used here)
  - kfp installed (tests skip gracefully if absent)
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

# ---------------------------------------------------------------------------
# KFP availability sentinel
# ---------------------------------------------------------------------------

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None

_skip_no_kfp = pytest.mark.skipif(
    not _KFP_AVAILABLE,
    reason="kfp is not installed — skipping compile tests. "
           "Install with: pip install kfp",
)

# ---------------------------------------------------------------------------
# Expected pipeline component identifiers in compiled YAML.
# KFP sanitises function names (lowercase, hyphens), so we match against
# the sanitised forms used by the pragma_pipeline.py components.
# ---------------------------------------------------------------------------

_EXPECTED_STAGE_NAMES = [
    "prepare-dataset",
    "upload-artifacts",
    "submit-pytorchjob",
    "run-pretraining",
    "export-checkpoint",
]


class TestDecoratedPipelineExample:
    """Level 2: example file and KFP compile path."""

    def test_decorated_pipeline_example_exists(self) -> None:
        """examples/workbench/05_decorated_pipeline.py must exist.

        This file is the canonical workbench authoring reference.
        If it is missing, the decorated pipeline path is untested end-to-end.
        """
        example_path = pathlib.Path("examples/workbench/05_decorated_pipeline.py")
        assert example_path.exists(), (
            f"Example file {example_path} not found. "
            "This file demonstrates the @pragma_pipeline decorator (ADR 004). "
            "It must exist as the canonical authoring reference."
        )

    @_skip_no_kfp
    def test_decorated_pipeline_compile_succeeds_if_kfp_installed(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """@pragma_pipeline.compile() must succeed when kfp is installed.

        Constructs a minimal decorated pipeline inline (not by running the
        example script — that would produce console output) and compiles it
        to a temporary YAML file. No cluster access is required.
        """
        from tools.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415

        @pragma_pipeline(name="pragma-s-test-compile")
        def _test_pipeline() -> None:
            ds = dataset("ibm-tabformer", prepare_if_missing=True)
            train(dataset=ds, model_size="S", epochs=1, max_steps=1)

        output_yaml = tmp_path / "pragma-s-test-compile.yaml"
        _test_pipeline.compile(str(output_yaml))  # must not raise

        assert output_yaml.exists(), (
            "compile() returned without error but the output YAML was not created."
        )

    @_skip_no_kfp
    def test_generated_pipeline_yaml_exists_after_compile(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Compiled YAML must exist and be non-empty after compile().

        An empty or missing file indicates compile() silently failed or
        wrote to the wrong path.
        """
        from tools.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415

        @pragma_pipeline(name="pragma-s-yaml-check")
        def _test_pipeline() -> None:
            ds = dataset("ibm-tabformer", prepare_if_missing=True)
            train(dataset=ds, model_size="S", epochs=1, max_steps=1)

        output_yaml = tmp_path / "pragma-s-yaml-check.yaml"
        _test_pipeline.compile(str(output_yaml))

        assert output_yaml.exists(), "YAML output file does not exist after compile()."
        size = output_yaml.stat().st_size
        assert size > 100, (  # noqa: PLR2004
            f"YAML output is suspiciously small ({size} bytes). "
            "Expected a non-trivial KFP pipeline definition."
        )

    @_skip_no_kfp
    def test_generated_pipeline_yaml_contains_expected_stages(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Compiled YAML must reference all five PRAGMA pipeline stages.

        KFP sanitises component function names (lowercase, hyphens).
        Expected identifiers in the YAML:
          prepare-dataset, upload-artifacts, submit-pytorchjob,
          run-pretraining, export-checkpoint

        If any are missing, the pipeline definition is incomplete and the
        submitted PipelineRun would not execute the full training workflow.
        """
        from tools.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415

        @pragma_pipeline(name="pragma-s-stages-check")
        def _test_pipeline() -> None:
            ds = dataset("ibm-tabformer", prepare_if_missing=True)
            train(dataset=ds, model_size="S", epochs=1, max_steps=1)

        output_yaml = tmp_path / "pragma-s-stages-check.yaml"
        _test_pipeline.compile(str(output_yaml))

        yaml_text = output_yaml.read_text()

        missing = [name for name in _EXPECTED_STAGE_NAMES if name not in yaml_text]
        assert not missing, (
            f"Compiled YAML is missing expected stage identifiers: {missing}. "
            "All five PRAGMA pipeline stages must appear in the compiled output: "
            f"{_EXPECTED_STAGE_NAMES}"
        )

    def test_compile_does_not_require_oc_or_cluster(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Pipeline compilation must not invoke oc or require cluster access.

        Monkeypatches the oc() function in tests.openshift.oc to raise
        AssertionError if called. The compile() call must not trigger it.

        This verifies the ADR 004 contract: compile() uses KFP only,
        with no cluster side-effects.

        If kfp is not installed, the test documents the expected behaviour
        via assertion without attempting the compile.
        """
        import tests.openshift.oc as oc_module  # noqa: PLC0415

        _oc_called: list[str] = []

        def _oc_must_not_be_called(
            args: list[str],
            **kwargs: object,
        ) -> None:
            _oc_called.append(str(args))
            raise AssertionError(
                f"oc() was called during compile(): {args}. "
                "Pipeline compilation must be a pure local KFP operation "
                "with no cluster access."
            )

        monkeypatch.setattr(oc_module, "oc", _oc_must_not_be_called)

        if not _KFP_AVAILABLE:
            # Without kfp, compile() raises ImportError before any oc call.
            # That is the correct and expected behaviour — document it.
            from tools.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415, I001

            @pragma_pipeline(name="pragma-s-no-oc-check")
            def _test_pipeline() -> None:
                ds = dataset("ibm-tabformer", prepare_if_missing=True)
                train(dataset=ds, model_size="S", epochs=1, max_steps=1)

            with pytest.raises((ImportError, Exception)):
                _test_pipeline.compile(str(tmp_path / "out.yaml"))

            assert not _oc_called, (
                "oc() was called before the ImportError was raised. "
                "compile() must not touch the cluster at any point."
            )
            return

        from tools.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415, I001

        @pragma_pipeline(name="pragma-s-no-oc-check")
        def _test_pipeline() -> None:
            ds = dataset("ibm-tabformer", prepare_if_missing=True)
            train(dataset=ds, model_size="S", epochs=1, max_steps=1)

        output_yaml = tmp_path / "pragma-s-no-oc.yaml"
        _test_pipeline.compile(str(output_yaml))  # must not call oc()

        assert not _oc_called, (
            f"oc() was called during compile(): {_oc_called}. "
            "compile() must be a pure KFP operation with no cluster access."
        )
