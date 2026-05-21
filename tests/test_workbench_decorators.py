"""Tests for the workbench-decorated pipeline authoring API.

Derived from ADR 004: Workbench-Decorated Pipeline Authoring.
Paper reference: Section 2.4 (Training Infrastructure)

The decorator API allows data scientists to express training intent in
normal Python. The decorated object captures that intent and can compile
it to a KFP v2 pipeline YAML for submission to OpenShift Pipelines.

Target UX:

    from pragma_encoder.workbench import pragma_pipeline, dataset, train

    @pragma_pipeline(name="pragma-s-ibm-tabformer")
    def run():
        ds = dataset("ibm-tabformer", prepare_if_missing=True)
        train(dataset=ds, model_size="S", epochs=1, max_steps=1)

    run.show_pipeline()
    run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

Test categories (adapted for a workbench API component):
    Contract tests:    verify decorator returns correct type, exposes correct methods
    Intent tests:      verify dataset() and train() record intent without side effects
    Interface tests:   compile(), show_pipeline(), submit() behaviour
    Integration tests: reuses pipeline/, no circular imports, modes unchanged

ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pathlib
import sys

import pytest

# These already exist and must remain importable.
from pragma_encoder.workbench._api import train_pragma
from pragma_encoder.workbench._decorators import PragmaPipeline, pragma_pipeline

# ---------------------------------------------------------------------------
# Imports from modules that do not exist yet (red phase).
# These will raise ImportError until implementation is provided.
# ---------------------------------------------------------------------------
from pragma_encoder.workbench._intent import DatasetIntent, TrainIntent, dataset, train
from pragma_encoder.workbench._run import PIPELINE_STEP_NAMES, PragmaRun

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_DATASET = "ibm-tabformer"


def _make_decorated_pipeline(name: str = "test-pipeline") -> PragmaPipeline:
    """Return a decorated pipeline with a simple ibm-tabformer + PRAGMA-S intent."""

    @pragma_pipeline(name=name)
    def _run():
        ds = dataset(_VALID_DATASET, prepare_if_missing=True)
        train(dataset=ds, model_size="S", epochs=1, max_steps=1)

    return _run


# ===========================================================================
# 1. TestPragmaPipelineDecorator
#    @pragma_pipeline returns a PragmaPipeline object; no training at authoring time.
# ===========================================================================

class TestPragmaPipelineDecorator:
    """Verify @pragma_pipeline returns a PragmaPipeline without side effects.

    ADR 004: decorating a function captures intent only. No adapter, no S3,
    no subprocess, no cluster is touched during decoration.
    """

    def test_decorator_returns_pragma_pipeline(self) -> None:
        """ADR 004: @pragma_pipeline must return a PragmaPipeline instance."""
        result = _make_decorated_pipeline()
        assert isinstance(result, PragmaPipeline), (
            f"@pragma_pipeline must return a PragmaPipeline, got {type(result)}"
        )

    def test_decorator_does_not_return_pragma_run(self) -> None:
        """ADR 004: decoration must not execute training; result is not a PragmaRun."""
        result = _make_decorated_pipeline()
        assert not isinstance(result, PragmaRun), (
            "@pragma_pipeline must not execute training at decoration time. "
            "The result must be a PragmaPipeline intent object, not a PragmaRun."
        )

    def test_pragma_pipeline_exposes_show_pipeline(self) -> None:
        """ADR 004: PragmaPipeline must expose show_pipeline()."""
        result = _make_decorated_pipeline()
        assert hasattr(result, "show_pipeline") and callable(result.show_pipeline), (
            "PragmaPipeline must expose show_pipeline() method"
        )

    def test_pragma_pipeline_exposes_compile(self) -> None:
        """ADR 004: PragmaPipeline must expose compile(path)."""
        result = _make_decorated_pipeline()
        assert hasattr(result, "compile") and callable(result.compile), (
            "PragmaPipeline must expose compile(path) method"
        )

    def test_pragma_pipeline_exposes_submit(self) -> None:
        """ADR 004: PragmaPipeline must expose submit() (even if NotImplementedError)."""
        result = _make_decorated_pipeline()
        assert hasattr(result, "submit") and callable(result.submit), (
            "PragmaPipeline must expose submit() method"
        )

    def test_pragma_pipeline_name_stored(self) -> None:
        """ADR 004: the pipeline name passed to @pragma_pipeline must be stored."""
        result = _make_decorated_pipeline(name="my-pipeline")
        assert result.name == "my-pipeline", (
            f"PragmaPipeline.name must equal the name passed to @pragma_pipeline, "
            f"got {result.name!r}"
        )

    def test_decorator_does_not_call_adapter_prepare(self, monkeypatch) -> None:
        """ADR 004: decorating a function must not call DatasetAdapter.prepare().

        Verified by monkeypatching get_adapter to a spy — if prepare() is
        called, the test fails.
        """
        from unittest.mock import MagicMock
        mock_prepare = MagicMock(side_effect=AssertionError(
            "adapter.prepare() must NOT be called during @pragma_pipeline decoration"
        ))
        mock_adapter = MagicMock()
        mock_adapter.prepare = mock_prepare
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        monkeypatch.setattr("pragma_encoder.workbench._api.get_adapter",
                            lambda _name: mock_adapter_cls)

        # Decorating must not raise (prepare must not be called)
        @pragma_pipeline(name="no-side-effects")
        def _run():
            ds = dataset(_VALID_DATASET)
            train(dataset=ds, model_size="S")

        mock_prepare.assert_not_called()

    def test_decorator_does_not_call_subprocess(self, monkeypatch) -> None:
        """ADR 004: decorating a function must not call subprocess.run()."""
        from unittest.mock import MagicMock
        mock_run = MagicMock(side_effect=AssertionError(
            "subprocess.run() must NOT be called during @pragma_pipeline decoration"
        ))
        monkeypatch.setattr("pragma_encoder.workbench._api.subprocess",
                            MagicMock(run=mock_run))

        @pragma_pipeline(name="no-subprocess")
        def _run():
            ds = dataset(_VALID_DATASET)
            train(dataset=ds, model_size="S")

        mock_run.assert_not_called()

    def test_decorator_does_not_access_s3(self, monkeypatch) -> None:
        """ADR 004: decorating a function must not access S3, even if creds absent."""
        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET",
                    "MODEL_REGISTRY_ACCESS_KEY", "MODEL_REGISTRY_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)

        # Must not raise — no S3 access at decoration time
        @pragma_pipeline(name="no-s3")
        def _run():
            ds = dataset(_VALID_DATASET)
            train(dataset=ds, model_size="S")


# ===========================================================================
# 2. TestDatasetIntent
#    dataset() records dataset name and options; no side effects.
# ===========================================================================

class TestDatasetIntent:
    """Verify dataset() captures intent correctly with no side effects.

    ADR 004: dataset() is an intent-capture function. It returns a value
    object recording the dataset name and options. No adapter is looked up,
    no data is read, no S3 is accessed.
    """

    def test_dataset_returns_dataset_intent(self) -> None:
        """ADR 004: dataset() must return a DatasetIntent."""
        result = dataset("ibm-tabformer")
        assert isinstance(result, DatasetIntent), (
            f"dataset() must return a DatasetIntent, got {type(result)}"
        )

    def test_dataset_intent_stores_name(self) -> None:
        """ADR 004: DatasetIntent must store the dataset name."""
        intent = dataset("ibm-tabformer")
        assert intent.name == "ibm-tabformer", (
            f"DatasetIntent.name must be 'ibm-tabformer', got {intent.name!r}"
        )

    def test_dataset_intent_stores_prepare_if_missing_true(self) -> None:
        """ADR 004: DatasetIntent must store prepare_if_missing=True."""
        intent = dataset("ibm-tabformer", prepare_if_missing=True)
        assert intent.prepare_if_missing is True, (
            f"DatasetIntent.prepare_if_missing must be True, got {intent.prepare_if_missing!r}"
        )

    def test_dataset_intent_stores_prepare_if_missing_false(self) -> None:
        """ADR 004: DatasetIntent must store prepare_if_missing=False."""
        intent = dataset("ibm-tabformer", prepare_if_missing=False)
        assert intent.prepare_if_missing is False

    def test_dataset_intent_prepare_if_missing_defaults_true(self) -> None:
        """ADR 004: prepare_if_missing must default to True."""
        intent = dataset("ibm-tabformer")
        assert intent.prepare_if_missing is True, (
            "dataset() prepare_if_missing must default to True"
        )

    def test_dataset_does_not_call_get_adapter(self, monkeypatch) -> None:
        """ADR 004: dataset() must not call get_adapter() — no registry lookup."""
        from unittest.mock import MagicMock
        spy = MagicMock(side_effect=AssertionError(
            "get_adapter() must NOT be called by dataset()"
        ))
        monkeypatch.setattr("pragma_encoder.workbench._api.get_adapter", spy)

        dataset("ibm-tabformer")  # must not raise

        spy.assert_not_called()

    def test_dataset_does_not_access_s3(self, monkeypatch) -> None:
        """ADR 004: dataset() must not access S3."""
        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET"):
            monkeypatch.delenv(var, raising=False)

        dataset("ibm-tabformer")  # must not raise


# ===========================================================================
# 3. TestTrainIntent
#    train() records model_size, epochs, max_steps; no side effects.
# ===========================================================================

class TestTrainIntent:
    """Verify train() captures training intent correctly with no side effects.

    ADR 004: train() is an intent-capture function. It records the training
    parameters. No subprocess is started, no adapter is called.
    """

    def _ds(self) -> DatasetIntent:
        return dataset(_VALID_DATASET)

    def test_train_returns_train_intent(self) -> None:
        """ADR 004: train() must return a TrainIntent."""
        result = train(dataset=self._ds(), model_size="S")
        assert isinstance(result, TrainIntent), (
            f"train() must return a TrainIntent, got {type(result)}"
        )

    def test_train_intent_stores_model_size(self) -> None:
        """ADR 004: TrainIntent must store model_size."""
        intent = train(dataset=self._ds(), model_size="S")
        assert intent.model_size == "S", (
            f"TrainIntent.model_size must be 'S', got {intent.model_size!r}"
        )

    def test_train_intent_stores_epochs(self) -> None:
        """ADR 004: TrainIntent must store epochs."""
        intent = train(dataset=self._ds(), model_size="S", epochs=5)
        assert intent.epochs == 5, (
            f"TrainIntent.epochs must be 5, got {intent.epochs!r}"
        )

    def test_train_intent_stores_max_steps(self) -> None:
        """ADR 004: TrainIntent must store max_steps."""
        intent = train(dataset=self._ds(), model_size="S", max_steps=10)
        assert intent.max_steps == 10, (
            f"TrainIntent.max_steps must be 10, got {intent.max_steps!r}"
        )

    def test_train_intent_max_steps_defaults_none(self) -> None:
        """ADR 004: max_steps must default to None."""
        intent = train(dataset=self._ds(), model_size="S")
        assert intent.max_steps is None, (
            f"TrainIntent.max_steps must default to None, got {intent.max_steps!r}"
        )

    def test_train_intent_stores_dataset_intent(self) -> None:
        """ADR 004: TrainIntent must store the DatasetIntent passed to it."""
        ds = self._ds()
        intent = train(dataset=ds, model_size="S")
        assert intent.dataset is ds or intent.dataset == ds, (
            "TrainIntent must store the DatasetIntent passed to train()"
        )

    def test_train_does_not_call_subprocess(self, monkeypatch) -> None:
        """ADR 004: train() must not call subprocess.run()."""
        from unittest.mock import MagicMock
        spy = MagicMock(side_effect=AssertionError(
            "subprocess.run() must NOT be called by train()"
        ))
        monkeypatch.setattr("pragma_encoder.workbench._api.subprocess", MagicMock(run=spy))

        train(dataset=self._ds(), model_size="S")

        spy.assert_not_called()

    def test_train_does_not_access_s3(self, monkeypatch) -> None:
        """ADR 004: train() must not access S3."""
        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET"):
            monkeypatch.delenv(var, raising=False)

        train(dataset=self._ds(), model_size="S")  # must not raise


# ===========================================================================
# 4. TestPipelineStages
#    show_pipeline() displays all five canonical §2.4 stage names.
# ===========================================================================

class TestPipelineStages:
    """Verify the decorated pipeline exposes the five §2.4 stages.

    ADR 004 / ADR 003: stage names are prepare/upload/submit/train/export.
    show_pipeline() must surface them consistently with PragmaRun.show_pipeline().
    """

    def test_show_pipeline_does_not_raise(self) -> None:
        """ADR 004: show_pipeline() must not raise on a decorated pipeline."""
        p = _make_decorated_pipeline()
        p.show_pipeline()  # must not raise

    def test_show_pipeline_includes_prepare(self, capsys) -> None:
        """ADR 004: show_pipeline() must include 'prepare' stage."""
        _make_decorated_pipeline().show_pipeline()
        assert "prepare" in capsys.readouterr().out

    def test_show_pipeline_includes_upload(self, capsys) -> None:
        """ADR 004: show_pipeline() must include 'upload' stage."""
        _make_decorated_pipeline().show_pipeline()
        assert "upload" in capsys.readouterr().out

    def test_show_pipeline_includes_submit(self, capsys) -> None:
        """ADR 004: show_pipeline() must include 'submit' stage."""
        _make_decorated_pipeline().show_pipeline()
        assert "submit" in capsys.readouterr().out

    def test_show_pipeline_includes_train(self, capsys) -> None:
        """ADR 004: show_pipeline() must include 'train' stage."""
        _make_decorated_pipeline().show_pipeline()
        assert "train" in capsys.readouterr().out

    def test_show_pipeline_includes_export(self, capsys) -> None:
        """ADR 004: show_pipeline() must include 'export' stage."""
        _make_decorated_pipeline().show_pipeline()
        assert "export" in capsys.readouterr().out

    def test_show_pipeline_does_not_say_dry_run(self, capsys) -> None:
        """ADR 004: a decorated pipeline is not a dry_run — must not show DRY RUN banner."""
        _make_decorated_pipeline().show_pipeline()
        assert "DRY RUN" not in capsys.readouterr().out, (
            "show_pipeline() for a decorated pipeline must not print the DRY RUN banner. "
            "DRY RUN is reserved for train_pragma(mode='dry_run') previews."
        )

    def test_pipeline_stage_names_match_pipeline_step_names(self) -> None:
        """ADR 004 / ADR 003: stage names must match PIPELINE_STEP_NAMES.

        The workbench show_pipeline() and the decorated pipeline must use
        identical stage names so they stay in sync.
        """
        p = _make_decorated_pipeline()
        # PragmaPipeline should expose a stages attribute or equivalent
        assert hasattr(p, "stages") or hasattr(p, "steps"), (
            "PragmaPipeline must expose stages/steps so callers can inspect them"
        )
        stage_names_attr = getattr(p, "stages", None) or getattr(p, "steps", None)
        stage_names = (
            [s.name for s in stage_names_attr]
            if hasattr(stage_names_attr[0], "name")
            else list(stage_names_attr)
        )
        assert stage_names == list(PIPELINE_STEP_NAMES), (
            f"PragmaPipeline stages must match PIPELINE_STEP_NAMES "
            f"{list(PIPELINE_STEP_NAMES)}, got {stage_names}"
        )


# ===========================================================================
# 5. TestCompileBehavior
#    compile() uses KFP when available; clear error when KFP is missing.
# ===========================================================================

class TestCompileBehavior:
    """Verify compile() uses KFP when available and fails clearly when absent.

    ADR 004: compile() is the compile-to-YAML step. It requires KFP.
    When KFP is absent, it must raise RuntimeError or ImportError with
    a message explaining how to install the dependency.
    """

    def test_compile_missing_kfp_raises(self, tmp_path) -> None:
        """ADR 004: compile() must raise when KFP is not installed.

        When kfp is absent, compile() must raise RuntimeError or ImportError
        with a message directing the user to install requirements.txt or
        use the prepared workbench image.
        """
        kfp_available = importlib.util.find_spec("kfp") is not None
        if kfp_available:
            pytest.skip("kfp is installed — this test only applies without kfp")

        p = _make_decorated_pipeline()
        output = tmp_path / "pipeline.yaml"

        with pytest.raises((RuntimeError, ImportError)) as exc_info:
            p.compile(str(output))

        msg = str(exc_info.value).lower()
        # Must mention some form of "install" or "kfp" so the user knows what to do
        assert "kfp" in msg or "install" in msg or "requirements" in msg, (
            f"compile() error message must mention kfp, install, or requirements.txt. "
            f"Got: {exc_info.value!r}"
        )

    def test_compile_kfp_available_writes_yaml(self, tmp_path) -> None:
        """ADR 004: compile() must write a non-empty YAML file when KFP is installed."""
        pytest.importorskip("kfp", reason="kfp not installed — skipping compile test")

        p = _make_decorated_pipeline()
        output = tmp_path / "pragma-s-ibm-tabformer.yaml"

        p.compile(str(output))

        assert output.exists(), "compile() must create the output file"
        assert output.stat().st_size > 0, "compile() must write a non-empty YAML file"

    def test_compile_accepts_string_path(self, tmp_path) -> None:
        """ADR 004: compile() must accept a string path argument."""
        pytest.importorskip("kfp", reason="kfp not installed")

        p = _make_decorated_pipeline()
        p.compile(str(tmp_path / "out.yaml"))  # must not raise TypeError

    def test_compile_accepts_pathlib_path(self, tmp_path) -> None:
        """ADR 004: compile() must accept a pathlib.Path argument."""
        pytest.importorskip("kfp", reason="kfp not installed")

        p = _make_decorated_pipeline()
        p.compile(tmp_path / "out.yaml")  # must not raise TypeError


# ===========================================================================
# 6. TestCompileNoSideEffects
#    compile() must not submit training, touch S3, or invoke oc/kubectl.
# ===========================================================================

class TestCompileNoSideEffects:
    """Verify compile() has no training or infrastructure side effects.

    ADR 004: compile() converts intent to YAML only. It must not start
    training, mutate S3, call oc/kubectl, or submit any cluster job.
    """

    def test_compile_does_not_call_subprocess_run(self, tmp_path) -> None:
        """ADR 004: compile() must not invoke subprocess.run()."""
        pytest.importorskip("kfp", reason="kfp not installed")
        from unittest.mock import patch

        p = _make_decorated_pipeline()
        output = str(tmp_path / "out.yaml")

        with patch("subprocess.run") as mock_run:
            p.compile(output)

        mock_run.assert_not_called()

    def test_compile_does_not_mutate_s3(self, tmp_path, monkeypatch) -> None:
        """ADR 004: compile() must not access S3, even if creds are absent."""
        pytest.importorskip("kfp", reason="kfp not installed")

        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET",
                    "MODEL_REGISTRY_ACCESS_KEY", "MODEL_REGISTRY_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)

        p = _make_decorated_pipeline()
        # Must not raise — no S3 access during compile
        p.compile(str(tmp_path / "out.yaml"))

    def test_compile_does_not_call_oc_kubectl(self, tmp_path) -> None:
        """ADR 004: compile() must not invoke oc or kubectl."""
        pytest.importorskip("kfp", reason="kfp not installed")
        from unittest.mock import patch

        p = _make_decorated_pipeline()
        output = str(tmp_path / "out.yaml")

        with patch("subprocess.run") as mock_run:
            p.compile(output)

        for call in mock_run.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("args", [])
            cmd_str = " ".join(str(c) for c in cmd)
            assert not cmd_str.startswith("oc") and "oc " not in cmd_str, (
                f"compile() must not call 'oc', got: {cmd_str!r}"
            )
            assert "kubectl" not in cmd_str, (
                f"compile() must not call 'kubectl', got: {cmd_str!r}"
            )

    def test_compile_does_not_call_adapter_prepare(self, tmp_path) -> None:
        """ADR 004: compile() must not call DatasetAdapter.prepare()."""
        pytest.importorskip("kfp", reason="kfp not installed")
        from unittest.mock import MagicMock, patch

        p = _make_decorated_pipeline()
        output = str(tmp_path / "out.yaml")

        with patch("pragma_encoder.workbench._api.get_adapter") as mock_get:
            mock_get.return_value = MagicMock(
                return_value=MagicMock(
                    prepare=MagicMock(side_effect=AssertionError(
                        "prepare() must not be called by compile()"
                    ))
                )
            )
            p.compile(output)  # must not raise AssertionError


# ===========================================================================
# 7. TestPipelineReuse
#    compile() delegates to existing pipeline components; no duplication.
# ===========================================================================

class TestPipelineReuse:
    """Verify compile() reuses existing pipeline/ components.

    ADR 004: the decorator layer must not duplicate training or tokeniser
    logic. compile() must reference pipeline.pragma_pipeline (or its
    components) as the lower-level implementation building block.
    """

    def test_decorators_source_does_not_duplicate_training_logic(self) -> None:
        """ADR 004: _decorators.py must not contain training loop code.

        The training implementation lives in scripts/train_pragma.py and
        pipeline/components_pragma.py. The decorator layer must not
        reimplement it.
        """
        src = pathlib.Path("src/pragma_encoder/workbench/_decorators.py").read_text()
        # These are signs of duplicated training logic:
        forbidden_patterns = [
            "DataLoader",     # training loop detail
            "optimizer",      # training loop detail
            "loss.backward",  # training loop detail
        ]
        for pattern in forbidden_patterns:
            assert pattern not in src, (
                f"_decorators.py must not contain training logic ({pattern!r}). "
                f"Delegate to pipeline/ components instead."
            )

    def test_compile_uses_pipeline_module(self) -> None:
        """ADR 004: compile() must reference pipeline.pragma_pipeline or components.

        The decorator layer must delegate to the existing pipeline implementation.
        Using importlib.import_module is the expected pattern to avoid the
        src -> pipeline circular import guard.
        """
        src = pathlib.Path("src/pragma_encoder/workbench/_decorators.py").read_text()
        # Either references the module by string (importlib) or via compile logic
        has_pipeline_ref = (
            "pipeline.pragma_pipeline" in src
            or "pragma_pipeline" in src
            or "pragma_pretraining_pipeline" in src
        )
        assert has_pipeline_ref, (
            "_decorators.py must reference pipeline.pragma_pipeline "
            "(via importlib or otherwise) so compile() delegates to the existing "
            "pipeline implementation rather than duplicating it."
        )

    def test_no_static_pipeline_import_in_decorators(self) -> None:
        """ADR 004: src/workbench/_decorators.py must not statically import pipeline/.

        The text 'from pipeline' and 'import pipeline' must not appear in
        _decorators.py source — this would fail the existing circular-dependency
        guard in tests/test_pipeline_components.py::TestNoCircularDependency.
        """
        src = pathlib.Path("src/pragma_encoder/workbench/_decorators.py").read_text()
        assert "from pipeline" not in src, (
            "_decorators.py must not contain 'from pipeline' — use "
            "importlib.import_module inside compile() to avoid the circular "
            "dependency guard."
        )
        assert "import pipeline" not in src, (
            "_decorators.py must not contain 'import pipeline' — use "
            "importlib.import_module inside compile() to avoid the circular "
            "dependency guard."
        )

    def test_no_static_pipeline_import_in_intent(self) -> None:
        """ADR 004: src/workbench/_intent.py must not import from pipeline/."""
        src = pathlib.Path("src/pragma_encoder/workbench/_intent.py").read_text()
        assert "from pipeline" not in src
        assert "import pipeline" not in src


# ===========================================================================
# 8. TestSubmitNotImplemented
#    submit() raises NotImplementedError with a helpful message.
# ===========================================================================

class TestSubmitNotImplemented:
    """Verify submit() raises NotImplementedError until explicitly implemented.

    ADR 004: compile() precedes submit(). submit() is a future extension.
    It must raise NotImplementedError with a message mentioning compile().
    """

    def test_submit_raises_not_implemented(self) -> None:
        """ADR 004: submit() must raise NotImplementedError."""
        p = _make_decorated_pipeline()
        with pytest.raises(NotImplementedError):
            p.submit()

    def test_submit_error_mentions_compile(self) -> None:
        """ADR 004: submit() NotImplementedError must mention compile()."""
        p = _make_decorated_pipeline()
        with pytest.raises(NotImplementedError) as exc_info:
            p.submit()
        msg = str(exc_info.value).lower()
        assert "compile" in msg, (
            f"submit() error must mention compile(). Got: {exc_info.value!r}"
        )


# ===========================================================================
# 9. TestTrainPragmaModeIntegration
#    train_pragma(mode="pipeline") uses the decorated/compiled path.
# ===========================================================================

class TestTrainPragmaModeIntegration:
    """Verify train_pragma(mode='pipeline') returns a pipeline-mode object.

    ADR 004: mode="pipeline" creates and returns a PragmaPipeline (or equivalent
    object) representing the compiled intent. It must not affect dry_run or
    local modes.
    """

    def test_train_pragma_pipeline_mode_returns_pipeline_object(self) -> None:
        """ADR 004: train_pragma(mode='pipeline') must return an object with compile()."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            epochs=1,
            mode="pipeline",
            max_steps=1,
        )
        assert hasattr(run, "compile") and callable(run.compile), (
            "train_pragma(mode='pipeline') must return an object with compile() method"
        )

    def test_train_pragma_pipeline_mode_exposes_show_pipeline(self) -> None:
        """ADR 004: train_pragma(mode='pipeline') result must have show_pipeline()."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            epochs=1,
            mode="pipeline",
        )
        assert hasattr(run, "show_pipeline") and callable(run.show_pipeline)

    def test_train_pragma_pipeline_mode_does_not_train(self) -> None:
        """ADR 004: train_pragma(mode='pipeline') must not execute training."""
        from unittest.mock import MagicMock, patch

        with patch("pragma_encoder.workbench._api.subprocess") as mock_subp:
            mock_subp.run.return_value = MagicMock(returncode=0)
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                epochs=1,
                mode="pipeline",
            )

        mock_subp.run.assert_not_called()

    def test_train_pragma_pipeline_mode_does_not_access_s3(self, monkeypatch) -> None:
        """ADR 004: train_pragma(mode='pipeline') must not access S3."""
        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET"):
            monkeypatch.delenv(var, raising=False)

        # Must not raise
        train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode="pipeline",
        )


# ===========================================================================
# 10. TestDryRunUnchanged
#     mode="dry_run" is unchanged by ADR 004.
# ===========================================================================

class TestDryRunUnchanged:
    """Verify mode='dry_run' is unaffected by the decorator changes.

    ADR 004: dry_run and local modes are explicitly unchanged.
    """

    def test_dry_run_still_returns_pragma_run(self) -> None:
        """ADR 004: train_pragma(mode='dry_run') must still return a PragmaRun."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode="dry_run",
        )
        assert isinstance(run, PragmaRun)

    def test_dry_run_still_shows_dry_run_banner(self, capsys) -> None:
        """ADR 004: dry_run show_pipeline() must still show DRY RUN banner."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode="dry_run",
        )
        run.show_pipeline()
        assert "DRY RUN" in capsys.readouterr().out

    def test_dry_run_still_all_steps_pending(self) -> None:
        """ADR 004: dry_run steps must all still be 'pending'."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode="dry_run",
        )
        for step in run.steps:
            assert step.status == "pending"


# ===========================================================================
# 11. TestLocalModeUnchanged
#     mode="local" is unchanged by ADR 004.
# ===========================================================================

class TestLocalModeUnchanged:
    """Verify mode='local' is unaffected by the decorator changes.

    ADR 004: dry_run and local modes are explicitly unchanged.
    KFP is not required for local mode.
    """

    _CSV = "tests/fixtures/ibm_tabformer_tiny.csv"
    _OUT = "/tmp/pragma-local-decorator-test"

    def _local_run(self) -> PragmaRun:
        from unittest.mock import MagicMock, patch

        from pragma_encoder.data.dataset_manifest import DatasetManifest, DatasetShard
        from pragma_encoder.model.config import PRAGMAConfig
        shard = DatasetShard(uri="local/shard.csv", format="csv", rows=4)
        manifest = DatasetManifest(
            dataset_name="ibm-tabformer", dataset_version="v1",
            prepared_prefix_uri="local/", shards=(shard,),
            vocab_uri=None, schema_uri=None, manifest_uri=None,
            row_count=4, source={"origin": "ibm-tabformer"},
            config=PRAGMAConfig.pragma_s(),
        )
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        with patch("pragma_encoder.workbench._api.get_adapter", return_value=mock_adapter_cls), \
             patch("pragma_encoder.workbench._api.subprocess") as mock_subp, \
             patch("pathlib.Path.mkdir"):
            mock_subp.run.return_value = MagicMock(returncode=0)
            run = train_pragma(
                dataset=_VALID_DATASET, model_size="S", epochs=1, mode="local",
                local_csv_path=self._CSV, output_dir=self._OUT, max_steps=1,
            )
        return run

    def test_local_mode_still_returns_pragma_run(self) -> None:
        """ADR 004: mode='local' must still return a PragmaRun."""
        run = self._local_run()
        assert isinstance(run, PragmaRun)

    def test_local_mode_run_mode_still_local(self) -> None:
        """ADR 004: mode='local' run_mode must still be 'local'."""
        run = self._local_run()
        assert run.run_mode == "local"

    def test_local_mode_does_not_require_kfp(self) -> None:
        """ADR 004: mode='local' must work without kfp installed."""
        kfp_before = "kfp" in sys.modules
        self._local_run()
        if not kfp_before:
            assert "kfp" not in sys.modules, (
                "local mode must not import kfp as a side effect"
            )


# ===========================================================================
# 12. TestNoPvcStorage
#     No PVC parameters appear in the decorator API.
# ===========================================================================

class TestNoPvcStorage:
    """Verify no PVC-based storage is introduced by the decorator API.

    ADR 004 / ADR 003 / openshift-storage-pattern.md:
    canonical data storage is S3. RWO PVCs must not appear.
    """

    _PVC_PATTERNS = ("pvc_path", "pvc_name", "pvc_uri", "data_pvc",
                     "dataset_pvc", "storage_pvc")

    def _has_pvc_param(self, sig: inspect.Signature) -> list[str]:
        found = []
        for name in sig.parameters:
            for pattern in self._PVC_PATTERNS:
                if pattern in name.lower():
                    found.append(name)
        return found

    def test_pragma_pipeline_decorator_no_pvc(self) -> None:
        """ADR 004: pragma_pipeline() must not accept PVC storage parameters."""
        sig = inspect.signature(pragma_pipeline)
        pvc = self._has_pvc_param(sig)
        assert not pvc, f"pragma_pipeline has PVC parameters: {pvc}"

    def test_dataset_function_no_pvc(self) -> None:
        """ADR 004: dataset() must not accept PVC storage parameters."""
        sig = inspect.signature(dataset)
        pvc = self._has_pvc_param(sig)
        assert not pvc, f"dataset() has PVC parameters: {pvc}"

    def test_train_function_no_pvc(self) -> None:
        """ADR 004: train() must not accept PVC storage parameters."""
        sig = inspect.signature(train)
        pvc = self._has_pvc_param(sig)
        assert not pvc, f"train() has PVC parameters: {pvc}"

    def test_decorators_source_no_pvc_strings(self) -> None:
        """ADR 004: _decorators.py source must not contain PVC-based path strings."""
        src = pathlib.Path("src/pragma_encoder/workbench/_decorators.py").read_text()
        pvc_strings = [s for s in ("ReadWriteOnce", "PersistentVolumeClaim", "pvc_path")
                       if s in src]
        assert not pvc_strings, (
            f"_decorators.py contains PVC references: {pvc_strings}"
        )

    def test_intent_source_no_pvc_strings(self) -> None:
        """ADR 004: _intent.py source must not contain PVC-based path strings."""
        src = pathlib.Path("src/pragma_encoder/workbench/_intent.py").read_text()
        pvc_strings = [s for s in ("ReadWriteOnce", "PersistentVolumeClaim", "pvc_path")
                       if s in src]
        assert not pvc_strings, (
            f"_intent.py contains PVC references: {pvc_strings}"
        )


# ===========================================================================
# 13. TestDecoratorValidation
#     @pragma_pipeline raises on zero or multiple train() calls.
# ===========================================================================

class TestDecoratorValidation:
    """Verify @pragma_pipeline raises instead of silently fabricating intent.

    ADR 004 hardening: a decorated function must call train() exactly once.
    Zero calls → the decorator cannot capture intent and must not fabricate.
    Multiple calls → ambiguous intent; the decorator must not silently pick one.
    Both cases raise ValueError with a helpful message that mentions train().
    """

    def test_zero_train_calls_raises_value_error(self) -> None:
        """ADR 004: @pragma_pipeline raises ValueError when train() is never called.

        The decorator must not silently fabricate a TrainIntent with
        name='unknown'. A missing train() call is a programming error
        and must be surfaced immediately at decoration time.
        """
        with pytest.raises(ValueError):
            @pragma_pipeline(name="no-train")
            def _run():
                dataset(_VALID_DATASET)
                # Deliberately omit train() call

    def test_zero_train_calls_error_mentions_train_function(self) -> None:
        """ADR 004: the ValueError for zero train() calls must mention train().

        The error message must guide the data scientist toward the fix:
        adding a train() call inside the decorated function body.
        """
        with pytest.raises(ValueError) as exc_info:
            @pragma_pipeline(name="no-train-message")
            def _run():
                dataset(_VALID_DATASET)
                # Deliberately omit train() call

        msg = str(exc_info.value).lower()
        assert "train" in msg, (
            f"ValueError for zero train() calls must mention 'train'. "
            f"Got: {exc_info.value!r}"
        )

    def test_multiple_train_calls_raises(self) -> None:
        """ADR 004: @pragma_pipeline raises when train() is called more than once.

        Multiple train() calls produce ambiguous intent — it is unclear which
        training configuration should be compiled. The decorator must reject
        this rather than silently picking the first (or any other) call.
        """
        with pytest.raises((ValueError, NotImplementedError)):
            @pragma_pipeline(name="two-trains")
            def _run():
                ds = dataset(_VALID_DATASET)
                train(dataset=ds, model_size="S", epochs=1)
                train(dataset=ds, model_size="M", epochs=2)  # second call — ambiguous

    def test_multiple_train_calls_error_is_helpful(self) -> None:
        """ADR 004: the error for multiple train() calls must be actionable.

        The error message must help the data scientist understand that only
        one train() call is allowed per @pragma_pipeline decorated function.
        It must mention either 'train' or 'once' or 'one'.
        """
        with pytest.raises((ValueError, NotImplementedError)) as exc_info:
            @pragma_pipeline(name="two-trains-message")
            def _run():
                ds = dataset(_VALID_DATASET)
                train(dataset=ds, model_size="S", epochs=1)
                train(dataset=ds, model_size="M", epochs=2)

        msg = str(exc_info.value).lower()
        assert "train" in msg or "once" in msg or "one" in msg, (
            f"Error for multiple train() calls must mention 'train', 'once', or 'one'. "
            f"Got: {exc_info.value!r}"
        )
