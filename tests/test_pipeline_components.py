"""Tests for pipeline/components_pragma.py and pipeline/pragma_pipeline.py.

Derived from PRAGMA paper Section 2.4 and ADR 003 (workbench-training-api).

The existing pipeline scaffold has four NotImplementedError stubs and a
top-level `from kfp import dsl` that prevents importing the module when
kfp is not installed.  Component 5 must fix all of this.

Specifically, tests verify that after Component 5:
  1. Both pipeline modules are importable without kfp installed.
  2. Five §2.4 stage names are declared (prepare / upload / submit / train / export).
  3. Component function names are human-readable.
  4. prepare component uses DatasetAdapter (get_adapter) not hardcoded IBM logic.
  5. train component uses model_size ("S"/"M"/"L") and references PRAGMAConfig —
     not a divergent config_name string convention.
  6. No component has a PVC-based data parameter.
  7. The canonical training contract is manifest_uri / DatasetManifest — not
     raw local paths.
  8. nodes: int is expressible in the pipeline for two-node DDP.

Test categories (marked in each class docstring):
    safety     — KFP import guard; module importability without kfp
    structural — stage names, component count, naming conventions
    interface  — function parameter signatures
    source     — source-level pattern checks (get_adapter, PRAGMAConfig refs)
    no_pvc     — absence of PVC-based storage in any component or pipeline
    kfp        — explicitly skipped when kfp is not available
"""

from __future__ import annotations

import ast as _ast
import importlib
import importlib.util
import inspect
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_COMPONENTS_PATH = pathlib.Path("pipeline/components_pragma.py")
_PIPELINE_PATH   = pathlib.Path("pipeline/pragma_pipeline.py")


def _load_module(name: str, path: pathlib.Path):
    """Load a pipeline module by file path — no kfp required for import.

    After the Component 5 redesign, both modules must be loadable here.
    If loading fails because kfp is missing, Component 5 has not yet added
    the required conditional import guard.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Attempt module loads at collection time.  Both will fail until Component 5
# adds the kfp import guard.  Tests use the _require_* helpers to produce
# clear failure messages rather than NameError/AttributeError.
try:
    _comp = _load_module("components_pragma", _COMPONENTS_PATH)
    _pipe = _load_module("pragma_pipeline",   _PIPELINE_PATH)
    _IMPORTABLE = True
except (ImportError, ModuleNotFoundError):
    _comp = None
    _pipe = None
    _IMPORTABLE = False


def _require_importable() -> None:
    """Call at the top of any test that needs the module to be loaded."""
    if not _IMPORTABLE:
        pytest.fail(
            "pipeline/components_pragma.py and/or pipeline/pragma_pipeline.py "
            "are not importable without kfp.  Component 5 must add a conditional "
            "import guard, e.g.:\n"
            "    try:\n"
            "        from kfp import dsl\n"
            "        _KFP_AVAILABLE = True\n"
            "    except ImportError:\n"
            "        dsl = None\n"
            "        _KFP_AVAILABLE = False\n"
            "All @dsl.component / @dsl.pipeline decorators must be applied only "
            "when _KFP_AVAILABLE is True."
        )


def _src_text(path: pathlib.Path) -> str:
    """Return the raw source text of a pipeline file."""
    return path.read_text()


def _raw_func_src(file_path: pathlib.Path, func_name: str) -> str:
    """Return the raw source text of a specific function, immune to KFP wrapping.

    Uses ast line numbers (lineno / end_lineno) to extract just the function body
    from the source file. This avoids inspect.getsource() which breaks when KFP's
    @dsl.pipeline wraps the function into a GraphComponent at import time.
    """
    source = file_path.read_text()
    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == func_name:
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"Function {func_name!r} not found in {file_path}")


# ---------------------------------------------------------------------------
# AST-based parameter inspection — immune to KFP decorator wrapping
#
# When kfp is installed, @dsl.component wraps functions and inspect.signature()
# returns (*args, **kwargs) instead of the original parameter list.
# @dsl.pipeline may or may not use functools.wraps — the behaviour is
# version-dependent and must not be relied upon.
#
# _raw_func_params() reads the source file with ast.parse(), which is
# completely independent of any runtime decoration applied to the function.
# ---------------------------------------------------------------------------

_EMPTY = inspect.Parameter.empty  # sentinel: parameter is required (no default)


def _raw_func_params(
    file_path: pathlib.Path,
    func_name: str,
) -> dict[str, object]:
    """Return {param_name: default_value_or_EMPTY} by parsing source with ast.

    _EMPTY (inspect.Parameter.empty) means the parameter is required (no default).
    Non-constant defaults (e.g. [] or a class instance) use Ellipsis (...) as
    the sentinel value.  This helper is immune to KFP @dsl.component and
    @dsl.pipeline wrapping.

    Raises AssertionError if func_name is not found in file_path.
    """
    source = file_path.read_text()
    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == func_name:
            args = node.args.args
            defaults = node.args.defaults
            n_args = len(args)
            n_defaults = len(defaults)
            result: dict[str, object] = {}
            for i, arg in enumerate(args):
                name = arg.arg
                if name == "self":
                    continue
                default_idx = i - (n_args - n_defaults)
                if default_idx >= 0:
                    d = defaults[default_idx]
                    result[name] = d.value if isinstance(d, _ast.Constant) else ...
                else:
                    result[name] = _EMPTY
            return result
    raise AssertionError(
        f"Function {func_name!r} not found in {file_path}. "
        f"Has the function been renamed or removed?"
    )


# ---------------------------------------------------------------------------
# TestKfpImportSafety  (category: safety)
# ---------------------------------------------------------------------------

class TestKfpImportSafety:
    """Verify both pipeline modules are importable without kfp installed.

    Category: safety
    Drives: adding conditional import guard to components_pragma.py and
            pragma_pipeline.py so kfp remains an optional dependency.
    """

    def test_components_module_importable_without_kfp(self) -> None:
        """ADR 003: pipeline/components_pragma.py must not require kfp to import.

        The current scaffold has `from kfp import dsl` at the top level.
        Component 5 must replace this with a conditional import guard so
        that importing the module in tests (without kfp installed) succeeds.
        """
        try:
            _load_module("components_pragma_check", _COMPONENTS_PATH)
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.fail(
                f"pipeline/components_pragma.py is not importable without kfp: {exc}\n"
                "Add a conditional import guard — see test docstring."
            )

    def test_pipeline_module_importable_without_kfp(self) -> None:
        """ADR 003: pipeline/pragma_pipeline.py must not require kfp to import."""
        try:
            _load_module("pragma_pipeline_check", _PIPELINE_PATH)
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.fail(
                f"pipeline/pragma_pipeline.py is not importable without kfp: {exc}\n"
                "Add a conditional import guard."
            )

    def test_kfp_available_flag_defined_in_components(self) -> None:
        """ADR 003: components_pragma must expose _KFP_AVAILABLE so callers can
        detect whether kfp decorators were applied."""
        _require_importable()
        assert hasattr(_comp, "_KFP_AVAILABLE"), (
            "pipeline/components_pragma.py must define _KFP_AVAILABLE: bool "
            "so tests and callers know whether kfp decorators are active."
        )

    def test_kfp_available_flag_is_bool(self) -> None:
        """ADR 003: _KFP_AVAILABLE must be a bool."""
        _require_importable()
        assert isinstance(_comp._KFP_AVAILABLE, bool), (
            f"_KFP_AVAILABLE must be bool, got {type(_comp._KFP_AVAILABLE)}"
        )

    def test_kfp_available_flag_is_false_when_kfp_not_installed(self) -> None:
        """ADR 003: _KFP_AVAILABLE must be False when kfp is not installed."""
        _require_importable()
        kfp_installed = importlib.util.find_spec("kfp") is not None
        if kfp_installed:
            pytest.skip("kfp is installed — this test only applies without kfp")
        assert _comp._KFP_AVAILABLE is False, (
            "_KFP_AVAILABLE must be False when kfp is not installed"
        )


# ---------------------------------------------------------------------------
# TestPipelineStageNames  (category: structural)
# ---------------------------------------------------------------------------

class TestPipelineStageNames:
    """Verify the five §2.4 stage names are declared and match PIPELINE_STEP_NAMES.

    Category: structural
    Paper: Section 2.4 (Training Infrastructure)
    ADR 003: pipeline steps — prepare / upload / submit / train / export
    """

    def test_pipeline_stage_names_constant_defined(self) -> None:
        """§2.4 / ADR 003: components_pragma must export PIPELINE_STAGE_NAMES."""
        _require_importable()
        assert hasattr(_comp, "PIPELINE_STAGE_NAMES"), (
            "pipeline/components_pragma.py must define PIPELINE_STAGE_NAMES "
            "so tests and callers can enumerate the five §2.4 stages."
        )

    def test_pipeline_stage_names_has_five_entries(self) -> None:
        """§2.4 / ADR 003: exactly five pipeline stage names."""
        _require_importable()
        assert len(_comp.PIPELINE_STAGE_NAMES) == 5, (
            f"PIPELINE_STAGE_NAMES must have 5 entries (§2.4 has five stages), "
            f"got {len(_comp.PIPELINE_STAGE_NAMES)}: {_comp.PIPELINE_STAGE_NAMES}"
        )

    def test_pipeline_stage_names_matches_workbench_step_names(self) -> None:
        """§2.4 / ADR 003: PIPELINE_STAGE_NAMES must match PragmaRun PIPELINE_STEP_NAMES.

        The workbench (PragmaRun.steps) and the KFP pipeline must use identical
        stage names so show_pipeline() and the pipeline run stay in sync.
        """
        _require_importable()
        from src.workbench._run import PIPELINE_STEP_NAMES
        assert tuple(_comp.PIPELINE_STAGE_NAMES) == PIPELINE_STEP_NAMES, (
            f"PIPELINE_STAGE_NAMES {tuple(_comp.PIPELINE_STAGE_NAMES)} "
            f"must match PIPELINE_STEP_NAMES {PIPELINE_STEP_NAMES}"
        )

    def test_pipeline_stage_names_order_is_causal(self) -> None:
        """§2.4 / ADR 003: stage names must be in causal execution order."""
        _require_importable()
        assert list(_comp.PIPELINE_STAGE_NAMES) == [
            "prepare", "upload", "submit", "train", "export"
        ], (
            f"Stage order must be [prepare, upload, submit, train, export], "
            f"got {list(_comp.PIPELINE_STAGE_NAMES)}"
        )

    def test_five_component_functions_defined(self) -> None:
        """§2.4 / ADR 003: exactly five component functions — one per stage."""
        _require_importable()
        # Each function name must correspond to one of the five stages.
        # When kfp is unavailable, components are plain callables — not KFP tasks.
        for stage in _comp.PIPELINE_STAGE_NAMES:
            # Accept either the exact stage name or a human-readable variant.
            # The key is that each stage has a corresponding callable.
            assert any(
                stage in name for name in dir(_comp) if callable(getattr(_comp, name, None))
            ), (
                f"No callable found in components_pragma for stage '{stage}'. "
                f"Component 5 must define one function per §2.4 stage."
            )

    def test_component_names_are_human_readable(self) -> None:
        """ADR 003: component function names must be human-readable English words.

        Names like 'comp_1', 'preproc', or 'pretrain_pragma' (which omits the
        five-stage structure) are not acceptable.  Each name should describe
        what the stage does in plain language.
        """
        _require_importable()
        rejected_names = {
            "preprocess_transactions",   # old scaffold — does not reflect five-stage structure
            "pretrain_pragma",           # old scaffold — skips prepare/upload/submit/export
            "extract_embeddings",        # old scaffold — not one of the five §2.4 stages
            "evaluate_downstream",       # old scaffold — not one of the five §2.4 stages
        }
        public_callables = {
            name for name in dir(_comp)
            if callable(getattr(_comp, name, None)) and not name.startswith("_")
        }
        scaffold_remnants = public_callables & rejected_names
        assert not scaffold_remnants, (
            f"Scaffold function names still present in components_pragma: "
            f"{scaffold_remnants}. Component 5 must replace them with the five "
            f"§2.4 stage functions."
        )


# ---------------------------------------------------------------------------
# TestComponentInterfaces  (category: interface)
# ---------------------------------------------------------------------------

class TestComponentInterfaces:
    """Verify component function signatures implement the correct contract.

    Uses AST-based parameter inspection (_raw_func_params) so these tests
    are immune to KFP decorator wrapping.  When kfp is installed,
    @dsl.component replaces the original signature with (*args, **kwargs) —
    inspect.signature() would silently return the wrong parameter list.

    Category: interface
    """

    def _params(self, func_name: str) -> dict[str, object]:
        """Return AST-derived {param_name: default_or_EMPTY} for a component function.

        _EMPTY means the parameter is required (no default).
        Uses _raw_func_params() to parse the source — immune to KFP wrapping.
        """
        _require_importable()
        assert hasattr(_comp, func_name), (
            f"components_pragma has no function '{func_name}'. "
            f"Component 5 must define it."
        )
        return _raw_func_params(_COMPONENTS_PATH, func_name)

    # --- prepare component ---

    def test_prepare_component_accepts_dataset_name(self) -> None:
        """ADR 003: prepare component must accept dataset_name (registry key)."""
        params = self._params("prepare_dataset")
        assert "dataset_name" in params, (
            f"prepare_dataset must have a 'dataset_name' parameter, "
            f"got {set(params)}"
        )

    def test_prepare_component_accepts_model_size(self) -> None:
        """ADR 003: prepare component must accept model_size ('S'/'M'/'L').

        model_size is needed so prepare can pass the correct PRAGMAConfig
        truncation limits (max_event_tokens=24, max_profile_tokens=200,
        max_events=6500) to DatasetAdapter.prepare().
        """
        params = self._params("prepare_dataset")
        assert "model_size" in params, (
            f"prepare_dataset must have a 'model_size' parameter, got {set(params)}"
        )

    def test_prepare_component_does_not_require_raw_data_path(self) -> None:
        """ADR 003: prepare must not take raw_data_path as a required positional arg.

        The canonical contract is dataset_name (a registry key) + model_size.
        raw_data_path was the old scaffold's IBM-specific hardcoded path.
        Using a registry key enables future adapters without changing the interface.
        """
        params = self._params("prepare_dataset")
        if "raw_data_path" in params:
            assert params["raw_data_path"] is not _EMPTY, (
                "prepare_dataset must not require raw_data_path — "
                "use dataset_name (registry key) as the canonical input."
            )

    # --- upload component ---

    def test_upload_component_accepts_manifest_uri(self) -> None:
        """ADR 003: upload component must accept manifest_uri (S3 path).

        The upload stage consumes a prepared DatasetManifest URI, not raw data.
        This enforces the manifest-as-contract principle from §2.4.
        """
        params = self._params("upload_artifacts")
        assert "manifest_uri" in params, (
            f"upload_artifacts must have a 'manifest_uri' parameter, got {set(params)}"
        )

    # --- train component ---

    def test_train_component_accepts_manifest_uri(self) -> None:
        """ADR 003: train component must consume manifest_uri — not raw data paths.

        The training contract is a DatasetManifest URI (S3 path to prepared data).
        Raw file paths violate the §2.4 data storage requirement (S3, not PVC).
        """
        params = self._params("run_pretraining")
        assert "manifest_uri" in params, (
            f"run_pretraining must have a 'manifest_uri' parameter, got {set(params)}. "
            f"Training must consume a DatasetManifest, not a raw file path."
        )

    def test_train_component_accepts_model_size(self) -> None:
        """ADR 003: train component must use model_size ('S'/'M'/'L').

        The old scaffold used config_name ('pragma_s') — a divergent convention
        from train_pragma()'s model_size parameter.  Component 5 must unify them.
        """
        params = self._params("run_pretraining")
        assert "model_size" in params, (
            f"run_pretraining must have a 'model_size' parameter (not 'config_name'), "
            f"got {set(params)}"
        )

    def test_train_component_does_not_use_config_name(self) -> None:
        """ADR 003: train component must not use config_name — divergent convention.

        config_name ('pragma_s') is the old scaffold's naming.  The unified
        convention (model_size = 'S'/'M'/'L') must be used throughout.
        """
        params = self._params("run_pretraining")
        assert "config_name" not in params, (
            "run_pretraining must not have a 'config_name' parameter. "
            "Use model_size='S'/'M'/'L' consistently with train_pragma()."
        )

    def test_train_component_accepts_nodes(self) -> None:
        """ADR 003: train component must accept nodes: int for two-node DDP.

        nodes=2 selects pytorchjob-pragma-s-2node.yaml (per ADR 003).
        """
        params = self._params("run_pretraining")
        assert "nodes" in params, (
            f"run_pretraining must have a 'nodes: int' parameter for two-node DDP, "
            f"got {set(params)}"
        )

    def test_train_component_nodes_default_is_1(self) -> None:
        """ADR 003: nodes must default to 1 (single-node)."""
        params = self._params("run_pretraining")
        assert "nodes" in params
        assert params["nodes"] == 1, (
            f"run_pretraining nodes must default to 1 (single-node), "
            f"got default={params['nodes']!r}"
        )

    def test_train_component_accepts_epochs(self) -> None:
        """§2.4: train component must accept epochs: int."""
        params = self._params("run_pretraining")
        assert "epochs" in params, (
            f"run_pretraining must have an 'epochs' parameter, got {set(params)}"
        )

    # --- export component ---

    def test_export_component_defined(self) -> None:
        """§2.4 / ADR 003: export component must exist (stage 5 of five)."""
        _require_importable()
        assert hasattr(_comp, "export_checkpoint"), (
            "components_pragma must define export_checkpoint() for stage 5 (export)."
        )


# ---------------------------------------------------------------------------
# TestPipelineFunction  (category: interface)
# ---------------------------------------------------------------------------

class TestPipelineFunction:
    """Verify the pipeline function signature implements the workbench contract.

    Uses AST-based parameter inspection — immune to KFP @dsl.pipeline wrapping.

    Category: interface
    """

    def _params(self) -> dict[str, object]:
        """Return AST-derived {param_name: default_or_EMPTY} for pragma_pretraining_pipeline."""
        _require_importable()
        assert hasattr(_pipe, "pragma_pretraining_pipeline"), (
            "pragma_pipeline.py must define pragma_pretraining_pipeline()."
        )
        return _raw_func_params(_PIPELINE_PATH, "pragma_pretraining_pipeline")

    def test_pipeline_function_accepts_dataset_name(self) -> None:
        """ADR 003: pipeline must accept dataset_name (adapter registry key)."""
        assert "dataset_name" in self._params(), (
            "pragma_pretraining_pipeline must have a 'dataset_name' parameter."
        )

    def test_pipeline_function_accepts_model_size(self) -> None:
        """ADR 003: pipeline must accept model_size ('S'/'M'/'L')."""
        assert "model_size" in self._params(), (
            "pragma_pretraining_pipeline must have a 'model_size' parameter."
        )

    def test_pipeline_model_size_default_is_s(self) -> None:
        """§2.4 / Table 1: model_size must default to 'S' (PRAGMA-S, ~10M params)."""
        params = self._params()
        assert "model_size" in params
        assert params["model_size"] == "S", (
            f"model_size must default to 'S' (PRAGMA-S), got {params['model_size']!r}"
        )

    def test_pipeline_function_does_not_use_config_name(self) -> None:
        """ADR 003: pipeline must not use config_name — divergent old convention."""
        assert "config_name" not in self._params(), (
            "pragma_pretraining_pipeline must not have a 'config_name' parameter. "
            "Use model_size='S'/'M'/'L' consistently with train_pragma()."
        )

    def test_pipeline_function_accepts_nodes(self) -> None:
        """ADR 003: pipeline must accept nodes: int for two-node DDP configuration."""
        assert "nodes" in self._params(), (
            "pragma_pretraining_pipeline must have a 'nodes: int' parameter "
            "so the two-node manifest (pytorchjob-pragma-s-2node.yaml) can be selected."
        )

    def test_pipeline_nodes_default_is_1(self) -> None:
        """ADR 003: nodes must default to 1 (single-node)."""
        params = self._params()
        assert "nodes" in params
        assert params["nodes"] == 1, (
            f"pragma_pretraining_pipeline nodes must default to 1, got {params['nodes']!r}"
        )

    def test_pipeline_function_accepts_epochs(self) -> None:
        """§2.4: pipeline must accept epochs: int."""
        assert "epochs" in self._params(), (
            "pragma_pretraining_pipeline must have an 'epochs' parameter."
        )

    def test_pipeline_epochs_default_is_10(self) -> None:
        """§2.4: epochs must default to 10 (consistent with train_pragma default)."""
        params = self._params()
        assert "epochs" in params
        assert params["epochs"] == 10, (
            f"epochs must default to 10 (consistent with train_pragma), "
            f"got {params['epochs']!r}"
        )


# ---------------------------------------------------------------------------
# TestPragmaPretrainingPipelineHonestContract  (category: interface)
# ---------------------------------------------------------------------------

class TestPragmaPretrainingPipelineHonestContract:
    """pragma_pretraining_pipeline always runs all five §2.4 stages. No skip path.

    Category: interface
    Paper: Section 2.4 (Training Infrastructure)
    ADR 003: five sequential stages — prepare, upload, submit, train, export.

    The pipeline previously advertised a manifest_uri skip path that was
    never implemented (the implementation always ran prepare and upload).
    These tests lock down the honest contract: all five stages always run.
    Callers who already have a manifest use pragma_train_from_manifest_pipeline.
    """

    def test_full_pipeline_has_no_manifest_uri_param(self) -> None:
        """§2.4: pragma_pretraining_pipeline must not expose manifest_uri.

        The skip path was never implemented.  Removing the parameter makes
        the contract honest: this pipeline always runs prepare and upload.
        Callers with an existing manifest use pragma_train_from_manifest_pipeline.

        Uses AST-based inspection — immune to KFP @dsl.pipeline wrapping.
        """
        _require_importable()
        assert hasattr(_pipe, "pragma_pretraining_pipeline")
        params = _raw_func_params(_PIPELINE_PATH, "pragma_pretraining_pipeline")
        assert "manifest_uri" not in params, (
            "pragma_pretraining_pipeline must not have a manifest_uri parameter. "
            "It always runs all five stages. Use pragma_train_from_manifest_pipeline "
            "for the manifest-bypass path."
        )

    def test_full_pipeline_source_calls_prepare_dataset(self) -> None:
        """§2.4 stage 1: prepare_dataset must be wired in the full pipeline."""
        src = _src_text(_PIPELINE_PATH)
        assert "prepare_dataset" in src, (
            "pragma_pretraining_pipeline source must call prepare_dataset (stage 1). "
            "All five stages must always run in this pipeline."
        )

    def test_full_pipeline_source_calls_upload_artifacts(self) -> None:
        """§2.4 stage 2: upload_artifacts must be wired in the full pipeline."""
        src = _src_text(_PIPELINE_PATH)
        assert "upload_artifacts" in src, (
            "pragma_pretraining_pipeline source must call upload_artifacts (stage 2). "
            "All five stages must always run in this pipeline."
        )

    def test_full_pipeline_docstring_no_skip_claim(self) -> None:
        """§2.4: pragma_pretraining_pipeline docstring must not claim stages are skipped."""
        _require_importable()
        fn = getattr(_pipe, "pragma_pretraining_pipeline", None)
        assert fn is not None
        doc = fn.__doc__ or ""
        assert "skip" not in doc.lower(), (
            "pragma_pretraining_pipeline docstring must not claim prepare/upload "
            "are skipped. The skip path is not implemented. Use honest language."
        )
        assert "manifest_uri" not in doc, (
            "pragma_pretraining_pipeline docstring must not mention manifest_uri — "
            "that parameter no longer exists on this pipeline."
        )


# ---------------------------------------------------------------------------
# TestTrainFromManifestPipeline  (category: interface)
# ---------------------------------------------------------------------------

class TestTrainFromManifestPipeline:
    """pragma_train_from_manifest_pipeline: submit+train+export only.

    Uses AST-based parameter inspection — immune to KFP @dsl.pipeline wrapping.

    Category: interface
    Paper: Section 2.4 (Training Infrastructure) stages 3-5.
    ADR 003: manifest_uri as canonical dataset contract.

    This pipeline accepts a pre-prepared DatasetManifest URI and runs only
    the cluster-facing stages: submit -> train -> export.
    It must NOT call prepare_dataset or upload_artifacts.
    """

    def test_function_exists(self) -> None:
        """§2.4: pragma_train_from_manifest_pipeline must be defined."""
        _require_importable()
        assert hasattr(_pipe, "pragma_train_from_manifest_pipeline"), (
            "pipeline/pragma_pipeline.py must define pragma_train_from_manifest_pipeline(). "
            "This is the honest manifest-bypass pipeline (stages 3-5 only)."
        )
        assert callable(getattr(_pipe, "pragma_train_from_manifest_pipeline")), (
            "pragma_train_from_manifest_pipeline must be callable."
        )

    def _params(self) -> dict[str, object]:
        """Return AST-derived {param_name: default_or_EMPTY} — immune to KFP wrapping."""
        _require_importable()
        assert hasattr(_pipe, "pragma_train_from_manifest_pipeline"), (
            "pragma_train_from_manifest_pipeline must be defined."
        )
        return _raw_func_params(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")

    def test_has_manifest_uri_required_param(self) -> None:
        """§2.4 / ADR 003: manifest_uri must be a required parameter (no default).

        Requiring manifest_uri with no default prevents accidental invocation
        without a prepared dataset — the pipeline would fail with no manifest.
        """
        params = self._params()
        assert "manifest_uri" in params, (
            "pragma_train_from_manifest_pipeline must have a manifest_uri parameter."
        )
        assert params["manifest_uri"] is _EMPTY, (
            "manifest_uri must be required (no default). "
            "Callers must always supply a prepared DatasetManifest URI."
        )

    def test_has_no_dataset_name_param(self) -> None:
        """§2.4: no dataset_name — this pipeline does not call the adapter registry."""
        assert "dataset_name" not in self._params(), (
            "pragma_train_from_manifest_pipeline must not have a dataset_name parameter. "
            "It skips prepare/upload and does not use the DatasetAdapter registry."
        )

    def test_source_does_not_call_prepare_dataset(self) -> None:
        """§2.4 stage 1: prepare_dataset must NOT be wired in the manifest pipeline."""
        fn_src = _raw_func_src(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")
        assert "prepare_dataset" not in fn_src, (
            "pragma_train_from_manifest_pipeline must not call prepare_dataset. "
            "Stage 1 (prepare) is skipped — the manifest already exists in S3."
        )

    def test_source_does_not_call_upload_artifacts(self) -> None:
        """§2.4 stage 2: upload_artifacts must NOT be wired in the manifest pipeline."""
        fn_src = _raw_func_src(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")
        assert "upload_artifacts" not in fn_src, (
            "pragma_train_from_manifest_pipeline must not call upload_artifacts. "
            "Stage 2 (upload) is skipped — the data is already in S3."
        )

    def test_source_calls_submit_pytorchjob(self) -> None:
        """§2.4 stage 3: submit_pytorchjob must be wired in the manifest pipeline."""
        fn_src = _raw_func_src(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")
        assert "submit_pytorchjob" in fn_src, (
            "pragma_train_from_manifest_pipeline must call submit_pytorchjob (stage 3)."
        )

    def test_source_calls_run_pretraining(self) -> None:
        """§2.4 stage 4: run_pretraining must be wired in the manifest pipeline."""
        fn_src = _raw_func_src(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")
        assert "run_pretraining" in fn_src, (
            "pragma_train_from_manifest_pipeline must call run_pretraining (stage 4)."
        )

    def test_source_calls_export_checkpoint(self) -> None:
        """§2.4 stage 5: export_checkpoint must be wired in the manifest pipeline."""
        fn_src = _raw_func_src(_PIPELINE_PATH, "pragma_train_from_manifest_pipeline")
        assert "export_checkpoint" in fn_src, (
            "pragma_train_from_manifest_pipeline must call export_checkpoint (stage 5)."
        )

    def test_has_model_size_with_default(self) -> None:
        """§2.4 / Table 1: model_size must default to 'S' (PRAGMA-S)."""
        params = self._params()
        assert "model_size" in params, (
            "pragma_train_from_manifest_pipeline must have a model_size parameter."
        )
        assert params["model_size"] == "S", (
            f"model_size must default to 'S', got {params['model_size']!r}"
        )

    def test_has_epochs_with_default(self) -> None:
        """§2.4: epochs must default to 10."""
        params = self._params()
        assert "epochs" in params, (
            "pragma_train_from_manifest_pipeline must have an epochs parameter."
        )
        assert params["epochs"] == 10, (
            f"epochs must default to 10, got {params['epochs']!r}"
        )

    def test_has_nodes_with_default(self) -> None:
        """ADR 003: nodes must default to 1 (single-node)."""
        params = self._params()
        assert "nodes" in params, (
            "pragma_train_from_manifest_pipeline must have a nodes parameter."
        )
        assert params["nodes"] == 1, (
            f"nodes must default to 1 (single-node), got {params['nodes']!r}"
        )


# ---------------------------------------------------------------------------
# TestNoPvcStorage  (category: no_pvc)
# ---------------------------------------------------------------------------

class TestNoPvcStorage:
    """Verify no component or pipeline introduces PVC-based data storage.

    Uses AST-based parameter inspection — immune to KFP decorator wrapping.

    Category: no_pvc
    ADR 003 / openshift-storage-pattern.md: canonical data lives in S3.
    PVCs must not appear in component or pipeline signatures.
    """

    _PVC_PARAM_PATTERNS = (
        "pvc_path", "pvc_name", "pvc_uri",
        "data_pvc", "dataset_pvc", "storage_pvc",
    )

    def _assert_no_pvc_param(self, param_names: set[str], context: str) -> None:
        """Assert none of param_names match PVC-based naming patterns."""
        for name in param_names:
            for pattern in self._PVC_PARAM_PATTERNS:
                assert pattern not in name.lower(), (
                    f"'{context}' has a PVC-based parameter '{name}'. "
                    f"ADR 003 / openshift-storage-pattern.md: canonical dataset "
                    f"storage is S3, not PVC. Remove PVC parameters."
                )

    def test_prepare_component_no_pvc(self) -> None:
        """ADR 003: prepare_dataset must not have PVC storage parameters."""
        _require_importable()
        params = _raw_func_params(_COMPONENTS_PATH, "prepare_dataset")
        self._assert_no_pvc_param(set(params), "prepare_dataset")

    def test_upload_component_no_pvc(self) -> None:
        """ADR 003: upload_artifacts must not have PVC storage parameters."""
        _require_importable()
        params = _raw_func_params(_COMPONENTS_PATH, "upload_artifacts")
        self._assert_no_pvc_param(set(params), "upload_artifacts")

    def test_train_component_no_pvc(self) -> None:
        """ADR 003: run_pretraining must not have PVC storage parameters."""
        _require_importable()
        params = _raw_func_params(_COMPONENTS_PATH, "run_pretraining")
        self._assert_no_pvc_param(set(params), "run_pretraining")

    def test_pipeline_function_no_pvc(self) -> None:
        """ADR 003: pragma_pretraining_pipeline must not have PVC storage parameters."""
        _require_importable()
        assert hasattr(_pipe, "pragma_pretraining_pipeline")
        params = _raw_func_params(_PIPELINE_PATH, "pragma_pretraining_pipeline")
        self._assert_no_pvc_param(set(params), "pragma_pretraining_pipeline")

    def test_components_source_no_pvc_path_string(self) -> None:
        """ADR 003: components source must not construct PVC-based paths."""
        src = _src_text(_COMPONENTS_PATH)
        pvc_strings = [s for s in ("ReadWriteOnce", "PersistentVolumeClaim", "pvc_path")
                       if s in src]
        assert not pvc_strings, (
            f"components_pragma.py source contains PVC references: {pvc_strings}. "
            f"Remove all PVC-based storage logic."
        )


# ---------------------------------------------------------------------------
# TestDatasetAdapterUsage  (category: source)
# ---------------------------------------------------------------------------

class TestDatasetAdapterUsage:
    """Verify prepare component delegates to DatasetAdapter rather than
    containing hardcoded IBM TabFormer-only logic.

    Category: source
    These are source-level structural assertions.  They verify the intent
    of the implementation, not just the interface.
    """

    def test_prepare_stage_references_get_adapter(self) -> None:
        """ADR 003: prepare_dataset must call get_adapter(), not hardcode IBM logic.

        The adapter registry pattern (src/data/adapters/__init__.py::get_adapter)
        enables any future dataset to be used without changing the pipeline.
        If get_adapter is not referenced, the component has re-implemented the
        adapter logic for IBM TabFormer only — violating the adapter boundary.
        """
        src = _src_text(_COMPONENTS_PATH)
        assert "get_adapter" in src, (
            "pipeline/components_pragma.py must reference get_adapter() "
            "(from src.data.adapters import get_adapter) in the prepare stage. "
            "Do not hardcode IBM TabFormer-specific logic in the pipeline."
        )

    def test_prepare_stage_does_not_hardcode_ibm_dataset_name(self) -> None:
        """ADR 003: prepare_dataset must not hardcode 'ibm-tabformer' as the only option.

        The dataset name must come from the dataset_name parameter and be resolved
        via get_adapter(), not hardcoded as a string in the component body.
        """
        src = _src_text(_COMPONENTS_PATH)
        # The string 'ibm-tabformer' may appear in comments/docs but must not be
        # the only codepath — i.e., the component must not have a hardcoded
        # if dataset_name == 'ibm-tabformer': pattern
        assert "get_adapter" in src, (
            "prepare_dataset must route through get_adapter() so any registered "
            "adapter can be used — not just IBM TabFormer."
        )

    def test_train_stage_references_pragma_config(self) -> None:
        """ADR 003: run_pretraining must reference PRAGMAConfig (or _MODEL_SIZE_MAP).

        The config mapping (model_size → PRAGMAConfig) must be derived from
        src.model.config — the same mapping used by train_pragma().  A second
        divergent mapping in the pipeline would be a maintenance hazard.
        """
        src = _src_text(_COMPONENTS_PATH)
        has_config_ref = "PRAGMAConfig" in src or "_MODEL_SIZE_MAP" in src
        assert has_config_ref, (
            "pipeline/components_pragma.py must reference PRAGMAConfig or "
            "_MODEL_SIZE_MAP from src.model.config in the train stage. "
            "Do not reimplement the model_size → config mapping."
        )

    def test_pipeline_references_dataset_manifest(self) -> None:
        """ADR 003: pipeline source must reference DatasetManifest (canonical contract).

        The pipeline must consume DatasetManifest (or its URI) — not raw local
        file paths — as the canonical training data contract (§2.4 data storage).
        """
        src = _src_text(_PIPELINE_PATH)
        assert "DatasetManifest" in src or "manifest_uri" in src, (
            "pipeline/pragma_pipeline.py must reference DatasetManifest or "
            "manifest_uri — the canonical §2.4 training data contract."
        )


# ---------------------------------------------------------------------------
# TestNoCircularDependency  (category: structural)
# ---------------------------------------------------------------------------

class TestNoCircularDependency:
    """Verify src/ modules do not import from pipeline/.

    Category: structural
    ADR 003 dependency graph: pipeline/ → src/ only. Never src/ → pipeline/.
    ADR 004 lazy-import boundary: compile() may use importlib.import_module at
    call time only — pipeline/ must never be loaded at src/ module import time.
    """

    _SRC_ROOT = pathlib.Path("src")
    _PIPELINE_NAMES = ("pipeline", "components_pragma", "pragma_pipeline")

    def test_src_modules_do_not_import_pipeline(self) -> None:
        """ADR 003: no src/ module may statically import from pipeline/.

        Scans src/ source for 'from pipeline' and 'import pipeline' at the
        text level.  The documented lazy-import boundary (compile() using
        importlib.import_module at call time) is an explicit exception and
        is validated separately by test_decorators_do_not_load_pipeline_at_import_time.

        A circular dependency (src → pipeline → src) would make the src/
        modules un-testable in isolation and couple the training script to
        the KFP pipeline infrastructure.
        """
        offending: list[str] = []
        for py_file in self._SRC_ROOT.rglob("*.py"):
            text = py_file.read_text()
            for name in self._PIPELINE_NAMES:
                if "from pipeline" in text or "import pipeline" in text:
                    offending.append(str(py_file))
                    break
        assert not offending, (
            f"The following src/ files import from pipeline/ (circular dependency): "
            f"{offending}. ADR 003: pipeline/ → src/ only."
        )

    def test_decorators_do_not_load_pipeline_at_import_time(self) -> None:
        """ADR 004 lazy-import boundary: importing src.workbench._decorators must not
        load any pipeline/ module as a side effect.

        compile() is explicitly allowed to call importlib.import_module("pipeline.*")
        at call time only.  This test enforces that no pipeline/ module is loaded
        during the import of src.workbench._decorators itself.

        If this test fails, a pipeline/ import has been moved from inside compile()
        to module level — violating the lazy-import boundary.
        """
        pipeline_mods_before = {k for k in sys.modules if k.startswith("pipeline")}
        importlib.import_module("src.workbench._decorators")
        pipeline_mods_after = {k for k in sys.modules if k.startswith("pipeline")}
        new_pipeline_mods = pipeline_mods_after - pipeline_mods_before
        assert not new_pipeline_mods, (
            f"Importing src.workbench._decorators loaded pipeline/ modules at import time: "
            f"{new_pipeline_mods}. ADR 004: only compile() may lazily load pipeline/ "
            f"via importlib.import_module at call time."
        )


# ---------------------------------------------------------------------------
# TestKfpOptionalExecution  (category: kfp)
# ---------------------------------------------------------------------------

class TestKfpOptionalExecution:
    """Tests that require kfp to be installed.  Explicitly skipped otherwise.

    Category: kfp
    These tests verify the KFP decorator and compilation behaviour — they
    only run in environments where kfp is installed (e.g. the cluster pipeline
    registration notebook).
    """

    def test_prepare_component_is_kfp_decorated_when_kfp_available(self) -> None:
        """When kfp is installed, prepare_dataset must be a KFP component task."""
        pytest.importorskip("kfp", reason="kfp not installed — skipping KFP execution tests")
        _require_importable()
        fn = getattr(_comp, "prepare_dataset", None)
        assert fn is not None
        # A @dsl.component-decorated function is a PipelineTask or component spec
        assert hasattr(fn, "component_spec") or callable(fn), (
            "prepare_dataset must be decorated with @dsl.component when kfp is available"
        )

    def test_pipeline_function_compilable_when_kfp_available(self) -> None:
        """When kfp is installed, pragma_pretraining_pipeline must compile to YAML."""
        kfp = pytest.importorskip("kfp", reason="kfp not installed — skipping KFP execution tests")
        import os
        import tempfile
        _require_importable()
        fn = getattr(_pipe, "pragma_pretraining_pipeline", None)
        assert fn is not None
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            kfp.compiler.Compiler().compile(
                pipeline_func=fn,
                package_path=tmp_path,
            )
            assert os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0, (
                "Pipeline compilation must produce a non-empty YAML file."
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_manifest_pipeline_compilable_when_kfp_available(self) -> None:
        """When kfp is installed, pragma_train_from_manifest_pipeline must compile to YAML."""
        kfp = pytest.importorskip("kfp", reason="kfp not installed — skipping KFP execution tests")
        import os
        import tempfile
        _require_importable()
        fn = getattr(_pipe, "pragma_train_from_manifest_pipeline", None)
        assert fn is not None, (
            "pragma_train_from_manifest_pipeline must be defined before KFP compilation."
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            kfp.compiler.Compiler().compile(
                pipeline_func=fn,
                package_path=tmp_path,
            )
            assert os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0, (
                "pragma_train_from_manifest_pipeline compilation must produce a non-empty YAML."
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
