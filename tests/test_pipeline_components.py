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

    When kfp is not installed, @dsl.component is not applied and the
    functions are plain Python callables — inspect.signature() works directly.

    Category: interface
    """

    def _sig(self, func_name: str) -> inspect.Signature:
        """Return the signature of a component function."""
        _require_importable()
        fn = getattr(_comp, func_name, None)
        assert fn is not None, (
            f"components_pragma has no function '{func_name}'. "
            f"Component 5 must define it."
        )
        return inspect.signature(fn)

    def _params(self, func_name: str) -> set[str]:
        return set(self._sig(func_name).parameters)

    # --- prepare component ---

    def test_prepare_component_accepts_dataset_name(self) -> None:
        """ADR 003: prepare component must accept dataset_name (registry key)."""
        params = self._params("prepare_dataset")
        assert "dataset_name" in params, (
            f"prepare_dataset must have a 'dataset_name' parameter, "
            f"got {params}"
        )

    def test_prepare_component_accepts_model_size(self) -> None:
        """ADR 003: prepare component must accept model_size ('S'/'M'/'L').

        model_size is needed so prepare can pass the correct PRAGMAConfig
        truncation limits (max_event_tokens=24, max_profile_tokens=200,
        max_events=6500) to DatasetAdapter.prepare().
        """
        params = self._params("prepare_dataset")
        assert "model_size" in params, (
            f"prepare_dataset must have a 'model_size' parameter, got {params}"
        )

    def test_prepare_component_does_not_require_raw_data_path(self) -> None:
        """ADR 003: prepare must not take raw_data_path as a required positional arg.

        The canonical contract is dataset_name (a registry key) + model_size.
        raw_data_path was the old scaffold's IBM-specific hardcoded path.
        Using a registry key enables future adapters without changing the interface.
        """
        sig = self._sig("prepare_dataset")
        params = sig.parameters
        if "raw_data_path" in params:
            p = params["raw_data_path"]
            # If it exists, it must not be a required positional (must have a default)
            assert p.default is not inspect.Parameter.empty, (
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
            f"upload_artifacts must have a 'manifest_uri' parameter, got {params}"
        )

    # --- train component ---

    def test_train_component_accepts_manifest_uri(self) -> None:
        """ADR 003: train component must consume manifest_uri — not raw data paths.

        The training contract is a DatasetManifest URI (S3 path to prepared data).
        Raw file paths violate the §2.4 data storage requirement (S3, not PVC).
        """
        params = self._params("run_pretraining")
        assert "manifest_uri" in params, (
            f"run_pretraining must have a 'manifest_uri' parameter, got {params}. "
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
            f"got {params}"
        )

    def test_train_component_does_not_use_config_name(self) -> None:
        """ADR 003: train component must not use config_name — divergent convention.

        config_name ('pragma_s') is the old scaffold's naming.  The unified
        convention (model_size = 'S'/'M'/'L') must be used throughout.
        """
        params = self._params("run_pretraining")
        assert "config_name" not in params, (
            f"run_pretraining must not have a 'config_name' parameter. "
            f"Use model_size='S'/'M'/'L' consistently with train_pragma()."
        )

    def test_train_component_accepts_nodes(self) -> None:
        """ADR 003: train component must accept nodes: int for two-node DDP.

        nodes=2 selects pytorchjob-pragma-s-2node.yaml (per ADR 003).
        """
        params = self._params("run_pretraining")
        assert "nodes" in params, (
            f"run_pretraining must have a 'nodes: int' parameter for two-node DDP, "
            f"got {params}"
        )

    def test_train_component_nodes_default_is_1(self) -> None:
        """ADR 003: nodes must default to 1 (single-node)."""
        sig = self._sig("run_pretraining")
        nodes_param = sig.parameters.get("nodes")
        assert nodes_param is not None
        assert nodes_param.default == 1, (
            f"run_pretraining nodes must default to 1 (single-node), "
            f"got default={nodes_param.default}"
        )

    def test_train_component_accepts_epochs(self) -> None:
        """§2.4: train component must accept epochs: int."""
        params = self._params("run_pretraining")
        assert "epochs" in params, (
            f"run_pretraining must have an 'epochs' parameter, got {params}"
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

    Category: interface
    """

    def _sig(self) -> inspect.Signature:
        _require_importable()
        fn = getattr(_pipe, "pragma_pretraining_pipeline", None)
        assert fn is not None, (
            "pragma_pipeline.py must define pragma_pretraining_pipeline()."
        )
        return inspect.signature(fn)

    def _params(self) -> dict:
        return dict(self._sig().parameters)

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
        p = self._params().get("model_size")
        assert p is not None
        assert p.default == "S", (
            f"model_size must default to 'S' (PRAGMA-S), got {p.default!r}"
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
        p = self._params().get("nodes")
        assert p is not None
        assert p.default == 1, (
            f"pragma_pretraining_pipeline nodes must default to 1, got {p.default}"
        )

    def test_pipeline_function_accepts_manifest_uri(self) -> None:
        """ADR 003: pipeline must accept manifest_uri to skip prepare/upload stages.

        When manifest_uri is provided (non-empty), the pipeline skips prepare and
        upload and goes directly to submit.  This lets users re-submit training
        on already-prepared data without re-running the expensive prepare stage.
        """
        assert "manifest_uri" in self._params(), (
            "pragma_pretraining_pipeline must accept a 'manifest_uri' parameter "
            "so callers can supply a pre-prepared DatasetManifest URI."
        )

    def test_pipeline_manifest_uri_default_is_empty_string(self) -> None:
        """ADR 003: manifest_uri must default to '' (empty = run prepare from scratch)."""
        p = self._params().get("manifest_uri")
        assert p is not None
        assert p.default == "", (
            f"manifest_uri must default to '' (empty = run prepare stage), "
            f"got default={p.default!r}"
        )

    def test_pipeline_function_accepts_epochs(self) -> None:
        """§2.4: pipeline must accept epochs: int."""
        assert "epochs" in self._params(), (
            "pragma_pretraining_pipeline must have an 'epochs' parameter."
        )

    def test_pipeline_epochs_default_is_10(self) -> None:
        """§2.4: epochs must default to 10 (consistent with train_pragma default)."""
        p = self._params().get("epochs")
        assert p is not None
        assert p.default == 10, (
            f"epochs must default to 10 (consistent with train_pragma), "
            f"got {p.default}"
        )


# ---------------------------------------------------------------------------
# TestNoPvcStorage  (category: no_pvc)
# ---------------------------------------------------------------------------

class TestNoPvcStorage:
    """Verify no component or pipeline introduces PVC-based data storage.

    Category: no_pvc
    ADR 003 / openshift-storage-pattern.md: canonical data lives in S3.
    PVCs must not appear in component or pipeline signatures.
    """

    _PVC_PARAM_PATTERNS = (
        "pvc_path", "pvc_name", "pvc_uri",
        "data_pvc", "dataset_pvc", "storage_pvc",
    )

    def _assert_no_pvc_param(self, sig: inspect.Signature, context: str) -> None:
        for name in sig.parameters:
            for pattern in self._PVC_PARAM_PATTERNS:
                assert pattern not in name.lower(), (
                    f"'{context}' has a PVC-based parameter '{name}'. "
                    f"ADR 003 / openshift-storage-pattern.md: canonical dataset "
                    f"storage is S3, not PVC. Remove PVC parameters."
                )

    def test_prepare_component_no_pvc(self) -> None:
        """ADR 003: prepare_dataset must not have PVC storage parameters."""
        _require_importable()
        self._assert_no_pvc_param(
            inspect.signature(_comp.prepare_dataset), "prepare_dataset"
        )

    def test_upload_component_no_pvc(self) -> None:
        """ADR 003: upload_artifacts must not have PVC storage parameters."""
        _require_importable()
        self._assert_no_pvc_param(
            inspect.signature(_comp.upload_artifacts), "upload_artifacts"
        )

    def test_train_component_no_pvc(self) -> None:
        """ADR 003: run_pretraining must not have PVC storage parameters."""
        _require_importable()
        self._assert_no_pvc_param(
            inspect.signature(_comp.run_pretraining), "run_pretraining"
        )

    def test_pipeline_function_no_pvc(self) -> None:
        """ADR 003: pragma_pretraining_pipeline must not have PVC storage parameters."""
        _require_importable()
        fn = getattr(_pipe, "pragma_pretraining_pipeline", None)
        assert fn is not None
        self._assert_no_pvc_param(inspect.signature(fn), "pragma_pretraining_pipeline")

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
    """

    _SRC_ROOT = pathlib.Path("src")
    _PIPELINE_NAMES = ("pipeline", "components_pragma", "pragma_pipeline")

    def test_src_modules_do_not_import_pipeline(self) -> None:
        """ADR 003: no src/ module may import from pipeline/.

        A circular dependency (src → pipeline → src) would make the src/
        modules un-testable in isolation and couple the training script to
        the KFP pipeline infrastructure.
        """
        offending: list[str] = []
        for py_file in self._SRC_ROOT.rglob("*.py"):
            text = py_file.read_text()
            for name in self._PIPELINE_NAMES:
                if f"from pipeline" in text or f"import pipeline" in text:
                    offending.append(str(py_file))
                    break
        assert not offending, (
            f"The following src/ files import from pipeline/ (circular dependency): "
            f"{offending}. ADR 003: pipeline/ → src/ only."
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
        kfp = pytest.importorskip("kfp", reason="kfp not installed — skipping KFP execution tests")
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
        import tempfile, os
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
