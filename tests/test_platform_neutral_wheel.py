"""Platform-neutrality boundary tests for the pragma_encoder core wheel.

The core ``pragma_encoder`` package must be platform-neutral:

- ``import pragma_encoder`` must not trigger kfp / kfp-kubernetes imports.
- Core subpackages (training, model, encoders, masking, data, etc.) must not
  contain kfp import statements — kfp belongs only in the ``[workbench]``
  optional extras.
- Non-workbench modules must not hardcode Kubernetes pod paths
  (``/var/run/secrets/kubernetes.io/``).
- OpenShift-specific platform resource CRD names (ArgoCD, InferenceService,
  DataSciencePipelinesApplication, ServingRuntime, HardwareProfile) must not
  appear in non-workbench core module source.

``pragma_encoder.workbench`` is the explicitly-labelled platform-aware exception:

- It is installed under ``[workbench]`` optional extras in pyproject.toml.
- It is NOT imported by ``import pragma_encoder`` (top-level ``__init__.py``
  does not reference workbench).
- It is used only inside OpenShift AI Workbench pods or KFP compile environments.
- Its ``__init__.py`` is annotated as a platform-aware optional subpackage.

TD-009: the eventual goal is to move the workbench subpackage to a separate
distribution so the core wheel has zero platform dependencies.
See docs/tech-debt.md — TD-009.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SRC_DIR = _REPO_ROOT / "src" / "pragma_encoder"
_WORKBENCH_DIR = _SRC_DIR / "workbench"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _core_py_files() -> list[pathlib.Path]:
    """Return all .py source files in src/pragma_encoder/ except workbench/."""
    return [
        f
        for f in sorted(_SRC_DIR.rglob("*.py"))
        if f.parent != _WORKBENCH_DIR
    ]


def _has_kfp_import_statement(py_file: pathlib.Path) -> list[str]:
    """Return lines containing kfp import statements (excluding comments).

    Detects lines where the stripped content starts with ``import kfp`` or
    ``from kfp`` — i.e. actual import statements, not comment mentions.
    Lines starting with ``#`` after stripping are excluded.
    """
    hits = []
    for line in py_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("import kfp") or stripped.startswith("from kfp"):
            hits.append(line)
    return hits


# ===========================================================================
# 1. TestTopLevelImportIsNeutral
#    Verify that importing core package modules does not pull in platform deps
# ===========================================================================


class TestTopLevelImportIsNeutral:
    """Core package imports must not trigger kfp / kfp-kubernetes loading.

    Structural tests: the top-level __init__.py and training __init__.py
    must not contain kfp import references. If ``import pragma_encoder``
    triggered kfp import, the training image (which does not install kfp)
    would fail at startup.
    """

    def test_toplevel_init_has_no_kfp_reference(self) -> None:
        """pragma_encoder/__init__.py must not reference kfp at all.

        The top-level package init is loaded on every ``import pragma_encoder``
        call. Any kfp reference here would force kfp as a non-optional dep.
        """
        init_path = _SRC_DIR / "__init__.py"
        text = init_path.read_text()
        assert "kfp" not in text, (
            f"pragma_encoder/__init__.py must not contain any 'kfp' reference. "
            f"kfp belongs in [workbench] optional extras only. "
            f"Found in: {init_path}"
        )

    def test_training_init_has_no_kfp_reference(self) -> None:
        """pragma_encoder/training/__init__.py must not reference kfp."""
        init_path = _SRC_DIR / "training" / "__init__.py"
        text = init_path.read_text()
        assert "kfp" not in text, (
            f"pragma_encoder/training/__init__.py must not contain any 'kfp' reference. "
            f"The training subpackage is loaded in the training image where kfp is absent. "
            f"Found in: {init_path}"
        )

    def test_import_pragma_encoder_does_not_require_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import pragma_encoder must succeed even when kfp is blocked.

        kfp is a [workbench] optional extra; it is not installed in the
        training image or in CI without the workbench extras. The top-level
        package import must succeed regardless.
        """
        import importlib  # noqa: PLC0415

        # Block kfp: Python treats sys.modules["kfp"] = None as a cached
        # import failure — any `import kfp` will raise ModuleNotFoundError.
        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)  # type: ignore[arg-type]

        # Force fresh import by removing cached pragma_encoder entries.
        for key in list(sys.modules):
            if key == "pragma_encoder" or key.startswith("pragma_encoder."):
                monkeypatch.delitem(sys.modules, key, raising=False)

        try:
            mod = importlib.import_module("pragma_encoder")
        except ImportError as exc:
            pytest.fail(
                f"import pragma_encoder raised ImportError when kfp is blocked: {exc}. "
                "The core pragma_encoder package must not import kfp at module level. "
                "kfp must be imported only inside pragma_encoder.workbench functions."
            )
        assert mod is not None

    def test_import_pragma_encoder_training_does_not_require_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import pragma_encoder.training must succeed when kfp is blocked.

        The training subpackage runs inside the training image where kfp is
        absent. Any kfp import at training module level would crash the image.
        """
        import importlib  # noqa: PLC0415

        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)  # type: ignore[arg-type]

        for key in list(sys.modules):
            if key == "pragma_encoder.training" or key.startswith("pragma_encoder.training."):
                monkeypatch.delitem(sys.modules, key, raising=False)

        try:
            mod = importlib.import_module("pragma_encoder.training")
        except ImportError as exc:
            pytest.fail(
                f"import pragma_encoder.training raised ImportError when kfp is blocked: {exc}. "
                "The training subpackage must not import kfp at module level."
            )
        assert mod is not None


# ===========================================================================
# 2. TestCoreModulesNoPlatformCode
#    Static source scans: platform-specific code must stay in workbench/
# ===========================================================================


class TestCoreModulesNoPlatformCode:
    """Core module source must not contain platform-specific code.

    Scans all .py files in src/pragma_encoder/ except workbench/ for:
    - kfp import statements (caught by _has_kfp_import_statement)
    - Hardcoded Kubernetes pod paths (/var/run/secrets/kubernetes.io/)
    - OpenShift-specific CRD resource names that have no business in
      platform-neutral training/model/data code

    Platform terms that appear only in docstrings/comments (explaining
    deployment context) are acceptable and are not flagged by these tests.
    """

    def test_no_kfp_import_in_core_modules(self) -> None:
        """No core module (outside workbench/) may contain a kfp import statement.

        kfp imports belong only in pragma_encoder.workbench — the explicitly
        labelled platform-aware optional subpackage. A kfp import anywhere
        else would make kfp a de-facto required dependency for the training image.
        """
        violations: list[str] = []
        for py_file in _core_py_files():
            hits = _has_kfp_import_statement(py_file)
            for hit in hits:
                rel = py_file.relative_to(_REPO_ROOT)
                violations.append(f"  {rel}: {hit.strip()!r}")

        assert not violations, (
            "These core (non-workbench) modules contain kfp import statements. "
            "kfp must only be imported inside pragma_encoder.workbench functions, "
            "never at module level or in training/model/data code:\n"
            + "\n".join(violations)
        )

    def test_no_kubernetes_pod_paths_in_core_modules(self) -> None:
        """No core module may hardcode Kubernetes pod paths.

        The path /var/run/secrets/kubernetes.io/ is a Kubernetes pod-mount
        path for service account tokens and namespace files. It belongs only
        in pragma_encoder.workbench._submit (which reads the SA token for
        DSPA authentication). Core training and model code must not assume
        they run inside a Kubernetes pod.
        """
        k8s_path = "/var/run/secrets/kubernetes.io"
        violations: list[str] = []
        for py_file in _core_py_files():
            text = py_file.read_text()
            if k8s_path in text:
                rel = py_file.relative_to(_REPO_ROOT)
                violations.append(str(rel))

        assert not violations, (
            f"These core (non-workbench) modules contain the Kubernetes pod path "
            f"'{k8s_path}'. This path belongs only in pragma_encoder.workbench. "
            f"Core training code must not assume it runs inside a Kubernetes pod:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_openshift_crd_resource_names_in_core_modules(self) -> None:
        """No core module may reference OpenShift CRD resource type names.

        The following terms represent OpenShift/RHOAI-specific Kubernetes CRD
        resource types. They have no business in platform-neutral core code:
          - ArgoCD
          - DataSciencePipelinesApplication
          - InferenceService
          - ServingRuntime
          - HardwareProfile

        Note: PyTorchJob and TrainJob are excluded from this scan because they
        appear in the pragma_encoder.training.train module docstring (explaining
        the distributed launch context). Docstring-only mentions are acceptable
        for context — they do not create platform dependencies.
        """
        forbidden_terms = [
            "ArgoCD",
            "DataSciencePipelinesApplication",
            "InferenceService",
            "ServingRuntime",
            "HardwareProfile",
        ]
        violations: list[str] = []
        for py_file in _core_py_files():
            text = py_file.read_text()
            for term in forbidden_terms:
                if term in text:
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(f"  {rel}: contains '{term}'")

        assert not violations, (
            "These core (non-workbench) modules reference OpenShift CRD resource "
            "names. These terms belong in openshift/, pipeline/, or docs/ only. "
            "Core training/model/data code must not reference platform resources:\n"
            + "\n".join(violations)
        )

    def test_no_pragma_workbench_env_in_core_modules(self) -> None:
        """No core module may reference the 'pragma-workbench-env' Secret name.

        'pragma-workbench-env' is a Kubernetes Secret name — a platform fixture
        in openshift/secrets/ and tests/openshift/fixtures/. Core modules must
        read MODEL_REGISTRY_* env vars without naming the Secret that provides them.

        This test is complementary to TestPlatformNameBoundary in
        test_checkpoint_resume.py, which scans the same directory. Both tests
        protect the boundary from different angles.
        """
        violations: list[str] = []
        for py_file in _core_py_files():
            if "pragma-workbench-env" in py_file.read_text():
                rel = py_file.relative_to(_REPO_ROOT)
                violations.append(str(rel))

        assert not violations, (
            "These core (non-workbench) modules reference 'pragma-workbench-env', "
            "which is a Kubernetes Secret name. Replace with platform-neutral "
            "language referencing MODEL_REGISTRY_* env vars:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ===========================================================================
# 3. TestWorkbenchIsLabeledPlatformAware
#    Confirm the workbench subpackage is properly isolated and documented
# ===========================================================================


class TestWorkbenchIsLabeledPlatformAware:
    """Verify the workbench subpackage is clearly labelled and isolated.

    The workbench subpackage is the approved platform-aware exception in the
    pragma_encoder distribution. These tests confirm the isolation is
    mechanical (not just convention):
    - The __init__.py docstring explicitly calls it out as platform-aware
    - kfp is in [workbench] optional extras, not core [project.dependencies]
    - The top-level __init__.py does not import workbench
    """

    def test_workbench_init_docstring_labels_it_platform_aware(self) -> None:
        """pragma_encoder.workbench.__init__.py must label itself platform-aware.

        The docstring must contain the phrase 'platform-aware' so that:
        - Readers scanning the package understand its special status
        - The TestWorkbenchIsLabeledPlatformAware test can verify the label
          mechanically (this very test)
        """
        import pragma_encoder.workbench as wb  # noqa: PLC0415

        docstring = wb.__doc__ or ""
        assert "platform-aware" in docstring.lower(), (
            "pragma_encoder.workbench.__init__.py must contain 'platform-aware' "
            "in its module docstring. This labels the subpackage as the approved "
            "platform-aware exception in the otherwise platform-neutral core wheel. "
            f"Current docstring (first 200 chars): {docstring[:200]!r}"
        )

    def test_workbench_init_docstring_references_td009(self) -> None:
        """pragma_encoder.workbench.__init__.py must reference TD-009.

        TD-009 is the tech debt entry documenting the eventual goal of moving
        workbench helpers to a separate distribution package. The reference in
        the module docstring links the code to the decision record.
        """
        import pragma_encoder.workbench as wb  # noqa: PLC0415

        docstring = wb.__doc__ or ""
        assert "TD-009" in docstring, (
            "pragma_encoder.workbench.__init__.py must reference 'TD-009' in its "
            "module docstring. TD-009 is the tech debt entry documenting the "
            "eventual separation of workbench helpers from the core wheel."
        )

    def test_kfp_is_in_optional_workbench_extras_not_core_deps(self) -> None:
        """kfp must be in [workbench] optional extras, not in core [project.dependencies].

        Core [project.dependencies] are installed in all environments including
        the training image. kfp must never be a core dependency — only an
        optional extra for workbench/compile environments.
        """
        text = _PYPROJECT.read_text()

        # Split into [project.dependencies] block and [project.optional-dependencies] block
        # Simple heuristic: find the core deps block before optional-dependencies
        lines = text.splitlines()

        in_core_deps = False
        core_dep_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == "[project.optional-dependencies]":
                break
            if stripped == "dependencies = [":
                in_core_deps = True
                continue
            if in_core_deps and stripped == "]":
                in_core_deps = False
                continue
            if in_core_deps:
                core_dep_lines.append(stripped)

        for dep_line in core_dep_lines:
            assert "kfp" not in dep_line, (
                f"kfp appears in core [project.dependencies]: {dep_line!r}. "
                "kfp must be in [project.optional-dependencies].workbench only. "
                "Core dependencies are installed in the training image where kfp is absent."
            )

    def test_toplevel_pragma_encoder_does_not_import_workbench(self) -> None:
        """The top-level pragma_encoder/__init__.py must not import workbench.

        If the top-level init imported workbench, then every ``import pragma_encoder``
        (including in the training image) would trigger kfp-dependent code paths.
        The workbench subpackage must be imported explicitly by consumers.

        Docstring mentions of 'workbench' are acceptable (they describe the
        package structure); only actual import statements are checked here.
        """
        init_path = _SRC_DIR / "__init__.py"
        text = init_path.read_text()
        assert "from pragma_encoder.workbench import" not in text, (
            "pragma_encoder/__init__.py must not contain "
            "'from pragma_encoder.workbench import ...'. "
            "The workbench subpackage must be imported explicitly by consumers."
        )
        assert "import pragma_encoder.workbench" not in text, (
            "pragma_encoder/__init__.py must not contain "
            "'import pragma_encoder.workbench'. "
            "The workbench subpackage must be imported explicitly by consumers."
        )

    def test_workbench_subpackage_is_importable(self) -> None:
        """pragma_encoder.workbench must be importable (basic smoke check).

        Confirms the subpackage is installed and its public surface is intact.
        kfp is imported lazily (inside make_kfp_client / compile), so this
        import succeeds even without kfp installed.
        """
        try:
            import pragma_encoder.workbench as wb  # noqa: PLC0415
        except ImportError as exc:
            pytest.fail(
                f"import pragma_encoder.workbench raised ImportError: {exc}. "
                "The workbench subpackage must be importable without kfp installed "
                "(kfp is imported lazily inside functions, not at module level)."
            )
        assert wb is not None

    def test_workbench_kfp_import_is_lazy(self) -> None:
        """kfp must not be imported at module load time in pragma_encoder.workbench.

        Importing pragma_encoder.workbench must succeed even when kfp is absent.
        All kfp usage must be inside function bodies (lazy imports), protected
        by try/except ImportError.
        """
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        import pytest  # noqa: PLC0415

        # We can only run this check if kfp is NOT currently importable.
        # If kfp is installed (e.g. in workbench extras CI), skip this variant.
        kfp_spec = importlib.util.find_spec("kfp")
        if kfp_spec is not None:
            pytest.skip("kfp is installed — lazy-import test is only meaningful without kfp")

        # kfp is not installed — importing workbench must still succeed.
        for key in list(sys.modules):
            if key.startswith("pragma_encoder.workbench"):
                pass  # don't remove — we just want to check it was loadable

        try:
            import pragma_encoder.workbench  # noqa: F401,PLC0415
        except ImportError as exc:
            pytest.fail(
                f"import pragma_encoder.workbench raised ImportError without kfp installed: "
                f"{exc}. kfp must be imported lazily inside functions, not at module level."
            )
