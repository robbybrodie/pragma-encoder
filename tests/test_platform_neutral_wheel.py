"""Platform-neutrality boundary tests for the pragma_encoder core wheel.

The core ``pragma_encoder`` package must be platform-neutral:

- ``import pragma_encoder`` must not trigger kfp / kfp-kubernetes imports.
- Core subpackages (training, model, encoders, masking, data, etc.) must not
  contain kfp import statements — kfp is an optional dep only for tools/.
- Core modules must not hardcode Kubernetes pod paths
  (``/var/run/secrets/kubernetes.io/``).
- OpenShift-specific platform resource CRD names (ArgoCD, InferenceService,
  DataSciencePipelinesApplication, ServingRuntime, HardwareProfile) must not
  appear in core module source.

``pragma_encoder.workbench`` has been REMOVED from the core wheel.
The workbench helpers now live under ``tools/openshift_ai/workbench/``:

- ``tools/`` is NOT part of the installed wheel (setuptools only discovers src/).
- Importable only when the repo root is on ``sys.path`` (``PYTHONPATH=.``).
- ``import pragma_encoder.workbench`` must raise ``ModuleNotFoundError``.
- ``import tools.openshift_ai.workbench`` succeeds from the repo root.

TD-009 resolved: workbench moved out of the wheel entirely.
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
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _core_py_files() -> list[pathlib.Path]:
    """Return all .py source files in src/pragma_encoder/.

    Workbench is no longer present in src/ so no exclusion filter is needed.
    """
    return sorted(_SRC_DIR.rglob("*.py"))


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
                "kfp must be imported only inside tools.openshift_ai.workbench functions."
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
#    Static source scans: platform-specific code must not be in src/
# ===========================================================================


class TestCoreModulesNoPlatformCode:
    """Core module source must not contain platform-specific code.

    Scans all .py files in src/pragma_encoder/ for:
    - kfp import statements (caught by _has_kfp_import_statement)
    - Hardcoded Kubernetes pod paths (/var/run/secrets/kubernetes.io/)
    - OpenShift-specific CRD resource names that have no business in
      platform-neutral training/model/data code

    Platform terms that appear only in docstrings/comments (explaining
    deployment context) are acceptable and are not flagged by these tests.
    Workbench is no longer in src/ so no exclusion is needed.
    """

    def test_no_kfp_import_in_core_modules(self) -> None:
        """No core module in src/pragma_encoder/ may contain a kfp import statement.

        kfp imports belong only in tools.openshift_ai.workbench — which is NOT
        part of the installed wheel. A kfp import in src/ would make kfp a
        de-facto required dependency for the training image.
        """
        violations: list[str] = []
        for py_file in _core_py_files():
            hits = _has_kfp_import_statement(py_file)
            for hit in hits:
                rel = py_file.relative_to(_REPO_ROOT)
                violations.append(f"  {rel}: {hit.strip()!r}")

        assert not violations, (
            "These core modules contain kfp import statements. "
            "kfp must only be imported inside tools.openshift_ai.workbench functions, "
            "never at module level or in training/model/data code:\n"
            + "\n".join(violations)
        )

    def test_no_kubernetes_pod_paths_in_core_modules(self) -> None:
        """No core module may hardcode Kubernetes pod paths.

        The path /var/run/secrets/kubernetes.io/ is a Kubernetes pod-mount
        path for service account tokens and namespace files. It belongs only
        in tools.openshift_ai.workbench._submit (which reads the SA token for
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
            f"These core modules contain the Kubernetes pod path "
            f"'{k8s_path}'. This path belongs only in tools.openshift_ai.workbench. "
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
        read native AWS_* env vars from the process environment without naming
        the Secret that provides them.

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
            "language referencing native AWS_* env vars:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ===========================================================================
# 3. TestWorkbenchRemovedFromWheel
#    Confirm pragma_encoder.workbench is gone; tools.openshift_ai.workbench works
# ===========================================================================


class TestWorkbenchRemovedFromWheel:
    """Verify pragma_encoder.workbench has been removed from the installed wheel.

    TD-009 resolved: workbench helpers live in tools/openshift_ai/workbench/
    which is NOT packaged into the wheel (setuptools only finds src/).

    Tests confirm:
    - import pragma_encoder.workbench raises ModuleNotFoundError
    - pragma_encoder has no .workbench attribute
    - The built wheel zip (if present in dist/) contains no workbench directory
    - tools.openshift_ai.workbench is importable from the repo root
    - kfp is not in core [project.dependencies]
    """

    def test_import_pragma_encoder_workbench_raises_module_not_found(self) -> None:
        """import pragma_encoder.workbench must raise ModuleNotFoundError.

        The workbench subpackage has been moved to tools/openshift_ai/workbench/
        and is no longer part of the installed pragma_encoder distribution.
        Any code that still uses 'import pragma_encoder.workbench' is broken
        and must be updated to 'import tools.openshift_ai.workbench'.
        """
        with pytest.raises(ModuleNotFoundError):
            import pragma_encoder.workbench  # noqa: F401,PLC0415

    def test_pragma_encoder_has_no_workbench_attribute(self) -> None:
        """pragma_encoder must not expose a .workbench attribute.

        After the wheel is installed, pragma_encoder.workbench must not exist
        as a namespace package or attribute. This confirms the removal is clean.
        """
        import pragma_encoder  # noqa: PLC0415

        assert not hasattr(pragma_encoder, "workbench"), (
            "pragma_encoder must not have a 'workbench' attribute. "
            "The workbench subpackage has been moved to tools/openshift_ai/workbench/ "
            "and is no longer part of the installed distribution."
        )

    def test_wheel_does_not_contain_workbench_directory(self) -> None:
        """Built wheel in dist/ must not contain a pragma_encoder/workbench/ directory.

        Scans the most recent .whl file in dist/ (if present) and asserts that
        no path within it starts with pragma_encoder/workbench/. If no wheel
        exists the test is skipped (build first with 'pip wheel .').
        """
        import zipfile  # noqa: PLC0415

        dist_dir = _REPO_ROOT / "dist"
        wheels = sorted(dist_dir.glob("pragma_encoder-*.whl")) if dist_dir.exists() else []
        if not wheels:
            pytest.skip("No wheel found in dist/ — build with 'pip wheel .' first")

        latest_wheel = wheels[-1]
        with zipfile.ZipFile(latest_wheel) as zf:
            workbench_entries = [
                name for name in zf.namelist()
                if "pragma_encoder/workbench" in name
            ]

        assert not workbench_entries, (
            f"Wheel {latest_wheel.name} contains workbench entries: {workbench_entries}. "
            "pragma_encoder/workbench must not be packaged into the wheel. "
            "Run 'pip wheel .' again after removing src/pragma_encoder/workbench/."
        )

    def test_tools_openshift_ai_workbench_is_importable(self) -> None:
        """tools.openshift_ai.workbench must be importable from the repo root.

        When PYTHONPATH=. (the standard test invocation), tools/ is on sys.path
        and tools.openshift_ai.workbench must import successfully. This confirms
        the workbench code is still available — just not from the core wheel.
        """
        try:
            import tools.openshift_ai.workbench as wb  # noqa: PLC0415
        except ImportError as exc:
            pytest.fail(
                f"import tools.openshift_ai.workbench raised ImportError: {exc}. "
                "The workbench helpers must be importable from the repo root "
                "when PYTHONPATH=. is set. Check that tools/__init__.py and "
                "tools/openshift_ai/__init__.py exist."
            )
        assert wb is not None

    def test_kfp_is_not_in_core_wheel_dependencies(self) -> None:
        """kfp must not appear in core [project.dependencies] in pyproject.toml.

        Core [project.dependencies] are installed in all environments including
        the training image. kfp must never be a core dependency — it may only
        appear in optional extras or in tools/ requirements.
        """
        text = _PYPROJECT.read_text()
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
                "kfp must not be a core wheel dependency. "
                "Core dependencies are installed in the training image where kfp is absent."
            )
