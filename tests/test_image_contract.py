"""Image contract tests for the PRAGMA two-image model.

Validates that the workbench image and training/component image have the
correct dependency split, build instructions, and guard behaviour.

See docs/openshift-image-contract.md for the authoritative contract.

Three categories:

  TestWorkbenchDependencies — pyproject.toml and notebook requirements.txt
      must declare kfp and kfp-kubernetes in the workbench extras/image only.

  TestTrainingImageContract — openshift/training/Dockerfile.training must
      bake src/ at build time and must not use runtime git clone.

  TestKfpKubernetesGuard — src/workbench/_submit._require_kfp_kubernetes()
      must raise a friendly ImportError (with install hint) when
      kfp_kubernetes is absent, and must never be called at module import
      time (only when secret injection is explicitly requested).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Paths resolved relative to the repo root (pyproject.toml testpaths = ["tests"])
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_NOTEBOOK_REQUIREMENTS = _REPO_ROOT / "openshift" / "notebook-image" / "requirements.txt"
_DOCKERFILE_TRAINING = _REPO_ROOT / "openshift" / "training" / "Dockerfile.training"
_SUBMIT_PY = _REPO_ROOT / "src" / "workbench" / "_submit.py"


# ---------------------------------------------------------------------------
# TestWorkbenchDependencies
# ---------------------------------------------------------------------------


class TestWorkbenchDependencies:
    """workbench extras / notebook image requirements include kfp-kubernetes.

    Category: image-contract / dependency-declaration
    """

    def test_pyproject_workbench_extras_includes_kfp(self) -> None:
        """pyproject.toml [project.optional-dependencies].workbench must include kfp>=2."""
        text = _PYPROJECT.read_text()
        assert "workbench" in text, "pyproject.toml must have [project.optional-dependencies]"
        workbench_line = next(
            (ln for ln in text.splitlines() if ln.strip().startswith("workbench")), None
        )
        assert workbench_line is not None, (
            "pyproject.toml must define workbench extras "
            "(line starting with 'workbench = ...')"
        )
        assert "kfp" in workbench_line, (
            f"pyproject.toml workbench extras must include kfp. "
            f"Got: {workbench_line!r}"
        )

    def test_pyproject_workbench_extras_includes_kfp_kubernetes(self) -> None:
        """pyproject.toml workbench extras must include kfp-kubernetes.

        kfp-kubernetes is needed in the compile/workbench environment for
        Kubernetes-native pipeline features (secret injection, PVC mounting).
        It is NOT a core dependency — only in [project.optional-dependencies].workbench.
        See docs/openshift-image-contract.md.
        """
        text = _PYPROJECT.read_text()
        workbench_line = next(
            (ln for ln in text.splitlines() if ln.strip().startswith("workbench")), None
        )
        assert workbench_line is not None
        assert "kfp-kubernetes" in workbench_line, (
            f"pyproject.toml workbench extras must include kfp-kubernetes. "
            f"Got: {workbench_line!r}. "
            "kfp-kubernetes belongs in the workbench/compile environment, "
            "not in core dependencies. See docs/openshift-image-contract.md."
        )

    def test_pyproject_core_dependencies_do_not_include_kfp(self) -> None:
        """kfp must NOT appear in [project.dependencies] (core deps).

        kfp and kfp-kubernetes are workbench-only. Core training and the
        workbench API must work without them installed.
        """
        text = _PYPROJECT.read_text()
        # Find the [project.dependencies] section (if it exists)
        in_deps = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[project.optional-dependencies]"):
                break  # optional-dependencies section starts — stop checking
            if stripped.startswith("[project.dependencies]"):
                in_deps = True
                continue
            if in_deps and "kfp" in stripped and not stripped.startswith("#"):
                pytest.fail(
                    f"kfp must not appear in [project.dependencies]. "
                    f"Found: {line!r}. "
                    "kfp belongs only in [project.optional-dependencies].workbench."
                )

    def test_notebook_requirements_includes_kfp(self) -> None:
        """openshift/notebook-image/requirements.txt must include kfp."""
        text = _NOTEBOOK_REQUIREMENTS.read_text()
        assert any(
            line.strip().startswith("kfp") and not line.strip().startswith("kfp-kubernetes")
            for line in text.splitlines()
        ), (
            "openshift/notebook-image/requirements.txt must include kfp. "
            "The workbench image is the compile environment for KFP pipelines."
        )

    def test_notebook_requirements_includes_kfp_kubernetes(self) -> None:
        """openshift/notebook-image/requirements.txt must include kfp-kubernetes.

        The workbench image is where pipelines are compiled. kfp-kubernetes
        is needed to annotate component pods with secrets, PVCs, etc.
        """
        text = _NOTEBOOK_REQUIREMENTS.read_text()
        assert any(
            line.strip().startswith("kfp-kubernetes")
            for line in text.splitlines()
        ), (
            "openshift/notebook-image/requirements.txt must include kfp-kubernetes. "
            "The workbench image must have kfp-kubernetes for secret injection. "
            "See docs/openshift-image-contract.md."
        )


# ---------------------------------------------------------------------------
# TestTrainingImageContract
# ---------------------------------------------------------------------------


class TestTrainingImageContract:
    """Training/component image Dockerfile must satisfy the image contract.

    Category: image-contract / build-instructions
    """

    def test_dockerfile_training_copies_src(self) -> None:
        """Dockerfile.training must COPY src/ into the image (bake in source)."""
        text = _DOCKERFILE_TRAINING.read_text()
        assert "COPY" in text and "src/" in text, (
            "Dockerfile.training must COPY src/ into the image. "
            "KFP component pods import from src.* — source must be baked in. "
            "See docs/openshift-image-contract.md."
        )
        copy_lines = [ln for ln in text.splitlines() if "COPY" in ln and "src/" in ln]
        assert copy_lines, (
            "No COPY src/ line found in Dockerfile.training. "
            "Without src/ baked in, component pods cannot import PRAGMA code."
        )

    def test_dockerfile_training_has_no_runtime_git_clone(self) -> None:
        """Dockerfile.training must NOT use runtime git clone.

        Source code is baked in at build time via COPY. Runtime git clone is
        not the default because:
          - Requires network egress from component pods.
          - Couples execution to GitHub availability.
          - git push silently changes running pipeline behaviour.
          - Build step provides image digest audit trail.
        See docs/openshift-image-contract.md.
        """
        text = _DOCKERFILE_TRAINING.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # skip comments
            if "git clone" in stripped:
                pytest.fail(
                    f"Dockerfile.training must not use runtime git clone. "
                    f"Found: {line!r}. "
                    "Source is baked in at build time via COPY src/. "
                    "See docs/openshift-image-contract.md."
                )

    def test_dockerfile_training_extends_workbench_image(self) -> None:
        """Dockerfile.training must extend the workbench image (FROM pragma-encoder-workbench).

        The training image inherits all Python deps (kfp, kfp-kubernetes,
        PyTorch, etc.) from the workbench image, then adds PRAGMA source.
        """
        text = _DOCKERFILE_TRAINING.read_text()
        from_lines = [
            ln for ln in text.splitlines()
            if ln.strip().startswith("FROM") and not ln.strip().startswith("#")
        ]
        assert from_lines, "Dockerfile.training must have a FROM directive."
        assert any("pragma-encoder-workbench" in ln for ln in from_lines), (
            f"Dockerfile.training must extend pragma-encoder-workbench. "
            f"FROM lines found: {from_lines}. "
            "The training image inherits workbench deps (kfp, kfp-kubernetes, …)."
        )

    def test_dockerfile_training_does_not_install_kfp_kubernetes_separately(self) -> None:
        """Dockerfile.training must not pip install kfp-kubernetes.

        kfp-kubernetes is inherited from the workbench base image.
        Installing it again in the training image would be redundant and
        could pin a different version.
        """
        text = _DOCKERFILE_TRAINING.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pip install" in stripped and "kfp-kubernetes" in stripped:
                pytest.fail(
                    f"Dockerfile.training must not pip install kfp-kubernetes separately. "
                    f"Found: {line!r}. "
                    "kfp-kubernetes is inherited from pragma-encoder-workbench base image."
                )


# ---------------------------------------------------------------------------
# TestKfpKubernetesGuard
# ---------------------------------------------------------------------------


class TestKfpKubernetesGuard:
    """_require_kfp_kubernetes() guard: lazy, friendly, only when enabled.

    Category: image-contract / guard-behaviour
    """

    def test_guard_not_imported_at_module_level_in_submit(self) -> None:
        """_submit.py must not import kfp_kubernetes at module level.

        The guard is ONLY called when secret injection is explicitly enabled.
        Any top-level import would fail in environments without kfp-kubernetes
        (e.g. local dev, CI without workbench extras installed).
        """
        text = _SUBMIT_PY.read_text()
        for line in text.splitlines():
            # Skip indented lines (they're inside function/class bodies)
            if line.startswith(" ") or line.startswith("\t"):
                continue
            stripped = line.strip()
            # Skip comments and blank lines
            if not stripped or stripped.startswith("#"):
                continue
            # Only flag actual import statements at module level
            is_import = stripped.startswith("import ") or stripped.startswith("from ")
            if is_import and "kfp_kubernetes" in stripped:
                pytest.fail(
                    f"kfp_kubernetes must not be imported at module level in _submit.py. "
                    f"Found top-level import: {line!r}. "
                    "Only call _require_kfp_kubernetes() inside functions that need it."
                )

    def test_guard_raises_importerror_when_kfp_kubernetes_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_require_kfp_kubernetes() raises ImportError when kfp_kubernetes is not installed."""
        # Block kfp_kubernetes import by placing a None sentinel in sys.modules.
        # Python's import system raises ImportError when sys.modules[name] is None.
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        from src.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        with pytest.raises(ImportError):
            _require_kfp_kubernetes("test feature")

    def test_guard_error_message_contains_feature_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ImportError message must include the feature name passed to the guard."""
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        from src.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        with pytest.raises(ImportError) as exc_info:
            _require_kfp_kubernetes("my-special-feature")

        assert "my-special-feature" in str(exc_info.value), (
            "Guard error message must include the feature name so callers see "
            f"which feature requires kfp-kubernetes. Got: {exc_info.value!r}"
        )

    def test_guard_error_message_contains_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ImportError message must include a pip install hint."""
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        from src.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        with pytest.raises(ImportError) as exc_info:
            _require_kfp_kubernetes()

        msg = str(exc_info.value)
        assert "pip install" in msg, (
            f"Guard error message must include pip install instructions. Got: {msg!r}"
        )
        assert "kfp-kubernetes" in msg, (
            f"Guard error message must name kfp-kubernetes. Got: {msg!r}"
        )

    def test_guard_returns_module_when_kfp_kubernetes_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When kfp_kubernetes is importable, the guard returns the module object."""
        # Create a lightweight stub module to simulate kfp_kubernetes being installed.
        import types  # noqa: PLC0415

        stub = types.ModuleType("kfp_kubernetes")
        stub.__version__ = "1.2.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", stub)

        from src.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        result = _require_kfp_kubernetes("test feature")
        assert result is stub, (
            "Guard must return the kfp_kubernetes module when present. "
            f"Got: {result!r}"
        )
