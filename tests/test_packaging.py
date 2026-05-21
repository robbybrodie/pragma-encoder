"""Packaging smoke tests — verify pyproject.toml build system configuration.

These tests confirm the project is correctly configured so that
``pip install -e .`` succeeds in a clean environment, and that the
installed wheel metadata contains the expected runtime dependencies.

Why this matters:
  ``build-backend = "setuptools.backends.legacy:build"`` is not a valid
  setuptools backend string. It causes ``BackendUnavailable`` when pip
  tries to resolve the build backend, which means ``pip install -e .``
  fails even though setuptools is installed.

  The correct backend is ``"setuptools.build_meta"``.

TestPackagingConfig:
  Fast static checks against pyproject.toml content.

TestSetupPy:
  Confirms setup.py is a minimal shim that does not duplicate metadata
  or override package discovery from pyproject.toml.

TestWheelMetadata:
  Verifies pyproject.toml lists core runtime dependencies and that the
  installed package metadata (importlib.metadata) exposes them as
  Requires-Dist entries.

TestCleanInstall:
  Creates a temporary virtualenv and runs ``pip install -e .`` inside it.
  This is a true clean-install test: no pre-existing packages, no PYTHONPATH.
  Slower than string-check tests (~5–30s with pip cache) but catches real
  install-time failures that string checks cannot detect.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import subprocess
import sys
import tempfile
import venv

import tomllib

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_SETUP_PY = pathlib.Path(__file__).parent.parent / "setup.py"


class TestPackagingConfig:
    """Verify pyproject.toml [build-system] configuration is correct."""

    def test_build_backend_is_setuptools_build_meta(self) -> None:
        """build-backend must be 'setuptools.build_meta'.

        'setuptools.backends.legacy:build' is not a valid setuptools
        backend string. It causes BackendUnavailable on pip install -e .

        Fix: change the build-backend line in [build-system] to:
            build-backend = "setuptools.build_meta"
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        backend = data["build-system"]["build-backend"]
        assert backend == "setuptools.build_meta", (
            f"pyproject.toml build-backend is {backend!r}. "
            "Must be 'setuptools.build_meta' for pip install -e . to work. "
            "Change [build-system] build-backend to 'setuptools.build_meta'."
        )

    def test_build_backend_module_is_importable(self) -> None:
        """setuptools.build_meta must be importable as a Python module.

        If the backend string were correct but setuptools were not installed
        (or were too old), this test would catch it before a user hits the
        error at install time.
        """
        try:
            importlib.import_module("setuptools.build_meta")
        except ImportError as exc:
            raise AssertionError(
                f"Cannot import setuptools.build_meta: {exc}. "
                "Ensure setuptools>=68 is installed (listed in pyproject.toml requires)."
            ) from exc

    def test_pyproject_requires_lists_setuptools(self) -> None:
        """[build-system].requires must include setuptools.

        setuptools.build_meta is part of setuptools — it must be listed
        as a build requirement so pip knows to install it in the build env.
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        requires = data["build-system"].get("requires", [])
        has_setuptools = any("setuptools" in r for r in requires)
        assert has_setuptools, (
            f"[build-system].requires does not include setuptools. "
            f"Got: {requires}. "
            "Add 'setuptools>=68' to the requires list."
        )


class TestSetupPy:
    """Verify setup.py is a minimal shim that delegates to pyproject.toml.

    setup.py must not contain find_packages() with stale src include-patterns,
    install_requires, or any other metadata that conflicts with pyproject.toml.
    All metadata is owned by pyproject.toml (PEP 621).
    """

    def test_setup_py_exists(self) -> None:
        """setup.py must exist for editable-install compatibility."""
        assert _SETUP_PY.exists(), (
            "setup.py is missing. A minimal setup.py shim is required for "
            "compatibility with tools that invoke setup.py directly."
        )

    def test_setup_py_does_not_contain_find_packages_with_src(self) -> None:
        """setup.py must not call find_packages(include=['src', 'src.*']).

        That pattern discovers src/ as a package, which is wrong after the
        namespace rename to pragma_encoder. Package discovery must live in
        pyproject.toml [tool.setuptools.packages.find].
        """
        text = _SETUP_PY.read_text()
        assert 'include=["src"' not in text and "include=['src'" not in text, (
            "setup.py still contains find_packages(include=['src', ...]). "
            "Remove all metadata from setup.py and replace with setup() only. "
            "Package discovery must be declared in pyproject.toml."
        )

    def test_setup_py_does_not_contain_install_requires(self) -> None:
        """setup.py must not declare install_requires.

        Runtime dependencies belong in pyproject.toml [project.dependencies].
        Duplicating them in setup.py causes them to be silently ignored
        (pyproject.toml takes precedence) and creates a maintenance hazard.
        """
        text = _SETUP_PY.read_text()
        assert "install_requires" not in text, (
            "setup.py still contains install_requires=[...]. "
            "Move all runtime dependencies to [project.dependencies] in "
            "pyproject.toml and remove install_requires from setup.py."
        )

    def test_setup_py_calls_setup_with_no_kwargs(self) -> None:
        """setup.py must contain only setup() with no metadata kwargs.

        A minimal shim calls setup() with no arguments so that setuptools
        reads all configuration from pyproject.toml.
        """
        text = _SETUP_PY.read_text()
        # Allow 'from setuptools import setup' and 'setup()' but not 'setup(name='
        assert "setup()" in text, (
            "setup.py does not contain a bare setup() call. "
            "The minimal shim must be: from setuptools import setup; setup()"
        )


class TestWheelMetadata:
    """Verify package metadata contains required runtime dependencies.

    Two complementary checks:
    1. Static: pyproject.toml [project.dependencies] lists each core dep.
    2. Dynamic: importlib.metadata.requires('pragma-encoder') exposes them
       after the editable install, i.e. the installed METADATA file is correct.
    """

    # The minimum set of core runtime dependencies that must appear in metadata.
    # These are the packages required for the three-encoder model to run.
    REQUIRED_DEPS = ["torch", "transformers", "numpy", "pandas", "peft", "scikit-learn"]

    @staticmethod
    def _dep_name(spec: str) -> str:
        """Extract the bare package name from a PEP 508 dependency specifier."""
        import re

        return re.split(r"[>=<!;\s\[]", spec.strip())[0].lower().replace("-", "_")

    def test_pyproject_lists_runtime_dependencies(self) -> None:
        """pyproject.toml [project.dependencies] must include core runtime deps.

        If this list is absent the wheel ships with no Requires-Dist entries
        and pip will not install torch, transformers, etc. when a user does
        ``pip install pragma-encoder``.
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        deps = data["project"].get("dependencies", [])
        assert deps, (
            "pyproject.toml [project.dependencies] is empty or missing. "
            "Add the core runtime dependencies (torch, transformers, etc.)."
        )
        dep_names = {self._dep_name(d) for d in deps}
        for pkg in self.REQUIRED_DEPS:
            assert pkg.lower().replace("-", "_") in dep_names, (
                f"pyproject.toml [project.dependencies] is missing: {pkg!r}. "
                f"Current deps: {sorted(dep_names)}"
            )

    def test_pyproject_package_discovery_uses_src_layout(self) -> None:
        """[tool.setuptools.packages.find].where must be ['src'].

        This is required for the src-layout: setuptools looks inside src/ and
        discovers pragma_encoder (not src itself).
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        find_cfg = (
            data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        )
        where = find_cfg.get("where", [])
        assert where == ["src"], (
            f"[tool.setuptools.packages.find].where = {where!r}. "
            "Must be ['src'] for the src-layout to work correctly."
        )

    def test_pyproject_no_src_in_package_name(self) -> None:
        """The installed package name must be 'pragma-encoder', not 'src'.

        [project.name] must be 'pragma-encoder' so the public import is
        pragma_encoder, not src.
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        name = data["project"]["name"]
        assert name == "pragma-encoder", (
            f"[project.name] = {name!r}. Must be 'pragma-encoder'."
        )
        assert name != "src", "[project.name] must not be 'src'."

    def test_installed_metadata_requires_dist(self) -> None:
        """Installed METADATA must list core runtime Requires-Dist entries.

        Reads importlib.metadata.requires('pragma-encoder') which parses the
        METADATA file written by pip during editable install. This is the same
        data that would appear in a built wheel's METADATA file.

        Failure here means the wheel ships without dependency declarations and
        users will hit ImportError at runtime after a fresh pip install.
        """
        try:
            reqs = importlib.metadata.requires("pragma-encoder") or []
        except importlib.metadata.PackageNotFoundError:
            raise AssertionError(
                "pragma-encoder is not installed in the current Python environment. "
                "Run 'pip install -e .' first, then re-run this test."
            )

        req_names = {self._dep_name(r) for r in reqs}
        for pkg in self.REQUIRED_DEPS:
            normalized = pkg.lower().replace("-", "_")
            assert normalized in req_names, (
                f"pragma-encoder installed METADATA is missing Requires-Dist: {pkg!r}. "
                f"Found Requires-Dist entries: {sorted(req_names)}. "
                "Check pyproject.toml [project.dependencies]."
            )

    def test_src_is_not_installed_as_package(self) -> None:
        """'src' must not be importable as an installed package.

        After the rename to pragma_encoder, 'src' should not appear as a
        top-level package in the installed distribution.
        """
        try:
            top_level = importlib.metadata.packages_distributions()
        except Exception:
            return  # importlib.metadata.packages_distributions not available; skip

        # 'src' must not map to pragma-encoder (or any distribution)
        # The installed top-level package is pragma_encoder, not src.
        assert "src" not in top_level or "pragma-encoder" not in top_level.get("src", []), (
            "'src' is registered as a top-level package of pragma-encoder. "
            "Check [tool.setuptools.packages.find] in pyproject.toml — "
            "where must be ['src'] not ['.'], and setup.py must not call "
            "find_packages(include=['src', 'src.*'])."
        )


class TestCleanInstall:
    """True clean-install verification using a temporary virtualenv.

    Creates a fresh venv, upgrades pip/setuptools/wheel, then runs
    ``pip install -e .`` against the project root.  Asserts exit 0.

    This test catches install-time failures that the string-check tests
    in TestPackagingConfig cannot detect — e.g. a broken [project] table,
    a missing package_dir, or a bad entry-point declaration.

    The test is slower than the other packaging tests (~5–30s with pip
    cache) but runs by default as part of ``pytest tests/``.

    On CI the packaging job runs this in isolation so a slow cold-start
    (no pip cache) does not block other jobs.
    """

    def test_clean_editable_install(self) -> None:
        """pip install -e . in a fresh venv must exit 0.

        Steps:
          1. Create a temporary virtualenv (stdlib venv, no pip cache sharing).
          2. Upgrade pip, setuptools, and wheel in the venv.
          3. Run ``pip install -e <project_root>`` inside the venv.
          4. Assert exit code 0.

        Failure means the package cannot be installed from a clean environment,
        which would block any user doing ``pip install -e .`` on a fresh clone.
        """
        project_root = pathlib.Path(__file__).parent.parent

        with tempfile.TemporaryDirectory() as tmp_dir:
            venv_dir = pathlib.Path(tmp_dir) / "venv"

            # Create venv with pip available.
            venv.create(str(venv_dir), with_pip=True)

            # Resolve venv executables (cross-platform: bin/ on Unix, Scripts/ on Windows).
            bin_dir = venv_dir / "bin" if (venv_dir / "bin").exists() else venv_dir / "Scripts"
            pip_exe = bin_dir / "pip"

            # Upgrade build tools so the venv has a modern pip that understands pyproject.toml.
            upgrade = subprocess.run(
                [str(pip_exe), "install", "-U", "pip", "setuptools", "wheel"],
                capture_output=True,
                text=True,
            )
            assert upgrade.returncode == 0, (
                "pip install -U pip setuptools wheel failed in fresh venv.\n"
                f"stdout:\n{upgrade.stdout[-2000:]}\n"
                f"stderr:\n{upgrade.stderr[-2000:]}"
            )

            # Install the project in editable mode.
            install = subprocess.run(
                [str(pip_exe), "install", "-e", str(project_root)],
                capture_output=True,
                text=True,
            )
            assert install.returncode == 0, (
                "pip install -e . failed in a fresh virtualenv.\n"
                "This means the project cannot be installed from a clean checkout.\n"
                f"stdout:\n{install.stdout[-2000:]}\n"
                f"stderr:\n{install.stderr[-2000:]}"
            )

            # Verify the installed package is importable using the venv Python.
            # The public Python package is 'pragma_encoder' (not 'src').
            python_exe = bin_dir / "python"
            import_check = subprocess.run(
                [str(python_exe), "-c", "import pragma_encoder"],
                capture_output=True,
                text=True,
            )
            assert import_check.returncode == 0, (
                "import pragma_encoder failed in the venv after pip install -e .\n"
                "The package was installed (exit 0) but 'pragma_encoder' is not "
                "importable. Check pyproject.toml [tool.setuptools.packages.find] — "
                "where must be ['src'] and src/pragma_encoder/__init__.py must exist.\n"
                f"stdout:\n{import_check.stdout}\n"
                f"stderr:\n{import_check.stderr}"
            )

        print(
            f"\n[TestCleanInstall] pip install -e . and import pragma_encoder succeeded "
            f"(Python {sys.version.split()[0]})."
        )
