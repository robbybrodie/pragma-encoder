"""Packaging smoke tests — verify pyproject.toml build system configuration.

These tests confirm the project is correctly configured so that
``pip install -e .`` succeeds in a clean environment.

Why this matters:
  ``build-backend = "setuptools.backends.legacy:build"`` is not a valid
  setuptools backend string. It causes ``BackendUnavailable`` when pip
  tries to resolve the build backend, which means ``pip install -e .``
  fails even though setuptools is installed.

  The correct backend is ``"setuptools.build_meta"``.

TestCleanInstall:
  Creates a temporary virtualenv and runs ``pip install -e .`` inside it.
  This is a true clean-install test: no pre-existing packages, no PYTHONPATH.
  Slower than string-check tests (~5–30s with pip cache) but catches real
  install-time failures that string checks cannot detect.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import venv

import tomllib

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"


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
