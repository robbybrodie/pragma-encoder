"""Packaging smoke tests — verify pyproject.toml build system configuration.

These tests confirm the project is correctly configured so that
``pip install -e .`` succeeds in a clean environment.

Why this matters:
  ``build-backend = "setuptools.backends.legacy:build"`` is not a valid
  setuptools backend string. It causes ``BackendUnavailable`` when pip
  tries to resolve the build backend, which means ``pip install -e .``
  fails even though setuptools is installed.

  The correct backend is ``"setuptools.build_meta"``.
"""

from __future__ import annotations

import importlib
import pathlib
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
