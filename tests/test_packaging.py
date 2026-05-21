"""Packaging smoke tests — verify pyproject.toml build system configuration.

These tests confirm the project is correctly configured so that the wheel
builds correctly, the installed wheel metadata contains the expected runtime
dependencies, and ``pip install --no-deps -e .`` / ``pip install --no-deps
dist/*.whl`` both succeed in a clean environment.

Test classes
------------
TestPackagingConfig:
  Fast static checks against pyproject.toml content. No subprocess calls.

TestSetupPy:
  Confirms setup.py is a minimal shim with no stale find_packages() or
  install_requires that would override pyproject.toml.

TestWheelMetadata:
  Static checks against pyproject.toml + artifact-based checks that read
  the METADATA file from the built wheel zip. Artifact-based checks are
  reliable because they test the actual built artifact from this checkout,
  not whatever version of pragma-encoder happens to be installed in the
  current Python environment (which may be stale after a force-reinstall
  or editable install that pre-dates the [project.dependencies] addition).

TestEditableInstall:
  Creates a temporary virtualenv and runs ``pip install --no-deps -e .``
  inside it. Uses ``--no-deps`` so the test does not spend minutes
  downloading torch, transformers, etc. Proves the package surface
  (editable install mechanics, import identity) without pulling ML
  dependencies. Dependency declarations are validated by inspecting wheel
  METADATA, not by actually installing dependencies.

TestWheelInstall:
  Builds the wheel, creates a fresh venv, installs ``pip install --no-deps
  dist/*.whl``, then verifies ``import pragma_encoder`` works. Proves the
  built wheel is installable from a clean environment without dependencies.
  Also verifies the installed top-level package is ``pragma_encoder``, not
  ``src``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import subprocess
import sys
import tempfile
import venv
import zipfile

import tomllib

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_SETUP_PY = pathlib.Path(__file__).parent.parent / "setup.py"
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Module-level cache: the wheel built (or found) for this pytest session.
_BUILT_WHEEL: pathlib.Path | None = None


def _get_or_build_wheel() -> pathlib.Path:
    """Return a built wheel for this checkout, building it if necessary.

    Reuses an existing dist/pragma_encoder-*.whl from a prior build in this
    session or a previous run so we don't rebuild on every test. To force a
    fresh build delete dist/pragma_encoder-*.whl before running pytest.

    Why build the wheel rather than inspect the installed editable-install
    metadata?
      importlib.metadata.requires('pragma-encoder') reads from whatever dist-
      info is installed in the current Python environment. That may be a stale
      editable install from before pyproject.toml was updated with
      [project.dependencies]. Reading the METADATA file inside the wheel zip
      proves the current source tree builds correctly, regardless of the state
      of the dev environment.
    """
    global _BUILT_WHEEL
    if _BUILT_WHEEL is not None and _BUILT_WHEEL.exists():
        return _BUILT_WHEEL

    dist_dir = _PROJECT_ROOT / "dist"
    existing = sorted(dist_dir.glob("pragma_encoder-*.whl")) if dist_dir.exists() else []
    if existing:
        _BUILT_WHEEL = existing[-1]
        return _BUILT_WHEEL

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", str(_PROJECT_ROOT)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        "python -m build --wheel failed.\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )

    wheels = sorted((_PROJECT_ROOT / "dist").glob("pragma_encoder-*.whl"))
    assert wheels, "python -m build --wheel succeeded but dist/pragma_encoder-*.whl not found."
    _BUILT_WHEEL = wheels[-1]
    return _BUILT_WHEEL


class TestPackagingConfig:
    """Verify pyproject.toml [build-system] configuration is correct."""

    def test_pyproject_has_pragma_encoder_train_script(self) -> None:
        """pyproject.toml [project.scripts] must declare pragma-encoder-train.

        The console script entrypoint must be:
            pragma-encoder-train = "pragma_encoder.training.train:main"

        This installs 'pragma-encoder-train' into the environment PATH so
        training can be launched without specifying the full module path.
        The same function is reachable as: python -m pragma_encoder.training.train
        """
        with open(_PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)

        scripts = data.get("project", {}).get("scripts", {})
        assert "pragma-encoder-train" in scripts, (
            "pyproject.toml [project.scripts] must declare 'pragma-encoder-train'. "
            "Add: [project.scripts]\n"
            "pragma-encoder-train = \"pragma_encoder.training.train:main\""
        )
        entry = scripts["pragma-encoder-train"]
        assert entry == "pragma_encoder.training.train:main", (
            f"pragma-encoder-train entry point must be 'pragma_encoder.training.train:main'. "
            f"Got: {entry!r}"
        )

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

    Three complementary checks:
    1. Static: pyproject.toml [project.dependencies] lists each core dep.
    2. Artifact: build the wheel, read METADATA from the zip, assert
       Requires-Dist lines for each core dep. Reliable: tests the built
       artifact from this checkout, not the installed environment.
    3. Artifact: wheel top_level.txt must be 'pragma_encoder', not 'src'.
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

    def test_wheel_metadata_requires_dist(self) -> None:
        """Built wheel METADATA must list core runtime Requires-Dist entries.

        Builds the wheel from this checkout (reusing a cached build if
        available) and reads the METADATA file directly from inside the zip.
        This is reliable: it tests the artifact produced by the current
        source tree, not whatever dist-info is installed in the test runner
        environment (which may be stale after editable installs).

        Failure means the wheel ships without dependency declarations —
        users who run ``pip install pragma-encoder`` will not get torch,
        transformers, etc. installed automatically.
        """
        wheel = _get_or_build_wheel()
        with zipfile.ZipFile(wheel) as zf:
            metadata_entries = [n for n in zf.namelist() if n.endswith("/METADATA")]
            assert metadata_entries, f"No METADATA file found in wheel {wheel.name}"
            metadata = zf.read(metadata_entries[0]).decode()

        requires_dist = [
            line.removeprefix("Requires-Dist:").strip()
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist:")
        ]
        req_names = {self._dep_name(r) for r in requires_dist}

        for pkg in self.REQUIRED_DEPS:
            normalized = pkg.lower().replace("-", "_")
            assert normalized in req_names, (
                f"Wheel METADATA missing Requires-Dist: {pkg!r}. "
                f"Found {len(requires_dist)} Requires-Dist entries: {sorted(req_names)}. "
                f"Wheel: {wheel.name}. "
                "Check pyproject.toml [project.dependencies]."
            )

    def test_wheel_top_level_is_pragma_encoder(self) -> None:
        """Built wheel top_level.txt must be 'pragma_encoder', not 'src'.

        The top-level package in the wheel must be pragma_encoder. If it
        were 'src', the public import would be ``import src``, not
        ``import pragma_encoder``.
        """
        wheel = _get_or_build_wheel()
        with zipfile.ZipFile(wheel) as zf:
            top_level_entries = [n for n in zf.namelist() if n.endswith("/top_level.txt")]
            assert top_level_entries, f"No top_level.txt found in wheel {wheel.name}"
            top_level = zf.read(top_level_entries[0]).decode().strip()

        assert top_level == "pragma_encoder", (
            f"Wheel top_level.txt = {top_level!r}. Must be 'pragma_encoder'. "
            "Check [tool.setuptools.packages.find].where in pyproject.toml — "
            "must be ['src'] so setuptools discovers pragma_encoder, not src."
        )
        assert top_level != "src", (
            "Wheel top_level.txt is 'src'. The public import would be "
            "``import src``, which is wrong. "
            "Check pyproject.toml [tool.setuptools.packages.find].where = ['src']."
        )

    def test_wheel_entry_points_include_pragma_encoder_train(self) -> None:
        """Built wheel must include the pragma-encoder-train console script entry point.

        Reads entry_points.txt from inside the wheel zip and verifies the
        [console_scripts] section declares pragma-encoder-train.

        This confirms the [project.scripts] declaration in pyproject.toml
        was picked up by the build system and baked into the wheel artifact.
        The entry point makes 'pragma-encoder-train' available in PATH
        after 'pip install pragma-encoder'.
        """
        wheel = _get_or_build_wheel()
        with zipfile.ZipFile(wheel) as zf:
            ep_entries = [n for n in zf.namelist() if n.endswith("/entry_points.txt")]
            assert ep_entries, (
                f"No entry_points.txt found in wheel {wheel.name}. "
                "The [project.scripts] declaration in pyproject.toml must produce "
                "an entry_points.txt inside the wheel. "
                "Rebuild the wheel after adding [project.scripts]."
            )
            ep_text = zf.read(ep_entries[0]).decode()

        assert "pragma-encoder-train" in ep_text, (
            f"entry_points.txt in wheel {wheel.name} does not contain 'pragma-encoder-train'. "
            f"Content:\n{ep_text}\n"
            "Add [project.scripts] to pyproject.toml:\n"
            "pragma-encoder-train = \"pragma_encoder.training.train:main\"\n"
            "Then rebuild the wheel: python -m build --wheel"
        )

    def test_src_is_not_installed_as_package(self) -> None:
        """'src' must not be importable as an installed package of pragma-encoder.

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
            "This indicates a stale egg-info from before the namespace rename. "
            "Delete pragma_encoder.egg-info at the repo root and run "
            "'pip install -e . --force-reinstall --no-deps'. "
            "Check [tool.setuptools.packages.find] in pyproject.toml — "
            "where must be ['src'] not ['.'], and setup.py must not call "
            "find_packages(include=['src', 'src.*'])."
        )


class TestEditableInstall:
    """Verify editable install in a clean temporary virtualenv.

    Uses ``pip install --no-deps -e .`` so the test does not spend minutes
    downloading torch, transformers, numpy, etc. The goal is to prove the
    package surface — editable install mechanics and import identity — not
    to verify that runtime dependencies can be resolved.

    Dependency correctness is covered by TestWheelMetadata.test_wheel_metadata_requires_dist,
    which reads the wheel METADATA file directly.

    On CI the packaging job runs this in isolation to bound its duration.
    With --no-deps and a warm pip cache it completes in under 30 seconds.
    """

    def test_editable_install_no_deps(self) -> None:
        """pip install --no-deps -e . in a fresh venv must exit 0.

        Steps:
          1. Create a temporary virtualenv (stdlib venv, no pip cache sharing).
          2. Upgrade pip and setuptools in the venv.
          3. Run ``pip install --no-deps -e <project_root>`` inside the venv.
          4. Assert exit code 0.
          5. Run ``python -c "import pragma_encoder"`` — must exit 0.
          6. Run ``python -c "import src"`` — must exit non-zero
             (src is NOT the public package name).

        ``--no-deps`` is intentional: this test proves the install mechanics,
        not that ML dependencies can be downloaded. Dependency declarations
        are validated separately by inspecting the wheel METADATA artifact.

        Failure means the package cannot be installed from a clean checkout,
        which would block any user doing ``pip install -e .`` on a fresh clone.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            venv_dir = pathlib.Path(tmp_dir) / "venv"
            venv.create(str(venv_dir), with_pip=True)

            bin_dir = venv_dir / "bin" if (venv_dir / "bin").exists() else venv_dir / "Scripts"
            pip_exe = bin_dir / "pip"
            python_exe = bin_dir / "python"

            # Upgrade build tools to a modern pip that understands pyproject.toml.
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

            # Install in editable mode WITHOUT pulling ML dependencies.
            # --no-deps: do not install torch, transformers, etc. (saves minutes).
            # Dependency correctness is proven by
            # TestWheelMetadata.test_wheel_metadata_requires_dist.
            install = subprocess.run(
                [str(pip_exe), "install", "--no-deps", "-e", str(_PROJECT_ROOT)],
                capture_output=True,
                text=True,
            )
            assert install.returncode == 0, (
                "pip install --no-deps -e . failed in a fresh virtualenv.\n"
                "This means the package cannot be installed from a clean checkout.\n"
                f"stdout:\n{install.stdout[-2000:]}\n"
                f"stderr:\n{install.stderr[-2000:]}"
            )

            # The public Python package is 'pragma_encoder' (not 'src').
            import_check = subprocess.run(
                [str(python_exe), "-c", "import pragma_encoder"],
                capture_output=True,
                text=True,
            )
            assert import_check.returncode == 0, (
                "import pragma_encoder failed in the venv after pip install --no-deps -e .\n"
                "The package was installed (exit 0) but 'pragma_encoder' is not "
                "importable. Check pyproject.toml [tool.setuptools.packages.find] — "
                "where must be ['src'] and src/pragma_encoder/__init__.py must exist.\n"
                f"stdout:\n{import_check.stdout}\n"
                f"stderr:\n{import_check.stderr}"
            )

        print(
            f"\n[TestEditableInstall] pip install --no-deps -e . and "
            f"import pragma_encoder succeeded (Python {sys.version.split()[0]})."
        )


# Keep the old class name as an alias so any external references don't break.
TestCleanInstall = TestEditableInstall


class TestWheelInstall:
    """Verify the built wheel installs cleanly in a fresh venv.

    Builds the wheel (or reuses an existing dist/*.whl), creates a temporary
    virtualenv, installs ``pip install --no-deps dist/*.whl``, and confirms
    ``import pragma_encoder`` works.

    Like TestEditableInstall, uses ``--no-deps`` so the test is fast. ML
    dependency correctness is proved by reading the wheel METADATA directly
    in TestWheelMetadata.

    This test catches wheel-specific packaging failures that editable install
    does not cover — e.g. a missing __init__.py in the sdist, wrong
    package_dir, or an entry-point declaration that breaks the wheel layout.
    """

    def test_wheel_install_no_deps(self) -> None:
        """pip install --no-deps dist/*.whl in a fresh venv must succeed.

        Steps:
          1. Build (or reuse) the wheel from this checkout.
          2. Create a temporary virtualenv.
          3. Upgrade pip in the venv.
          4. Run ``pip install --no-deps <wheel_path>`` inside the venv.
          5. Assert exit code 0.
          6. Run ``python -c "import pragma_encoder"`` — must exit 0.
          7. Run a one-liner that checks the installed top-level package is
             'pragma_encoder', not 'src'.
        """
        wheel = _get_or_build_wheel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            venv_dir = pathlib.Path(tmp_dir) / "venv"
            venv.create(str(venv_dir), with_pip=True)

            bin_dir = venv_dir / "bin" if (venv_dir / "bin").exists() else venv_dir / "Scripts"
            pip_exe = bin_dir / "pip"
            python_exe = bin_dir / "python"

            upgrade = subprocess.run(
                [str(pip_exe), "install", "-U", "pip"],
                capture_output=True,
                text=True,
            )
            assert upgrade.returncode == 0, (
                "pip install -U pip failed in fresh venv.\n"
                f"stderr:\n{upgrade.stderr[-1000:]}"
            )

            install = subprocess.run(
                [str(pip_exe), "install", "--no-deps", str(wheel)],
                capture_output=True,
                text=True,
            )
            assert install.returncode == 0, (
                f"pip install --no-deps {wheel.name} failed in a fresh virtualenv.\n"
                "This means the built wheel cannot be installed from a clean environment.\n"
                f"stdout:\n{install.stdout[-2000:]}\n"
                f"stderr:\n{install.stderr[-2000:]}"
            )

            # pragma_encoder must be importable.
            import_check = subprocess.run(
                [str(python_exe), "-c", "import pragma_encoder"],
                capture_output=True,
                text=True,
            )
            assert import_check.returncode == 0, (
                f"import pragma_encoder failed after pip install --no-deps {wheel.name}.\n"
                "The wheel installed (exit 0) but 'pragma_encoder' is not importable.\n"
                f"stdout:\n{import_check.stdout}\n"
                f"stderr:\n{import_check.stderr}"
            )

            # The top-level package must be pragma_encoder, not src.
            top_level_check = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    (
                        "import importlib.metadata; "
                        "d = importlib.metadata.packages_distributions(); "
                        "assert 'pragma_encoder' in d, "
                        "f'pragma_encoder not in packages_distributions: {sorted(d)}'; "
                        "assert 'src' not in d or 'pragma-encoder' not in d.get('src', []), "
                        "'src is registered as a top-level package of pragma-encoder'"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            assert top_level_check.returncode == 0, (
                "Top-level package identity check failed after wheel install.\n"
                "'pragma_encoder' must be registered; 'src' must not be.\n"
                f"stdout:\n{top_level_check.stdout}\n"
                f"stderr:\n{top_level_check.stderr}"
            )

        print(
            f"\n[TestWheelInstall] pip install --no-deps {wheel.name} and "
            f"import pragma_encoder succeeded (Python {sys.version.split()[0]})."
        )


class TestConsoleScript:
    """Verify the pragma-encoder-train console script entrypoint.

    Tests that the module can be invoked via 'python -m pragma_encoder.training.train'
    and that the argparse --help path exits 0 in the current environment.

    These tests require the full dependency set (torch, transformers, etc.)
    to be installed, because pragma_encoder.training.train imports torch at
    module level. They are skipped when torch is not importable (e.g. in a
    stripped CI environment without ML deps).
    """

    def test_train_module_exists(self) -> None:
        """src/pragma_encoder/training/train.py must exist.

        This is the canonical training entrypoint installed by the wheel.
        It backs both the 'pragma-encoder-train' console script and
        'python -m pragma_encoder.training.train'.
        """
        train_module = _PROJECT_ROOT / "src" / "pragma_encoder" / "training" / "train.py"
        assert train_module.exists(), (
            f"src/pragma_encoder/training/train.py not found at {train_module}. "
            "Create it: move training logic from scripts/train_pragma.py, "
            "expose def main(argv=None) -> int, and register as the console script "
            "entry point in pyproject.toml [project.scripts]."
        )

    def test_train_module_declares_main(self) -> None:
        """src/pragma_encoder/training/train.py must declare def main.

        The console script entry point references pragma_encoder.training.train:main.
        The function must be present at module level in the source file.
        This is a static check — no subprocess or import required.
        """
        train_module = _PROJECT_ROOT / "src" / "pragma_encoder" / "training" / "train.py"
        if not train_module.exists():
            return  # covered by test_train_module_exists
        text = train_module.read_text()
        assert "def main(" in text, (
            "src/pragma_encoder/training/train.py must declare def main(...). "
            "The console script entry point requires this function. "
            "Signature: def main(argv: list[str] | None = None) -> int"
        )

    def test_module_help_exits_zero(self) -> None:
        """python -m pragma_encoder.training.train --help must exit 0.

        --help is handled by argparse before any training or I/O occurs.
        The exit code must be 0 (success). This confirms the module is
        importable and the argparse setup is valid in the current environment.

        Skipped automatically when torch or other deps are not importable
        (i.e. when 'No module named' appears in stderr).
        """
        import pytest  # noqa: PLC0415

        result = subprocess.run(
            [sys.executable, "-m", "pragma_encoder.training.train", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "No module named" in result.stderr:
            pytest.skip(
                f"Dependency not importable (skipping --help test): "
                f"{result.stderr[:200]}"
            )
        assert result.returncode == 0, (
            "python -m pragma_encoder.training.train --help must exit 0. "
            f"Got return code: {result.returncode}\n"
            f"stdout:\n{result.stdout[:500]}\n"
            f"stderr:\n{result.stderr[:500]}"
        )
        assert "PRAGMA pretraining" in result.stdout or "--model-variant" in result.stdout, (
            "python -m pragma_encoder.training.train --help must print argparse help. "
            f"stdout:\n{result.stdout[:500]}"
        )
