"""Image contract tests for the PRAGMA two-image model.

Validates that the workbench image and training/component image have the
correct dependency split, build instructions, and guard behaviour.

See docs/openshift-image-contract.md for the authoritative contract.

Four categories:

  TestWorkbenchDependencies — pyproject.toml must NOT declare kfp/kfp-kubernetes
      in optional-deps (TD-009 resolved). notebook requirements.txt must
      declare kfp and kfp-kubernetes for the workbench image.

  TestTrainingImageContract — openshift/training/Dockerfile.training must
      install the built wheel (not an editable src/ install) and must not
      use runtime git clone.

  TestKfpKubernetesGuard — tools/openshift_ai/workbench/_submit._require_kfp_kubernetes()
      must raise a friendly ImportError (with install hint) when
      kfp_kubernetes is absent, and must never be called at module import
      time (only when secret injection is explicitly requested).

  TestKfpKubernetesBoundary — src/pragma_encoder/training/checkpoints.py and
      scripts/train_pragma.py must not import kfp or kfp_kubernetes.
      These run in the training image where kfp is a workbench-only dep.
      KFP/kfp-kubernetes belong only in the workbench compile environment.
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
_SUBMIT_PY = _REPO_ROOT / "tools" / "openshift_ai" / "workbench" / "_submit.py"
_CHECKPOINTS_PY = _REPO_ROOT / "src" / "pragma_encoder" / "training" / "checkpoints.py"
_TRAIN_MODULE = _REPO_ROOT / "src" / "pragma_encoder" / "training" / "train.py"
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_pragma.py"  # compatibility wrapper


# ---------------------------------------------------------------------------
# TestWorkbenchDependencies
# ---------------------------------------------------------------------------


class TestWorkbenchDependencies:
    """workbench extras / notebook image requirements include kfp-kubernetes.

    Category: image-contract / dependency-declaration
    """

    def test_pyproject_has_no_workbench_optional_extras(self) -> None:
        """pyproject.toml must NOT have [project.optional-dependencies].workbench.

        TD-009 resolved: workbench helpers moved to tools/openshift_ai/workbench/
        and removed from the wheel. kfp and kfp-kubernetes are now exclusively
        declared in openshift/notebook-image/requirements.txt.
        """
        text = _PYPROJECT.read_text()
        workbench_line = next(
            (ln for ln in text.splitlines() if ln.strip().startswith("workbench =")), None
        )
        assert workbench_line is None, (
            "pyproject.toml must NOT have workbench optional-dependencies. "
            "TD-009 resolved: kfp/kfp-kubernetes are declared in "
            "openshift/notebook-image/requirements.txt only. "
            f"Found unexpected line: {workbench_line!r}"
        )

    def test_kfp_not_in_pyproject_optional_dependencies(self) -> None:
        """kfp must NOT appear in pyproject.toml [project.optional-dependencies].

        TD-009 resolved: the workbench extras were removed. kfp is now
        exclusively declared in openshift/notebook-image/requirements.txt
        for the workbench image. This test ensures the extras were not
        accidentally re-added.
        """
        text = _PYPROJECT.read_text()
        in_optional_deps = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[project.optional-dependencies]"):
                in_optional_deps = True
                continue
            if in_optional_deps and stripped.startswith("["):
                in_optional_deps = False
                continue
            if in_optional_deps and "kfp" in stripped and not stripped.startswith("#"):
                pytest.fail(
                    f"kfp must not appear in [project.optional-dependencies]. "
                    f"Found: {line!r}. "
                    "kfp is declared in openshift/notebook-image/requirements.txt only "
                    "(TD-009 resolved)."
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

    The image install model is wheel-based (not editable src/ install):
      1. Build the wheel: python -m build --wheel
      2. COPY dist/ into the image context
      3. RUN pip install --no-deps dist/pragma_encoder-*.whl

    This replaces the old COPY src/ + pip install -e . pattern.
    The pragma_encoder package is in site-packages, not a src/ tree.
    """

    def test_dockerfile_training_installs_wheel(self) -> None:
        """Dockerfile.training must install pragma_encoder from a built wheel.

        The image must COPY dist/ (or a .whl file) and pip install it.
        This replaces the old 'COPY src/ + pip install -e .' pattern.

        The wheel provides a clean, reproducible install: the same artifact
        tested by TestWheelMetadata and TestWheelInstall is what runs in the
        training pod. No source tree, no editable-install symlinks.

        Build flow before oc start-build:
          python -m build --wheel   # creates dist/pragma_encoder-*.whl
          oc start-build pragma-encoder-training --from-dir=. --follow -n pragma-encoder
        """
        text = _DOCKERFILE_TRAINING.read_text()
        # Must have a pip install line that references a .whl file
        pip_lines = [
            ln for ln in text.splitlines()
            if "pip install" in ln and ".whl" in ln and not ln.strip().startswith("#")
        ]
        assert pip_lines, (
            "Dockerfile.training must install pragma_encoder from a built wheel. "
            "Expected a 'pip install ... .whl' line. "
            "Build the wheel first (python -m build --wheel), then COPY dist/ and "
            "RUN pip install --no-deps dist/pragma_encoder-*.whl. "
            "See docs/openshift-image-contract.md."
        )

    def test_dockerfile_training_does_not_use_editable_install(self) -> None:
        """Dockerfile.training must NOT use pip install -e (editable install).

        The old model was: COPY src/ + pip install --no-deps -e .
        The new model is:  COPY dist/*.whl + pip install --no-deps *.whl

        Editable installs create a .pth file pointing into the source tree.
        With the wheel model, the source tree is not in the image — the
        package is installed cleanly into site-packages.
        """
        text = _DOCKERFILE_TRAINING.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pip install" in stripped and " -e " in stripped:
                pytest.fail(
                    f"Dockerfile.training must not use 'pip install -e' (editable install). "
                    f"Found: {line!r}. "
                    "Replace with: pip install --no-deps dist/pragma_encoder-*.whl. "
                    "The wheel must be built before the image: python -m build --wheel."
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
    """tools/openshift_ai/workbench/_submit._require_kfp_kubernetes() guard.

    Lazy, friendly, only when enabled.
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

        from tools.openshift_ai.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        with pytest.raises(ImportError):
            _require_kfp_kubernetes("test feature")

    def test_guard_error_message_contains_feature_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ImportError message must include the feature name passed to the guard."""
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        from tools.openshift_ai.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

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

        from tools.openshift_ai.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

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

        from tools.openshift_ai.workbench._submit import _require_kfp_kubernetes  # noqa: PLC0415

        result = _require_kfp_kubernetes("test feature")
        assert result is stub, (
            "Guard must return the kfp_kubernetes module when present. "
            f"Got: {result!r}"
        )


# ---------------------------------------------------------------------------
# TestKfpKubernetesBoundary
# ---------------------------------------------------------------------------


class TestKfpKubernetesBoundary:
    """KFP/kfp-kubernetes must remain isolated to the workbench compile path.

    Category: image-contract / kfp-boundary

    The training image runtime must NOT import kfp or kfp-kubernetes:
      - src/pragma_encoder/training/checkpoints.py runs inside training pods (emptyDir, CPU/GPU)
      - src/pragma_encoder/training/train.py — the canonical training entrypoint (wheel module)
      - scripts/train_pragma.py — compatibility wrapper (delegates to train.py)

    kfp and kfp-kubernetes are compile-time workbench dependencies only.
    They are installed in the workbench/notebook image and used to author and
    compile KFP pipelines. They must not be required at training time.

    See docs/openshift-image-contract.md — KFP boundary.
    """

    def test_checkpoints_module_exists(self) -> None:
        """src/pragma_encoder/training/checkpoints.py must exist (TD-006 fix implementation).

        This test will fail until checkpoints.py is created (expected red phase).
        Once created it acts as a guard against accidental deletion.
        """
        assert _CHECKPOINTS_PY.exists(), (
            f"src/pragma_encoder/training/checkpoints.py not found at {_CHECKPOINTS_PY}. "
            "Create it to implement the all-rank S3 download pattern (TD-006 fix). "
            "See tests/test_checkpoint_resume.py for the full API contract."
        )

    def test_checkpoints_does_not_import_kfp_at_module_level(self) -> None:
        """src/pragma_encoder/training/checkpoints.py must not have top-level kfp imports.

        This module runs inside training pods that may not have kfp installed.
        Any top-level kfp import would cause ImportError at job startup.
        """
        if not _CHECKPOINTS_PY.exists():
            pytest.skip("src/pragma_encoder/training/checkpoints.py does not exist yet.")
        text = _CHECKPOINTS_PY.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            # Only catch non-indented (module-level) import lines
            if not line.startswith(" ") and not line.startswith("\t"):
                if stripped.startswith("import kfp") or stripped.startswith("from kfp"):
                    pytest.fail(
                        f"src/pragma_encoder/training/checkpoints.py must not import kfp at module level. "
                        f"Found: {line!r}. "
                        "kfp belongs only in the workbench compile environment. "
                        "See docs/openshift-image-contract.md."
                    )

    def test_checkpoints_does_not_import_kfp_kubernetes_at_module_level(self) -> None:
        """src/pragma_encoder/training/checkpoints.py must not have top-level kfp_kubernetes imports."""
        if not _CHECKPOINTS_PY.exists():
            pytest.skip("src/pragma_encoder/training/checkpoints.py does not exist yet.")
        text = _CHECKPOINTS_PY.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if not line.startswith(" ") and not line.startswith("\t"):
                if "kfp_kubernetes" in stripped and (
                    stripped.startswith("import") or stripped.startswith("from")
                ):
                    pytest.fail(
                        f"src/pragma_encoder/training/checkpoints.py must not import kfp_kubernetes. "
                        f"Found: {line!r}. "
                        "kfp-kubernetes belongs only in the workbench compile environment."
                    )

    def test_checkpoints_importable_when_kfp_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pragma_encoder.training.checkpoints must be importable when kfp is absent.

        Simulates the training image environment where kfp is not installed.
        The checkpoints module must import cleanly — no kfp-related ImportError.
        """
        if not _CHECKPOINTS_PY.exists():
            pytest.skip("src/pragma_encoder/training/checkpoints.py does not exist yet.")
        # Simulate kfp being absent
        monkeypatch.setitem(sys.modules, "kfp", None)
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        # Force reimport to test clean import without kfp.
        # Use monkeypatch.delitem so pytest restores the original module object
        # after this test, preventing cross-test pollution when other tests
        # patch src.training.checkpoints._make_s3_client.
        monkeypatch.delitem(sys.modules, "pragma_encoder.training.checkpoints", raising=False)

        try:
            import pragma_encoder.training.checkpoints  # noqa: F401,PLC0415
        except ImportError as exc:
            if "kfp" in str(exc).lower():
                pytest.fail(
                    f"pragma_encoder.training.checkpoints raised kfp-related ImportError: {exc}. "
                    "The checkpoints module must not require kfp or kfp-kubernetes. "
                    "See docs/openshift-image-contract.md."
                )
            # Other ImportErrors (e.g. boto3 not installed) are acceptable —
            # the training image has boto3 but a local dev env may not.

    def test_train_module_exists(self) -> None:
        """src/pragma_encoder/training/train.py must exist (canonical training entrypoint).

        This is the wheel-installed module that backs pragma-encoder-train (console script)
        and python -m pragma_encoder.training.train. It replaces the runtime dependency
        on scripts/train_pragma.py in the training image.
        """
        assert _TRAIN_MODULE.exists(), (
            f"src/pragma_encoder/training/train.py not found at {_TRAIN_MODULE}. "
            "The canonical training entrypoint must be the installed wheel module. "
            "See: pragma_encoder.training.train (console script: pragma-encoder-train)."
        )

    def test_train_module_does_not_import_kfp_at_module_level(self) -> None:
        """src/pragma_encoder/training/train.py must not have top-level kfp imports.

        This module runs inside training pods (as pragma-encoder-train or via torchrun
        -m pragma_encoder.training.train). kfp must not be required at training time.
        """
        if not _TRAIN_MODULE.exists():
            pytest.skip("src/pragma_encoder/training/train.py does not exist yet.")
        text = _TRAIN_MODULE.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if not line.startswith(" ") and not line.startswith("\t"):
                if stripped.startswith("import kfp") or stripped.startswith("from kfp"):
                    pytest.fail(
                        f"src/pragma_encoder/training/train.py must not import kfp at module level. "
                        f"Found: {line!r}. "
                        "kfp belongs only in the workbench compile environment."
                    )

    def test_train_script_does_not_import_kfp_at_module_level(self) -> None:
        """scripts/train_pragma.py must not have top-level kfp imports.

        The compatibility wrapper delegates to pragma_encoder.training.train.
        It must not introduce any kfp imports.
        """
        text = _TRAIN_SCRIPT.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            # Only flag non-indented (module-level) import lines
            if not line.startswith(" ") and not line.startswith("\t"):
                if stripped.startswith("import kfp") or stripped.startswith("from kfp"):
                    pytest.fail(
                        f"scripts/train_pragma.py must not import kfp at module level. "
                        f"Found: {line!r}. "
                        "kfp belongs only in the workbench compile environment."
                    )

    def test_train_module_references_resolve_resume_checkpoint(self) -> None:
        """src/pragma_encoder/training/train.py must call resolve_resume_checkpoint.

        This confirms the TD-006 fix is integrated in the canonical training module:
        all ranks download the checkpoint from S3 independently (not just rank 0).
        Without this, workers start from global_step=0 and model states diverge.
        """
        if not _TRAIN_MODULE.exists():
            pytest.skip("src/pragma_encoder/training/train.py does not exist yet.")
        text = _TRAIN_MODULE.read_text()
        assert "resolve_resume_checkpoint" in text, (
            "src/pragma_encoder/training/train.py must call resolve_resume_checkpoint() "
            "from src/pragma_encoder/training/checkpoints.py. "
            "The TD-006 fix requires ALL ranks to independently download the checkpoint. "
            "Until this is integrated, Level 5 (S3 resume smoke) cannot pass. "
            "See tests/openshift/test_05_s3_checkpoint_resume.py."
        )

    def test_train_script_references_resolve_resume_checkpoint(self) -> None:
        """scripts/train_pragma.py (compatibility wrapper) must delegate to train.py.

        The wrapper must import from pragma_encoder.training.train so that the
        resolve_resume_checkpoint behaviour is preserved via the module.
        """
        text = _TRAIN_SCRIPT.read_text()
        assert "pragma_encoder.training.train" in text, (
            "scripts/train_pragma.py must import from pragma_encoder.training.train. "
            "The file is now a compatibility wrapper — actual logic lives in train.py. "
            "See: src/pragma_encoder/training/train.py."
        )
