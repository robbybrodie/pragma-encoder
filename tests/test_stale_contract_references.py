"""Guard tests for stale platform-contract references.

These tests run in the default suite (no cluster, no credentials) and guard
against stale naming conventions being re-introduced:

- ``MODEL_REGISTRY_*`` env var names must not appear in active source files
  (``scripts/``, ``pipeline/``, ``src/``, ``tools/``). They were replaced by
  canonical RHOAI S3 Connection names (``AWS_*``) in TD-010 (2026-05-22).

- ``scripts/train_pragma.py`` must remain a thin compatibility wrapper.
  Primary training logic belongs in ``pragma_encoder.training.train``.
  Canonical entrypoints: ``pragma-encoder-train`` and
  ``python -m pragma_encoder.training.train``.

References:
- TD-010: docs/tech-debt.md
- docs/openshift-ai-3.3-alignment.md §Env Var Contract
- docs/openshift-ai-primitives.md §Object Storage Connection Contract
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# The old env var names replaced by AWS_* in TD-010.
_FORBIDDEN_ENV_PATTERNS = [
    "MODEL_REGISTRY_BUCKET",
    "MODEL_REGISTRY_ENDPOINT",
    "MODEL_REGISTRY_ACCESS_KEY",
    "MODEL_REGISTRY_SECRET_KEY",
]


def _py_source_files(directory: Path) -> list[Path]:
    """Return .py files under directory, excluding __pycache__ subdirectories."""
    if not directory.exists():
        return []
    return [
        p for p in directory.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


class TestModelRegistryEnvVarsAbsent:
    """MODEL_REGISTRY_* env var names must not appear in active Python source.

    Allowed locations (guard/historical):
    - This file itself (it documents what is forbidden)
    - tests/test_smoke_pipeline.py (forbidden-list guard)
    - tests/test_openshift_ai_fixtures.py (fixture exclusion guard)
    - tests/test_openshift_ai_primitive_contract.py (primitive contract guard)
    - Historical docs (docs/ is not scanned here — only .py files)
    """

    def _forbidden_in(self, directory: Path) -> list[str]:
        offenders: list[str] = []
        for py_file in _py_source_files(directory):
            text = py_file.read_text(encoding="utf-8")
            for pattern in _FORBIDDEN_ENV_PATTERNS:
                if pattern in text:
                    offenders.append(
                        f"{py_file.relative_to(REPO_ROOT)}: contains '{pattern}'"
                    )
        return offenders

    def test_no_model_registry_env_vars_in_scripts(self) -> None:
        """scripts/*.py must not reference MODEL_REGISTRY_* env var names.

        The canonical env vars injected by an RHOAI S3 Connection are:
        AWS_S3_BUCKET, AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
        """
        offenders = self._forbidden_in(REPO_ROOT / "scripts")
        assert not offenders, (
            "scripts/ files must use canonical AWS_* env var names (TD-010 resolved).\n"
            "Replace MODEL_REGISTRY_* with:\n"
            "  AWS_S3_BUCKET / AWS_S3_ENDPOINT / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY\n\n"
            "Found:\n" + "\n".join(f"  {o}" for o in offenders)
        )

    def test_no_model_registry_env_vars_in_pipeline(self) -> None:
        """pipeline/*.py must not reference MODEL_REGISTRY_* env var names."""
        offenders = self._forbidden_in(REPO_ROOT / "pipeline")
        assert not offenders, (
            "pipeline/ files must use canonical AWS_* env var names (TD-010 resolved).\n\n"
            "Found:\n" + "\n".join(f"  {o}" for o in offenders)
        )

    def test_no_model_registry_env_vars_in_src(self) -> None:
        """src/pragma_encoder/ (the wheel) must not reference MODEL_REGISTRY_* names.

        The wheel reads AWS_S3_BUCKET and AWS_S3_ENDPOINT from the process
        environment. The old MODEL_REGISTRY_* naming is fully removed (TD-010).
        """
        offenders = self._forbidden_in(REPO_ROOT / "src")
        assert not offenders, (
            "src/ (wheel) files must use canonical AWS_* env var names (TD-010 resolved).\n\n"
            "Found:\n" + "\n".join(f"  {o}" for o in offenders)
        )

    def test_no_model_registry_env_vars_in_tools(self) -> None:
        """tools/*.py must not reference MODEL_REGISTRY_* env var names."""
        offenders = self._forbidden_in(REPO_ROOT / "tools")
        assert not offenders, (
            "tools/ files must use canonical AWS_* env var names (TD-010 resolved).\n\n"
            "Found:\n" + "\n".join(f"  {o}" for o in offenders)
        )


class TestTrainPragmaCompatWrapper:
    """scripts/train_pragma.py must remain a thin compatibility wrapper.

    The canonical training entrypoints are:
    - ``pragma-encoder-train`` (installed console script from wheel)
    - ``python -m pragma_encoder.training.train``

    scripts/train_pragma.py is a transitional compatibility shim only.
    It must not accumulate primary training logic.
    """

    _WRAPPER = REPO_ROOT / "scripts" / "train_pragma.py"

    def test_train_pragma_exists(self) -> None:
        """scripts/train_pragma.py must exist as a compatibility wrapper."""
        assert self._WRAPPER.exists(), (
            "scripts/train_pragma.py is missing.\n"
            "It is the compatibility wrapper for users invoking "
            "'python scripts/train_pragma.py'. Restore it as a thin shim."
        )

    def test_train_pragma_is_small(self) -> None:
        """scripts/train_pragma.py must be small — a wrapper, not an implementation.

        Threshold: at most 15 non-blank, non-comment lines. A compat wrapper has
        essentially: a docstring, one import, one call. If the file grows beyond
        this, primary logic has leaked into a file that should only delegate.
        """
        text = self._WRAPPER.read_text(encoding="utf-8")
        code_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#") and ln.strip() != '"""'
        ]
        assert len(code_lines) <= 20, (
            f"scripts/train_pragma.py has {len(code_lines)} non-blank/non-comment lines "
            f"(threshold: 20).\n"
            "Primary training logic must live in pragma_encoder.training.train, not here.\n"
            "Canonical entrypoints: pragma-encoder-train / python -m pragma_encoder.training.train"
        )

    def test_train_pragma_delegates_to_canonical_module(self) -> None:
        """scripts/train_pragma.py must delegate to pragma_encoder.training.train."""
        text = self._WRAPPER.read_text(encoding="utf-8")
        assert "pragma_encoder.training.train" in text, (
            "scripts/train_pragma.py must import from pragma_encoder.training.train.\n"
            "It must not contain its own training logic — it is a compatibility shim."
        )
