"""Offline tests for the release-assets build script.

Tests run the build script with --no-wheel (requires a pre-built wheel in
dist/ or dist/release-assets/) and verify the outputs without any network
access or Docker.

Run with::

    pytest tests/test_release_assets.py -q

The tests are skipped if no wheel exists (wheel is built separately by the
packaging CI lane or by ``python -m build``).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_release_assets.py"

# Paths that must appear inside the Workbench zip
REQUIRED_ZIP_PREFIXES = [
    "notebooks/",
    "tools/workbench/",
    "examples/workbench/",
]

# Paths that must NOT appear inside the Workbench zip (security / cleanliness)
EXCLUDED_FROM_ZIP = [
    "__pycache__",
    ".pyc",
    ".ipynb_checkpoints",
    "src/",        # wheel carries the package — raw src should not be duplicated
    "openshift/",  # cluster manifests are not for Workbench users
    "tests/",      # test suite is not part of the Workbench distribution
]

# Stale import strings that must not appear in any example shipped in the zip
STALE_IMPORT_PATTERNS = [
    "from pragma_encoder.workbench",
    "from src.workbench",
    "import pragma_encoder.workbench",
    "scripts/train_pragma.py",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def release_assets_dir(tmp_path_factory) -> pathlib.Path:
    """Run the build script and return the output directory."""
    out_dir = tmp_path_factory.mktemp("release-assets")

    # Check a wheel exists somewhere in dist/
    dist_candidates = list((REPO_ROOT / "dist").glob("pragma_encoder-*.whl")) if (
        REPO_ROOT / "dist"
    ).exists() else []
    ra_candidates = list((REPO_ROOT / "dist" / "release-assets").glob("pragma_encoder-*.whl"))
    all_wheels = dist_candidates + ra_candidates

    if not all_wheels:
        pytest.skip(
            "No wheel found in dist/ — run 'python -m build --wheel' first, "
            "or let the packaging CI lane produce it."
        )

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--no-wheel",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(
            f"build_release_assets.py exited {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return out_dir


@pytest.fixture(scope="module")
def zip_path(release_assets_dir) -> pathlib.Path:
    zips = sorted(release_assets_dir.glob("pragma-encoder-workbench-*.zip"))
    assert zips, f"No workbench zip found in {release_assets_dir}"
    return zips[0]


@pytest.fixture(scope="module")
def zip_names(zip_path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


@pytest.fixture(scope="module")
def manifest(zip_path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("manifest.json"))


# ── Output file checks ────────────────────────────────────────────────────────

def test_wheel_present(release_assets_dir):
    wheels = list(release_assets_dir.glob("pragma_encoder-*.whl"))
    assert wheels, "Wheel not found in release-assets output dir"


def test_workbench_zip_present(release_assets_dir):
    zips = list(release_assets_dir.glob("pragma-encoder-workbench-*.zip"))
    assert zips, "Workbench zip not found in release-assets output dir"


def test_sha256sums_present(release_assets_dir):
    assert (release_assets_dir / "SHA256SUMS").exists(), "SHA256SUMS not found"


def test_sha256sums_lists_both_assets(release_assets_dir):
    text = (release_assets_dir / "SHA256SUMS").read_text()
    assert ".whl" in text, "SHA256SUMS missing wheel entry"
    assert ".zip" in text, "SHA256SUMS missing zip entry"
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 2, f"Expected 2 lines in SHA256SUMS, got {len(lines)}"


def test_sha256sums_correct(release_assets_dir):
    import hashlib

    def sha256(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    text = (release_assets_dir / "SHA256SUMS").read_text()
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        fpath = release_assets_dir / name.strip()
        assert fpath.exists(), f"SHA256SUMS references missing file: {name}"
        actual = sha256(fpath)
        assert actual == digest, (
            f"SHA-256 mismatch for {name}: expected {digest}, got {actual}"
        )


# ── Zip content checks ────────────────────────────────────────────────────────

def test_manifest_present(zip_names):
    assert "manifest.json" in zip_names, "manifest.json missing from zip"


@pytest.mark.parametrize("prefix", REQUIRED_ZIP_PREFIXES)
def test_required_prefix_present(zip_names, prefix):
    matches = [n for n in zip_names if n.startswith(prefix)]
    assert matches, f"No files with prefix '{prefix}' found in zip"


@pytest.mark.parametrize("excluded", EXCLUDED_FROM_ZIP)
def test_excluded_path_absent(zip_names, excluded):
    matches = [n for n in zip_names if excluded in n]
    assert not matches, (
        f"'{excluded}' should not be in the zip, but found: {matches[:5]}"
    )


def test_no_pyc_files(zip_names):
    pyc = [n for n in zip_names if n.endswith(".pyc")]
    assert not pyc, f".pyc files found in zip: {pyc[:5]}"


def test_notebooks_present(zip_names):
    nbs = [n for n in zip_names if n.endswith(".ipynb")]
    assert len(nbs) >= 5, f"Expected ≥5 notebooks, found {len(nbs)}: {nbs}"


def test_workbench_tools_present(zip_names):
    wb_files = [n for n in zip_names if n.startswith("tools/workbench/") and n.endswith(".py")]
    assert wb_files, "No .py files found under tools/workbench/ in zip"


def test_examples_present(zip_names):
    ex_files = [n for n in zip_names if n.startswith("examples/workbench/") and n.endswith(".py")]
    assert ex_files, "No .py files found under examples/workbench/ in zip"


# ── Manifest field checks ─────────────────────────────────────────────────────

def test_manifest_version(manifest):
    assert "version" in manifest, "manifest missing 'version'"
    assert manifest["version"], "manifest version is empty"
    # Must match semver-ish pattern (at least X.Y.Z)
    parts = manifest["version"].split(".")
    assert len(parts) >= 3, f"version '{manifest['version']}' is not X.Y.Z"


def test_manifest_git_commit(manifest):
    assert "git_commit" in manifest, "manifest missing 'git_commit'"
    assert manifest["git_commit"], "manifest git_commit is empty"


def test_manifest_built_at(manifest):
    assert "built_at" in manifest, "manifest missing 'built_at'"
    # Must be an ISO-8601 datetime
    from datetime import datetime
    datetime.fromisoformat(manifest["built_at"])


def test_manifest_wheel_sha256(manifest):
    assert "wheel_sha256" in manifest, "manifest missing 'wheel_sha256'"
    sha = manifest["wheel_sha256"]
    assert len(sha) == 64, f"wheel_sha256 should be 64 hex chars, got {len(sha)}"


def test_manifest_wheel_sha256_matches_actual(release_assets_dir, manifest):
    import hashlib

    wheel = next(release_assets_dir.glob("pragma_encoder-*.whl"), None)
    assert wheel is not None
    h = hashlib.sha256()
    with wheel.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert h.hexdigest() == manifest["wheel_sha256"], (
        "manifest wheel_sha256 does not match actual wheel"
    )


# ── Stale import / reference checks ──────────────────────────────────────────

def test_no_stale_imports_in_examples(zip_path):
    """Verify shipped examples do not reference removed or renamed modules."""
    violations: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            for pattern in STALE_IMPORT_PATTERNS:
                if pattern in text:
                    violations.append((name, pattern))

    assert not violations, (
        "Stale references found in shipped files:\n"
        + "\n".join(f"  {f}: '{p}'" for f, p in violations)
    )


def test_no_stale_imports_in_notebooks(zip_path):
    """Verify shipped notebooks do not reference removed or renamed modules."""
    violations: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".ipynb"):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            for pattern in STALE_IMPORT_PATTERNS:
                if pattern in text:
                    violations.append((name, pattern))

    assert not violations, (
        "Stale references found in shipped notebooks:\n"
        + "\n".join(f"  {f}: '{p}'" for f, p in violations)
    )


# ── Version consistency ───────────────────────────────────────────────────────

def test_wheel_version_matches_manifest(release_assets_dir, manifest):
    wheel = next(release_assets_dir.glob("pragma_encoder-*.whl"), None)
    assert wheel is not None
    # Wheel filename format: pragma_encoder-VERSION-py3-none-any.whl
    wheel_version = wheel.stem.split("-")[1]
    assert wheel_version == manifest["version"], (
        f"Wheel version '{wheel_version}' != manifest version '{manifest['version']}'"
    )


def test_version_matches_pyproject(manifest):
    pyproject = REPO_ROOT / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        line = line.strip()
        if line.startswith("version") and "=" in line:
            pyproject_version = line.split("=", 1)[1].strip().strip('"').strip("'")
            assert manifest["version"] == pyproject_version, (
                f"manifest version '{manifest['version']}' != "
                f"pyproject.toml version '{pyproject_version}'"
            )
            return
    pytest.fail("Could not find version in pyproject.toml")
