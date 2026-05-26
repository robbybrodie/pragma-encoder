"""Build release assets for a pragma-encoder version tag.

Produces ``dist/release-assets/`` containing:

    pragma_encoder-<VERSION>-py3-none-any.whl   — pure-Python wheel
    pragma-encoder-workbench-<VERSION>.zip       — Workbench assets (notebooks,
                                                   tools/workbench, examples)
    SHA256SUMS                                   — integrity file for both assets

The Workbench zip embeds a ``manifest.json`` describing the version, git
commit, build timestamp, and the wheel SHA-256 so the Workbench user can
cross-check what they downloaded.

Usage (from repo root)::

    python scripts/build_release_assets.py            # build wheel + zip
    python scripts/build_release_assets.py --no-wheel # use existing wheel in dist/

The script is intentionally dependency-free beyond the standard library so it
can be called from CI without installing extras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

# ── Repo layout constants ──────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Paths included in the Workbench assets zip (relative to repo root)
WORKBENCH_ASSET_PATHS: list[str] = [
    "notebooks",
    "tools/workbench",
    "examples/workbench",
]

# Paths / suffixes to exclude from the zip even if they appear inside the above
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".mypy_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _read_version() -> str:
    """Read version from pyproject.toml without installing any extras."""
    pyproject = REPO_ROOT / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        line = line.strip()
        if line.startswith("version") and "=" in line:
            # version = "0.1.0"
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("Could not find version in pyproject.toml")


def _find_wheel(dist_dir: pathlib.Path, version: str) -> pathlib.Path | None:
    pattern = f"pragma_encoder-{version}-*.whl"
    matches = sorted(dist_dir.glob(pattern))
    return matches[-1] if matches else None


def _build_wheel(dist_dir: pathlib.Path) -> pathlib.Path:
    print("Building wheel …")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("pragma_encoder-*.whl"))
    if not wheels:
        raise RuntimeError("build produced no wheel in dist/")
    return wheels[-1]


def _should_exclude(path: pathlib.Path) -> bool:
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def _add_tree_to_zip(
    zf: zipfile.ZipFile,
    src_dir: pathlib.Path,
    zip_prefix: str,
) -> int:
    """Add all non-excluded files under src_dir into the zip at zip_prefix/."""
    count = 0
    for fpath in sorted(src_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(src_dir)
        if _should_exclude(rel):
            continue
        arcname = f"{zip_prefix}/{rel}"
        zf.write(fpath, arcname)
        count += 1
    return count


def build_workbench_zip(
    out_dir: pathlib.Path,
    version: str,
    wheel_sha256: str,
) -> pathlib.Path:
    """Build the Workbench assets zip and return its path."""
    zip_name = f"pragma-encoder-workbench-{version}.zip"
    zip_path = out_dir / zip_name

    manifest = {
        "version": version,
        "git_commit": _git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": wheel_sha256,
        "notes": (
            "Install the wheel first: pip install pragma_encoder-*.whl. "
            "Then set PYTHONPATH=tools in your Workbench session so that "
            "'from tools.workbench import train_pragma' resolves."
        ),
    }

    print(f"Building Workbench zip → {zip_path.name} …")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest first
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )

        for rel_path_str in WORKBENCH_ASSET_PATHS:
            src = REPO_ROOT / rel_path_str
            if not src.exists():
                print(f"  WARNING: {rel_path_str} not found — skipping")
                continue
            n = _add_tree_to_zip(zf, src, rel_path_str)
            print(f"  {rel_path_str}: {n} files")
            file_count += n

    print(f"  Total: {file_count} files in zip")
    return zip_path


def write_sha256sums(out_dir: pathlib.Path, files: list[pathlib.Path]) -> pathlib.Path:
    lines = []
    for f in files:
        lines.append(f"{_sha256(f)}  {f.name}\n")
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text("".join(lines))
    return sums_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pragma-encoder release assets into dist/release-assets/"
    )
    parser.add_argument(
        "--no-wheel",
        action="store_true",
        help="Skip building the wheel; use an existing wheel found in dist/",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "dist" / "release-assets",
        help="Output directory (default: dist/release-assets/)",
    )
    args = parser.parse_args()

    version = _read_version()
    print(f"pragma-encoder version: {version}")

    dist_dir = REPO_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Wheel ─────────────────────────────────────────────────────────────────
    if args.no_wheel:
        wheel_path = _find_wheel(dist_dir, version)
        if wheel_path is None:
            # Try release-assets dir too
            wheel_path = _find_wheel(out_dir, version)
        if wheel_path is None:
            print(
                f"ERROR: --no-wheel specified but no pragma_encoder-{version}-*.whl "
                f"found in {dist_dir} or {out_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using existing wheel: {wheel_path.name}")
    else:
        wheel_path = _build_wheel(dist_dir)
        print(f"Wheel built: {wheel_path.name}")

    # Copy wheel into out_dir (idempotent)
    wheel_dest = out_dir / wheel_path.name
    if wheel_dest.resolve() != wheel_path.resolve():
        shutil.copy2(wheel_path, wheel_dest)
    wheel_sha256 = _sha256(wheel_dest)
    print(f"Wheel SHA-256: {wheel_sha256[:16]}…")

    # ── Workbench zip ─────────────────────────────────────────────────────────
    zip_path = build_workbench_zip(out_dir, version, wheel_sha256)
    zip_sha256 = _sha256(zip_path)
    print(f"Zip SHA-256:   {zip_sha256[:16]}…")

    # ── SHA256SUMS ────────────────────────────────────────────────────────────
    sums_path = write_sha256sums(out_dir, [wheel_dest, zip_path])
    print(f"SHA256SUMS written: {sums_path}")

    print()
    print("Release assets ready:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<55} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
