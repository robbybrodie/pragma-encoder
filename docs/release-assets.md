# Release Assets

This document explains how pragma-encoder is distributed, what each release
artifact contains, and how to install it in the three supported usage modes.

---

## Three usage modes

| Mode | Who | What they install |
|---|---|---|
| **Developer** | Contributors to this repo | `pip install -e .` from the repo root (editable install) |
| **Workbench user** | ML practitioners running notebooks in OpenShift AI | Wheel + Workbench assets zip from GitHub Releases |
| **Runtime** | Training containers on the cluster | Wheel baked into the training image |

This document covers the **Workbench user** and **runtime** modes. Developers
use the standard contributor setup described in `DEVELOPMENT_PROCESS.md`.

---

## What each release produces

Every version tag (`v0.1.0`, `v0.2.0`, …) triggers the
`.github/workflows/release-assets.yml` CI job, which attaches three files to
the GitHub Release:

| File | Contents |
|---|---|
| `pragma_encoder-<VERSION>-py3-none-any.whl` | Pure-Python wheel — the `pragma_encoder` package only |
| `pragma-encoder-workbench-<VERSION>.zip` | Workbench assets: `notebooks/`, `tools/workbench/`, `examples/workbench/`, `manifest.json` |
| `SHA256SUMS` | SHA-256 digests of both files for integrity verification |

The wheel is **platform-neutral** (`py3-none-any`). It contains no compiled
extensions, no cluster tooling, and no workbench helpers. This keeps the wheel
small and installable everywhere.

The Workbench zip contains everything a Workbench user needs to run the
learning-journey notebooks and submit pipeline runs — without cloning the
full repository.

---

## Workbench user install

### Step 1 — Download the release assets

Replace `<VERSION>` with the actual version (e.g. `0.1.0`):

```bash
# Using the GitHub CLI (recommended)
gh release download v<VERSION> \
  --repo <ORG>/<REPO> \
  --pattern "pragma_encoder-<VERSION>-*.whl" \
  --pattern "pragma-encoder-workbench-<VERSION>.zip" \
  --pattern "SHA256SUMS" \
  --dir ~/pragma-release
```

Or download manually from the GitHub Releases page and place the files in
`~/pragma-release/`.

### Step 2 — Verify integrity

```bash
cd ~/pragma-release
sha256sum --check SHA256SUMS
```

Expected output: both files should show `OK`.

### Step 3 — Install the wheel

```bash
pip install ~/pragma-release/pragma_encoder-<VERSION>-py3-none-any.whl
```

### Step 4 — Unpack the Workbench assets

```bash
cd ~
unzip ~/pragma-release/pragma-encoder-workbench-<VERSION>.zip -d pragma-workbench
```

This creates:

```
~/pragma-workbench/
  manifest.json
  notebooks/
    01_pragma_tokenization.ipynb
    02_pragma_pretraining.ipynb
    03_pragma_embedding_probe.ipynb
    04_pragma_lora_finetuning.ipynb
    05_pragma_evaluation.ipynb
  tools/
    workbench/
      _api.py
      _decorators.py
      _intent.py
      _run.py
      _submit.py
  examples/
    workbench/
      01_train_ibm_tabformer.py
      ...
```

### Step 5 — Set PYTHONPATH in your Workbench session

In a notebook cell or at the top of your script:

```python
import sys, pathlib
wb_root = pathlib.Path.home() / "pragma-workbench"
if str(wb_root) not in sys.path:
    sys.path.insert(0, str(wb_root))
```

This allows `from tools.workbench import train_pragma` to resolve.

Alternatively, set `PYTHONPATH` in your Workbench environment:

```bash
export PYTHONPATH="$HOME/pragma-workbench:$PYTHONPATH"
```

### Step 6 — Run the notebooks

Open `~/pragma-workbench/notebooks/01_pragma_tokenization.ipynb` in
JupyterLab and run cells in order.

---

## Runtime (training image) install

The training image bakes the wheel in at build time:

```dockerfile
COPY pragma_encoder-<VERSION>-py3-none-any.whl /tmp/
RUN pip install /tmp/pragma_encoder-<VERSION>-py3-none-any.whl
```

The workbench zip is not needed in the training image — it contains only
notebook and helper files, not the `pragma_encoder` package itself.

---

## Verifying a release asset manually

The `manifest.json` inside the Workbench zip records the build provenance:

```bash
unzip -p pragma-encoder-workbench-<VERSION>.zip manifest.json | python -m json.tool
```

Output example:

```json
{
  "version": "0.1.0",
  "git_commit": "a1b2c3d",
  "built_at": "2026-05-26T10:00:00+00:00",
  "wheel_sha256": "abc123...",
  "notes": "Install the wheel first..."
}
```

The `wheel_sha256` field must match the SHA-256 of the wheel you downloaded
(cross-check against `SHA256SUMS`).

---

## Building release assets locally

To build the assets without creating a tag (e.g. to test the build script):

```bash
# From repo root
python -m build --wheel                         # build wheel into dist/
python scripts/build_release_assets.py          # build wheel + zip + SHA256SUMS

# Or, if a wheel already exists in dist/:
python scripts/build_release_assets.py --no-wheel
```

Output is written to `dist/release-assets/`.

---

## CI workflow

`.github/workflows/release-assets.yml` runs on every `v*` tag push:

1. **Lint** — `ruff check src/ tests/ scripts/ tools/`
2. **Build** — `python -m build --wheel` then `python scripts/build_release_assets.py --no-wheel`
3. **Test** — `pytest tests/test_packaging.py tests/test_platform_neutral_wheel.py tests/test_release_assets.py`
4. **Upload** — `gh release upload <TAG> dist/release-assets/*`

The workflow can also be triggered manually from the GitHub Actions UI (dry run
mode — builds but does not upload).
