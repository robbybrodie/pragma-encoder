# PRAGMA Notebooks — A Guided Journey

**Who this is for:** Anyone new to PRAGMA (or new to Jupyter) who wants to understand the model end-to-end by running real code, not just reading docs.

**What you need:** Python, this repo cloned, a Jupyter kernel. No GPU, no S3, no cluster credentials required. Everything runs on CPU with synthetic data.

---

## The five notebooks

| # | Notebook | What you'll understand by the end |
|---|----------|----------------------------------|
| 01 | `01_pragma_tokenization.ipynb` | Why tabular financial data needs its own tokenisation, and how PRAGMA turns each transaction field into tokens |
| 02 | `02_pragma_pretraining.ipynb` | How PRAGMA learns from masked events, and what the loss curve looks like on a real training loop |
| 03 | `03_pragma_embedding_probe.ipynb` | How to use frozen PRAGMA embeddings for a downstream classification task with a linear probe |
| 04 | `04_pragma_lora_finetuning.ipynb` | How LoRA fine-tunes PRAGMA with 2–4% of parameters, and why it beats a frozen probe |
| 05 | `05_pragma_evaluation.ipynb` | How to evaluate a PRAGMA encoder end-to-end, and a recap of every number you produced across the journey |

---

## How to start

1. Open `01_pragma_tokenization.ipynb` in JupyterLab
2. Press **Shift+Enter** to run each cell top to bottom
3. Continue through notebooks 02–05 in order

Each notebook opens with a short **"What this notebook teaches"** section and closes with a pointer to the next one.

---

## Prerequisites

- Python environment with `pragma_encoder` installed (run `pip install -e ".[dev]"` from the repo root, or use the workbench image which has it pre-installed)
- The repo must be on your `sys.path` — each notebook handles this automatically in its setup cell
- No S3 credentials, no GPU, no cluster access required

---

## Runtime

Each notebook runs top-to-bottom in **under two minutes on CPU**. Notebooks 02–04 run short training loops on synthetic data (30–50 steps) to keep things fast while producing real, visible output.

---

## Going deeper

- **Narrative walkthrough:** `docs/workbench-journey.md` — the guided story behind the pipeline
- **Real training on IBM TabFormer data:** `docs/training-guide.md`
- **Paper-to-code mapping:** `docs/paper-to-code.md` — every paper section mapped to its source file
- **Paper:** Ostroukhov et al. (2026), arXiv:2604.08649v1
