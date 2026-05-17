# Section 3.1 — Evaluation Protocol

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 3.1

---

## 3.1.1 Embedding Probing

Embeddings extracted from: **History Encoder output (`zh`)**

Use cases for probing:
- Rapid iteration during experimentation
- Gauging whether a new feature brings expected gain
- Checkpoint selection after pre-training run
- Determining whether a task is worth pursuing

**Tokens evaluated:**
- [USR] token
- Final [EVT] token
- Combination of both

**Probe type:** Standard linear probe (logistic regression or linear regression)

**Pre-processing:** Embeddings are **standard-scaled** before probe fitting.
(Because architecture is "inherently pre-norm".)

**Optimiser:** L-BFGS (Liu et al., 1989) — yields best results, converges quickly.
Fitting takes a couple of minutes.

**Note on GBDTs:**
> "While Gradient Boosted Decision Trees (GBDT) perform well on lower-dimensional
> embeddings (e.g., 192-d), the requirement for per-task hyper-parameter tuning and
> the increased time-to-fit make them less practical than linear probing for
> high-velocity model evaluation."

---

## 3.1.2 Downstream Adaptation with LoRA

LoRA introduces **2–4% parameter overhead** only.

**Target modules:** QKV projections and MLP layers within encoder layers

**Default configuration:**
- rank = **8**
- α (alpha) = **8**

**Rank sweep for smaller datasets:** {4, 8, 16}

**Optimiser:** Adam (Kingma et al., 2015)

**Training time:** typically 1/8 of pre-training wall-clock time
→ 12 hours to a few days depending on dataset size.

---

## 3.1.3 Preparing Downstream Datasets

Each downstream dataset is built by:
1. Taking a unique identifier (profile id + evaluation point)
2. Gathering event history and profile attributes directly preceding the evaluation point
3. Following pre-defined folds and splits for each task
4. Process mirrors pre-training dataset collection

---

## Key Numbers (used in tests)

| Number | Value | Implementation |
|--------|-------|----------------|
| lora_rank | 8 | `LoRAAdapter` / `PRAGMAConfig.lora_rank` |
| lora_alpha | 8 | `LoRAAdapter` / `PRAGMAConfig.lora_alpha` |
| lora_param_fraction | 2–4% | `tests/test_lora.py::TestPaperSpecifications` |
| lora_target_modules | QKV + MLP | `LoRAAdapter` |
| probe_optimiser | L-BFGS | `EmbeddingProbe` |
| probe_scaling | standard-scaled | `EmbeddingProbe` |

---

## Implementation Notes

### EmbeddingProbe (`src/adaptation/probe.py`)

Must implement:
1. Extract `zh` from frozen History Encoder
2. Standard-scale the embeddings
3. Fit linear probe using L-BFGS
4. Support [USR] token, final [EVT] token, and combination

### LoRAAdapter (`src/adaptation/lora.py`)

Must implement:
1. Apply LoRA to QKV projections in all encoder layers
2. Apply LoRA to MLP layers in all encoder layers
3. Default rank=8, alpha=8
4. Keep backbone frozen (only LoRA parameters update)
5. Verify 2–4% trainable parameter fraction

**Use PEFT library** (ADR 006). Do not implement custom LoRA.
