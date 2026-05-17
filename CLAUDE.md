# CLAUDE.md — Context for Claude Code

This file provides context for Claude Code working in this repository.

---

## What this repo is

An **independent open-source implementation** of the PRAGMA foundation model
architecture described in:

> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
> arXiv:2604.08649v1

This is NOT Revolut's code. This implements the architecture from the paper.
No pretrained weights or proprietary data from Revolut are included or referenced.

---

## Adjacent repo

```
../transaction-foundation-model-openshiftai/
```

Reference that repo for:
- OpenShift AI deployment patterns (GitOps, ArgoCD, namespace/RBAC)
- KFP SDK v2 pipeline patterns (decorated components, pipeline definition)
- Sealed Secrets pattern (never commit populated secrets)
- Hardware Profile and KFTO PyTorchJob patterns
- Pre-commit hook security scanning pattern

Do NOT copy code from the adjacent repo without explicit instruction.
Reference it for patterns only, then implement fresh for PRAGMA.

---

## Paper sections → code mapping

| Section | Topic | Code |
|---|---|---|
| 2.2 | Key-value-time tokenisation | `src/tokenizer/` |
| 2.3.1 | Architecture config | `src/model/config.py` |
| 2.3.2 | Profile State Encoder | `src/encoders/profile_state_encoder.py` |
| 2.3.3 | Event Encoder | `src/encoders/event_encoder.py` |
| 2.3.4 | History Encoder | `src/encoders/history_encoder.py` |
| 2.3.4 | RoPE | `src/encoders/rope.py` |
| 2.3.5 | Three-strategy masking | `src/masking/strategy.py` |
| 2.3.5 | MLM head | `src/model/mlm_head.py` |
| 2.3.5 | MLM loss | `src/training/objective.py` |
| 2.4 | Sequence packing | `src/training/packing.py` |
| 2.4 | Dynamic batching | `src/training/batching.py` |
| 3 | Full model | `src/model/pragma.py` |
| 3.1.1 | Linear probe | `src/adaptation/probe.py` |
| 3.1.2 | LoRA fine-tuning | `src/adaptation/lora.py` |
| 3 | Evaluation | `src/evaluation/` |

---

## Key architectural constraints — NON-NEGOTIABLE

These come directly from the paper and must not be violated:

### 1. Encoder-only architecture
PRAGMA is **encoder-only**. It is NOT decoder-only (like the NVIDIA blueprint).

- **NEVER** use a causal (left-to-right) attention mask anywhere in this codebase.
- **NEVER** add causal masking to `ProfileStateEncoder`, `EventEncoder`, or `HistoryEncoder`.
- All three encoders use **full bidirectional self-attention**.

### 2. Three separate encoders with separate weights
The three encoders have **no shared weights**:
- `ProfileStateEncoder` — profile fields
- `EventEncoder` — per-event token sequence
- `HistoryEncoder` — event sequence (conditioned on `[USR]`)

Do NOT merge them into a single encoder.

### 3. RoPE positional encoding
Use Rotary Positional Embedding (RoPE) in:
- `ProfileStateEncoder` — for timestamp positions
- `HistoryEncoder` — for event sequence positions

Do NOT use:
- Sinusoidal positional encoding
- Absolute learned positional embeddings
- ALiBi or other alternatives (unless explicitly instructed)

`EventEncoder` uses **calendar token embeddings** (not RoPE) — see Section 2.3.3.

### 4. Key-value-time tokenisation (not text serialisation)
Each transaction field is tokenised with a field-type-specific strategy:
- Numerical → percentile buckets (`NumericalTokenizer`)
- Categorical → single token (`CategoricalTokenizer`)
- Text → BPE subwords (`TextualTokenizer`)
- Timestamp → log-seconds + calendar (`TemporalTokenizer`)

Do NOT serialise transactions as free text (e.g. "amount: 50.00, currency: GBP").
That is the NVIDIA blueprint approach. PRAGMA is different.

### 5. Masked event modelling objective
Pretraining uses **masked** (BERT-style) objective — NOT causal language modelling.
Three strategies: token masking, field masking, event masking (Section 2.3.5).

### 6. Cross-attention in History Encoder
The `HistoryEncoder` has **cross-attention from event positions to the `[USR]` vector**.
This is how the profile conditions the history encoder.
Do NOT replace this with concatenation or addition.

---

## Training stack

| Component | Tool |
|---|---|
| Model implementation | PyTorch (pure) |
| Training orchestration | NeMo AutoModel |
| Distributed training | KFTO PyTorchJob (`kubeflow.org/v1`) |
| Pipeline orchestration | KFP SDK v2 |
| LoRA fine-tuning | Hugging Face PEFT library |
| Embedding extraction | Custom script + numpy/parquet |

**NeMo wraps the training loop only.** The model itself is pure PyTorch.

---

## OpenShift AI deployment rules

Same rules as the adjacent repo:

1. **GA APIs only** — use `kubeflow.org/v1` PyTorchJob, NOT TrainJob (Tech Preview).
2. **Never commit populated secrets** — only `.template.*` files are committed.
3. **Sealed Secrets** for production secret management.
4. **ArgoCD** for GitOps — manifests live in `openshift/gitops/`.
5. **KFP SDK v2** for pipeline components — not v1.

---

## What NOT to do

| Do not | Why |
|---|---|
| Use causal attention mask | PRAGMA is encoder-only, bidirectional |
| Use sinusoidal/absolute positional embeddings | Paper specifies RoPE |
| Serialise transactions as text | That's tabular-GPT; PRAGMA uses key-value-time |
| Copy code from adjacent repo | Reference patterns only |
| Include Revolut weights or data | Not available, not permitted |
| Claim to reproduce Revolut's results | We don't have their data |
| Use TrainJob API | Tech Preview on RHOAI; use PyTorchJob |
| Merge the three encoders into one | Architecture requires three separate encoders |

---

## Documentation requirement

Every `src/` Python file MUST have a module-level docstring that:
1. States what the module implements
2. References the specific PRAGMA paper section
3. Describes the key design choices or constraints

This is enforced by convention. Do not add `src/` files without docstrings.
See existing files for the pattern.

---

## Security

- Pre-commit hook scans for secrets (`.githooks/pre-commit`)
- Hook is active: `git config core.hooksPath .githooks`
- Never commit API keys, tokens, kubeconfig, or `.env` files
- Use `git commit --no-verify` only for confirmed false positives

---

## Development branch

All implementation work happens on `pragma-implementation`.
`main` is protected — PRs only, no direct pushes.

```bash
git checkout pragma-implementation
```
