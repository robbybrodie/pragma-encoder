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

## MANDATORY — Read Before Any Implementation

### DEVELOPMENT_PROCESS.md
Read DEVELOPMENT_PROCESS.md in full before implementing
any component. It contains:
  - The five-step TDD process (tests before implementation)
  - Interface contracts (Protocols before code)
  - Naming conventions (fixed, from the paper)
  - Configuration rules (no hardcoded values)
  - Dependency graph (one direction, no circular imports)
  - Type hint requirements (complete, with shape comments)
  - Pre-implementation checklist (eight items, all required)
  - Lessons from prior projects (hard-won, not theoretical)

### docs/decisions/
Read docs/decisions/ before making any architectural decision.
Every major design choice is documented with:
  - The paper section that supports it
  - The context and reasoning
  - What future sessions must not contradict

If you are about to do something that contradicts a decision:
STOP. Raise it with the human. Do not proceed silently.

### The non-negotiables
  1. Encoder-only. Never decoder-only.
  2. Bidirectional attention. Never causal mask.
  3. Three separate encoders. Never one combined encoder.
  4. Tests before implementation. Always.
  5. Names from naming conventions. Always.
  6. Parameters from PRAGMAConfig. Always.
  7. Dependency graph respected. Always.
  8. Interface Protocol defined before implementation. Always.

---

## Paper Knowledge Base

The PRAGMA paper is extracted into `docs/paper/`.
Before implementing any component, read the relevant file.

| Component | Primary reference |
|-----------|-------------------|
| `PRAGMAConfig` | `docs/paper/table-1-model-sizes.md` |
| `PRAGMATokenizer` | `docs/paper/02-02-tokenisation.md` |
| `RoPEEncoding` | `docs/paper/02-03-architecture.md` |
| `ProfileStateEncoder` | `docs/paper/02-03-architecture.md` |
| `EventEncoder` | `docs/paper/02-03-architecture.md` |
| `HistoryEncoder` | `docs/paper/02-03-architecture.md` |
| `MaskingStrategy` | `docs/paper/02-03-architecture.md` |
| `MLMHead` | `docs/paper/02-03-architecture.md` |
| `LoRAAdapter` | `docs/paper/03-01-protocol.md` |
| `EmbeddingProbe` | `docs/paper/03-01-protocol.md` |
| All numbers | `docs/paper/key-numbers.md` |
| All equations | `docs/paper/equations.md` |

### The rule for test values

When writing tests, every expected value must appear in
`docs/paper/key-numbers.md` with its source section.

**If a value is not in `key-numbers.md` it is not from the paper.
Do not use it.**

Example — correct:
```python
assert config.d_model == 192  # key-numbers.md: d_model (PRAGMA-S), Table 1
```

Example — wrong:
```python
assert config.d_model == 256  # not in key-numbers.md — where does 256 come from?
```

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

### 6. History Encoder — bidirectional self-attention over concatenated [USR:EVT]

The `HistoryEncoder` receives the concatenated sequence `z = [za : ze]`
(Equation 6, §2.3.4) and applies **bidirectional self-attention** over it.

- `za` — the `[USR]` token from `ProfileStateEncoder` — is placed at position 0.
- `ze` — the `[EVT]` tokens from `EventEncoder` — occupy positions 1..ne.
- Profile conditioning happens because `[USR]` sits at position 0 of the input
  sequence. Every `[EVT]` position attends to it through standard bidirectional
  self-attention. No separate sublayer is needed or specified.

**There is NO cross-attention sublayer in `HistoryEncoder`.**

Do NOT add a cross-attention sublayer.
Do NOT add dedicated cross-attention from `[EVT]` positions to `[USR]`.
Do NOT revert to the old (incorrect) stub that had a cross-attention sublayer.

Reference: §2.3.4, Equation 6, `docs/paper/02-03-architecture.md` §2.3.4,
`docs/paper-to-code.md` §2.3.4, `src/encoders/history_encoder.py` module docstring.

---

## Training stack

| Component | Tool |
|---|---|
| Model implementation | PyTorch (pure) |
| Training orchestration | Custom PyTorch loop (`scripts/train_pragma.py`) |
| Distributed training | torchrun / DDP via KFTO PyTorchJob (`kubeflow.org/v1`) |
| Pipeline orchestration | KFP SDK v2 |
| LoRA fine-tuning | Hugging Face PEFT library |
| Embedding extraction | Custom script + numpy/parquet |

**The training loop is pure PyTorch** (`scripts/train_pragma.py`). The model itself is pure PyTorch.
NeMo AutoModel is not currently used. See ADR 005 (`docs/decisions/005-training-orchestration.md`).

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

## Development Process — Non-Negotiable

Every component implementation in this repo follows
Test-Driven Development (TDD) derived from the PRAGMA paper.

This is not optional. It is the process.

The full process is documented in DEVELOPMENT_PROCESS.md.
Read it before implementing anything.

The short version:
  Tests come from the paper. Always.
  Tests come before implementation. Always.
  Never fix a test to match implementation. Ever.
  The comparison test (PRAGMA vs NVIDIA blueprint AUC)
  is the final arbiter of correctness.

---

## Development branch

All implementation work happens on `pragma-implementation`.
`main` is protected — PRs only, no direct pushes.

```bash
git checkout pragma-implementation
```
