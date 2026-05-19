# Decision 001: PRAGMA Encoder Architecture

Status: Accepted
Date: 2026-05-17
Paper reference: Sections 2.1–2.3
Supersedes: nothing

---

## Context

The adjacent NVIDIA blueprint uses a decoder-only (GPT/Llama-style) architecture
with causal language modelling. A decision was needed on whether to follow the
same approach or implement the PRAGMA architecture as described in the paper.

The PRAGMA paper (Ostroukhov et al., 2026, arXiv:2604.08649v1) describes a
purpose-built encoder-only foundation model for financial transaction data, with
a distinctive key-value-time tokenisation scheme and three separate encoder branches.

This ADR captures the core architecture decisions derived from Sections 2.1–2.3.
They are non-negotiable: contradicting them produces a model that is not PRAGMA.

---

## Decision

**Use the encoder-only (BERT-style) bidirectional Transformer architecture
exactly as described in the paper. Do not implement a decoder-only or causal
architecture.**

The following architectural constraints all follow from this primary decision
and are equally binding.

---

## Architectural Constraints

### 1. Encoder-only, bidirectional attention (§2.3)

Use bidirectional self-attention throughout. Every token attends to every other
token in both directions. Do not use a causal (left-to-right) attention mask
anywhere in the model.

**What future sessions must not contradict:**
- Do not add causal masking to any attention layer.
- Do not switch to a decoder architecture.
- Do not use GPT-style autoregressive generation.

---

### 2. Three separate encoder branches (§2.3.2–2.3.4)

Use three separate encoder branches with independent weights:

1. **Profile State Encoder** (§2.3.2) — static customer attributes
2. **Event Encoder** (§2.3.3) — per-event token sequence
3. **History Encoder** (§2.3.4) — event sequence, conditioned on `[USR]`

Each is a separate bidirectional Transformer. The `[USR]` token produced by the
Profile State Encoder and the `[EVT]` tokens produced by the Event Encoder are
the only outputs passed to the History Encoder. Raw profile or event tokens are
never passed directly to the History Encoder.

Code locations:
- `src/encoders/profile_state_encoder.py`
- `src/encoders/event_encoder.py`
- `src/encoders/history_encoder.py`

**What future sessions must not contradict:**
- Do not merge the three encoders into one.
- Do not pass raw profile tokens to the History Encoder.
- Do not pass raw event tokens to the History Encoder.
- Only the aggregated `[USR]` and `[EVT]` tokens are passed.

---

### 3. RoPE positional encoding (§2.3.2, Su et al. 2024)

Use Rotary Position Embeddings (RoPE) for temporal coordinate encoding in:
- Profile State Encoder (timestamp positions)
- History Encoder (event sequence positions)

Use **calendar feature embeddings** (periodic functions) for the Event Encoder's
within-event temporal signals — not RoPE.

RoPE encodes relative temporal positions naturally and extrapolates to unseen
temporal gaps. The temporal coordinate `t` (log-seconds to most recent event)
must be provided to every RoPE-using encoder as a separate argument.

Code: `src/encoders/rope.py`

**What future sessions must not contradict:**
- Do not replace RoPE with sinusoidal positional encoding.
- Do not replace RoPE with learned positional embeddings.
- Do not mix the temporal RoPE with within-field PosEmb.
- Within-field positions use standard sinusoidal PosEmb per §2.3.1.
  This is separate from temporal RoPE and must not be confused with it.

---

### 4. Masked event modelling objective (§2.3.5)

Use masked modelling (MLM) with three masking strategies:

| Strategy | Probability |
|----------|-------------|
| Token-level masking | 15% |
| Event-level masking | 10% |
| Semantic-type (key) level masking | 10% |

A small fraction of selected positions are replaced with `[UNK]` instead of
`[MASK]` for an input dropout effect.

The MLM head receives a concatenation of three `d_model`-dimensional vectors:
- Event Encoder token-level output at masked position
- History Encoder `[EVT]` token for cross-event context
- History Encoder `[USR]` token for user-level context

This is a pre-training objective. Do not switch to causal language modelling.

Code: `src/masking/strategy.py`, `src/model/mlm_head.py`

**What future sessions must not contradict:**
- Do not switch to a causal LM objective.
- Do not change masking rates without a new ADR.
- Do not simplify the MLM head to use one input vector.
- All three strategies (token, event, key) must be implemented.

---

### 5. Key-value-time tokenisation (§2.2)

Use PRAGMA's key-value-time tokenisation scheme:

| Field type | Encoding |
|------------|----------|
| Numerical | Percentile bucket token |
| Categorical | Single token |
| Textual | BPE subword tokens |
| Timestamp | Log-seconds + calendar features |

Each transaction field produces a (key, value) token pair. Key and value are
separate embedding lookups in the same embedding table `E`. This is fundamentally
different from the NVIDIA blueprint (which serialises transactions as text strings).

Vocabulary structure (§2.2):
- ~60 key tokens (one per field type)
- ~28k value tokens (numerical buckets, categoricals, BPE subwords)
- Total embedding vocab = key_size + value_size + special tokens

Equation 1 (§2.3.1): `x_ij = PosEmb_j(E(k_i) + E(v_ij))`

Code: `src/tokenizer/`

**What future sessions must not contradict:**
- Do not reuse the NVIDIA blueprint tokeniser.
- Do not serialise records as text strings.
- Keys and values must be separate embedding lookups in the same table `E`.
- Within-field positions index values within one field only (not across fields).

---

### 6. Profile state is a separate input stream (§2.1.2)

Profile state (static customer attributes: plan, region, balance quantile,
life-long events such as `first_topup`, `account_age`) is a separate input
to the Profile State Encoder. It is never mixed with the event sequence.

Life-long events are encoded as profile state with individual timestamps, not
as regular events in the history.

**What future sessions must not contradict:**
- Do not merge profile state into the event sequence.
- Do not pass profile tokens to the Event Encoder.
- Profile State Encoder and Event Encoder are independent.
- Two separate inputs to the model: profile state and event history.

---

## Consequences

Enables:
- Bidirectional context for each masked position during pre-training
- Clean ablation (removing profile branch) as tested in §3.4.2
- Direct implementation of the paper as published
- Better embedding quality for downstream fine-tuning tasks

Constrains:
- Cannot generate text (no autoregressive decoding)
- Three separate forward passes required per inference
- Training uses MLM objective, not next-token prediction
- NeMo and other causal-LM frameworks cannot be used for pre-training without adaptation

---

## What this ADR does NOT cover

- Training orchestration (framework, distributed strategy, cluster) → ADR 005
- LoRA / PEFT fine-tuning → ADR 006
- Workbench Python API → ADR 003
- Decorated pipeline authoring → ADR 004
- Embedding assembler architecture → ADR 002

---

## References

- Paper: Ostroukhov et al. (2026), arXiv:2604.08649v1, Sections 2.1–2.3
- `docs/paper/02-03-architecture.md`
- `src/encoders/`, `src/masking/`, `src/model/mlm_head.py`, `src/tokenizer/`
