# Section 2.3 — Model Architecture (CRITICAL)

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.3

**This is the primary implementation reference.** Read before implementing any encoder.

---

## Overview

PRAGMA is an **encoder-only Transformer** with:
- Masked modelling (MLM) pre-training objective
- Three encoder blocks: ProfileStateEncoder, EventEncoder, HistoryEncoder
- Output: dense record-level embeddings

All size variants share:
- **Activation:** GELU (Hendrycks et al., 2016)
- **Normalisation:** pre-norm LayerNorm (Xiong et al., 2020)
- **Dropout:** 0.1

Parametric family defined in Table 1 (see `docs/paper/table-1-model-sizes.md`).

---

## 2.3.1 Token Embedding

**Equation 1:** `x = PosEmb(E(k) + E(v)), x ∈ R^(n×d)`

- Shared embedding table `E` maps key and value token IDs to d-dimensional vectors
- Key and value embeddings are **summed** (not concatenated)
- Augmented with **static sine/cosine PosEmb** (Vaswani et al., 2017)
- Positions index **within-field**, not across fields
- For multi-valued fields: key token is replicated to match each value
- A learnable `[USR]` (or `[EVT]`) token is **prepended** to each sequence

Notation:
- `xa ∈ R^(na×d)` — profile state token embeddings
- `xe ∈ R^(ne×d)` — event token embeddings (ne = number of events)

**Implementation:** `src/tokenizer/pipeline.py`

---

## 2.3.2 Profile State Encoder

**What it is:** Bidirectional Transformer

**Input:**
- `xa ∈ R^(na×d)` — profile state token embeddings
- `ta ∈ R^na` — temporal coordinates (log-seconds since life-long event; 0 for non-life-long)

**Positional encoding:** RoPE (Su et al., 2024) applied to `ta`

**Key constraint:** RoPE temporal encoding is **separate** from the within-field
PosEmb (Equation 1) to avoid semantic and scale mismatch.

**Output:**
- `za ∈ R^(na×d)` — full encoder output sequence
- Only `za[0] ∈ R^(1×d)` (the `[USR]` token) is passed to the History Encoder

**Implementation:** `src/encoders/profile_state_encoder.py`

---

## 2.3.3 Event Encoder

**What it is:** Bidirectional Transformer (same architecture as Profile State Encoder)

**Input:**
- `xe = (xe,1, xe,2, ..., xe,ne)` — sequence of event token embeddings
- Each event `xe,i ∈ R^(ni×d)` (different number of tokens per event)
- Each event processed **independently** of all other events

**Independence constraint:** No attention across events at this stage.
Implemented with a varlen attention kernel (Dao et al., 2022).

**Outputs:**
1. `z_hat_e` — token-level embeddings for all events (used by MLM head during pre-training)
2. `z'e ∈ R^(ne×d)` — the `[EVT]` token (first token) for each event

**Calendar feature addition:**
- Calendar features `xt ∈ R^(ne×3)` → sine/cosine → 2-layer MLP → `zt ∈ R^(ne×d)`
- Final event representation: `ze = z'e + zt`

**Implementation:** `src/encoders/event_encoder.py`

---

## 2.3.4 History Encoder

**What it is:** Bidirectional Transformer

**Input:**
- `z = [za : ze] ∈ R^((1+ne)×d)` — concatenation of [USR] token + calendar-augmented [EVT] tokens
- `te ∈ R^(1+ne)` — temporal coordinates:
  - Position 0 (za/[USR]): te = 0
  - Positions 1..ne ([EVT] tokens): log-seconds to most recent event

**Positional encoding:** RoPE applied to `te`

**Output:**
- `zh ∈ R^((1+ne)×d)` — full history encoder output
- `zh,0` — [USR] token (user-level representation)
- `zh,1, ..., zh,ne` — [EVT] tokens (per-event representations)

**Usage:**
- MLM head during pre-training (uses `zh` and `z_hat_e`)
- Embedding probe (uses `zh,0` = [USR] and/or `zh,ne` = final [EVT])
- Classification head during fine-tuning (uses `zh,0`)

**Implementation:** `src/encoders/history_encoder.py`

---

## 2.3.5 Training

### Pre-training Objective (MLM)

Following BERT. A random subset of event input tokens is masked. The model reconstructs
the original tokens.

**MLM head input for each masked token at position j in event i:**
```
[z_hat_{e,i,j} : zh,i : zh,0] ∈ R^(3d)
```
- `z_hat_{e,i,j}` — Event Encoder token-level output at masked position (local context)
- `zh,i` — History Encoder [EVT] output for event i (cross-event context)
- `zh,0` — History Encoder [USR] output (user-level context)

**Projection:** 3d → d → logits over value vocabulary (~28k)

**Loss:** Cross-entropy with label smoothing

### Masking Strategy

Three sources (all applied during pre-training):

| Strategy | Probability | What is masked |
|----------|-------------|----------------|
| Token-level | **15%** | Individual tokens within events |
| Event-level | **10%** | All tokens within an entire event |
| Semantic-type (key)-level | **10%** | All values of selected keys |

Additionally: a small fraction of selected positions replaced with `[UNK]` instead
of `[MASK]`. These are excluded from MLM objective (no gradient) → acts as input dropout.
Trains model to recover values under stronger corruption. Avoids `[MASK]`-at-inference mismatch.

### Downstream Adaptation

Two modes:

1. **Embedding probe:** frozen `zh` → linear probe (logistic/linear regression)
2. **LoRA fine-tuning:** update ~2–4% of weights via Low-Rank Adaptation

**Implementation:**
- `src/masking/strategy.py` → `MaskingStrategy`
- `src/model/mlm_head.py` → `MLMHead`
- `src/adaptation/probe.py` → `EmbeddingProbe`
- `src/adaptation/lora.py` → `LoRAAdapter`

---

## Key Numbers (used in tests)

| Number | Value | Implementation |
|--------|-------|----------------|
| token_mask_prob | 0.15 | `PRAGMAConfig.token_mask_prob` |
| event_mask_prob | 0.10 | `PRAGMAConfig.event_mask_prob` |
| key_mask_prob | 0.10 | `PRAGMAConfig.key_mask_prob` |
| mlm_head_input_dim | 3 × d_model | `MLMHead` |
| dropout | 0.1 | `PRAGMAConfig.dropout` |

---

## Implementation Notes

The three encoders must remain **separate**:
- They have independent weights
- Independent positional encoding strategies
- Independent forward passes

Only aggregated tokens pass between encoders:
- Profile State Encoder → [USR] token only → History Encoder
- Event Encoder → [EVT] tokens only → History Encoder
- Raw tokens never cross encoder boundaries (ADR 001 §2)

The MLM head receives concatenation of **three** d-dimensional vectors (not one).
Do not simplify. The 3d input is specified in the paper (ADR 001 §4).
