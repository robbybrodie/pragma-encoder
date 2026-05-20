# Section 2.2 — Tokenisation (CRITICAL)

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.2

**This section is critical for implementation.** ADR 001 §5 is derived entirely from it.
Do not reuse the NVIDIA blueprint tokeniser.

---

## Core Concept: Disentangled Embedding Space

Unlike standard LLMs that treat everything as text, PRAGMA uses a **disentangled
embedding space** that represents each data point by three components:

1. **Semantic type (key)** — the meaning of a field
2. **Value** — the content of the field
3. **Temporal coordinate** — when the event occurred

Example: `Channel: email at 24-04-07 19:20:18`
→ key=`Channel`, value=`email`, temporal=log-seconds

---

## Semantic Type (Key) Tokenisation

- All semantic types (keys) tokenised as **single tokens**
- Profile state and event semantic types encoded the same way
- Vocabulary size: **~60 tokens** ← `PRAGMAConfig.key_vocab_size`
- One token per field, regardless of value count

---

## Value Tokenisation

Three value types (determined by cardinality threshold):

### 1. Numerical values
- Mapped to **percentile buckets**
- Bin boundaries learned from training data
- Extra bucket for zero
- **One token per numerical value**

### 2. Categorical values
- Cardinality below threshold → treated as categorical
- Manually selected from text fields to prevent splitting common values
  (e.g., MCC codes)
- **One token per categorical value**

### 3. Textual values
- Cardinality above threshold → treated as textual
- Tokenised with **BPE-style subword tokeniser** (Sennrich et al., 2016)
- Reserved `[UNK]` token for rare unseen fragments
- BPE tokeniser trained on the transaction corpus

Total value vocabulary: **~28,000 tokens** ← `PRAGMAConfig.value_vocab_size`

---

## Position Indexing (Critical detail — Equation 1)

**Positions index values WITHIN a field, not across fields.**

Example:
```
Currency: eur         → key=Currency (pos 0), value=eur (pos 0)
Description: metal plan → key=Description (pos 0,1,2), values=met/al/plan (pos 0,1,2)
```

The key token is **replicated** to match each of its values for multi-valued fields.
This yields `n` key–value pairs total for a field with n value tokens.

For the embedding equation: see `docs/paper/equations.md` Equation 1.

---

## Temporal Information — Two Encoding Schemes

### 1. Log-seconds (RoPE temporal coordinate)

Applied to: all events (both profile state and event history)

Transform (Equation 2):
$$t' = 8 \cdot \ln\!\left(1 + \frac{t}{8}\right)$$

where `t` = elapsed time since most recent event, in seconds.

Purpose: Compresses dynamic range while preserving linear granularity for recent events.
Used as the temporal coordinate for RoPE positional encoding.

### 2. Calendar features (Event Encoder only)

Applied to: event history only (not profile state)

Features:
- Hour of day
- Day of week
- Day of month

Encoding: periodic (sine/cosine functions) with **periods fixed to known calendar cycles**
(not learned). Then embedded with a 2-layer MLP into `d_model` dimensions.

These capture daily and weekly temporal cycles. Not applied to profile state (life-long
events already captured by log-seconds RoPE).

---

## Key Numbers (used in tests)

| Number | Value | Implementation |
|--------|-------|----------------|
| key_vocab_size | ~60 | `PRAGMAConfig.key_vocab_size` |
| value_vocab_size | ~28,000 | `PRAGMAConfig.value_vocab_size` |
| temporal_transform_scale | 8 | `src/tokenizer/pipeline.py` |
| calendar_feature_dims | 3 | `EventEncoder` calendar MLP input |
| calendar_mlp_layers | 2 | `EventEncoder` calendar MLP |

---

## Implementation Notes

`src/tokenizer/pipeline.py` → `PRAGMATokenizer`

The tokeniser must implement:
1. Key tokenisation (single token lookup from key vocabulary)
2. Value tokenisation (numerical → percentile bucket; categorical → single token; text → BPE)
3. Temporal coordinate computation: `8 * ln(1 + t/8)` where t is in seconds
4. Calendar feature extraction: (hour, day_of_week, day_of_month) → periodic functions

The tokeniser produces:
- Key token IDs (from key vocabulary, ~60 tokens)
- Value token IDs (from value vocabulary, ~28k tokens)
- Position indices (within-field, not across fields)
- Temporal coordinates (log-seconds, one per event)
- Calendar features (3 per event, for EventEncoder only)

**ADR 001 §5:** Do not reuse the NVIDIA blueprint tokeniser. Do not serialise records as text strings.
Keys and values must be separate embedding lookups. Within-field positions index values within one field only.
