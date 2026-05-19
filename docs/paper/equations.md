# PRAGMA Equations

All equations from the paper with LaTeX source, variable definitions,
and implementation mapping.

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

---

## Equation 1 — Token Embedding (Section 2.3.1)

$$x = \text{PosEmb}(E(k) + E(v)), \quad x \in \mathbb{R}^{n \times d}$$

**Variables:**
- `E` — shared embedding table mapping key/value token IDs to d-dimensional vectors
- `k` — key token ID (semantic type; one token per field)
- `v` — value token ID (one or more per field, type-dependent)
- `PosEmb` — static sine/cosine positional encoding (Vaswani et al., 2017)
- `n` — number of key–value pairs for this field
- `d` — model hidden dimension (`d_model` in `PRAGMAConfig`)
- `x` — resulting token embedding

**Position indexing:** Positions index values **within a field**, not across fields.
- `Currency: eur` → key=Currency (position 0), value=eur (position 0)
- `Description: metal plan` → key=Description (positions 0,1,2), values=met/al/plan (positions 0,1,2)

**Applies to:** Profile state tokens (`xa ∈ R^(na×d)`) and event tokens (`xe ∈ R^(ne×d)`)

**Implementation:** `src/tokenizer/pipeline.py`
**Test:** `tests/test_tokenizer.py::TestMathProperties::test_embedding_equation`

---

## Equation 2 — Temporal Log Transform (Section 2.2)

$$t' = 8 \cdot \ln\!\left(1 + \frac{t}{8}\right)$$

**Variables:**
- `t` — elapsed time since most recent event, in seconds
- `t'` — transformed temporal coordinate (log-seconds)
- `8` — scale constant that controls the linear-to-log crossover point

**Purpose:** Compresses dynamic range of long historical gaps while preserving
linear granularity for recent events. Prevents aliasing in positional embeddings
caused by extreme temporal gaps.

**Applies to:**
- Profile state encoder temporal coordinates `ta` (log-seconds since life-long event)
- History encoder temporal coordinates `te` (log-seconds to most recent event)
- Non-life-long profile attributes use `t' = 0`
- `[USR]` position in history encoder uses `t' = 0`

**Implementation:** `src/tokenizer/pipeline.py` (temporal coordinate preprocessing)
**Test:** `tests/test_tokenizer.py::TestMathProperties::test_temporal_transform`

---

## Equation 3 — Calendar Feature Embedding (Section 2.3.3)

$$z_t = \text{MLP}_2(\text{sincos}(x_t)), \quad x_t \in \mathbb{R}^{n_e \times 3}$$

**Variables:**
- `xt` — raw calendar features: hour of day, day of week, day of month
- `sincos(·)` — periodic embedding: each scalar → (sin, cos) pair
- `MLP_2` — 2-layer MLP projecting to `d_model` dimensions
- `zt ∈ R^(ne × d)` — calendar feature embeddings (one per event)
- `ne` — number of events in history

**Purpose:** Captures daily and weekly temporal cycles. Periods fixed to known
calendar cycles (not learned). Applied to event history only — not profile state.

**Addition to Event Encoder output:**
$$z_e = z'_e + z_t$$

where `z'e` is the [EVT] token output of the Event Encoder.

**Implementation:** `src/encoders/event_encoder.py` (calendar MLP)
**Test:** `tests/test_event_encoder.py::TestMathProperties::test_calendar_feature_embedding`

---

## Equation 4 — Profile State Encoder Output (Section 2.3.2)

$$z_a \in \mathbb{R}^{n_a \times d}, \quad \text{pass only } z_a[0] \in \mathbb{R}^{1 \times d} \text{ to History Encoder}$$

**Variables:**
- `xa ∈ R^(na×d)` — profile state token embeddings (input)
- `ta ∈ R^na` — temporal coordinates (log-seconds since life-long event; 0 for non-life-long)
- `za ∈ R^(na×d)` — full profile state encoder output
- `za[0] ∈ R^(1×d)` — the [USR] token (first position); the only output passed forward
- RoPE applied to `ta` for positional encoding

**Key constraint:** Only the [USR] token `za` is passed to the History Encoder.
The remaining `na - 1` positions are discarded.

**Implementation:** `src/encoders/profile_state_encoder.py`
**Test:** `tests/test_profile_state_encoder.py::TestShapes::test_usr_token_output_shape`

---

## Equation 5 — Event Encoder Output (Section 2.3.3)

$$\hat{z}_e = \text{EventEncoder}(x_e), \quad z'_e = \hat{z}_e[\text{EVT positions}] \in \mathbb{R}^{n_e \times d}$$

**Variables:**
- `xe = (xe,1, xe,2, ..., xe,ne)` — sequence of event token embeddings
- Each `xe,i ∈ R^(ni×d)` — tokens for event i (different lengths per event)
- `z_hat_e` — full token-level output from Event Encoder (used by MLM head)
- `z'e ∈ R^(ne×d)` — [EVT] tokens (first token of each event); aggregated representations
- `ze = z'e + zt` — calendar-augmented event representations

**Key constraint:** Each event processed **independently** — no attention across events.
Varlen attention kernel used in practice.

**Implementation:** `src/encoders/event_encoder.py`
**Test:** `tests/test_event_encoder.py::TestShapes::test_evt_token_output_shape`

---

## Equation 6 — History Encoder Input (Section 2.3.4)

$$z = [z_a : z_e] \in \mathbb{R}^{(1 + n_e) \times d}$$

**Variables:**
- `za ∈ R^(1×d)` — the [USR] token from Profile State Encoder
- `ze ∈ R^(ne×d)` — calendar-augmented [EVT] tokens from Event Encoder
- `z` — concatenated input to History Encoder (profile + events)
- `te ∈ R^(1+ne)` — temporal coordinates: 0 for [USR] position, log-seconds to last event for each [EVT]

**RoPE applied to `te` for positional encoding.**

**Implementation:** `src/encoders/history_encoder.py`
**Test:** `tests/test_history_encoder.py::TestShapes::test_input_concatenation`

---

## Equation 7 — History Encoder Output (Section 2.3.4)

$$z_h \in \mathbb{R}^{(1 + n_e) \times d}$$

**Variables:**
- `zh` — full History Encoder output sequence
- `zh,0 ∈ R^d` — [USR] token embedding (position 0); user-level representation
- `zh,1, ..., zh,ne ∈ R^d` — [EVT] token embeddings (positions 1..ne); event-level representations
- Used by: MLM head during pre-training; embedding probe; classification head during fine-tuning

**Implementation:** `src/encoders/history_encoder.py`
**Test:** `tests/test_history_encoder.py::TestShapes::test_history_encoder_output_shape`

---

## Equation 8 — MLM Head Input (Section 2.3.5)

For a masked token at position `j` within event `i`:

$$\text{input} = [\hat{z}_{e,i,j} \, : \, z_{h,i} \, : \, z_{h,0}] \in \mathbb{R}^{3d}$$

**Variables:**
- `z_hat_{e,i,j} ∈ R^d` — Event Encoder token-level output at masked position j within event i (local context)
- `zh,i ∈ R^d` — History Encoder [EVT] output for event i (cross-event context)
- `zh,0 ∈ R^d` — History Encoder [USR] output (user-level context)
- `3d` — concatenated dimension fed to MLM head projection

**Projection:** 3d → d → logits over value vocabulary (~28k tokens)

**Implementation:** `src/model/mlm_head.py`
**Test:** `tests/test_mlm_head.py::TestShapes::test_mlm_input_is_3d`

---

## Equation 9 — RoPE Positional Encoding (Section 2.3.2, Su et al. 2024)

RoPE encodes position `t` by rotating query/key vectors:

$$q_m = R(t_m) \cdot q, \quad k_n = R(t_n) \cdot k$$

$$\text{attention}(q_m, k_n) \propto q_m^T k_n = q^T R(t_m)^T R(t_n) k = q^T R(t_n - t_m) k$$

**Key property:** The dot product depends only on the **relative** temporal distance `(tn - tm)`,
not absolute positions. Closer events produce higher dot-product similarity.

**Variables:**
- `R(t)` — rotation matrix parameterised by temporal coordinate t
- `qm, kn` — rotated query and key vectors at temporal positions m, n
- `tm, tn` — temporal coordinates (log-seconds)

**Used in:**
- `ProfileStateEncoder`: RoPE on `ta` (log-seconds since life-long event)
- `HistoryEncoder`: RoPE on `te` (log-seconds to most recent event)
- NOT used in `EventEncoder` (uses calendar features instead)

**Key constraint (ADR 001 §3):** Must NOT mix temporal RoPE with within-field PosEmb.
Within-field positions use standard sine/cosine PosEmb (Equation 1).

**Implementation:** `src/encoders/rope.py`
**Test:** `tests/test_rope.py::TestMathProperties::test_rope_temporal_decay`
