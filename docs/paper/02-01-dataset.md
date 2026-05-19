# Section 2.1 — Dataset

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.1

---

## Dataset Scale

| Metric | Value |
|--------|-------|
| User records | 26 M |
| Countries | 111 |
| Total events | 24 B |
| Total tokens | 207 B |
| Pre-training time range | 25 months (2023–2025) |

All data is fully anonymised with no personally identifiable information.

---

## 2.1.1 Event History

An event is defined by:
- A `created` timestamp
- A set of key–value pairs (e.g., `Direction: out`)

Event source types (broadly grouped):
- **transactions** (card payments, transfers, topups)
- **app** (in-app navigation, screen views)
- **trading** (buy/sell orders)
- **communications** (emails, push notifications)

Event schemas are specific to source type. Example: `Symbol` key is unique to trading events.

No additional statistical filtering or pre-processing (no outlier removal, no vocabulary
pruning) applied beyond anonymisation and eligibility criteria.

---

## 2.1.2 Profile State

Profile state = static customer attributes at the evaluation point, e.g.:
- `Plan: metal`
- `Region: uk`
- `Balance: 6,012.54`
- `Currency: gbp`
- `Age: 25–35`

Profile state is represented in **event-like format** (key–value pairs with timestamp).

**Life-long events:** Profile state is augmented with life-long events — key–value pairs
that each carry an individual timestamp recording a first occurrence.

Example: `Lifelong: first_topup at 20-11-02 12:09:04`

Life-long events enable the model to encode the timing of historical milestones
(e.g., account age) even when those events fall outside the truncated event history window.

**ADR 001 §6:** Profile state is a separate input to the Profile State Encoder.
It is never mixed with the event sequence. Life-long events are encoded as profile state
with individual timestamps, not as regular events in the history.

---

## 2.1.3 Pre-training Time Range

Selected range: **25 months (2023–2025)**

Trade-offs considered:
- Older events may reflect obsolete patterns → distribution shift at inference time
- Shorter range risks missing out-of-distribution temporal patterns
- Transformer context spans are hardware-bounded

---

## Key Numbers (used in tests)

| Number | Value | Implementation |
|--------|-------|----------------|
| max_events | 6,500 | `PRAGMAConfig.max_events` |
| max_event_tokens | 24 | `PRAGMAConfig.max_event_tokens` |
| max_profile_tokens | 200 | `PRAGMAConfig.max_profile_tokens` |

*The truncation numbers are in §2.4 but apply to the dataset described here.*

---

## Implementation Notes

The dataset section defines the two input streams to PRAGMA:

1. **Event history** → processed by `EventEncoder` (§2.3.3)
2. **Profile state** (including life-long events) → processed by `ProfileStateEncoder` (§2.3.2)

These are never merged into one sequence (ADR 001 §6).

The `PRAGMATokenizer` must handle all event source types with the same tokenisation scheme
described in §2.2. No special handling per source type at the tokeniser level.
