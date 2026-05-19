# Section 2.4 — Training Infrastructure

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4

---

## Scale Challenge

Pre-training on **207B tokens spanning 24B user events** requires specialised
storage, batching, and truncation strategies.

---

## Data Storage

Two-level structure:

1. **User index** — LMDB-backed key-value store
   - Maps each user to their tokenised profile state
   - Per-user token statistics

2. **Event shards** — Parquet files
   - Partitioned by event count (each file: users with same number of events)
   - Workers stream event shards independently
   - Profile state looked up on demand from user index

---

## Batching — Dynamic Batching with Fixed Token Budget

Problem: event histories vary from a handful to thousands of events.
Naive padding wastes majority of compute on padding tokens.

Solution: **Shard-based dynamic batching**
- Records from same shard (same event count) → uniform-length History Encoder tensor
- No ragged or padded dimensions for History Encoder
- Greedy packing: records packed until fixed GPU token budget reached

---

## Sequence Packing

Problem: within a batch, individual events still vary in token count.

Solution: **Varlen attention kernel** (Dao et al., 2022 — FlashAttention)
- All event tokens packed into flat buffer
- Variable-length attention: tokens from different events do not attend to each other
- Eliminates padding along both event and token axes

**Result:** 2–5× throughput improvement vs padded baseline (depending on sequence length distribution)

---

## Truncation Limits

| Level | Limit | Fraction affected |
|-------|-------|------------------|
| Event-level (tokens per event) | **24 tokens** | 0.01% of events |
| Profile state | **200 tokens** | — |
| History (events per user) | **6,500 events** | subsampled by recency |
| Users with zero events | discarded | — |

---

## Pre-training Compute

| Variant | GPUs | Convergence |
|---------|------|-------------|
| PRAGMA-S (10M) | 16× NVIDIA H100 | ~2 days |
| PRAGMA-M (100M) | 16× NVIDIA H100 | ~2 weeks |
| PRAGMA-L (1B) | 32× NVIDIA H100 | ~2 weeks |

- **Precision:** bf16 mixed precision
- **Optimiser:** Muon + AdamW (Loshchilov et al., 2019; Jordan, 2024; Liu et al., 2025)

---

## Key Numbers (used in PRAGMAConfig)

| Number | Value | Implementation |
|--------|-------|----------------|
| max_event_tokens | 24 | `PRAGMAConfig.max_event_tokens` |
| max_profile_tokens | 200 | `PRAGMAConfig.max_profile_tokens` |
| max_events | 6,500 | `PRAGMAConfig.max_events` |
| throughput_improvement | 2–5× | documentation only |

---

## Implementation Notes

For this open-source implementation, a custom PyTorch training loop
(`scripts/train_pragma.py`) handles training orchestration (ADR 005).
The model itself is pure PyTorch. NeMo AutoModel is not currently used;
it remains a future option if scaling requirements demand it.

The varlen attention kernel detail is an infrastructure optimisation that can be
implemented in a later phase. Initial implementation can use standard padded attention
for correctness, then optimise.

The truncation limits (max_event_tokens=24, max_profile_tokens=200, max_events=6500)
are configuration values that must live in `PRAGMAConfig`.

The training infrastructure section is documented here for reference.
The pipeline/ directory handles data loading. See `src/training/` for training code.
