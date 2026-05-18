# PRAGMA Architecture

Maps the PRAGMA paper architecture to this repository's implementation.

> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
> arXiv:2604.08649v1

---

## Overview

PRAGMA is an **encoder-only** foundation model for financial transaction data.
It is NOT a decoder-only model. It uses **bidirectional attention** throughout.

```
Input: Customer profile + transaction history
         │
         ├─► Profile State Encoder ──► [USR] ──────────────────┐
         │                                                       │ z = [USR : EVT₁ : EVT₂ : ...]
         └─► Event Encoder (×N events) ──► [EVT] reprs ────────┘
                                                                 │
                                           History Encoder (bidirectional self-attention,
                                                            [USR] at position 0)
                                                                 │
                                           Contextualised history (Section 2.3.4)
                                                                 │
                                       ┌─────────────────────────┴─────────────────────────┐
                                   Linear probe                               LoRA fine-tuning
                                  (Section 3.1.1)                           (Section 3.1.2)
```

---

## Paper Section → Code Mapping

| Paper Section | Description | Code |
|---|---|---|
| 2.2 | Key-value-time tokenisation | `src/tokenizer/` |
| 2.2 | Numerical (percentile buckets) | `src/tokenizer/numerical.py` |
| 2.2 | Categorical (single token) | `src/tokenizer/categorical.py` |
| 2.2 | Textual (BPE subword) | `src/tokenizer/textual.py` |
| 2.2 | Temporal (log-seconds + calendar) | `src/tokenizer/temporal.py` |
| 2.2 | Unified pipeline | `src/tokenizer/pipeline.py` |
| 2.3.1 | Architecture hyperparameters | `src/model/config.py` |
| 2.3.2 | Profile State Encoder | `src/encoders/profile_state_encoder.py` |
| 2.3.3 | Event Encoder | `src/encoders/event_encoder.py` |
| 2.3.3 | Calendar embeddings | `src/encoders/event_encoder.py::CalendarEmbedding` |
| 2.3.4 | History Encoder | `src/encoders/history_encoder.py` |
| 2.3.4 | RoPE positional encoding | `src/encoders/rope.py` |
| 2.3.5 | Three-strategy masking | `src/masking/strategy.py` |
| 2.3.5 | MLM head | `src/model/mlm_head.py` |
| 2.3.5 | MLM loss with label smoothing | `src/training/objective.py` |
| 2.4 | Sequence packing | `src/training/packing.py` |
| 2.4 | Dynamic batching | `src/training/batching.py` |
| 3 | Full PRAGMA model | `src/model/pragma.py` |
| 3.1.1 | Linear embedding probe | `src/adaptation/probe.py` |
| 3.1.2 | LoRA fine-tuning | `src/adaptation/lora.py` |
| 3 | Evaluation metrics | `src/evaluation/metrics.py` |
| 3 | Downstream evaluator | `src/evaluation/downstream.py` |

---

## Comparison with NVIDIA Blueprint (adjacent repo)

| Property | PRAGMA (this repo) | NVIDIA Blueprint (adjacent) |
|---|---|---|
| Architecture | Encoder-only | Decoder-only (GPT-style) |
| Attention | Bidirectional | Causal (left-to-right) |
| Objective | Masked modelling | Causal language modelling |
| Tokenisation | Key-value-time | Tabular text serialisation |
| Positional encoding | RoPE | Absolute (learned) |
| Encoders | 3 separate (profile/event/history) | 1 monolithic |
| Fine-tuning | LoRA via PEFT | Full fine-tuning |
| Scale (paper) | 207B tokens, 10M–1B params | Varies |

---

## Key Architectural Constraints

These are non-negotiable from the paper:

1. **Encoder-only** — Never use a causal attention mask.
2. **Bidirectional** — All three encoders see the full context.
3. **RoPE** — Not sinusoidal, not absolute positional embeddings.
4. **Three separate encoders** — Profile, Event, History have separate weights.
5. **Key-value-time tokenisation** — Not text serialisation.
6. **Masked modelling** — Not causal language modelling.
