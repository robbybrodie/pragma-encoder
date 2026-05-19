# Decision 006: LoRA / PEFT Fine-Tuning

Status: Accepted
Date: 2026-05-17 (supersedes embedded Decision 006 in old ADR 001)
Paper reference: Section 3.1.2
Supersedes: embedded "Decision 006: PEFT/LoRA for Fine-Tuning" in the original ADR 001 file

---

## Context

The PRAGMA paper (§3.1.2) describes parameter-efficient fine-tuning via
Low-Rank Adaptation (LoRA), updating only 2–4% of model weights for
downstream tasks (credit scoring, fraud detection, churn prediction).

A decision was needed on whether to implement LoRA from scratch or use
an existing library.

---

## Decision

Use the **Hugging Face PEFT library** for LoRA fine-tuning. Do not implement
LoRA from scratch.

PEFT provides production-quality LoRA with support for adapter saving,
loading, and merging. It handles the low-rank delta injection, gradient
masking of frozen weights, and adapter serialisation consistently with the
broader Hugging Face ecosystem.

---

## Implementation Status

**Implemented** in `src/adaptation/lora.py`.

The implementation uses `peft.LoraConfig` with `task_type=TaskType.FEATURE_EXTRACTION`
(PRAGMA is encoder-only; there is no causal generation head).

### LoRA configuration (sourced from `PRAGMAConfig`)

| Parameter | Value | Source |
|-----------|-------|--------|
| `r` (rank) | 8 | Paper §3.1.2 |
| `lora_alpha` | 8 | Paper §3.1.2 |
| `lora_dropout` | 0.0 | Implementation choice; paper does not specify |
| `bias` | `"none"` | Standard PEFT default |

### Target modules (verified against PRAGMA module paths)

| Module | Layer |
|--------|-------|
| `q_proj` | Q projection in `_RoPEMultiheadAttention` / `_EventAttention` |
| `k_proj` | K projection |
| `v_proj` | V projection |
| `out_proj` | Output projection |
| `ff.0` | First FFN linear (`d_model → d_ffn`) |
| `ff.2` | Second FFN linear (`d_ffn → d_model`) |

PEFT uses suffix matching, so `"q_proj"` matches all three encoders
(profile, event, history) without per-encoder path prefixes.

### Excluded modules (by design)

| Module | Reason |
|--------|--------|
| `event_encoder.calendar_mlp.mlp.*` | Feature embedding, not a transformer layer |
| `mlm_head.proj` / `mlm_head.decoder` | Prediction head, not an encoder |

### Parameter coverage

For PRAGMA-S with `rank=8`, `alpha=8`:

> LoRA params: 221,184 / 9,334,432 total = **2.37%** — within paper's 2–4% range.

---

## Companion: EmbeddingProbe (§3.1.1)

`src/adaptation/probe.py` implements the linear embedding probe described in
§3.1.1. The probe is separate from LoRA: it is used for evaluation
(embedding quality assessment) not fine-tuning.

---

## Consequences

Enables:
- Minimal parameter overhead per downstream task (2–4% of total)
- Frozen backbone shared across tasks
- Fast specialisation without catastrophic forgetting
- Adapter save/load for task-specific deployments
- Compatible with `peft.get_peft_model()` workflow

Constrains:
- Must use PEFT library, not a custom LoRA implementation
- `rank` and `alpha` defaults are 8, sourced from `PRAGMAConfig`
- Target modules must include QKV and FFN layers
- Cannot use full fine-tuning without a new ADR
- `task_type=FEATURE_EXTRACTION` must be maintained (encoder-only; no causal head)

---

## What future sessions must not contradict

- Do not implement custom LoRA. Use PEFT.
- Do not change `rank` / `alpha` defaults without paper justification.
- Do not add causal generation head adapters (contradicts ADR 001).
- `lora_dropout` may be tuned, but changes must be documented.

---

## References

- Paper: §3.1.2 (LoRA fine-tuning protocol)
- LoRA paper: Hu et al. (2022), arXiv:2106.09685
- PEFT library: https://github.com/huggingface/peft
- `src/adaptation/lora.py`
- `src/adaptation/probe.py`
- ADR 001: `docs/decisions/001-encoder-architecture.md` (encoder-only constraint)
