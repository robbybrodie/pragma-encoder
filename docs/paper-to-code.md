# Paper-to-Code Mapping

Explicit section-by-section mapping from arXiv:2604.08649v1 to this codebase.

> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
> arXiv:2604.08649v1

This document helps a reader who has read the paper navigate to the
corresponding implementation in this repository.

---

## Section 2.2 — Key-Value-Time Tokenisation

The core innovation of PRAGMA's input representation. Each transaction field
is tokenised independently using a field-type-specific strategy.

| Paper element | This repo |
|---|---|
| Numerical fields → percentile buckets | `src/tokenizer/numerical.py::NumericalTokenizer` |
| Categorical fields → single token | `src/tokenizer/categorical.py::CategoricalTokenizer` |
| Text fields → BPE subwords | `src/tokenizer/textual.py::TextualTokenizer` |
| Timestamp → log-seconds + calendar | `src/tokenizer/temporal.py::TemporalTokenizer` |
| (key, value, time) assembly | `src/tokenizer/pipeline.py::TokenizerPipeline` |
| Financial-domain pipeline | `src/tokenizer/financial_pipeline.py` |

---

## Section 2.3.2 — Profile State Encoder

Processes static customer attributes. Outputs [USR] token.

| Paper element | This repo |
|---|---|
| Bidirectional Transformer | `src/encoders/profile_state_encoder.py::ProfileStateEncoder` |
| [USR] token | `ProfileStateEncoder.usr_token` parameter |
| RoPE on timestamp positions | `src/encoders/rope.py::RotaryPositionalEmbedding` |

---

## Section 2.3.3 — Event Encoder

Processes a single transaction. Outputs [EVT] token.

| Paper element | This repo |
|---|---|
| Bidirectional Transformer | `src/encoders/event_encoder.py::EventEncoder` |
| [EVT] token | `EventEncoder.evt_token` parameter |
| Calendar token embeddings | `src/encoders/event_encoder.py::CalendarEmbedding` |

---

## Section 2.3.4 — History Encoder

Processes the concatenated sequence `z = [USR : EVT₁ : EVT₂ : ...]` with bidirectional
self-attention. `[USR]` at position 0 conditions all event representations naturally —
there is no cross-attention sublayer and no separate HIST summary token.

| Paper element | This repo |
|---|---|
| Bidirectional self-attention on [USR:EVT] | `src/encoders/history_encoder.py::HistoryEncoder` |
| RoPE on event sequence | `HistoryEncoder.rope` |
| [MASK] embedding for MLM | `HistoryEncoder.mask_embedding` |

---

## Section 2.3.5 — Three-Strategy Masking

| Paper element | This repo |
|---|---|
| All three masking strategies | `src/masking/strategy.py::MaskingStrategy` |
| MLM prediction head | `src/model/mlm_head.py::MLMHead` |
| MLM loss with label smoothing | `src/training/objective.py::MaskedEventModellingLoss` |

---

## Section 2.4 — Training Setup

| Paper element | This repo |
|---|---|
| Sequence packing | `src/training/packing.py::SequencePacker` |
| Dynamic batching | `src/training/batching.py::DynamicBatchSampler` |
| Training entrypoint | `scripts/train_pragma.py` |
| KFP pipeline | `pipeline/pragma_pipeline.py` |
| KFTO PyTorchJob | `openshift/gitops/training/pytorchjob-pragma-s.yaml` |

---

## Section 3.1.1 — Linear Embedding Probe

| Paper element | This repo |
|---|---|
| Frozen encoder + linear head | `src/adaptation/probe.py::EmbeddingProbe` |
| sklearn logistic regression | `src/evaluation/downstream.py::evaluate_linear_probe` |
| Interactive notebook | `notebooks/03_pragma_embedding_probe.ipynb` |

---

## Section 3.1.2 — LoRA Fine-tuning

| Paper element | This repo |
|---|---|
| LoRA adapters | `src/adaptation/lora.py::LoRAAdapter` |
| Interactive notebook | `notebooks/04_pragma_lora_finetuning.ipynb` |

---

## Section 3 — Evaluation Metrics

| Paper metric | This repo |
|---|---|
| AUC (AUROC) | `src/evaluation/metrics.py::compute_auc` |
| PR-AUC | `src/evaluation/metrics.py::compute_pr_auc` |
| F1 score | `src/evaluation/metrics.py::compute_f1` |
| KS statistic | `src/evaluation/metrics.py::compute_ks` |
| Full evaluator | `src/evaluation/downstream.py::DownstreamEvaluator` |
