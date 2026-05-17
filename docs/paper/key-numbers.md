# PRAGMA Key Numbers

Every specific number from the paper with its source section and implementation target.

**Rule:** If a value is not in this table, it is not from the paper. Do not use it in tests.

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

---

## Architectural Parameters (Table 1)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| d_model (PRAGMA-S) | 192 | Table 1 | `PRAGMAConfig.pragma_s()` |
| d_ffn (PRAGMA-S) | 768 | Table 1 | `PRAGMAConfig.pragma_s()` |
| profile_encoder_layers (PRAGMA-S) | 1 | Table 1 | `PRAGMAConfig.pragma_s()` |
| event_encoder_layers (PRAGMA-S) | 5 | Table 1 | `PRAGMAConfig.pragma_s()` |
| history_encoder_layers (PRAGMA-S) | 2 | Table 1 | `PRAGMAConfig.pragma_s()` |
| n_heads (PRAGMA-S) | 3 | Table 1 | `PRAGMAConfig.pragma_s()` |
| params (PRAGMA-S) | 10 M | Table 1 | `tests/test_config.py::TestPaperSpecifications` |
| d_model (PRAGMA-M) | 512 | Table 1 | `PRAGMAConfig.pragma_m()` |
| d_ffn (PRAGMA-M) | 2048 | Table 1 | `PRAGMAConfig.pragma_m()` |
| profile_encoder_layers (PRAGMA-M) | 3 | Table 1 | `PRAGMAConfig.pragma_m()` |
| event_encoder_layers (PRAGMA-M) | 16 | Table 1 | `PRAGMAConfig.pragma_m()` |
| history_encoder_layers (PRAGMA-M) | 6 | Table 1 | `PRAGMAConfig.pragma_m()` |
| n_heads (PRAGMA-M) | 8 | Table 1 | `PRAGMAConfig.pragma_m()` |
| params (PRAGMA-M) | 100 M | Table 1 | `tests/test_config.py::TestPaperSpecifications` |
| d_model (PRAGMA-L) | 1024 | Table 1 | `PRAGMAConfig.pragma_l()` |
| d_ffn (PRAGMA-L) | 4096 | Table 1 | `PRAGMAConfig.pragma_l()` |
| profile_encoder_layers (PRAGMA-L) | 9 | Table 1 | `PRAGMAConfig.pragma_l()` |
| event_encoder_layers (PRAGMA-L) | 45 | Table 1 | `PRAGMAConfig.pragma_l()` |
| history_encoder_layers (PRAGMA-L) | 18 | Table 1 | `PRAGMAConfig.pragma_l()` |
| n_heads (PRAGMA-L) | 16 | Table 1 | `PRAGMAConfig.pragma_l()` |
| params (PRAGMA-L) | 1 B | Table 1 | `tests/test_config.py::TestPaperSpecifications` |

### Derived invariants (all variants)

| Number | Value | Derived from | Implementation |
|--------|-------|-------------|----------------|
| head_dimension | 64 | d_model / n_heads | `tests/test_config.py::TestMathProperties::test_head_dimension_is_64` |
| ffn_ratio | 4 | d_ffn / d_model | `tests/test_config.py::TestMathProperties::test_ffn_is_four_times_dmodel` |

---

## Activation and Normalisation (Section 2.3)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| activation | GELU | §2.3 | All encoder modules |
| normalisation | pre-norm LayerNorm | §2.3 | All transformer layers |
| dropout | 0.1 | §2.3 | `PRAGMAConfig.dropout` |

---

## Tokenisation (Section 2.2)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| key_vocab_size | ~60 | §2.2 | `PRAGMATokenizer` / `PRAGMAConfig.key_vocab_size` |
| value_vocab_size | ~28,000 | §2.2 | `PRAGMATokenizer` / `PRAGMAConfig.value_vocab_size` |
| temporal_transform | 8 · ln(1 + t/8) | §2.2 | `src/tokenizer/pipeline.py` |
| calendar_feature_dims | 3 (hour, day_of_week, day_of_month) | §2.2 | `EventEncoder` calendar MLP |
| calendar_feature_embedding_layers | 2 (MLP) | §2.3.3 | `EventEncoder` |

---

## Truncation Limits (Section 2.4)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| max_event_tokens | 24 | §2.4 | `PRAGMAConfig.max_event_tokens` |
| max_profile_tokens | 200 | §2.4 | `PRAGMAConfig.max_profile_tokens` |
| max_events | 6,500 | §2.4 | `PRAGMAConfig.max_events` |
| truncation_affected_fraction | 0.01% | §2.4 | documentation only |
| throughput_improvement | 2–5× | §2.4 | training infrastructure |

---

## Masking Strategy (Section 2.3.5)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| token_mask_prob | 0.15 | §2.3.5 | `MaskingStrategy` / `PRAGMAConfig.token_mask_prob` |
| event_mask_prob | 0.10 | §2.3.5 | `MaskingStrategy` / `PRAGMAConfig.event_mask_prob` |
| key_mask_prob | 0.10 | §2.3.5 | `MaskingStrategy` / `PRAGMAConfig.key_mask_prob` |

---

## MLM Head (Section 2.3.5)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| mlm_head_input_dim | 3 × d_model | §2.3.5 | `MLMHead` — concatenation of 3 d-dimensional vectors |
| mlm_head_output_dim | d_model | §2.3.5 | `MLMHead` — projected back to d |
| mlm_loss | cross-entropy with label smoothing | §2.3.5 | `MLMHead` |

---

## LoRA Adaptation (Section 3.1.2)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| lora_rank (default) | 8 | §3.1.2 | `LoRAAdapter` / `PRAGMAConfig.lora_rank` |
| lora_alpha (default) | 8 | §3.1.2 | `LoRAAdapter` / `PRAGMAConfig.lora_alpha` |
| lora_param_fraction | 2–4% | §3.1.2 | `tests/test_lora.py::TestPaperSpecifications` |
| lora_rank_sweep | {4, 8, 16} | §3.1.2 | ablation only |
| lora_target_modules | QKV projections + MLP layers | §3.1.2 | `LoRAAdapter` |
| lora_optimiser | Adam | §3.1.2 | training config |

---

## Embedding Probe (Section 3.1.1)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| probe_type | linear (logistic / linear regression) | §3.1.1 | `EmbeddingProbe` |
| probe_optimiser | L-BFGS | §3.1.1 | `EmbeddingProbe` |
| probe_scaling | standard-scaled (pre-norm architecture) | §3.1.1 | `EmbeddingProbe` |
| probe_tokens_evaluated | [USR], final [EVT], combination | §3.1.1 | `EmbeddingProbe` |

---

## Dataset Scale (Section 2.1)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| pre_training_records | 26 M user records | §2.1 | documentation only |
| pre_training_countries | 111 | §2.1 | documentation only |
| pre_training_events | 24 B events | §2.1 | documentation only |
| pre_training_tokens | 207 B tokens | §2.1 / §2.4 | documentation only |
| pre_training_months | 25 months (2023–2025) | §2.1.3 | documentation only |

---

## Training Infrastructure (Section 2.4)

| Number | Value | Section | Implementation |
|--------|-------|---------|----------------|
| training_precision | bf16 | §2.4 | training config |
| optimiser_pretraining | Muon + AdamW | §2.4 | training config |
| gpu_count_s_m | 16× NVIDIA H100 | §2.4 | documentation only |
| gpu_count_l | 32× NVIDIA H100 | §2.4 | documentation only |
| convergence_s | ~2 days | §2.4 | documentation only |
| convergence_m_l | ~2 weeks | §2.4 | documentation only |

---

## Main Results vs Task-Specific Baseline (Table 2, Section 3.3)

These numbers are reference only — not used in unit tests.

| Task | Metric | Improvement | Section |
|------|--------|-------------|---------|
| Credit scoring | PR-AUC | +130.2% | Table 2 |
| Credit scoring | ROC-AUC | +12.4% | Table 2 |
| Communication engagement | PR-AUC | +79.4% | Table 2 |
| Communication engagement | ROC-AUC | +20.4% | Table 2 |
| External fraud | Precision | +16.7% | Table 2 |
| External fraud | Recall | +64.7% | Table 2 |
| Product recommendation | mAP | +40.5% | Table 2 |
| Recurrent transactions | F1 | +5.8% | Table 2 |
| Lifetime value | PR-AUC | +1.8% | Table 2 |
| Lifetime value | ROC-AUC | +2.6% | Table 2 |

---

## Ablation Results (Section 3.4)

These numbers are reference only.

| Ablation | Task | Metric | Improvement | Section |
|----------|------|--------|-------------|---------|
| +profile state (vs event-only) | Credit scoring | PR-AUC | +31.8% | §3.4.2 / Table 6 |
| +profile state (vs event-only) | Credit scoring | ROC-AUC | +4.9% | §3.4.2 / Table 6 |
| Comm. engagement uplift | AUUC | +163.7% | §3.4.3 / Table 7 |
| Comm. engagement uplift | SNIPS | +10.8% | §3.4.3 / Table 7 |
| +Nemotron text encoder | Credit scoring | PR-AUC | +16.1% | §3.4.4 / Table 8 |
| +Nemotron text encoder | Credit scoring | ROC-AUC | +2.8% | §3.4.4 / Table 8 |
| Nemotron training overhead | — | +18% latency | §3.4.4 | documentation only |
| AML (limitation) | F0.5 | −47.1% | §3.4.5 / Table 9 |
| LoRA vs embedding (Comm. Eng. Small) | PR-AUC | +72.9% | §3.4.1 / Table 5 |
| LoRA vs embedding (Credit M) | PR-AUC | +20.4% | §3.4.1 / Table 5 |
| LoRA vs embedding (Recurrent/LTV) | various | +2.3% to +4.7% | §3.4.1 / Table 5 |
