# Section 4 — Related Work

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 4

---

## 4.1 Transformer

Key references:
- **Transformer** (Vaswani et al., 2017) — self-attention mechanism
- **BERT** (Devlin et al., 2019) — encoder-only, MLM objective
- **GPT-3** (Brown et al., 2020) — decoder-only, generative
- **ViT** (Dosovitskiy et al., 2021) — Vision Transformer
- **FlashAttention** (Dao et al., 2022) — hardware-aware attention (used in varlen kernel)

PRAGMA positioned as: **encoder foundation model for heterogeneous tabular event streams**.
Inherits bidirectional contextualisation of encoder-only Transformers.
Adapts to heterogeneous fields, explicit time signals, and reusable record-level representations.

---

## 4.2 Masked Modelling

Key references:
- **BERT** (Devlin et al., 2019) — MLM for text
- **RoBERTa** (Liu et al., 2019) — dynamic masking, optimised training
- **BEiT** (Bao et al., 2021) — Masked Image Modelling
- **MAE** (He et al., 2022) — Masked Autoencoders for vision
- **Data2Vec** (Baevski et al., 2022) — cross-modal unification
- **I-JEPA** (Assran et al., 2023) — latent feature prediction

PRAGMA extends masked modelling from text/images to **heterogeneous financial records**.
Masks: individual tokens + whole events + semantic types.

---

## 4.3 Transformers for Tabular Data

Key references:
- **TabTransformer** (Huang et al., 2020)
- **FT-Transformer** (Gorishniy et al., 2021)
- **SAINT** (Somepalli et al., 2021)
- **TabPFN** (Hollmann et al., 2023) — foundation model on synthetic data

PRAGMA differs: does not operate on fixed-schema single rows.
Unlike TabPFN: pre-trained with self-supervision on real financial ledgers.
Models variable-length histories with hierarchical encoder.

---

## 4.4 Modelling for Recommender Systems

Key references:
- **SASRec** (Kang et al., 2018) — self-attention over interaction history
- **BERT4Rec** (Sun et al., 2019) — bidirectional masked item prediction
- **P5** (Geng et al., 2022) — unified text-to-text recommendation
- **Generative Recommenders** (Zhai et al., 2024) — causal sequence, trillion parameters
- **TransAct** (Xia et al., 2023; 2025) — composite action embeddings

PRAGMA differs: models richer financial events with typed fields, amounts, free text,
temporal coordinates. Adapted to broader banking tasks beyond ranking.

---

## 4.5 Foundation Models for Finance

Key references:
- **FinBERT** (Yang et al., 2020) — encoder-only for financial text
- **BloombergGPT** (Wu et al., 2023) — generative at scale
- **FinGPT** (Yang et al., 2023) — lightweight LoRA fine-tuning
- **Time-LLM** (Jin et al., 2024) — time series as tokens
- **Chronos** (Ansari et al., 2024) — zero-shot forecasting
- **nuFormer** (Braithwaite et al., 2025) — transaction sequences + tabular features
- **TransactionGPT** (Dou et al., 2025) — 3D-Transformer for payment trajectories

PRAGMA closest to: **nuFormer** and **TransactionGPT** (transaction-ledger models).
But PRAGMA aims for: reusable encoder backbone over **multi-source** events with
explicit profile state and lightweight adaptation across **diverse discriminative tasks**.

---

## Implementation Notes

The adjacent NVIDIA blueprint repo is the baseline being outperformed.
That repo uses:
- Decoder-only (GPT/Llama-style) architecture
- Causal language modelling objective
- Text serialisation tokenisation

PRAGMA uses the opposite for all three:
- Encoder-only bidirectional
- Masked modelling
- Key–value–time tokenisation

Do not import or reuse code from the NVIDIA blueprint (ADR 001 §5).
