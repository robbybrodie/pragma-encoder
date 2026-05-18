# Section 1 — Introduction

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 1

---

## Key Context

Banking user histories differ from text in three ways (Section 1):

1. Each event is a variable-length record with **mixed categorical, numerical, and free-text fields**
2. Histories are **long-tailed in length and irregular in time**, with strong daily/weekly cycles
3. **Privacy and regulatory constraints** limit what can be reported and which features can be used

Standard text LLMs are unsuitable because:
- Serialising structured records as text **inflates sequence lengths**
- Field names and delimiters become multiple subword tokens
- Numerical values split into digit fragments, **discarding magnitude and ordering**

---

## Three Contributions (exact from paper)

1. **PRAGMA architecture:** encoder-style foundation models for multi-source banking user
   histories, scaling from 10M to 1B parameters — to our knowledge, the largest published
   encoder backbone for consumer banking event sequences. Architecture combines:
   - Key–value–time tokenisation scheme
   - Two-branch design: profile-state encoder + event encoder feeding a history encoder

2. **Pre-training recipe:** masked modelling on long, irregular banking user histories with
   sequence packing and dynamic batching. LoRA fine-tuning consistently matches or
   outperforms full training from scratch.

3. **Evaluation scope:** single pre-trained backbone across six diverse downstream tasks
   (credit scoring, fraud detection, lifetime value, communication engagement, recurrent
   transaction detection, product recommendation) — broader scope than prior models.

---

## Why Encoder-Only (exact from paper, Section 1)

> "We choose an encoder-only, bidirectional design because our primary goal is
> transferable representations for discriminative financial tasks, rather than
> open-ended generation. Masked modelling enables each token to attend to both
> past and future context, which is particularly useful when reconstructing
> partially observed event records and learning record-level representations
> from complete histories."

This is the paper's justification for ADR 001 (encoder-only architecture).

---

## Implementation Notes

The introduction establishes the problem framing that motivates every architectural decision:

- The **key–value–time tokenisation** (§2.2) solves the text serialisation problem
- The **three-encoder architecture** (§2.3) handles the heterogeneity of profile + events
- The **masked modelling objective** (§2.3.5) enables bidirectional attention
- The **two adaptation modes** (§3.1) address the regulatory constraint on labels

The NVIDIA blueprint (decoder-only, text serialisation) is the explicit baseline being outperformed.
Do not import or reuse the NVIDIA blueprint's tokeniser or architecture.
