# Abstract and Key Claims

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Authors: Ruslan Mikhailov, Maxim Ostroukhov, Artem Sokolov, Andrei Akshonov,
Vitaly Protasov, Dmitrii Beloborodov (Revolut Research);
Vince Mullin, Roman Y. Enzmann, Georgios Kolovos, Jason Renders (NVIDIA);
Anton Repushko, Pavel Nesterov, Vladimir Iashin (Revolut Research)

---

## Abstract (exact)

Modern financial systems generate vast quantities of transactional and event-level
data that encode rich economic signals. This paper presents PRAGMA, a family of
foundation models for multi-source banking event sequences. Our approach pre-trains
a Transformer-based architecture with masked modelling on a large-scale,
heterogeneous banking event corpus using a self-supervised objective tailored to the
discrete, variable-length nature of financial records. The resulting model supports
a wide range of downstream tasks such as credit scoring, fraud detection, and
lifetime value prediction: strong performance can be achieved by training a simple
linear model on top of the extracted embeddings and can be further improved with
lightweight fine-tuning. Through extensive evaluation on downstream tasks, we
demonstrate that PRAGMA achieves superior performance across multiple domains
directly from raw event sequences, providing a general-purpose representation layer
for financial applications.

*Disclaimer: We report only relative improvements, as absolute metrics are
commercially sensitive. All examples are synthetic and not from real production data.*

---

## Key Claims

1. **PRAGMA is a family of encoder-style foundation models** for multi-source banking
   user histories, scaling from 10M to 1B parameters.

2. **Pre-training objective:** Masked modelling (MLM) on heterogeneous banking event
   corpus. Self-supervised. Does not require task labels.

3. **Two downstream adaptation modes:**
   - Embedding probe: frozen backbone + linear head (minutes to train)
   - LoRA fine-tuning: 2–4% parameter update (hours to train)

4. **Downstream tasks evaluated:** credit scoring, fraud detection, lifetime value,
   communication engagement, recurrent transaction detection, product recommendation.

5. **Performance:** Consistently outperforms task-specific baselines across nearly
   all domains (Table 2). Most striking: +130.2% PR-AUC on Credit Scoring.

6. **Single architecture from 10M to 1B parameters** that outperforms task-specific
   models across tasks (Figure 1).

---

## Implementation Notes

The abstract establishes the non-negotiable design choices:

- Encoder-only (bidirectional) architecture — confirmed in ADR 001
- Masked modelling pre-training objective — confirmed in ADR 004
- Self-supervised (no labels required for pre-training)
- Two adaptation modes: embedding probe + LoRA
- General-purpose: single backbone for all tasks

None of these choices can be changed without creating a new ADR.
