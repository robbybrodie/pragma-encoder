# Section 3.4 — Additional Experiments and Ablations

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 3.4

---

## 3.4.1 Effect of Low-Rank Adaptation (Table 5)

LoRA-tuned variants consistently outperform embedding-only baselines across all tasks and scales.

Most substantial improvement: Communication Engagement
- PRAGMA-S: +72.9% PR-AUC with LoRA vs embedding-only
- Medium/Large: significant leads maintained

Credit Scoring:
- Peak: +20.4% PR-AUC for Medium model with LoRA

More modest gains (but consistent):
- Recurrent Transactions: +2.3% to +4.7%
- LTV: +2.3% to +4.7%

**Key insight:** LoRA is not optional — it adds substantial task-specific value
beyond what frozen embeddings provide.

---

## 3.4.2 Effect of Profile State (Table 6)

Compares full PRAGMA-S vs variant with Profile State Encoder removed entirely.

| Task | Metric | Gain from profile state |
|------|--------|------------------------|
| Credit scoring | PR-AUC | **+31.8%** |
| Credit scoring | ROC-AUC | +4.9% |
| Lifetime value | PR-AUC | +2.2% |
| Lifetime value | ROC-AUC | +2.0% |
| Recurrent txns | F1 | +2.4% |
| Comm. engagement | PR-AUC | −3.0% (slight regression) |
| Comm. engagement | ROC-AUC | +1.3% |
| External fraud | Precision | +46.8% |
| External fraud | Recall | +85.6% |
| Product rec. | mAP | +3.5% |

**Key insight (paper quote):**
> "Profile state is particularly valuable for identifying the minority default class,
> where static signals such as account tenure and onboarding characteristics provide
> discriminative context that event sequences alone cannot fully capture."

**Validates ADR 002:** The dedicated Profile State Encoder adds significant value
for static-signal-dependent tasks, while the architecture degrades gracefully when
those signals are less relevant (Communication Engagement).

---

## 3.4.3 Communication Engagement Uplift (Table 7)

Task: identify which messaging strategy best re-engages users (treatment selection)

PRAGMA used as **frozen feature extractor** feeding a meta-learner (no fine-tuning).

| Metric | Improvement |
|--------|-------------|
| AUUC | **+163.7%** |
| SNIPS | +10.8% |

Model: PRAGMA-L

---

## 3.4.4 Effect of Pre-trained Text Encoder (Table 8)

Optional extension: replace BPE tokeniser + embedding table for text fields with
frozen pre-trained Nemotron-1B-v2.

Results on PRAGMA-M:
- Credit Scoring: +16.1% PR-AUC, +2.8% ROC-AUC
- Recurrent Transactions: +0.1% F1 (near flat)
- LTV: +0.8% PR-AUC, +0.6% ROC-AUC
- External Fraud: +3.8% Precision, −0.7% Recall
- Product Recommendation: **−6.4% mAP** (loss)
- Training latency: **+18%**

**Decision:** Kept as opt-in module for text-heavy tasks.
Not baked into default architecture (our implementation uses standard BPE).

---

## 3.4.5 Limitations — Anti-Money Laundering (Table 9)

| Metric | Result |
|--------|--------|
| F0.5 | **−47.1%** vs baseline |

**Why PRAGMA fails on AML:**

1. AML dataset is large enough for task-specific baseline to learn without pre-training
2. AML is **inherently relational** — baseline uses cross-record network-level features
3. PRAGMA processes event histories **in isolation** — cannot capture cross-record dependencies

**Known limitation:** Tasks that depend on cross-record relational structure are out of reach.
Extending to relational tasks is identified as future work.

---

## Implementation Notes

Ablation 3.4.2 is the most important for our architecture:

The +31.8% PR-AUC gain from profile state on Credit Scoring confirms that the
Profile State Encoder is not optional. It must be implemented correctly.

The clean ablation is possible precisely because of ADR 002 (three separate encoders):
removing the profile branch requires only removing the ProfileStateEncoder input to
the HistoryEncoder — no other changes needed.

If the profile branch cannot be ablated cleanly, the implementation violates ADR 002.
