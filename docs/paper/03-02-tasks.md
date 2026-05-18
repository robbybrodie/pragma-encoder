# Section 3.2 — Downstream Tasks

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 3.2

---

## Six Downstream Tasks

### Credit Scoring
- **Goal:** Predict probability of default within first 12 months of use
- **Formulation:** Binary classification (minority class)
- **Metrics:** ROC-AUC, PR-AUC
- **Characteristics:** Multiple years, diverse records across countries

### Communication Engagement
- **Goal:** Predict whether user who abandoned credit application will open re-engagement communication
- **Formulation:** Binary classification
- **Metrics:** ROC-AUC, PR-AUC
- **Characteristics:** Severely limited sample size — model must capture nuanced signals from minimal data

### External Fraud
- **Goal:** Representative fraud detection use case
- **Formulation:** Binary classification
- **Metrics:** Precision, Recall
- **Characteristics:** Imbalanced classes; focus on precision-recall trade-off

### Product Recommendation
- **Goal:** Predict which products user is likely to adopt after receiving communication
- **Formulation:** Multilabel classification (independent probability per product)
- **Metric:** Mean Average Precision (mAP)
- **Characteristics:** Multi-product, conditioning on communication type

### Recurrent Transactions
- **Goal:** Predict whether a transaction corresponds to recurring subscription that will repeat next month
- **Formulation:** Binary classification
- **Metric:** Macro-averaged F1-score (accounts for class imbalance)
- **Characteristics:** Must distinguish true recurring patterns from irregular/one-off payments

### Lifetime Value (LTV)
- **Goal:** Predict probability of user generating positive gross profit
- **Formulation:** Binary classification
- **Metrics:** ROC-AUC, PR-AUC
- **Characteristics:** Users have shorter event histories (couple of weeks), long prediction horizon (6+ months)

---

## Implementation Notes

These are evaluation tasks — they do not affect the PRAGMA model implementation.
They are referenced here for:

1. Understanding what the model's representations must capture
2. Selecting which token to use as the embedding (`[USR]` vs final `[EVT]`)
3. Interpreting ablation results (e.g., why profile state matters for credit scoring)

The end-to-end comparison test (DEVELOPMENT_PROCESS.md) uses TabFormer fraud detection
as a proxy task. This corresponds to the External Fraud task described here.
