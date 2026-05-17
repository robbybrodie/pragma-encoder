"""Evaluation package — downstream task evaluation and metrics.

Implements the evaluation framework from PRAGMA paper Section 3.

PRAGMA is evaluated on three downstream financial tasks:
    1. Fraud detection         — binary classification, AUC metric
    2. Churn prediction        — binary classification, AUC metric
    3. Credit risk assessment  — binary classification, PR-AUC metric

Two evaluation protocols (Section 3.1):
    3.1.1 — Linear embedding probe: frozen encoder + linear head
    3.1.2 — LoRA fine-tuning: parameter-efficient full adaptation

Metrics used (from the paper):
    - AUC (Area Under the ROC Curve)       — primary metric for fraud/churn
    - PR-AUC (Precision-Recall AUC)        — primary metric for credit risk
    - F1 score                             — secondary metric
    - KS statistic                         — for credit risk evaluation

Reference: Ostroukhov et al. (2026), Section 3
"""

from .downstream import DownstreamEvaluator
from .metrics import compute_auc, compute_pr_auc, compute_f1, compute_ks

__all__ = [
    "DownstreamEvaluator",
    "compute_auc",
    "compute_pr_auc",
    "compute_f1",
    "compute_ks",
]
