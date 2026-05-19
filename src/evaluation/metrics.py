"""Evaluation metrics for PRAGMA downstream tasks.

Implements the metrics reported in PRAGMA paper Section 3:
    - AUC (AUROC)   — fraud detection, churn prediction
    - PR-AUC        — credit risk assessment (imbalanced datasets)
    - F1 score      — secondary classification metric
    - KS statistic  — Kolmogorov-Smirnov, used in credit risk evaluation

All functions accept numpy arrays or PyTorch tensors and return
scalar float values.

Reference: Ostroukhov et al. (2026), Section 3
"""

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Coerce input to numpy array."""
    if hasattr(x, "cpu"):  # PyTorch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute Area Under the ROC Curve (AUROC).

    Primary metric for fraud detection and churn prediction (Section 3).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Predicted probability scores for the positive class.

    Returns:
        AUROC in [0, 1]. Random = 0.5, perfect = 1.0.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(_to_numpy(y_true), _to_numpy(y_score)))


def compute_pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute Precision-Recall AUC.

    Primary metric for credit risk assessment where positive class
    (defaults) is heavily imbalanced (Section 3).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Predicted probability scores for the positive class.

    Returns:
        PR-AUC in [0, 1].
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(_to_numpy(y_true), _to_numpy(y_score)))


def compute_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: float = 0.5,
    average: str = "binary",
) -> float:
    """Compute F1 score.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred: Predicted probabilities or binary predictions.
        threshold: Threshold for converting probabilities to binary. Default: 0.5.
        average: Averaging strategy. Default: 'binary'.

    Returns:
        F1 score in [0, 1].
    """
    from sklearn.metrics import f1_score
    y_pred_np = _to_numpy(y_pred)
    if y_pred_np.max() <= 1.0 and y_pred_np.dtype != int:
        y_pred_np = (y_pred_np >= threshold).astype(int)
    return float(f1_score(_to_numpy(y_true), y_pred_np, average=average))


def compute_ks(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic.

    The KS statistic measures the maximum separation between the
    cumulative distributions of positive and negative scores.
    Widely used in credit risk model validation (Section 3).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Predicted probability scores for the positive class.

    Returns:
        KS statistic in [0, 1]. Higher is better.
    """
    from sklearn.metrics import roc_curve
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
    return float(np.max(tpr - fpr))
