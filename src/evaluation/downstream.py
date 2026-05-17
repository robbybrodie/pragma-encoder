"""Downstream task evaluator for PRAGMA.

Implements the evaluation protocol from PRAGMA paper Section 3.

Supports two evaluation modes:
    1. Linear probe (Section 3.1.1): Frozen PRAGMA encoder + logistic regression.
       Calls PRAGMA forward pass to extract embeddings, then fits sklearn
       LogisticRegression. Fast and interpretable.

    2. LoRA fine-tuning (Section 3.1.2): PRAGMA with LoRA adapters + linear head.
       End-to-end training with the LoRA-adapted PRAGMA model.

The DownstreamEvaluator orchestrates both protocols and reports the
metrics from Section 3: AUC, PR-AUC, F1, KS.

Evaluated downstream tasks (Table 2 of the paper):
    - Fraud detection
    - Churn prediction
    - Credit risk (loan default)
    - Card decline prediction
    - Merchant category prediction

Reference: Ostroukhov et al. (2026), Section 3
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_auc, compute_f1, compute_ks, compute_pr_auc


class DownstreamEvaluator:
    """Orchestrates downstream task evaluation for PRAGMA embeddings.

    Supports linear probing (scikit-learn) and PyTorch-based evaluation.

    Args:
        task_name: Name of the downstream task (for logging). Default: 'task'.
        metric: Primary metric to optimise. One of: 'auc', 'pr_auc', 'f1', 'ks'.
        device: Torch device for inference. Default: 'cpu'.
    """

    def __init__(
        self,
        task_name: str = "task",
        metric: str = "auc",
        device: str = "cpu",
    ):
        self.task_name = task_name
        self.metric = metric
        self.device = device

    def evaluate_linear_probe(
        self,
        embeddings_train: np.ndarray,
        labels_train: np.ndarray,
        embeddings_test: np.ndarray,
        labels_test: np.ndarray,
        max_iter: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate PRAGMA embeddings via logistic regression linear probe.

        Fits a logistic regression on training embeddings and evaluates
        on the test set. This is the protocol from Section 3.1.1.

        Args:
            embeddings_train: (n_train, d_model) training embeddings.
            labels_train: (n_train,) binary training labels.
            embeddings_test: (n_test, d_model) test embeddings.
            labels_test: (n_test,) binary test labels.
            max_iter: Max LogisticRegression iterations. Default: 1000.

        Returns:
            Dict with keys: 'auc', 'pr_auc', 'f1', 'ks'.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Standardise embeddings
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings_train)
        X_test = scaler.transform(embeddings_test)

        # Fit logistic regression
        clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        clf.fit(X_train, labels_train)

        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        return {
            "auc": compute_auc(labels_test, y_score),
            "pr_auc": compute_pr_auc(labels_test, y_score),
            "f1": compute_f1(labels_test, y_pred),
            "ks": compute_ks(labels_test, y_score),
        }

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: Any,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Evaluate a PRAGMA model (with head) on a DataLoader.

        Args:
            model: PRAGMA model with classification head attached.
            dataloader: PyTorch DataLoader yielding (inputs, labels) batches.
            criterion: Optional loss function for computing eval loss.

        Returns:
            Dict with keys: 'auc', 'pr_auc', 'f1', 'ks', and optionally 'loss'.
        """
        model.eval()
        all_scores: list = []
        all_labels: list = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                logits = model(inputs)
                scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())

                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    n_batches += 1

        y_score = np.concatenate(all_scores)
        y_true = np.concatenate(all_labels)

        results = {
            "auc": compute_auc(y_true, y_score),
            "pr_auc": compute_pr_auc(y_true, y_score),
            "f1": compute_f1(y_true, y_score),
            "ks": compute_ks(y_true, y_score),
        }
        if criterion is not None and n_batches > 0:
            results["loss"] = total_loss / n_batches

        return results
