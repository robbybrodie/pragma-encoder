"""Linear embedding probe for rapid PRAGMA evaluation.

Implements the linear probing approach from PRAGMA paper Section 3.1.1.

A linear probe trains a single linear layer on top of frozen PRAGMA
embeddings to evaluate how much task-relevant information the encoder
has captured during pretraining. Linear probes are:

    - Fast: Only the linear layer is trained (encoder frozen).
    - Interpretable: Performance directly reflects embedding quality.
    - Diagnostic: Used to identify which encoder layer is most
      informative for a given task.

Evaluation protocol from the paper (Section 3.1.1):
    1. Extract embeddings from a frozen PRAGMA model for labelled samples.
    2. Train a logistic regression (or shallow MLP) on the embeddings.
    3. Report AUC / PR-AUC on the held-out test set.

This module implements both scikit-learn-compatible (LogisticRegression)
and PyTorch (LinearProbe) variants.

Reference: Ostroukhov et al. (2026), Section 3.1.1
"""

from typing import Optional

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Lightweight linear classifier on top of frozen PRAGMA embeddings.

    A single linear + sigmoid layer for binary classification tasks
    (fraud detection, churn prediction, credit risk).

    Args:
        d_model: Dimensionality of the input embeddings (PRAGMA d_model).
        n_classes: Number of output classes. Default: 2 (binary).
        dropout: Dropout on the input embeddings. Default: 0.1.
        pooling: How to pool the history sequence for classification.
            'hist': Use the [HIST] summary token (recommended).
            'mean': Mean-pool over all event representations.
            'last': Use the last event representation.
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = "hist",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.pooling = pooling

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        history_reprs: torch.Tensor,
        hist_repr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify from PRAGMA embeddings.

        Args:
            history_reprs: (batch, n_events, d_model) — event representations.
            hist_repr: (batch, d_model) — [HIST] summary token (used if
                       pooling='hist').
            attention_mask: (batch, n_events) — False for valid events.

        Returns:
            (batch, n_classes) — classification logits.
        """
        if self.pooling == "hist" and hist_repr is not None:
            pooled = hist_repr
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = (~attention_mask).float().unsqueeze(-1)
                pooled = (history_reprs * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = history_reprs.mean(1)
        elif self.pooling == "last":
            pooled = history_reprs[:, -1, :]
        else:
            pooled = hist_repr if hist_repr is not None else history_reprs.mean(1)

        return self.classifier(self.drop(pooled))
