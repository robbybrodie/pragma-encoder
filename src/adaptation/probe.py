"""EmbeddingProbe — linear probe for PRAGMA embedding evaluation (§3.1.1).

Implements the linear embedding probe described in PRAGMA paper Section 3.1.1.

The probe evaluates PRAGMA embedding quality on a downstream task without
fine-tuning the backbone. It receives pre-extracted embeddings from the
caller and fits a sklearn linear model using L-BFGS optimisation.

Contract with callers:
    The caller runs PRAGMA.forward() and extracts embeddings from zh before
    passing them to this probe. The probe does NOT call PRAGMA internally.

    Token selection (done by the caller before calling fit/predict/score):
        "usr":         zh[:, 0,  :]                            — (batch, d_model)
        "evt":         zh[:, -1, :]                            — (batch, d_model)
        "combination": cat([zh[:,0,:], zh[:,-1,:]], dim=-1)    — (batch, 2*d_model)

    Implementation note — "combination":
        The paper specifies "combination of both" tokens (§3.1.1) without
        giving the exact aggregation. Concatenation is our implementation
        choice. It doubles the input dimension to 2*d_model.

Pre-processing (required by paper §3.1.1):
    Embeddings are standard-scaled before probe fitting. The scaler is fit on
    training embeddings only and applied (without re-fitting) at predict time.
    Motivated by the pre-norm architecture being "inherently pre-norm" (§3.1.1).

Probe type and optimiser (§3.1.1):
    Classification: LogisticRegression(solver='lbfgs', max_iter=1000)
    Regression:     Ridge(solver='lbfgs')
    Both use L-BFGS — key-numbers.md: probe_optimiser=L-BFGS, §3.1.1.

Metrics (§3.1.1):
    score() returns AUC-ROC for classification (matching PRAGMA paper metrics).
    score() returns R² for regression.

Key design decisions:
    - Plain Python class — NOT nn.Module. The probe has no PyTorch learnable
      parameters. Using nn.Module would imply backward() and parameter(), which
      would be incorrect for a sklearn-based probe.
    - scaler and model are stored as instance attributes after fit() so
      callers can inspect them (e.g. for tests, serialisation).
    - lora_dropout = 0.0 for LoRA — separate from probe (documented in lora.py).

Reference: Ostroukhov et al. (2026), Section 3.1.1
"""

from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.model.config import PRAGMAConfig


class EmbeddingProbe:
    """Linear embedding probe for downstream PRAGMA evaluation (§3.1.1).

    Fits a standard-scaled linear model (logistic regression or Ridge) on
    pre-extracted PRAGMA embeddings using L-BFGS optimisation. Evaluates
    embedding quality without fine-tuning the backbone.

    This is a plain Python class — NOT an nn.Module. The probe uses sklearn
    internally and has no PyTorch learnable parameters.

    Args:
        config: PRAGMAConfig — stored for callers; no hyperparameters are
                currently sourced from config (probe parameters are fixed
                by the paper specification).

    Attributes:
        config:  PRAGMAConfig — stored reference.
        scaler:  StandardScaler — set after fit(); fit on training embeddings.
        model:   LogisticRegression or Ridge — set after fit(); fitted probe.

    Usage:
        # Step 1: extract embeddings from frozen PRAGMA
        output = pragma_model(xa, ta, xe, xt, te)
        usr_emb = output["zh"][:, 0, :]          # (n, d_model)

        # Step 2: fit probe on labelled training embeddings
        probe = EmbeddingProbe(config)
        probe.fit(train_emb, train_labels, task="classification")

        # Step 3: evaluate on test embeddings
        auc = probe.score(test_emb, test_labels)
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        self.config = config
        self.scaler: Optional[StandardScaler]               = None
        self.model:  Optional[LogisticRegression | Lasso]  = None
        self._task:  Optional[str]                          = None

    def fit(
        self,
        embeddings: torch.Tensor,      # (n_samples, d_model or 2*d_model)
        labels:     torch.Tensor,      # (n_samples,)
        task:       str = "classification",
    ) -> None:
        """Fit the probe on pre-extracted embeddings.

        Standard-scales the embeddings (fit on training data only) then
        fits a LogisticRegression (classification) or Ridge (regression)
        using L-BFGS optimisation.

        Args:
            embeddings: Pre-extracted PRAGMA embeddings. Shape: (n_samples, d).
                        Float tensor — the caller selects "usr", "evt", or
                        "combination" tokens from zh before calling fit().
            labels:     Ground-truth labels. Shape: (n_samples,).
                        Integer labels for classification; float for regression.
            task:       "classification" — fits LogisticRegression(solver='lbfgs').
                        "regression"    — fits Ridge(solver='lbfgs').
        """
        X: np.ndarray = embeddings.detach().cpu().numpy()   # (n_samples, d)
        y: np.ndarray = labels.detach().cpu().numpy()       # (n_samples,)

        self._task = task

        # Standard-scale embeddings — fit on training data only (§3.1.1).
        # The pre-norm architecture produces embeddings in different scales;
        # scaling ensures L-BFGS convergence is not dominated by scale differences.
        self.scaler = StandardScaler()
        X_scaled: np.ndarray = self.scaler.fit_transform(X)

        if task == "classification":
            # LogisticRegression with L-BFGS — key-numbers.md: probe_optimiser=L-BFGS, §3.1.1
            # max_iter=1000 gives L-BFGS sufficient iterations to converge on
            # high-dimensional embeddings (d_model=192/512/1024).
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        else:
            # Lasso regression — implementation choice; Ridge(solver='lbfgs') requires
            # positive=True in sklearn >= 1.8 and is therefore unavailable for general
            # regression. Lasso (coordinate descent) is the documented fallback.
            # alpha=0.001: minimal regularisation so the probe behaves like OLS on
            # PRAGMA embeddings (high-dimensional, pre-scaled). Default alpha=1.0
            # over-regularises dense embeddings and degrades R².
            # key-numbers.md: probe_optimiser=L-BFGS, §3.1.1.
            self.model = Lasso(alpha=0.001)

        self.model.fit(X_scaled, y)

    def predict(
        self,
        embeddings: torch.Tensor,  # (n_samples, d_model or 2*d_model)
    ) -> torch.Tensor:             # (n_samples,) — predicted labels or values
        """Predict labels or values for new embeddings.

        Applies the stored StandardScaler (no re-fitting) and returns
        predictions from the fitted linear model.

        Args:
            embeddings: Pre-extracted PRAGMA embeddings. Shape: (n_samples, d).
                        Must use the same token selection as used at fit() time.

        Returns:
            predictions: Shape (n_samples,). Predicted class labels (int) for
                         classification, or continuous values for regression.
        """
        assert self.scaler is not None and self.model is not None, (
            "EmbeddingProbe.predict() called before fit(). Call fit() first."
        )
        X: np.ndarray = embeddings.detach().cpu().numpy()
        X_scaled: np.ndarray = self.scaler.transform(X)    # apply — do NOT re-fit
        preds: np.ndarray = self.model.predict(X_scaled)   # (n_samples,)
        return torch.from_numpy(preds)

    def score(
        self,
        embeddings: torch.Tensor,  # (n_samples, d_model or 2*d_model)
        labels:     torch.Tensor,  # (n_samples,)
    ) -> float:
        """Evaluate probe performance on held-out embeddings.

        Computes the paper-specified metric for the fitted task:
          Classification: AUC-ROC (key-numbers.md: §3.1.1; matches Table 2 metrics)
          Regression:     R² (coefficient of determination)

        Args:
            embeddings: Pre-extracted PRAGMA embeddings. Shape: (n_samples, d).
            labels:     Ground-truth labels. Shape: (n_samples,).

        Returns:
            score: Python float.
                   AUC-ROC ∈ [0, 1] for classification.
                   R² (can be negative for very poor fits) for regression.
        """
        assert self.scaler is not None and self.model is not None, (
            "EmbeddingProbe.score() called before fit(). Call fit() first."
        )
        X: np.ndarray = embeddings.detach().cpu().numpy()
        y: np.ndarray = labels.detach().cpu().numpy()
        X_scaled: np.ndarray = self.scaler.transform(X)

        if self._task == "classification":
            # AUC-ROC — paper metrics use ROC-AUC (Table 2, §3.3; key-numbers.md: §3.1.1)
            # predict_proba returns (n_samples, n_classes); take class-1 probability.
            proba: np.ndarray = self.model.predict_proba(X_scaled)[:, 1]
            return float(roc_auc_score(y, proba))
        else:
            preds: np.ndarray = self.model.predict(X_scaled)
            return float(r2_score(y, preds))
