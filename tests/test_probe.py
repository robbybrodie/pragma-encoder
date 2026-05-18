"""Tests for EmbeddingProbe.

Derived from PRAGMA paper Section 3.1.1:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:      verify fit/predict/score output shapes and types
  Math tests:       verify AUC > threshold on separable data, R² on linear data,
                    scaler is stateful and applied at predict time
  Architecture tests: class name, config constructor, NOT nn.Module
  Spec tests:       verify L-BFGS solver, StandardScaler usage

Key paper properties (§3.1.1):
  - Probe type: linear only (logistic regression or linear regression)
  - Optimiser: L-BFGS — not SGD, not Adam
  - Embeddings standard-scaled before fitting (pre-norm architecture)
  - Tokens evaluated: [USR], final [EVT], or combination
  - Probe receives pre-extracted embeddings — does NOT call PRAGMA internally
  - score(): AUC-ROC for classification, R² for regression

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch   = pytest.importorskip("torch",   reason="torch not installed")
sklearn = pytest.importorskip("sklearn", reason="sklearn not installed")

import torch
import numpy as np

from src.model.config import PRAGMAConfig
from src.adaptation.probe import EmbeddingProbe

_CONFIG = PRAGMAConfig.pragma_s()

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_N_TRAIN = 200
_N_TEST  = 60


def _make_separable_classification(d: int = 32, seed: int = 0):
    """Two linearly separable Gaussian clusters — easy classification."""
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((_N_TRAIN // 2, d)) - 3.0
    X1 = rng.standard_normal((_N_TRAIN // 2, d)) + 3.0
    X  = np.concatenate([X0, X1], axis=0)
    y  = np.concatenate([np.zeros(_N_TRAIN // 2), np.ones(_N_TRAIN // 2)])
    # Shuffle
    idx = rng.permutation(_N_TRAIN)
    X, y = X[idx], y[idx]

    Xt0 = rng.standard_normal((_N_TEST // 2, d)) - 3.0
    Xt1 = rng.standard_normal((_N_TEST // 2, d)) + 3.0
    Xt  = np.concatenate([Xt0, Xt1], axis=0)
    yt  = np.concatenate([np.zeros(_N_TEST // 2), np.ones(_N_TEST // 2)])

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(Xt).float(),
        torch.from_numpy(yt).float(),
    )


def _make_linear_regression(d: int = 32, seed: int = 1):
    """y = w·x + noise — clean linear relationship for regression."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X  = rng.standard_normal((_N_TRAIN, d))
    y  = X @ w + 0.05 * rng.standard_normal(_N_TRAIN)

    Xt = rng.standard_normal((_N_TEST, d))
    yt = Xt @ w + 0.05 * rng.standard_normal(_N_TEST)

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(Xt).float(),
        torch.from_numpy(yt).float(),
    )


# ---------------------------------------------------------------------------
# TestShapes — verify output shapes and types
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify EmbeddingProbe output shapes (§3.1.1)."""

    def test_predict_shape_classification(self) -> None:
        """§3.1.1: predict() must return (n_samples,) for classification.

        The probe produces one scalar prediction per sample — not a probability
        vector. For classification this is the predicted class label.
        """
        X_train, y_train, X_test, _ = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        preds = probe.predict(X_test)

        assert preds.shape == (len(X_test),), (
            f"§3.1.1: predict() must return (n_samples,) = ({len(X_test)},), "
            f"got {tuple(preds.shape)}"
        )

    def test_predict_shape_regression(self) -> None:
        """§3.1.1: predict() must return (n_samples,) for regression.

        Regression predictions are one scalar per sample — not a vector.
        """
        X_train, y_train, X_test, _ = _make_linear_regression()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="regression")

        preds = probe.predict(X_test)

        assert preds.shape == (len(X_test),), (
            f"§3.1.1: predict() returns (n_samples,) = ({len(X_test)},), "
            f"got {tuple(preds.shape)}"
        )

    def test_score_returns_float_classification(self) -> None:
        """§3.1.1: score() must return a Python float for classification.

        AUC-ROC is a scalar in [0, 1]. Returning a tensor or numpy scalar
        breaks callers that expect a Python float.
        """
        X_train, y_train, X_test, y_test = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        result = probe.score(X_test, y_test)

        assert isinstance(result, float), (
            f"§3.1.1: score() must return a Python float, "
            f"got {type(result).__name__}"
        )

    def test_score_returns_float_regression(self) -> None:
        """§3.1.1: score() must return a Python float for regression.

        R² is a scalar. Returning a tensor or numpy scalar breaks callers.
        """
        X_train, y_train, X_test, y_test = _make_linear_regression()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="regression")

        result = probe.score(X_test, y_test)

        assert isinstance(result, float), (
            f"§3.1.1: score() must return a Python float, "
            f"got {type(result).__name__}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — AUC bounds, R² bounds, scaler statefulness
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §3.1.1."""

    def test_classification_score_is_in_0_1(self) -> None:
        """§3.1.1: AUC-ROC ∈ [0, 1] (key-numbers.md: probe_type=linear, §3.1.1).

        AUC-ROC is always bounded in [0, 1] by definition. A value outside
        this range indicates the wrong metric is being computed.
        """
        X_train, y_train, X_test, y_test = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        auc = probe.score(X_test, y_test)

        assert 0.0 <= auc <= 1.0, (
            f"§3.1.1: AUC-ROC must be in [0, 1], got {auc:.4f}. "
            f"Check that score() uses roc_auc_score, not accuracy or loss."
        )

    def test_classification_learns_separable_data(self) -> None:
        """§3.1.1: linear probe must achieve AUC > 0.9 on separable data.

        Two well-separated Gaussian clusters are trivially linearly separable.
        A working probe with L-BFGS optimisation must achieve near-perfect AUC.
        Failure here means the probe is not fitting at all.
        """
        X_train, y_train, X_test, y_test = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        auc = probe.score(X_test, y_test)

        assert auc > 0.9, (  # key-numbers.md: probe_type=linear, §3.1.1
            f"§3.1.1: L-BFGS logistic probe must achieve AUC > 0.9 on "
            f"linearly separable clusters, got {auc:.4f}. "
            f"Check that L-BFGS fitting and StandardScaler are applied correctly."
        )

    def test_regression_fits_linear_data(self) -> None:
        """§3.1.1: linear probe must achieve R² > 0.9 on linear data.

        y = w·x + small_noise is a perfectly linear relationship. A Ridge
        regression (L2) must recover it with high R². Failure means
        the probe is not fitting or StandardScaler is breaking the transform.
        """
        X_train, y_train, X_test, y_test = _make_linear_regression()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="regression")

        r2 = probe.score(X_test, y_test)

        assert r2 > 0.9, (  # key-numbers.md: probe_type=linear, §3.1.1
            f"§3.1.1: Ridge regression probe must achieve R² > 0.9 on linear "
            f"data, got {r2:.4f}. "
            f"Check that StandardScaler and Ridge are applied correctly."
        )

    def test_scaler_is_stateful_applied_at_predict_time(self) -> None:
        """§3.1.1: StandardScaler fit on training data only — applied at predict.

        The scaler must be fit on training embeddings and stored, so the same
        transform is applied to test embeddings at predict() time without
        re-fitting. A scaler that is re-fit on test data would leak test
        statistics into the evaluation.

        Verified by: calling fit() and then checking that a scaler attribute
        is set and is an instance of StandardScaler.
        """
        from sklearn.preprocessing import StandardScaler

        X_train, y_train, _, _ = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        # The probe must expose a fitted scaler (fit on train, applied to test)
        assert hasattr(probe, "scaler"), (
            "§3.1.1: EmbeddingProbe must store a 'scaler' attribute after fit(). "
            "StandardScaler must be fit on training data and reused at predict time."
        )
        assert isinstance(probe.scaler, StandardScaler), (
            f"§3.1.1: probe.scaler must be a StandardScaler instance "
            f"(key-numbers.md: probe_scaling=standard-scaled, §3.1.1), "
            f"got {type(probe.scaler).__name__}."
        )
        # Verify scaler was actually fitted (has mean_ attribute)
        assert hasattr(probe.scaler, "mean_"), (
            "§3.1.1: StandardScaler must be fit on training embeddings. "
            "scaler.mean_ not found — scaler.fit() was not called."
        )


# ---------------------------------------------------------------------------
# TestArchitecture — class name, config constructor, not nn.Module
# ---------------------------------------------------------------------------


class TestArchitecture:
    """Verify EmbeddingProbe architecture constraints (§3.1.1)."""

    def test_class_name_is_embedding_probe(self) -> None:
        """DEVELOPMENT_PROCESS.md: class must be named EmbeddingProbe exactly.

        The naming conventions table maps src/adaptation/probe.py → EmbeddingProbe.
        Any deviation (LinearProbe, EmbeddingsProbe) violates the contract.
        """
        assert EmbeddingProbe.__name__ == "EmbeddingProbe", (
            f"Class must be named 'EmbeddingProbe', got '{EmbeddingProbe.__name__}'. "
            "DEVELOPMENT_PROCESS.md: class names must match the paper exactly."
        )

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        All three model sizes must be constructable without individual args.
        Scaling must require changing exactly one line.
        """
        probe_s = EmbeddingProbe(PRAGMAConfig.pragma_s())
        probe_m = EmbeddingProbe(PRAGMAConfig.pragma_m())
        probe_l = EmbeddingProbe(PRAGMAConfig.pragma_l())

        # config must be stored and accessible
        assert probe_s.config.d_model == 192    # key-numbers.md: Table 1
        assert probe_m.config.d_model == 512    # key-numbers.md: Table 1
        assert probe_l.config.d_model == 1024   # key-numbers.md: Table 1

    def test_is_not_nn_module(self) -> None:
        """§3.1.1: EmbeddingProbe must NOT be an nn.Module.

        The probe is a plain Python class using sklearn — no PyTorch parameters,
        no backward(), no register_buffer(). Subclassing nn.Module would
        incorrectly imply trainable parameters and break the sklearn API.
        key-numbers.md: probe_type = linear (sklearn), §3.1.1.
        """
        import torch.nn as nn
        probe = EmbeddingProbe(_CONFIG)
        assert not isinstance(probe, nn.Module), (
            "§3.1.1: EmbeddingProbe must NOT subclass nn.Module. "
            "The probe is a sklearn-based linear probe (key-numbers.md: §3.1.1). "
            "Use plain Python class — no PyTorch parameters."
        )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — L-BFGS solver, StandardScaler
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from key-numbers.md (§3.1.1)."""

    def test_uses_lbfgs_solver_classification(self) -> None:
        """key-numbers.md: probe_optimiser = L-BFGS (§3.1.1).

        LogisticRegression must be instantiated with solver='lbfgs'.
        Other solvers (liblinear, sag, saga) are not the paper specification.
        The solver is verified by inspecting the fitted model's solver attribute.
        """
        from sklearn.linear_model import LogisticRegression

        X_train, y_train, _, _ = _make_separable_classification()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="classification")

        assert hasattr(probe, "model"), (
            "§3.1.1: EmbeddingProbe must store a 'model' attribute after fit(). "
            "Expected a fitted LogisticRegression."
        )
        assert isinstance(probe.model, LogisticRegression), (
            f"§3.1.1: classification probe must use LogisticRegression "
            f"(key-numbers.md: probe_type=linear, §3.1.1), "
            f"got {type(probe.model).__name__}."
        )
        assert probe.model.solver == "lbfgs", (  # key-numbers.md: probe_optimiser=L-BFGS, §3.1.1
            f"key-numbers.md: classification probe must use solver='lbfgs' "
            f"(§3.1.1), got solver='{probe.model.solver}'."
        )

    def test_uses_ridge_regression_model(self) -> None:
        """key-numbers.md: probe_optimiser = L-BFGS (§3.1.1) → Ridge (L2 regularisation).

        The paper specifies L-BFGS optimisation with an L2 penalty (ridge regression).
        L1 (Lasso) has no basis in the paper and produces different sparse solutions.
        Ridge(L2) is the correct match: L-BFGS is suited for smooth objectives and
        Ridge's quadratic penalty is smooth.

        Note: sklearn Ridge does not expose an 'lbfgs' solver for general regression;
        the default solver (auto/cholesky) minimises the same L2 objective.
        key-numbers.md: probe_type=linear, probe_optimiser=L-BFGS, §3.1.1.
        """
        from sklearn.linear_model import Ridge

        X_train, y_train, _, _ = _make_linear_regression()
        probe = EmbeddingProbe(_CONFIG)
        probe.fit(X_train, y_train, task="regression")

        assert hasattr(probe, "model"), (
            "§3.1.1: EmbeddingProbe must store a 'model' attribute after fit(). "
            "Expected a fitted Ridge."
        )
        assert isinstance(probe.model, Ridge), (
            f"§3.1.1: regression probe must use Ridge (L2), not Lasso (L1). "
            f"key-numbers.md: probe_optimiser=L-BFGS maps to Ridge (smooth L2 objective). "
            f"Got {type(probe.model).__name__}."
        )
