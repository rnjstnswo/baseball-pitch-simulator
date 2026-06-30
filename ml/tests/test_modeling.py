"""Unit tests for ml/modeling.py — metrics and the train→calibrate→predict path."""

import numpy as np
import pytest

from ml.modeling import calibrate, compute_ece, evaluate, train_baselines, tune_lgbm


def _toy_classification(n=400, n_classes=3, n_features=6, seed=0):
    """Separable-ish multiclass data so models train to non-trivial accuracy."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # class signal on the first two features
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 0] > 1.0).astype(int)
    y = np.clip(y, 0, n_classes - 1)
    return X, y


def test_compute_ece_perfectly_calibrated_is_low():
    # One-hot probabilities that are always correct => zero gap.
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 1, 0, 2])
    proba = np.eye(3)[y]
    assert compute_ece(y, proba, classes) == pytest.approx(0.0)


def test_compute_ece_overconfident_wrong_is_high():
    classes = np.array([0, 1])
    y = np.array([0, 0, 0, 0])
    # Always predicts class 1 with full confidence but is always wrong.
    proba = np.tile([0.0, 1.0], (4, 1))
    assert compute_ece(y, proba, classes) == pytest.approx(1.0)


def test_evaluate_returns_finite_metrics():
    X, y = _toy_classification()
    models = train_baselines(X, y)
    metrics = evaluate(models["logistic"], X, y)
    assert set(metrics) == {"log_loss", "macro_f1", "ece"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_tune_and_calibrate_end_to_end():
    X, y = _toy_classification(n=600)
    Xtr, ytr = X[:400], y[:400]
    Xval, yval = X[400:], y[400:]
    booster = tune_lgbm(Xtr, ytr, Xval, yval, n_trials=3)
    cal = calibrate(booster, Xval, yval)
    proba = cal.predict_proba(Xval)
    assert proba.shape == (len(Xval), len(np.unique(y)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # Calibrated model should beat the prior-only dummy on log-loss.
    dummy = train_baselines(Xtr, ytr)["dummy"]
    assert (
        evaluate(cal, Xval, yval)["log_loss"] < evaluate(dummy, Xval, yval)["log_loss"]
    )
