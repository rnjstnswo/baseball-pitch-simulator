"""Unit tests for ml/explain.py — SHAP factor extraction and phrasing."""

import time

import lightgbm as lgb
import numpy as np

from ml.explain import (
    build_explainer,
    compute_shap_values,
    to_plain_english,
    top_factors,
)

FEATURE_NAMES = [f"f{i}" for i in range(6)]


def _fitted_booster(n=300, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 0] > 1.0).astype(int)
    y = np.clip(y, 0, n_classes - 1)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1, random_state=seed)
    model.fit(X, y)
    return model, X


def test_top_factors_sorted_and_sized():
    model, X = _fitted_booster()
    explainer = build_explainer(model)
    shap_values = compute_shap_values(explainer, X[:1])
    factors = top_factors(shap_values, FEATURE_NAMES, class_idx=0, n=3)
    assert len(factors) == 3
    magnitudes = [abs(f["shap_value"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(f["direction"] in {"increases", "decreases"} for f in factors)
    assert all(f["feature"] in FEATURE_NAMES for f in factors)


def test_to_plain_english_non_empty_and_mentions_prediction():
    factors = [
        {"feature": "strikes", "shap_value": 0.2, "direction": "increases"},
        {"feature": "plate_x", "shap_value": -0.1, "direction": "decreases"},
    ]
    text = to_plain_english("swinging_strike", factors)
    assert "swinging strike" in text
    assert "strikes" in text and "plate_x" in text


def test_to_plain_english_handles_no_factors():
    assert to_plain_english("ball", []).strip() != ""


def test_shap_latency_under_200ms():
    model, X = _fitted_booster()
    explainer = build_explainer(model)
    compute_shap_values(explainer, X[:1])  # warm up
    start = time.perf_counter()
    compute_shap_values(explainer, X[:1])
    assert (time.perf_counter() - start) < 0.2
