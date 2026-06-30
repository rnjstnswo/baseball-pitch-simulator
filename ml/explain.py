"""
SHAP-based explanation generation.

Computes SHAP values for Model 1 (pitch outcome) and optionally Model 2
(BIP outcome), then converts the top drivers into a plain-English sentence
suitable for display in the frontend.

SHAP runs against the *raw* LightGBM booster, not the CalibratedClassifierCV
wrapper: TreeExplainer cannot explain the calibrator, and isotonic calibration
is monotonic per class, so it does not change which features drive a class —
only the final probability. `build_explainer` unwraps the booster automatically.

Public interface:
    build_explainer(model, background_data) -> shap.TreeExplainer
    compute_shap_values(explainer, X) -> np.ndarray
    top_factors(shap_values, feature_names, class_idx, n) -> list[dict]
    to_plain_english(prediction, top_factors) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

N_BACKGROUND_SAMPLES = 500


def _unwrap_booster(model: object) -> object:
    """Return the underlying tree model, unwrapping a CalibratedClassifierCV.

    Handles both a bare booster (passed straight through) and a calibrated
    model whose first calibrator wraps a FrozenEstimator around the booster.
    """
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator
        return getattr(estimator, "estimator", estimator)  # FrozenEstimator.estimator
    return model


def build_explainer(
    model: object, background_data: pd.DataFrame | None = None
) -> object:
    """Build a SHAP TreeExplainer for the model's underlying booster.

    Args:
        model: Fitted calibrated classifier (or a bare booster).
        background_data: Optional representative sample used as the background
            distribution. If omitted, TreeExplainer uses the tree-path-dependent
            estimator, which needs no background data.

    Returns:
        Fitted shap.TreeExplainer instance.
    """
    booster = _unwrap_booster(model)
    if background_data is not None and len(background_data) > N_BACKGROUND_SAMPLES:
        background_data = shap.sample(
            background_data, N_BACKGROUND_SAMPLES, random_state=0
        )
    return shap.TreeExplainer(booster, data=background_data)


def compute_shap_values(explainer: object, X) -> np.ndarray:
    """Run SHAP inference on X.

    Args:
        explainer: Fitted shap.TreeExplainer.
        X: Feature matrix (1 or more rows), already preprocessed.

    Returns:
        SHAP values array of shape (n_samples, n_features, n_classes).
    """
    return explainer(X, check_additivity=False).values


def top_factors(
    shap_values: np.ndarray,
    feature_names: list[str],
    class_idx: int,
    n: int = 4,
) -> list[dict]:
    """Return the top-n SHAP contributors for a given predicted class.

    Args:
        shap_values: Array of shape (1, n_features, n_classes) for a single prediction.
        feature_names: Feature names aligned with axis 1.
        class_idx: Index of the predicted class to explain.
        n: Number of top factors to return.

    Returns:
        List of dicts [{"feature", "shap_value", "direction"}], sorted by
        abs(shap_value) descending.
    """
    contributions = np.asarray(shap_values)[0, :, class_idx]
    order = np.argsort(np.abs(contributions))[::-1][:n]
    factors = []
    for idx in order:
        value = float(contributions[idx])
        factors.append(
            {
                "feature": feature_names[idx],
                "shap_value": value,
                "direction": "increases" if value >= 0 else "decreases",
            }
        )
    return factors


def to_plain_english(prediction: str, factors: list[dict]) -> str:
    """Convert a prediction and its top SHAP factors to a plain-English sentence.

    Args:
        prediction: Predicted outcome label (e.g. "swinging_strike").
        factors: Output of top_factors().

    Returns:
        1–3 sentence string suitable for the frontend explanation panel.
    """
    label = prediction.replace("_", " ")
    if not factors:
        return f"This pitch is most likely to result in a {label}."
    drivers = ", ".join(
        f"{f['feature']} ({'+' if f['shap_value'] >= 0 else '−'})" for f in factors
    )
    return (
        f"This pitch is most likely to result in a {label}. "
        f"The biggest factors are: {drivers}."
    )
