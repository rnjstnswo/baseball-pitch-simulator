"""
SHAP-based explanation generation.

Computes SHAP values for Model 1 (pitch outcome) and optionally Model 2
(BIP outcome), then converts the top drivers into a plain-English sentence
suitable for display in the frontend.

Public interface:
    build_explainer(model, background_data) -> shap.Explainer
    compute_shap_values(explainer, X) -> np.ndarray
    top_factors(shap_values, feature_names, class_idx, n) -> list[dict]
    to_plain_english(prediction, top_factors) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BACKGROUND_SAMPLES = 500


def build_explainer(model: object, background_data: pd.DataFrame) -> object:
    """Build a SHAP TreeExplainer (or KernelExplainer fallback) for the given model.

    Args:
        model: Fitted calibrated classifier.
        background_data: Representative sample (N_BACKGROUND_SAMPLES rows) used
                         as the SHAP background distribution.

    Returns:
        Fitted shap.Explainer instance.
    """
    raise NotImplementedError


def compute_shap_values(explainer: object, X: pd.DataFrame) -> np.ndarray:
    """Run SHAP inference on X.

    Args:
        explainer: Fitted shap.Explainer.
        X: Feature matrix (1 or more rows), already preprocessed.

    Returns:
        SHAP values array of shape (n_samples, n_features, n_classes).
    """
    raise NotImplementedError


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
        List of dicts: [{"feature": str, "shap_value": float, "direction": str}, ...]
        sorted by abs(shap_value) descending.
    """
    raise NotImplementedError


def to_plain_english(prediction: str, factors: list[dict]) -> str:
    """Convert a prediction and its top SHAP factors to a plain-English sentence.

    Args:
        prediction: Predicted outcome label (e.g. "swinging_strike").
        factors: Output of top_factors().

    Returns:
        1–3 sentence string suitable for display in the frontend explanation panel.
    """
    raise NotImplementedError
