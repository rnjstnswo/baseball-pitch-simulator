"""
Chained model inference for the /predict endpoint.

Loads serialized artifacts, runs the two-model chain (pitch outcome →
optional BIP outcome), generates SHAP explanations, and attaches usage
context from the precomputed arsenal/usage tables.

Public interface:
    load_artifacts(artifacts_dir) -> ModelBundle
    run_inference(request, bundle) -> PredictResponse
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.schemas import PredictRequest, PredictResponse

ARTIFACTS_DIR = Path(__file__).parent.parent / "ml" / "artifacts"


@dataclass
class ModelBundle:
    """Container for all loaded artifacts needed at inference time."""

    pitch_outcome_model: object
    bip_model: object
    preprocessor: object
    pitch_outcome_explainer: object
    bip_explainer: object
    arsenal_df: pd.DataFrame
    usage_df: pd.DataFrame
    feature_names: list[str]


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelBundle:
    """Load all serialized model artifacts and lookup tables.

    Args:
        artifacts_dir: Directory containing .joblib files and processed Parquets.

    Returns:
        Populated ModelBundle ready for inference.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    raise NotImplementedError


def run_inference(request: PredictRequest, bundle: ModelBundle) -> PredictResponse:
    """Execute the full prediction chain for a single request.

    Steps:
        1. Build feature row from request fields
        2. Attach arsenal context features (avg_speed, avg_spin, usage_pct, etc.)
        3. Apply preprocessor transform
        4. Run Model 1 (pitch outcome) → probabilities
        5. If predicted outcome == "in_play", run Model 2 (BIP)
        6. Compute SHAP values for Model 1 (and Model 2 if applicable)
        7. Look up usage context from usage_df
        8. Compute updated game state
        9. Return PredictResponse

    Args:
        request: Validated PredictRequest from the API layer.
        bundle: Loaded ModelBundle.

    Returns:
        Fully populated PredictResponse.

    Raises:
        ValueError: If pitcher_id is not in the arsenal table.
        ValueError: If pitch_type is not in the pitcher's arsenal.
    """
    raise NotImplementedError


def _build_feature_row(request: PredictRequest, bundle: ModelBundle) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row feature DataFrame.

    Attaches arsenal-derived features (avg_speed, avg_spin, usage_pct, etc.)
    from the precomputed arsenal table.
    """
    raise NotImplementedError


def _compute_updated_state(
    request: PredictRequest, pitch_prediction: str, bip_prediction: str | None
) -> dict:
    """Determine the resulting count / base / out state after the pitch."""
    raise NotImplementedError
