"""
Training script for Model 1: Pitch Outcome Classifier.

Trains a multiclass classifier over {ball, called_strike, swinging_strike,
foul, in_play, hit_by_pitch} using the feature pipeline from features.py.

Pipeline:
    1. Load labeled Parquet
    2. Engineer features and split (train/val/test by year)
    3. Fit baselines (dummy, logistic regression, single tree)
    4. Tune XGBoost or LightGBM with Optuna
    5. Calibrate with CalibratedClassifierCV (isotonic)
    6. Evaluate: log-loss, macro-F1, ECE on val and test splits
    7. Serialize: ml/artifacts/pitch_outcome_model.joblib
                  ml/artifacts/preprocessor.joblib

Public interface:
    train(X_train, y_train, X_val, y_val) -> CalibratedClassifierCV
    evaluate(model, X, y) -> dict[str, float]
    serialize(model, preprocessor, output_dir) -> None
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
logger = logging.getLogger(__name__)


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Fit dummy, logistic regression, and single decision tree baselines.

    Returns:
        Dict mapping model name → fitted model, for later evaluation logging.
    """
    raise NotImplementedError


def tune_boosted_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> object:
    """Run Optuna hyperparameter search over XGBoost or LightGBM.

    Args:
        n_trials: Number of Optuna trials.

    Returns:
        Best fitted (uncalibrated) booster model.
    """
    raise NotImplementedError


def calibrate(model: object, X_val: pd.DataFrame, y_val: pd.Series) -> object:
    """Wrap model in CalibratedClassifierCV with isotonic regression.

    Returns:
        Fitted CalibratedClassifierCV.
    """
    raise NotImplementedError


def evaluate(model: object, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute log-loss, macro-F1, and ECE for a fitted model.

    Returns:
        Dict with keys: log_loss, macro_f1, ece.
    """
    raise NotImplementedError


def serialize(model: object, preprocessor: object, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save model and preprocessor to output_dir with joblib."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 1: Pitch Outcome")
    parser.add_argument("--input", type=Path, required=True, help="Path to labeled Parquet")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Training pitch outcome model — not yet implemented.")
