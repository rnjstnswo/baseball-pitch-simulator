"""
Training script for Model 2: Ball-in-Play Outcome Classifier.

Trains a multiclass classifier over {out, single, double, triple, home_run}
using only the subset of pitches where pitch_outcome == "in_play".

Same pipeline structure as train_pitch_outcome.py:
    baselines → tuned booster → calibration → evaluation → serialization

Serializes to: ml/artifacts/bip_model.joblib

Public interface:
    train(X_train, y_train, X_val, y_val) -> CalibratedClassifierCV
    evaluate(model, X, y) -> dict[str, float]
    serialize(model, output_dir) -> None
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
    """Fit dummy, logistic regression, and single decision tree baselines on BIP rows.

    Returns:
        Dict mapping model name → fitted model.
    """
    raise NotImplementedError


def tune_boosted_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> object:
    """Run Optuna hyperparameter search over XGBoost or LightGBM on BIP rows.

    Returns:
        Best fitted (uncalibrated) booster.
    """
    raise NotImplementedError


def calibrate(model: object, X_val: pd.DataFrame, y_val: pd.Series) -> object:
    """Wrap model in CalibratedClassifierCV with isotonic regression.

    Returns:
        Fitted CalibratedClassifierCV.
    """
    raise NotImplementedError


def evaluate(model: object, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute log-loss, macro-F1, and ECE for a fitted BIP model.

    Returns:
        Dict with keys: log_loss, macro_f1, ece.
    """
    raise NotImplementedError


def serialize(model: object, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save BIP model to output_dir with joblib."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2: Ball-in-Play Outcome")
    parser.add_argument("--input", type=Path, required=True, help="Path to labeled in-play Parquet")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Training BIP outcome model — not yet implemented.")
