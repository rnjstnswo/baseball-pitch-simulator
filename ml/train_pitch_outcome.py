"""
Training script for Model 1: Pitch Outcome Classifier.

Trains a multiclass classifier over {ball, called_strike, swinging_strike,
foul, in_play, hit_by_pitch} using the feature pipeline from features.py.

Pipeline:
    1. Load labeled Parquet
    2. Engineer features and split (train/val/test by year)
    3. Fit preprocessor on the train split only
    4. Fit baselines (dummy, logistic regression, single tree)
    5. Tune LightGBM with Optuna
    6. Calibrate with CalibratedClassifierCV (isotonic) on the validation split
    7. Evaluate: log-loss, macro-F1, ECE on the held-out 2024 test split
    8. Serialize: ml/artifacts/pitch_outcome_model.joblib
                  ml/artifacts/preprocessor.joblib

Public interface:
    train_baselines / tune_boosted_model / calibrate / evaluate / serialize
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib

try:
    from ml import modeling
except ImportError:  # running as a script: ml/ is on sys.path, project root is not
    import modeling

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TARGET = "pitch_outcome"
logger = logging.getLogger(__name__)

# Thin re-exports so the documented public interface is importable from here.
train_baselines = modeling.train_baselines
tune_boosted_model = modeling.tune_lgbm
calibrate = modeling.calibrate
evaluate = modeling.evaluate


def serialize(model, preprocessor, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save the calibrated model and the fitted preprocessor with joblib."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "pitch_outcome_model.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    logger.info("Serialized pitch_outcome_model.joblib + preprocessor.joblib")


def _run(input_path: Path, output_dir: Path, n_trials: int, sample_frac: float | None):
    train, val, test = modeling.load_features(
        input_path, TARGET, sample_frac=sample_frac, in_play_only=False
    )
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    preprocessor = modeling.build_pipeline()
    X_train = preprocessor.fit_transform(train)
    X_val = preprocessor.transform(val)
    X_test = preprocessor.transform(test)
    logger.info(
        "Feature matrix: train=%s val=%s test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    results: dict[str, dict] = {}

    baselines = modeling.train_baselines(X_train, y_train)
    for name, model in baselines.items():
        results[name] = modeling.evaluate(model, X_test, y_test)

    booster = modeling.tune_lgbm(X_train, y_train, X_val, y_val, n_trials=n_trials)
    results["lgbm_tuned"] = modeling.evaluate(booster, X_test, y_test)

    calibrated = modeling.calibrate(booster, X_val, y_val)
    results["lgbm_calibrated"] = modeling.evaluate(calibrated, X_test, y_test)

    _log_results(results)
    _check_exit_criteria(results)

    serialize(calibrated, preprocessor, output_dir)
    metrics_path = output_dir / "metrics_pitch_outcome.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote metrics to %s", metrics_path)


def _log_results(results: dict[str, dict]) -> None:
    logger.info("=== Model 1 (pitch outcome) — test metrics ===")
    logger.info("%-18s %10s %10s %8s", "model", "log_loss", "macro_f1", "ece")
    for name, m in results.items():
        logger.info(
            "%-18s %10.4f %10.4f %8.4f", name, m["log_loss"], m["macro_f1"], m["ece"]
        )


def _check_exit_criteria(results: dict[str, dict]) -> None:
    tuned, logistic = results["lgbm_tuned"], results["logistic"]
    if tuned["log_loss"] >= logistic["log_loss"]:
        logger.warning(
            "Tuned log-loss %.4f did NOT beat logistic %.4f",
            tuned["log_loss"],
            logistic["log_loss"],
        )
    ece = results["lgbm_calibrated"]["ece"]
    if ece >= 0.05:
        logger.warning("Calibrated ECE %.4f exceeds target 0.05", ece)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 1: Pitch Outcome")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "labeled.parquet",
        help="Path to labeled Parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of rows to use (smoke runs only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _run(args.input, args.output_dir, args.n_trials, args.sample_frac)
