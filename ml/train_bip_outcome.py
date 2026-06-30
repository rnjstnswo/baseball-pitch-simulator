"""
Training script for Model 2: Ball-in-Play Outcome Classifier.

Trains a multiclass classifier over {out, single, double, triple, home_run}
using only the subset of pitches where pitch_outcome == "in_play".

Features are engineered on the *full* frame before the in-play filter so the
§4e arsenal-context denominators stay correct, then the rows are subset. The
preprocessor fit by train_pitch_outcome.py is reused (transform only) so both
models share one feature space — matching how the API serves a single
preprocessed row to both models.

Serializes to: ml/artifacts/bip_model.joblib

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
TARGET = "bip_outcome"
logger = logging.getLogger(__name__)

train_baselines = modeling.train_baselines
tune_boosted_model = modeling.tune_lgbm
calibrate = modeling.calibrate
evaluate = modeling.evaluate


def serialize(model, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save the calibrated BIP model with joblib."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "bip_model.joblib")
    logger.info("Serialized bip_model.joblib")


def _run(input_path: Path, output_dir: Path, n_trials: int, sample_frac: float | None):
    preprocessor_path = output_dir / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"{preprocessor_path} not found — run train_pitch_outcome.py first "
            "so both models share one fitted preprocessor."
        )
    preprocessor = joblib.load(preprocessor_path)

    train, val, test = modeling.load_features(
        input_path, TARGET, sample_frac=sample_frac, in_play_only=True
    )
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    X_train = preprocessor.transform(train)
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

    serialize(calibrated, output_dir)
    metrics_path = output_dir / "metrics_bip_outcome.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote metrics to %s", metrics_path)


def _log_results(results: dict[str, dict]) -> None:
    logger.info("=== Model 2 (ball-in-play) — test metrics ===")
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
    parser = argparse.ArgumentParser(description="Train Model 2: Ball-in-Play Outcome")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "labeled.parquet",
        help="Path to labeled Parquet (full set; in-play filter applied internally)",
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
