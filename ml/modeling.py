"""
Shared modeling helpers for both training scripts (Model 1 and Model 2).

Holds the logic that is identical across pitch-outcome and ball-in-play
training: data preparation, baseline fitting, LightGBM tuning with Optuna,
isotonic calibration, and metric computation (log-loss / macro-F1 / ECE).

Public interface:
    load_features(path, target, sample_frac, in_play_only) -> (train, val, test)
    train_baselines(X_train, y_train) -> dict[str, estimator]
    tune_lgbm(X_train, y_train, X_val, y_val, n_trials, class_weight) -> LGBMClassifier
    calibrate(model, X_val, y_val) -> CalibratedClassifierCV
    evaluate(model, X, y) -> dict[str, float]
    compute_ece(y_true, proba, classes, n_bins) -> float
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.tree import DecisionTreeClassifier

try:
    from ml.features import build_pipeline, engineer_features, make_train_val_test_split
except ImportError:  # running as a script: ml/ is on sys.path, project root is not
    from features import build_pipeline, engineer_features, make_train_val_test_split

__all__ = [
    "build_pipeline",
    "load_features",
    "train_baselines",
    "tune_lgbm",
    "calibrate",
    "evaluate",
    "compute_ece",
]

logger = logging.getLogger(__name__)


def load_features(
    path,
    target: str,
    sample_frac: float | None = None,
    in_play_only: bool = False,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labeled Parquet, engineer features, and split chronologically.

    Features are always engineered on the full frame *before* any in-play
    filtering so the §4e arsenal-context denominators count every pitch
    (filtering first would corrupt usage %). The target column is then
    required non-null and the rows are split by year via features.py.

    Args:
        path: Path to labeled Parquet (the full set, not the in-play subset).
        target: Label column to require non-null ("pitch_outcome" / "bip_outcome").
        sample_frac: If set, sample this fraction of rows up front (smoke runs only).
        in_play_only: If True, keep only pitch_outcome == "in_play" rows (Model 2).

    Returns:
        (train_df, val_df, test_df) with derived feature columns attached.
    """
    df = pd.read_parquet(path)
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(
            drop=True
        )
    logger.info("Loaded %d rows from %s", len(df), path)

    df = engineer_features(df)
    if in_play_only:
        df = df[df["pitch_outcome"] == "in_play"].copy()
    df = df[df[target].notna()].reset_index(drop=True)
    logger.info("After engineering/filtering: %d rows (target=%s)", len(df), target)

    return make_train_val_test_split(df)


def train_baselines(X_train, y_train) -> dict[str, object]:
    """Fit dummy, logistic-regression, and single-tree baselines on the train split.

    Returns:
        Dict mapping baseline name -> fitted estimator.
    """
    models: dict[str, object] = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=500),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=0),
    }
    for name, model in models.items():
        logger.info("Fitting baseline: %s", name)
        model.fit(X_train, y_train)
    return models


def tune_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
    class_weight: str | None = None,
    random_state: int = 0,
) -> lgb.LGBMClassifier:
    """Optuna search over LightGBM, scored by validation multiclass log-loss.

    Each trial fits with native early stopping on the validation set; the best
    params are then refit (also with early stopping) and returned uncalibrated.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    classes = np.unique(y_train)
    fixed = dict(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=1000,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        subsample_freq=1,
    )

    def _fit(model: lgb.LGBMClassifier) -> lgb.LGBMClassifier:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        model = _fit(lgb.LGBMClassifier(**fixed, **params))
        return log_loss(y_val, model.predict_proba(X_val), labels=model.classes_)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(
        "Best trial val log-loss=%.4f params=%s", study.best_value, study.best_params
    )

    return _fit(lgb.LGBMClassifier(**fixed, **study.best_params))


def calibrate(model, X_val, y_val) -> CalibratedClassifierCV:
    """Isotonic-calibrate an already-fitted model on the validation split.

    Uses FrozenEstimator so the underlying booster is not refit (the modern
    replacement for the removed cv="prefit" option).
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    return calibrated


def compute_ece(y_true, proba, classes, n_bins: int = 15) -> float:
    """Expected Calibration Error using top-label confidence binning.

    Args:
        y_true: True labels (same dtype as `classes`).
        proba: Predicted probability matrix (n_samples, n_classes).
        classes: Class labels aligned with proba columns (model.classes_).
        n_bins: Number of equal-width confidence bins on [0, 1].

    Returns:
        Weighted average gap between bin accuracy and bin confidence.
    """
    proba = np.asarray(proba)
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        ece += (count / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def evaluate(model, X, y) -> dict[str, float]:
    """Compute log-loss, macro-F1, and ECE for a fitted classifier on (X, y)."""
    classes = model.classes_
    proba = model.predict_proba(X)
    predictions = model.predict(X)
    return {
        "log_loss": float(log_loss(y, proba, labels=classes)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "ece": compute_ece(y, proba, classes),
    }
