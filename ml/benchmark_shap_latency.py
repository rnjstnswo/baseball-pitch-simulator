"""
Quick latency check for SHAP explanations against the *real* serialized models.

Mirrors the /predict explanation path: load the production artifacts, build a
TreeExplainer, and time single-row SHAP inference on a real preprocessed row.
Confirms Phase 3 exit criterion "SHAP explanations generate in < 200ms per
prediction" against the actual booster (not a toy model).

NOTE: the explainer is built WITHOUT a background dataset, so SHAP uses the
`tree_path_dependent` perturbation. This is required to hit the latency budget —
passing an interventional background sample (even ~500 rows) pushes a single
pitch-outcome explanation past 1.5s. /predict must build the explainer the same
way.

Usage:
    python ml/benchmark_shap_latency.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from ml.explain import build_explainer, compute_shap_values
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

ARTIFACTS = Path("ml/artifacts")
LABELED = Path("data/processed/labeled.parquet")
BUDGET_S = 0.2
N_ROWS = 2000  # engineer_features needs enough rows for its groupby aggregates


def _sample_features(n: int) -> pd.DataFrame:
    df = pd.read_parquet(LABELED).tail(n)
    return engineer_features(df)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _time_single_row(model_path: Path, X_transformed) -> float:
    model = joblib.load(model_path)
    explainer = build_explainer(model)  # no background -> tree_path_dependent
    compute_shap_values(explainer, X_transformed[:1])  # warm up
    start = time.perf_counter()
    compute_shap_values(explainer, X_transformed[:1])
    return time.perf_counter() - start


def main() -> None:
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
    X = preprocessor.transform(_sample_features(N_ROWS))

    ok = True
    for name, path in [
        ("pitch_outcome", ARTIFACTS / "pitch_outcome_model.joblib"),
        ("bip", ARTIFACTS / "bip_model.joblib"),
    ]:
        elapsed_ms = _time_single_row(path, X) * 1000
        status = "OK" if elapsed_ms < BUDGET_S * 1000 else "FAIL"
        ok &= elapsed_ms < BUDGET_S * 1000
        print(f"[{status}] {name}: {elapsed_ms:.1f}ms (budget {BUDGET_S * 1000:.0f}ms)")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
