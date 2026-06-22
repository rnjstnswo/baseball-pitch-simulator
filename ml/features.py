"""
Feature engineering and preprocessing pipeline.

Transforms raw Statcast rows into model-ready feature matrices via a
reproducible sklearn Pipeline. Covers all features defined in
docs/PROJECT_SPEC.md §4.

Public interface:
    build_pipeline() -> sklearn.pipeline.Pipeline
    engineer_features(df) -> pd.DataFrame
    get_feature_names() -> list[str]
    make_train_val_test_split(df, val_year, test_year) -> tuple[...]
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

WOBA_TIER_THRESHOLDS = {
    "below_avg": (float("-inf"), 0.310),
    "average": (0.310, 0.340),
    "above_avg": (0.340, 0.370),
    "elite": (0.370, float("inf")),
}

WOBA_TIER_ORDINAL = {"below_avg": 0, "average": 1, "above_avg": 2, "elite": 3}


def build_pipeline() -> Pipeline:
    """Build the full sklearn preprocessing Pipeline.

    Steps:
        1. Derived feature construction (dist_from_center, runners_encoded, etc.)
        2. Median imputation for spin_rate, spin_axis
        3. One-hot encoding for pitch_type, zone, count_state
        4. Binary encoding for p_throws, stand
        5. Passthrough for numeric columns

    Returns:
        Unfitted sklearn Pipeline. Call .fit_transform(X_train) to fit,
        then .transform(X_val/X_test).
    """
    raise NotImplementedError


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived columns before the sklearn Pipeline runs.

    Derived columns added:
        - dist_from_center: Euclidean distance from strike zone center
        - in_zone: bool flag (zone 1–9)
        - count_state: "{balls}-{strikes}" string
        - runners_encoded: 3-bit integer (0–7)
        - score_diff: clipped to [-10, 10]
        - inning_capped: min(inning, 9)

    Args:
        df: Raw Statcast DataFrame with original column names.

    Returns:
        DataFrame with derived columns appended. Original columns preserved.
    """
    raise NotImplementedError


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names output by the pipeline.

    Used to align SHAP values with human-readable names.
    """
    raise NotImplementedError


def make_train_val_test_split(
    df: pd.DataFrame,
    val_year: int = 2023,
    test_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological date-based train / validation / test split.

    Args:
        df: Full labeled DataFrame with a `game_date` column (datetime).
        val_year: Season year used for validation.
        test_year: Season year used for final test evaluation.

    Returns:
        (train_df, val_df, test_df) — no row overlap guaranteed by date boundary.
    """
    raise NotImplementedError
