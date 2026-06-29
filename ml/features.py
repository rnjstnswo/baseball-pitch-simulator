"""
Feature engineering and preprocessing pipeline.

Transforms raw Statcast rows into model-ready feature matrices via a
reproducible sklearn Pipeline. Covers all features defined in
docs/PROJECT_SPEC.md §4.

Public interface:
    compute_season_woba(df, id_col) -> pd.DataFrame
    woba_to_ordinal(woba) -> float
    engineer_features(df) -> pd.DataFrame
    build_pipeline() -> sklearn.pipeline.Pipeline
    get_feature_names(pipeline) -> list[str]
    make_train_val_test_split(df, val_year, test_year) -> tuple[...]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WOBA_TIER_THRESHOLDS = {
    "below_avg": (float("-inf"), 0.310),
    "average": (0.310, 0.340),
    "above_avg": (0.340, 0.370),
    "elite": (0.370, float("inf")),
}

WOBA_TIER_ORDINAL = {"below_avg": 0, "average": 1, "above_avg": 2, "elite": 3}

# Feature columns produced by engineer_features() and consumed by the pipeline.
NUMERIC_FEATURES = [
    # 4a pitch characteristics
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "spin_axis",
    # 4b strike-zone location
    "plate_x",
    "plate_z",
    "in_zone",
    "dist_from_center",
    "sz_top",
    "sz_bot",
    # 4c count & game situation
    "balls",
    "strikes",
    "outs_when_up",
    "inning_capped",
    "inning_top",
    "score_diff",
    "on_1b",
    "on_2b",
    "on_3b",
    "runners_encoded",
    # 4d identity
    "p_throws_enc",
    "stand_enc",
    "batter_woba_tier",
    "pitcher_woba_tier",
    # 4e arsenal context
    "pitch_usage_pct",
    "pitch_usage_in_count_pct",
    "avg_speed_for_pitch",
    "avg_spin_for_pitch",
]
CATEGORICAL_FEATURES = ["pitch_type", "zone_str", "count_state"]


def woba_to_ordinal(woba: float) -> float:
    """Map a season wOBA value to its ordinal tier (0–3), or NaN if undefined."""
    if pd.isna(woba):
        return np.nan
    for label, (lo, hi) in WOBA_TIER_THRESHOLDS.items():
        if lo <= woba < hi:
            return WOBA_TIER_ORDINAL[label]
    return np.nan


def compute_season_woba(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Compute per-(player, season) wOBA from Statcast woba_value / woba_denom.

    Args:
        df: Statcast DataFrame with `woba_value`, `woba_denom`, `game_year`.
        id_col: Player id column to group by (`batter` or `pitcher`).

    Returns:
        DataFrame with columns [id_col, "game_year", "woba"].
    """
    ev = df[df["woba_denom"].notna() & (df["woba_denom"] > 0)]
    grouped = (
        ev.groupby([id_col, "game_year"])
        .agg(_num=("woba_value", "sum"), _den=("woba_denom", "sum"))
        .reset_index()
    )
    grouped["woba"] = grouped["_num"] / grouped["_den"]
    return grouped[[id_col, "game_year", "woba"]]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived columns before the sklearn Pipeline runs.

    Args:
        df: Raw Statcast DataFrame with original column names.

    Returns:
        DataFrame with derived columns appended. Original columns preserved.
    """
    df = df.copy()

    # 4b strike-zone location
    df["pitch_type"] = df["pitch_type"].astype("string").fillna("UNK")
    df["zone_str"] = df["zone"].astype("string").fillna("UNK")
    df["in_zone"] = df["zone"].isin(range(1, 10)).astype(int)
    sz_mid = (df["sz_top"] + df["sz_bot"]) / 2
    df["dist_from_center"] = np.sqrt(df["plate_x"] ** 2 + (df["plate_z"] - sz_mid) ** 2)

    # 4c count & game situation
    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["inning_capped"] = df["inning"].clip(upper=9)
    df["inning_top"] = (df["inning_topbot"] == "Top").astype(int)
    df["score_diff"] = (df["bat_score"] - df["fld_score"]).clip(-10, 10)
    on_1b = df["on_1b"].notna().astype(int)
    on_2b = df["on_2b"].notna().astype(int)
    on_3b = df["on_3b"].notna().astype(int)
    df["on_1b"] = on_1b
    df["on_2b"] = on_2b
    df["on_3b"] = on_3b
    df["runners_encoded"] = 4 * on_3b + 2 * on_2b + on_1b

    # 4d identity
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["stand_enc"] = (df["stand"] == "R").astype(int)
    for id_col, feat in [
        ("batter", "batter_woba_tier"),
        ("pitcher", "pitcher_woba_tier"),
    ]:
        woba = compute_season_woba(df, id_col)
        woba[feat] = woba["woba"].map(woba_to_ordinal)
        df = df.merge(
            woba[[id_col, "game_year", feat]], on=[id_col, "game_year"], how="left"
        )

    # 4e arsenal context (season-level aggregates, computed per row)
    by_pt = df.groupby(["pitcher", "game_year", "pitch_type"])
    by_p = df.groupby(["pitcher", "game_year"])["pitch_type"].transform("size")
    df["pitch_usage_pct"] = by_pt["pitch_type"].transform("size") / by_p
    df["avg_speed_for_pitch"] = by_pt["release_speed"].transform("mean")
    df["avg_spin_for_pitch"] = by_pt["release_spin_rate"].transform("mean")
    by_count = df.groupby(["pitcher", "game_year", "count_state"])[
        "pitch_type"
    ].transform("size")
    by_count_pt = df.groupby(["pitcher", "game_year", "count_state", "pitch_type"])[
        "pitch_type"
    ].transform("size")
    df["pitch_usage_in_count_pct"] = by_count_pt / by_count

    return df


def build_pipeline() -> Pipeline:
    """Build the full sklearn preprocessing Pipeline.

    Median-imputes + scales numeric features and one-hot encodes the
    categorical features. Fit on the training split only.

    Returns:
        Unfitted sklearn Pipeline. Call .fit_transform(X_train) to fit,
        then .transform(X_val/X_test).
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the ordered output feature names of a fitted pipeline.

    Used to align SHAP values with human-readable names.
    """
    return pipeline.named_steps["preprocess"].get_feature_names_out().tolist()


def make_train_val_test_split(
    df: pd.DataFrame,
    val_year: int = 2023,
    test_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological date-based train / validation / test split.

    Args:
        df: Full labeled DataFrame with a `game_date` column.
        val_year: Season year used for validation.
        test_year: Season year used for final test evaluation.

    Returns:
        (train_df, val_df, test_df) — no row overlap guaranteed by date boundary.
    """
    year = pd.to_datetime(df["game_date"]).dt.year
    train = df[year < val_year]
    val = df[year == val_year]
    test = df[year == test_year]
    return train, val, test
