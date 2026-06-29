"""
Precomputes per-pitcher arsenal and count-level usage tables.

Reads processed Statcast data and writes three Parquet files to data/processed/:
    arsenal.parquet       — pitch type stats per pitcher per season
    usage.parquet         — pitch type usage % per pitcher per count state
    pitcher_woba.parquet  — pitcher wOBA-against tier lookup (for serving)

Public interface:
    build_arsenal_table(df) -> pd.DataFrame
    build_usage_table(df) -> pd.DataFrame
    build_pitcher_woba_table(df) -> pd.DataFrame
    save_tables(arsenal_df, usage_df, woba_df, output_dir) -> None
    load_arsenal(output_dir) -> pd.DataFrame
    load_usage(output_dir) -> pd.DataFrame
    load_pitcher_woba(output_dir) -> pd.DataFrame
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from ml.features import compute_season_woba, woba_to_ordinal
except ImportError:  # running as a script: ml/ is on sys.path, project root is not
    from features import compute_season_woba, woba_to_ordinal

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
logger = logging.getLogger(__name__)


def build_arsenal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pitcher, per-pitch-type season statistics.

    Output columns:
        pitcher_id, season, pitch_type, pitch_name,
        usage_pct, avg_speed, avg_spin, avg_pfx_x, avg_pfx_z, sample_size
    """
    arsenal = (
        df.groupby(["pitcher", "game_year", "pitch_type"])
        .agg(
            pitch_name=("pitch_name", "first"),
            avg_speed=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_pfx_x=("pfx_x", "mean"),
            avg_pfx_z=("pfx_z", "mean"),
            sample_size=("pitch_type", "size"),
        )
        .reset_index()
    )
    total = arsenal.groupby(["pitcher", "game_year"])["sample_size"].transform("sum")
    arsenal["usage_pct"] = arsenal["sample_size"] / total
    return arsenal.rename(columns={"pitcher": "pitcher_id", "game_year": "season"})


def build_usage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch type usage % per pitcher per count state.

    Output columns:
        pitcher_id, season, count_state, pitch_type, usage_pct, sample_size
    """
    df = df.copy()
    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    usage = (
        df.groupby(["pitcher", "game_year", "count_state", "pitch_type"])
        .size()
        .reset_index(name="sample_size")
    )
    total = usage.groupby(["pitcher", "game_year", "count_state"])[
        "sample_size"
    ].transform("sum")
    usage["usage_pct"] = usage["sample_size"] / total
    return usage.rename(columns={"pitcher": "pitcher_id", "game_year": "season"})


def build_pitcher_woba_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pitcher, per-season wOBA-against and its ordinal tier.

    Output columns:
        pitcher_id, season, woba, woba_tier
    """
    woba = compute_season_woba(df, "pitcher")
    woba["woba_tier"] = woba["woba"].map(woba_to_ordinal)
    return woba.rename(columns={"pitcher": "pitcher_id", "game_year": "season"})


def save_tables(
    arsenal_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    woba_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write arsenal, usage, and pitcher-wOBA DataFrames to Parquet in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arsenal_df.to_parquet(output_dir / "arsenal.parquet", index=False)
    usage_df.to_parquet(output_dir / "usage.parquet", index=False)
    woba_df.to_parquet(output_dir / "pitcher_woba.parquet", index=False)
    logger.info(
        "Wrote arsenal (%d rows), usage (%d rows), pitcher_woba (%d rows) to %s",
        len(arsenal_df),
        len(usage_df),
        len(woba_df),
        output_dir,
    )


def load_arsenal(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed arsenal table from Parquet."""
    return pd.read_parquet(output_dir / "arsenal.parquet")


def load_usage(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed usage table from Parquet."""
    return pd.read_parquet(output_dir / "usage.parquet")


def load_pitcher_woba(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed pitcher wOBA-against tier table from Parquet."""
    return pd.read_parquet(output_dir / "pitcher_woba.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute arsenal and usage tables")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to labeled Parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    arsenal_df = build_arsenal_table(df)
    usage_df = build_usage_table(df)
    woba_df = build_pitcher_woba_table(df)
    save_tables(arsenal_df, usage_df, woba_df, args.output_dir)
