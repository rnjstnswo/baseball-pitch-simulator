"""
Precomputes per-pitcher arsenal and count-level usage tables.

Reads processed Statcast data and writes two Parquet files to data/processed/:
    arsenal.parquet  — pitch type stats per pitcher per season
    usage.parquet    — pitch type usage % per pitcher per count state

Public interface:
    build_arsenal_table(df) -> pd.DataFrame
    build_usage_table(df) -> pd.DataFrame
    save_tables(arsenal_df, usage_df, output_dir) -> None
    load_arsenal(output_dir) -> pd.DataFrame
    load_usage(output_dir) -> pd.DataFrame
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
logger = logging.getLogger(__name__)


def build_arsenal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pitcher, per-pitch-type season statistics.

    Output columns:
        pitcher_id, season, pitch_type, pitch_name,
        usage_pct, avg_speed, avg_spin, avg_pfx_x, avg_pfx_z, sample_size

    Args:
        df: Labeled Statcast DataFrame with pitch_type, release_speed, etc.

    Returns:
        Arsenal summary DataFrame (one row per pitcher × pitch_type × season).
    """
    raise NotImplementedError


def build_usage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch type usage % per pitcher per count state.

    Output columns:
        pitcher_id, season, count_state, pitch_type, usage_pct, sample_size

    Args:
        df: Labeled Statcast DataFrame.

    Returns:
        Usage summary DataFrame (one row per pitcher × count × pitch_type × season).
    """
    raise NotImplementedError


def save_tables(
    arsenal_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write arsenal and usage DataFrames to Parquet in output_dir."""
    raise NotImplementedError


def load_arsenal(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed arsenal table from Parquet."""
    raise NotImplementedError


def load_usage(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed usage table from Parquet."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute arsenal and usage tables")
    parser.add_argument("--input", type=Path, required=True, help="Path to labeled Parquet")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.input)
    arsenal_df = build_arsenal_table(df)
    usage_df = build_usage_table(df)
    save_tables(arsenal_df, usage_df, args.output_dir)
