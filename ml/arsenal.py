"""
Precomputes per-pitcher arsenal and count-level usage tables.

Reads processed Statcast data and writes four Parquet files to data/processed/:
    arsenal.parquet       — pitch type stats per pitcher per season
    usage.parquet         — pitch type usage % per pitcher per count state
    pitcher_woba.parquet  — pitcher wOBA-against tier lookup (for serving)
    pitchers.parquet      — pitcher metadata (name, team, handedness) for the API

Public interface:
    build_arsenal_table(df) -> pd.DataFrame
    build_usage_table(df) -> pd.DataFrame
    build_pitcher_woba_table(df) -> pd.DataFrame
    build_pitchers_table(df) -> pd.DataFrame
    save_tables(arsenal_df, usage_df, woba_df, pitchers_df, output_dir) -> None
    load_arsenal(output_dir) -> pd.DataFrame
    load_usage(output_dir) -> pd.DataFrame
    load_pitcher_woba(output_dir) -> pd.DataFrame
    load_pitchers(output_dir) -> pd.DataFrame
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
        usage_pct, avg_speed, avg_spin, avg_pfx_x, avg_pfx_z, sample_size,
        median_release_pos_x, median_release_pos_z, median_release_extension,
        median_spin_axis

    The four median_* columns supply pitch-characteristic features that the
    /predict feature row cannot get from the request; see api/predict.py.
    """
    arsenal = (
        df.groupby(["pitcher", "game_year", "pitch_type"])
        .agg(
            pitch_name=("pitch_name", "first"),
            avg_speed=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_pfx_x=("pfx_x", "mean"),
            avg_pfx_z=("pfx_z", "mean"),
            median_release_pos_x=("release_pos_x", "median"),
            median_release_pos_z=("release_pos_z", "median"),
            median_release_extension=("release_extension", "median"),
            median_spin_axis=("spin_axis", "median"),
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


def _mode(s: pd.Series):
    """Most frequent value in a Series (first if tied), or None if empty."""
    m = s.mode()
    return m.iloc[0] if len(m) else None


def build_pitchers_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-pitcher metadata for the /pitchers API endpoint.

    Output columns:
        pitcher_id, full_name, team, p_throws, season

    `full_name` reformats Statcast's "Last, First" to "First Last". `team` is
    the pitcher's most frequent team (home team when pitching in the top half,
    away team in the bottom). `season` is the latest season the pitcher appears in.
    """
    df = df.copy()
    df["_team"] = df["home_team"].where(df["inning_topbot"] == "Top", df["away_team"])
    pitchers = (
        df.groupby("pitcher")
        .agg(
            _name=("player_name", _mode),
            team=("_team", _mode),
            p_throws=("p_throws", _mode),
            season=("game_year", "max"),
        )
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
    )

    def _reformat(name: object) -> object:
        if isinstance(name, str) and ", " in name:
            last, first = name.split(", ", 1)
            return f"{first} {last}"
        return name

    pitchers["full_name"] = pitchers["_name"].map(_reformat)
    return pitchers[["pitcher_id", "full_name", "team", "p_throws", "season"]]


def save_tables(
    arsenal_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    woba_df: pd.DataFrame,
    pitchers_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write arsenal, usage, pitcher-wOBA, and pitchers tables to Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arsenal_df.to_parquet(output_dir / "arsenal.parquet", index=False)
    usage_df.to_parquet(output_dir / "usage.parquet", index=False)
    woba_df.to_parquet(output_dir / "pitcher_woba.parquet", index=False)
    pitchers_df.to_parquet(output_dir / "pitchers.parquet", index=False)
    logger.info(
        "Wrote arsenal (%d), usage (%d), pitcher_woba (%d), pitchers (%d) to %s",
        len(arsenal_df),
        len(usage_df),
        len(woba_df),
        len(pitchers_df),
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


def load_pitchers(output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load precomputed pitcher metadata table from Parquet."""
    return pd.read_parquet(output_dir / "pitchers.parquet")


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
    pitchers_df = build_pitchers_table(df)
    save_tables(arsenal_df, usage_df, woba_df, pitchers_df, args.output_dir)
