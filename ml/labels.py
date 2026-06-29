"""
Outcome taxonomy mapping for Model 1 (pitch outcome) and Model 2 (ball-in-play).

Maps raw Statcast `description` and `events` columns to the model's label space:

  Model 1 — pitch_outcome (6 classes):
      ball | called_strike | swinging_strike | foul | in_play | hit_by_pitch

  Model 2 — bip_outcome (5 classes, conditioned on pitch_outcome == "in_play"):
      out | single | double | triple | home_run

Public interface:
    map_pitch_outcome(df) -> pd.Series
    map_bip_outcome(df) -> pd.Series
    add_labels(df) -> pd.DataFrame
    validate_coverage(df) -> None
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
logger = logging.getLogger(__name__)

PITCH_OUTCOME_MAP: dict[str, str] = {
    # ball
    "ball": "ball",
    "blocked_ball": "ball",
    "pitchout": "ball",
    "automatic_ball": "ball",
    # called strike
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    # swinging strike
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    # foul (contact made)
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "foul_pitchout": "foul",
    "bunt_foul_tip": "foul",
    # in play
    "hit_into_play": "in_play",
    "hit_into_play_no_out": "in_play",
    "hit_into_play_score": "in_play",
    # hit by pitch
    "hit_by_pitch": "hit_by_pitch",
}

BIP_OUTCOME_MAP: dict[str, str] = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    # everything else (incl. field_error) is "out"
}

PITCH_OUTCOME_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
]
BIP_OUTCOME_CLASSES = ["out", "single", "double", "triple", "home_run"]


def map_pitch_outcome(df: pd.DataFrame) -> pd.Series:
    """Map Statcast `description` column to Model 1 label.

    Args:
        df: Raw Statcast DataFrame containing a `description` column.

    Returns:
        Series of string labels aligned with df's index. Rows with unmapped
        descriptions are set to NaN (caller should drop or inspect them).
    """
    return df["description"].map(PITCH_OUTCOME_MAP)


def map_bip_outcome(df: pd.DataFrame) -> pd.Series:
    """Map Statcast `events` column to Model 2 label for in-play rows only.

    Args:
        df: Statcast DataFrame containing an `events` column.
            Should already be filtered to in-play pitches.

    Returns:
        Series of string labels. Events not in BIP_OUTCOME_MAP map to "out".
    """
    return df["events"].map(BIP_OUTCOME_MAP).fillna("out")


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add `pitch_outcome` and `bip_outcome` columns to a raw Statcast DataFrame.

    Args:
        df: Raw Statcast DataFrame.

    Returns:
        DataFrame with two new columns: `pitch_outcome` (all rows) and
        `bip_outcome` (non-null only for in-play rows).
    """
    df = df.copy()
    df["pitch_outcome"] = map_pitch_outcome(df)
    in_play = df["pitch_outcome"] == "in_play"
    df["bip_outcome"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[in_play, "bip_outcome"] = map_bip_outcome(df.loc[in_play])
    return df


def validate_coverage(df: pd.DataFrame) -> None:
    """Assert that all `description` values in df are covered by PITCH_OUTCOME_MAP.

    Raises:
        ValueError: If any unmapped `description` values are found, listing them.
    """
    present = df["description"].dropna().unique()
    unmapped = sorted(set(present) - set(PITCH_OUTCOME_MAP))
    if unmapped:
        counts = (
            df.loc[df["description"].isin(unmapped), "description"]
            .value_counts()
            .to_dict()
        )
        raise ValueError(f"Unmapped `description` values: {counts}")


def _log_distribution(series: pd.Series, name: str) -> None:
    counts = series.value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(1)
    logger.info(
        "%s distribution:\n%s", name, pd.DataFrame({"count": counts, "pct": pct})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map raw Statcast to model labels")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Raw Parquet file. Default: all data/raw/statcast_*.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.input is not None:
        paths = [args.input]
    else:
        paths = sorted(RAW_DIR.glob("statcast_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No raw Parquet files found in {RAW_DIR}")

    logger.info("Loading %d raw file(s): %s", len(paths), [p.name for p in paths])
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    logger.info("Loaded %d rows", len(df))

    validate_coverage(df)
    df = add_labels(df)

    n_unlabeled = df["pitch_outcome"].isna().sum()
    if n_unlabeled:
        logger.info(
            "Dropping %d rows with null pitch_outcome (null description)", n_unlabeled
        )
        df = df[df["pitch_outcome"].notna()].reset_index(drop=True)

    _log_distribution(df["pitch_outcome"], "Model 1 — pitch_outcome")
    bip = df[df["pitch_outcome"] == "in_play"].reset_index(drop=True)
    _log_distribution(bip["bip_outcome"], "Model 2 — bip_outcome")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = args.output_dir / "labeled.parquet"
    bip_path = args.output_dir / "labeled_bip.parquet"
    df.to_parquet(labeled_path, index=False)
    bip.to_parquet(bip_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), labeled_path)
    logger.info("Wrote %d in-play rows to %s", len(bip), bip_path)
