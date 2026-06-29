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

import pandas as pd

PITCH_OUTCOME_MAP: dict[str, str] = {
    # ball
    "ball": "ball",
    "blocked_ball": "ball",
    "pitchout": "ball",
    # called strike
    "called_strike": "called_strike",
    # swinging strike
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    # foul
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
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
    # everything else is "out"
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
    raise NotImplementedError


def map_bip_outcome(df: pd.DataFrame) -> pd.Series:
    """Map Statcast `events` column to Model 2 label for in-play rows only.

    Args:
        df: Statcast DataFrame containing an `events` column.
            Should already be filtered to in-play pitches.

    Returns:
        Series of string labels. Events not in BIP_OUTCOME_MAP map to "out".
    """
    raise NotImplementedError


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add `pitch_outcome` and `bip_outcome` columns to a raw Statcast DataFrame.

    Args:
        df: Raw Statcast DataFrame.

    Returns:
        DataFrame with two new columns: `pitch_outcome` (all rows) and
        `bip_outcome` (non-null only for in-play rows).
    """
    raise NotImplementedError


def validate_coverage(df: pd.DataFrame) -> None:
    """Assert that all `description` values in df are covered by PITCH_OUTCOME_MAP.

    Raises:
        ValueError: If any unmapped `description` values are found, listing them.
    """
    raise NotImplementedError
