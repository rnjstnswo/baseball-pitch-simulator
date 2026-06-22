"""
Statcast data ingestion pipeline.

Pulls pitch-level Statcast data from MLB via pybaseball and caches it as
Parquet files in data/raw/. Designed for deterministic, idempotent runs:
re-running with the same arguments produces the same output files.

Public interface:
    pull_statcast(start_year, end_year, force_refresh) -> pd.DataFrame
    load_raw(year) -> pd.DataFrame
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
logger = logging.getLogger(__name__)


def pull_statcast(
    start_year: int = 2021,
    end_year: int = 2024,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull Statcast pitch data for the given year range and cache to Parquet.

    Args:
        start_year: First season to pull (inclusive).
        end_year: Last season to pull (inclusive).
        force_refresh: If True, re-pull even if cached file exists.

    Returns:
        Combined DataFrame of all seasons.
    """
    raise NotImplementedError


def load_raw(year: int) -> pd.DataFrame:
    """Load a single season's cached Parquet from data/raw/.

    Args:
        year: Season year.

    Returns:
        DataFrame of raw Statcast rows for that season.

    Raises:
        FileNotFoundError: If the Parquet file for the given year does not exist.
    """
    raise NotImplementedError


def _pull_single_year(year: int) -> pd.DataFrame:
    """Pull one season from pybaseball.statcast() month by month to avoid timeouts."""
    raise NotImplementedError


def _log_summary(df: pd.DataFrame, year: int) -> None:
    """Log row count, date range, and top null-rate columns for a season DataFrame."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Statcast data to data/raw/")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pull_statcast(args.start_year, args.end_year, args.force_refresh)
