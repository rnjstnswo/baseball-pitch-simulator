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
import pybaseball

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
logger = logging.getLogger(__name__)

# Pull month by month to avoid pybaseball timeouts on large date ranges
_MONTHS = [
    ("01-01", "01-31"),
    ("02-01", "02-28"),
    ("03-01", "03-31"),
    ("04-01", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
    ("10-01", "10-31"),
]


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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pybaseball.cache.enable()

    all_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        dest = RAW_DIR / f"statcast_{year}.parquet"
        if dest.exists() and not force_refresh:
            logger.info("Year %d: loading from cache (%s)", year, dest)
            df = pd.read_parquet(dest)
        else:
            logger.info("Year %d: pulling from pybaseball...", year)
            df = _pull_single_year(year)
            df.to_parquet(dest, index=False)
            logger.info("Year %d: saved to %s", year, dest)
        _log_summary(df, year)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total rows across all years: %d", len(combined))
    return combined


def load_raw(year: int) -> pd.DataFrame:
    """Load a single season's cached Parquet from data/raw/.

    Args:
        year: Season year.

    Returns:
        DataFrame of raw Statcast rows for that season.

    Raises:
        FileNotFoundError: If the Parquet file for the given year does not exist.
    """
    path = RAW_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for {year}. Run pull_statcast() first. Expected: {path}"
        )
    return pd.read_parquet(path)


def _pull_single_year(year: int) -> pd.DataFrame:
    """Pull one season from pybaseball.statcast() month by month to avoid timeouts."""
    frames: list[pd.DataFrame] = []
    for start_md, end_md in _MONTHS:
        start_date = f"{year}-{start_md}"
        end_date = f"{year}-{end_md}"
        try:
            chunk = pybaseball.statcast(
                start_dt=start_date, end_dt=end_date, verbose=False
            )
        except Exception as exc:
            logger.warning(
                "Year %d %s–%s: fetch failed (%s), skipping",
                year,
                start_md,
                end_md,
                exc,
            )
            continue
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
            logger.debug("Year %d %s–%s: %d rows", year, start_md, end_md, len(chunk))

    if not frames:
        raise RuntimeError(f"No data returned for year {year}")

    df = pd.concat(frames, ignore_index=True)
    # Sort chronologically for deterministic output
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def _log_summary(df: pd.DataFrame, year: int) -> None:
    """Log row count, date range, and top null-rate columns for a season DataFrame."""
    n_rows = len(df)
    date_min = df["game_date"].min() if "game_date" in df.columns else "?"
    date_max = df["game_date"].max() if "game_date" in df.columns else "?"
    logger.info("Year %d: %d rows | %s → %s", year, n_rows, date_min, date_max)

    null_rates = df.isnull().mean().sort_values(ascending=False)
    top_nulls = null_rates[null_rates > 0].head(10)
    if not top_nulls.empty:
        logger.info("Year %d top null-rate columns:\n%s", year, top_nulls.to_string())
    else:
        logger.info("Year %d: no null values found", year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Statcast data to data/raw/")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    pull_statcast(args.start_year, args.end_year, args.force_refresh)
