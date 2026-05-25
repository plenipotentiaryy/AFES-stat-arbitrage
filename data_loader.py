"""
data_loader.py — Unified Parquet/CSV loader for intraday OHLCV data.

Priority: Parquet (column-pruning, ~10x faster) → CSV fallback.

Usage:
    from data_loader import load_closes, load_volumes

    closes = load_closes()                            # all 233 tickers
    closes = load_closes(["JPM", "BAC"])              # 2 columns only
    closes = load_closes(start="2015-01-01")          # date slice
    closes = load_closes(["SPY"], freq="daily")       # resample to daily
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    DATA_DIR, RTH_START, RTH_END,
    CLOSES_FILE, VOLUMES_FILE, BAR_MINUTES
)

CLOSES_PARQUET  = DATA_DIR / f"closes_{BAR_MINUTES}min.parquet"
VOLUMES_PARQUET = DATA_DIR / f"volumes_{BAR_MINUTES}min.parquet"
CLOSES_CSV      = DATA_DIR / CLOSES_FILE
VOLUMES_CSV     = DATA_DIR / VOLUMES_FILE

_TZ = "US/Eastern"


# ── Core loader ───────────────────────────────────────────────────────────────

def _load_parquet(path: Path, columns: list[str] | None) -> pd.DataFrame:
    """Read Parquet with optional column pruning (pyarrow column-pushdown)."""
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def _load_csv(path: Path, columns: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df


def _normalise_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(_TZ)
    else:
        df.index = df.index.tz_convert(_TZ)
    return df


def _apply_filters(df: pd.DataFrame,
                   start: str | None,
                   end:   str | None,
                   rth:   bool,
                   freq:  str | None) -> pd.DataFrame:
    if rth:
        df = df.between_time(RTH_START, RTH_END)
    if start:
        df = df[df.index >= pd.Timestamp(start).tz_localize(_TZ)]
    if end:
        df = df[df.index <= pd.Timestamp(end).tz_localize(_TZ)]
    if freq and freq != f"{BAR_MINUTES}min":
        df = df.resample(freq).last().dropna(how="all")
    return df


def _load(parquet_path: Path, csv_path: Path,
          tickers: list[str] | None,
          start:   str | None,
          end:     str | None,
          rth:     bool,
          freq:    str | None) -> pd.DataFrame:
    if parquet_path.exists():
        df = _load_parquet(parquet_path, tickers)
    elif csv_path.exists():
        df = _load_csv(csv_path, tickers)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} found. "
            "Run download_av.py to build the dataset."
        )
    df = _normalise_tz(df)
    df = _apply_filters(df, start, end, rth, freq)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_closes(tickers: list[str] | None = None,
                start:   str | None = None,
                end:     str | None = None,
                rth:     bool = True,
                freq:    str | None = None) -> pd.DataFrame:
    """
    Load intraday close prices.

    Parameters
    ----------
    tickers : list of str, optional
        Column subset — uses Parquet column pruning when available.
        None = all 233 tickers.
    start / end : "YYYY-MM-DD" strings, optional
    rth   : filter to Regular Trading Hours (09:30–16:00 ET)
    freq  : resample rule, e.g. "1D", "1W". None = keep default intraday bars.
    """
    return _load(CLOSES_PARQUET, CLOSES_CSV, tickers, start, end, rth, freq)


def load_volumes(tickers: list[str] | None = None,
                 start:   str | None = None,
                 end:     str | None = None,
                 rth:     bool = True,
                 freq:    str | None = None) -> pd.DataFrame:
    """Load intraday volume data. Same signature as load_closes."""
    return _load(VOLUMES_PARQUET, VOLUMES_CSV, tickers, start, end, rth, freq)


def load_pair(t1: str, t2: str,
              start: str | None = None,
              end:   str | None = None) -> pd.DataFrame:
    """
    Load closes for exactly two tickers and drop rows where either is NaN.
    Convenience wrapper for the most common backtest use-case.
    """
    df = load_closes([t1, t2], start=start, end=end)
    return df.dropna()


# ── Daily returns helper (for pair screening / clustering) ────────────────────

def load_daily_returns(tickers: list[str] | None = None,
                       start: str | None = None) -> pd.DataFrame:
    """
    Resample intraday closes to daily and compute log-returns.
    Used by pairs_screener.py for the K-Means clustering step.
    """
    closes = load_closes(tickers, start=start, rth=True, freq="1D")
    return np.log(closes / closes.shift(1)).dropna(how="all")


# ── Pair-loop memory helper ───────────────────────────────────────────────────

def release(*dfs) -> None:
    """Delete DataFrames and trigger GC. Call at end of each pair iteration."""
    for df in dfs:
        del df
    gc.collect()
