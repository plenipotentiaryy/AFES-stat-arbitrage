"""cross_listing.py — build USD-normalised local legs of cross-listed pairs.

Shared by `universe_expand.py` (USD-per-local FX quotes, e.g. AUDUSD=X) and
`universe_expand_v2.py` (local-per-USD quotes, e.g. USDCAD=X).

Corrected path (audit task T2): every series keeps its own trading dates and
its exchange's close time (`close_time`, `close_timestamps`); a missing print
is NaN and an exactly unchanged close with zero or missing volume is no print
(`mask_stale_prints`); a local close is converted with the intraday FX quote
observed at that close (`build_cross_listed_usd_at_close`) — without intraday
FX the leg is not converted and is flagged fx_unaligned; legs are joined on
the union calendar without any fill (`assemble_panel`).

Legacy path (`align_to_ref_calendar`, `to_usd`, `build_cross_listed_usd`;
the `--legacy-daily-fx` output of the universe_expand scripts): the logic
those scripts had inline — local closes and daily FX closes are aligned to
the reference (US) calendar by date with a forward-fill of up to `max_stale`
days, then the local price is converted with the FX close of the same date.
That is look-ahead (a local close up to 15 h older than the US close, an FX
close of another time) and trades stale holiday prints.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

ET = "America/New_York"


@dataclass(frozen=True)
class Exchange:
    """Where a series prints: exchange name, IANA time zone, local close time,
    whether its session overlaps the NYSE session, and its region."""
    name: str
    tz: str
    close: time
    overlaps_nyse: bool
    region: str


# Ticker suffix (after stripping a trailing "_USD") → exchange; "" = US listing.
EXCHANGES: dict[str, Exchange] = {
    "":    Exchange("NYSE/Nasdaq", "America/New_York", time(16, 0), True, "North America"),
    ".TO": Exchange("TSX", "America/Toronto", time(16, 0), True, "North America"),
    ".AX": Exchange("ASX", "Australia/Sydney", time(16, 0), False, "Asia-Pacific"),
    ".DE": Exchange("Xetra", "Europe/Berlin", time(17, 30), True, "Europe"),
    ".PA": Exchange("Euronext Paris", "Europe/Paris", time(17, 30), True, "Europe"),
    ".SW": Exchange("SIX", "Europe/Zurich", time(17, 30), True, "Europe"),
    ".L":  Exchange("LSE", "Europe/London", time(16, 30), True, "Europe"),
    ".T":  Exchange("TSE", "Asia/Tokyo", time(15, 0), False, "Asia-Pacific"),
}


def close_time(ticker: str) -> Exchange:
    """Exchange, time zone and local close time of a series, by its ticker
    suffix after stripping a trailing "_USD" (no suffix → US).  Unknown
    suffix → KeyError."""
    base = ticker[:-len("_USD")] if ticker.endswith("_USD") else ticker
    suffix = base[base.rfind("."):] if "." in base else ""
    if suffix not in EXCHANGES:
        raise KeyError(f"{ticker}: unknown exchange suffix {suffix!r} (see cross_listing.EXCHANGES)")
    return EXCHANGES[suffix]


def close_timestamps(ticker: str, dates, exchange: Exchange | None = None) -> pd.DatetimeIndex:
    """Print time of each date's close (tz-naive dates in the exchange's own
    calendar) as tz-aware America/New_York timestamps; DST-correct.
    `exchange` overrides the suffix rule (e.g. a synthetic exchange)."""
    ex = exchange if exchange is not None else close_time(ticker)
    local = (pd.DatetimeIndex(dates).normalize()
             + pd.Timedelta(hours=ex.close.hour, minutes=ex.close.minute))
    return local.tz_localize(ex.tz).tz_convert(ET)


def repeats_previous_print(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """True where a price equals exactly its series' previous non-NaN price
    (a zero return); False on NaN and on a first print.  The one definition
    behind the stale-print mask, the engine's zero-return entry guard and
    diag_stale_entries.py."""
    return prices.eq(prices.ffill().shift(1))     # ffill only to find the previous print


def mask_stale_prints(closes: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    """`closes` with NaN where a close equals the column's previous non-NaN
    close and that day's volume is 0, NaN, or the column has no volume series
    (missing volume): such a close is no print.  Nothing is forward-filled."""
    unchanged = repeats_previous_print(closes)
    if volumes is None:
        no_volume = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    else:
        vol = volumes.reindex(index=closes.index, columns=closes.columns)
        no_volume = vol.isna() | vol.eq(0)
    return closes.mask(unchanged & no_volume)


def build_cross_listed_usd_at_close(local_df: pd.DataFrame, fx_intraday: pd.DataFrame | None,
                                    adrs: dict[str, tuple[str, str]], fx_quote: str,
                                    exchanges: dict[str, Exchange] | None = None,
                                    fx_max_age: str = "1h") -> tuple[pd.DataFrame, dict[str, bool]]:
    """USD-normalised local legs, each print converted at its own close time.

    local_df:    local-currency closes on the local legs' own trading dates
                 (no reindex, no fill); NaN = no print.
    fx_intraday: intraday FX quotes (tz-aware DatetimeIndex × FX tickers), or None.
    adrs:        {adr: (local_ticker, fx_ticker)}.
    exchanges:   {local_ticker: Exchange} overriding the suffix rule.

    For a local print on date d the FX rate is the last quote at or before
    its close τ = close_timestamps(local, [d]); NaN if that quote is older
    than `fx_max_age`.  Without the FX series (`fx_intraday` None or no such
    column) no f"{local}_USD" column is produced and the leg is flagged
    fx_unaligned — never a fallback to a daily FX close.  An `fx_intraday`
    index that is not tz-aware or not increasing raises ValueError.
    Returns (USD frame on local_df's dates, {local_ticker: fx_unaligned}).
    """
    if fx_intraday is not None:
        if getattr(fx_intraday.index, "tz", None) is None:
            raise ValueError("fx_intraday needs a tz-aware DatetimeIndex (quote times)")
        if not fx_intraday.index.is_monotonic_increasing:
            raise ValueError("fx_intraday's index must be sorted in increasing time")
    exchanges = exchanges or {}
    max_age = pd.Timedelta(fx_max_age)
    out = pd.DataFrame(index=local_df.index)
    fx_unaligned: dict[str, bool] = {}
    for loc, fx in adrs.values():
        if fx_intraday is None or fx not in fx_intraday.columns:
            fx_unaligned[loc] = True
            continue
        if loc not in local_df.columns:
            continue
        quotes = fx_intraday[fx].dropna()
        tau = close_timestamps(loc, local_df.index, exchanges.get(loc)).tz_convert(quotes.index.tz)
        pos = quotes.index.searchsorted(tau, side="right") - 1      # last quote at or before τ
        found = pos >= 0
        rate = np.full(len(tau), np.nan)
        fresh = (tau[found] - quotes.index[pos[found]]) <= max_age
        rate[np.flatnonzero(found)[fresh]] = quotes.to_numpy()[pos[found]][fresh]
        out[f"{loc}_USD"] = to_usd(local_df[loc], pd.Series(rate, index=local_df.index), fx_quote)
        fx_unaligned[loc] = False
    return out, fx_unaligned


def assemble_panel(*frames: pd.DataFrame) -> pd.DataFrame:
    """Outer-join frames on their dates (the union calendar), sorted, without
    any fill: a column is NaN on the dates its market did not print."""
    return pd.concat(frames, axis=1, join="outer", sort=True)


def reindex_to_calendar(df: pd.DataFrame, ref_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex (tz stripped) to the reference calendar WITHOUT any fill: a
    date without a print stays NaN.  For series that trade on that calendar."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.reindex(ref_index)


# ── Legacy (pre-audit) path ─────────────────────────────────────────────────


def align_to_ref_calendar(df: pd.DataFrame, ref_index: pd.DatetimeIndex,
                          max_stale: int = 3) -> pd.DataFrame:
    """Reindex foreign-market data to the reference (US) trading calendar.
    Forward-fill up to `max_stale` days for foreign holidays / late prints."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.reindex(ref_index).ffill(limit=max_stale)


def to_usd(local: pd.Series, fx: pd.Series, fx_quote: str) -> pd.Series:
    """Convert local-currency prices to USD.

    fx_quote="usd_per_local": fx = USD per 1 local unit (AUDUSD=X) → local * fx.
    fx_quote="local_per_usd": fx = local units per 1 USD (USDCAD=X) → local / fx.
    """
    if fx_quote == "usd_per_local":
        return local * fx
    if fx_quote == "local_per_usd":
        return local / fx
    raise ValueError(f"unknown fx_quote {fx_quote!r}")


def build_cross_listed_usd(local_df: pd.DataFrame, fx_df: pd.DataFrame,
                           adrs: dict[str, tuple[str, str]],
                           ref_index: pd.DatetimeIndex, fx_quote: str,
                           max_stale: int = 3) -> pd.DataFrame:
    """USD-normalised local legs on `ref_index`.

    adrs: {adr: (local_ticker, fx_ticker)}.  One column f"{local}_USD" per
    entry whose local and FX columns both exist (others are skipped).
    """
    out = pd.DataFrame(index=ref_index)
    usable = {adr: (loc, fx) for adr, (loc, fx) in adrs.items()
              if loc in local_df.columns and fx in fx_df.columns}
    if not usable:
        return out
    local_aligned = align_to_ref_calendar(local_df, ref_index, max_stale)
    fx_aligned = align_to_ref_calendar(fx_df, ref_index, max_stale)
    for adr, (loc, fx) in usable.items():
        out[f"{loc}_USD"] = to_usd(local_aligned[loc], fx_aligned[fx], fx_quote)
    return out
