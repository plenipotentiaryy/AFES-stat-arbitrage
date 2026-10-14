"""cross_listing.py — build USD-normalised local legs of cross-listed pairs.

Shared by `universe_expand.py` (USD-per-local FX quotes, e.g. AUDUSD=X) and
`universe_expand_v2.py` (local-per-USD quotes, e.g. USDCAD=X).  The logic is
the one those scripts had inline: local closes and daily FX closes are
aligned to the reference (US) calendar by date with a forward-fill of up to
`max_stale` days, then the local price is converted with the FX close of the
same date.  (Known look-ahead / stale-price issues — audit task T2.)
"""
from __future__ import annotations

import pandas as pd


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
