"""Synthetic price panels for the placebo harness (audit task T0).

  independent_rw       two independent zero-drift random walks per pair — no edge.
  async_same_asset     one efficient price per pair printed on two exchanges
                       whose closes are hours apart, a daily FX close sampled
                       at yet another time, and local holidays — no edge.
                       Panels: `build_async_panel` (legacy builder) or
                       `build_async_panel_fixed` (corrected builder, audit T2).
  ou_positive_control  synchronous legs whose log-spread is an OU process —
                       a real edge.

All generators are deterministic given `seed` and return plain pandas objects.
Ticker names never contain "-" (pairs are split on "-").
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cross_listing import (  # noqa: E402
    Exchange, align_to_ref_calendar, assemble_panel, build_cross_listed_usd,
    build_cross_listed_usd_at_close,
)

START = pd.Timestamp("2005-01-03")
ET = "America/New_York"
GRID_FREQ = pd.Timedelta(minutes=30)
STEPS_PER_DAY = 48                      # 30-minute steps per calendar day


def business_days(n_years: int) -> pd.DatetimeIndex:
    """Weekdays of the `n_years` calendar years starting at START (2005-01-03)."""
    end = pd.Timestamp(year=START.year + n_years - 1, month=12, day=31)
    return pd.bdate_range(START, end, name="Date")


def independent_rw(n_years: int = 10, n_pairs: int = 20, vol: float = 0.015,
                   seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Two independent zero-drift random walks in log price per pair."""
    rng = np.random.default_rng(seed)
    idx = business_days(n_years)
    log_p = np.cumsum(rng.normal(0.0, vol, size=(len(idx), 2 * n_pairs)), axis=0)
    cols = [f"RW{leg}{i}" for i in range(n_pairs) for leg in ("A", "B")]
    closes = pd.DataFrame(100.0 * np.exp(log_p), index=idx, columns=cols)
    return closes, [f"RWA{i}-RWB{i}" for i in range(n_pairs)]


@dataclass
class AsyncRaw:
    """Raw per-exchange series of `async_same_asset`, before any alignment."""
    us: pd.DataFrame                    # US closes XU{i}, every weekday
    local: pd.DataFrame                 # local-currency closes XL{i}.SY, local trading days only
    fx_daily: pd.DataFrame              # "daily close" SYUSD=X (USD per local unit), every weekday
    fx_grid: pd.Series                  # SYUSD=X on the tz-aware 30-min ET grid
    adrs: dict[str, tuple[str, str]]    # {us: (local, fx)} as in universe_expand
    close_time_et: dict[str, float]     # sample time per ticker, hours after midnight ET
    holidays: pd.DatetimeIndex          # local holidays (no local print)


def async_same_asset(n_years: int = 10, n_pairs: int = 10,
                     close_lag_hours: float = 4.5, holidays_per_year: int = 5,
                     fx_lag_hours: float = 2, print_noise_bp: float = 10,
                     seed: int = 0, daily_vol: float = 0.015,
                     fx_daily_vol: float = 0.006) -> AsyncRaw:
    """One efficient USD price per pair, printed by a US and a local exchange.

    The efficient log price (and one shared log FX path, USD per local unit,
    starting at 0.75) follow random walks on a 30-minute ET grid over all
    calendar days.  The US leg prints at 16:00 ET on every weekday; the local
    leg prints `close_lag_hours` earlier, in local currency, on weekdays that
    are not local holidays; the daily FX close is sampled `fx_lag_hours`
    after the local close.  Prints carry i.i.d. noise of `print_noise_bp`.
    Both legs are the same asset, so no edge exists.
    """
    rng = np.random.default_rng(seed)
    dates = business_days(n_years)
    grid = pd.date_range(dates[0].tz_localize(ET),
                         (dates[-1] + pd.Timedelta(days=1)).tz_localize(ET),
                         freq=GRID_FREQ, inclusive="left", name="time")
    eff = np.log(100.0) + np.cumsum(
        rng.normal(0.0, daily_vol / np.sqrt(STEPS_PER_DAY), size=(len(grid), n_pairs)), axis=0)
    fx = 0.75 * np.exp(np.cumsum(
        rng.normal(0.0, fx_daily_vol / np.sqrt(STEPS_PER_DAY), size=len(grid))))

    def grid_pos(times: pd.DatetimeIndex) -> np.ndarray:
        pos = grid.get_indexer(times)
        if (pos < 0).any():
            raise ValueError("sample times must lie on the 30-minute grid")
        return pos

    us_time = (dates + pd.Timedelta(hours=16)).tz_localize(ET)
    loc_time = us_time - pd.Timedelta(hours=close_lag_hours)
    fx_time = loc_time + pd.Timedelta(hours=fx_lag_hours)
    noise_sd = print_noise_bp / 1e4

    holidays = pd.DatetimeIndex(sorted(
        d for year in np.unique(dates.year)
        for d in rng.choice(dates[dates.year == year], size=holidays_per_year, replace=False)
    ), name="Date")
    is_local_day = ~dates.isin(holidays)

    us_px = np.exp(eff[grid_pos(us_time)]) * np.exp(
        rng.normal(0.0, noise_sd, size=(len(dates), n_pairs)))
    loc_pos = grid_pos(loc_time[is_local_day])
    loc_px = (np.exp(eff[loc_pos]) / fx[loc_pos][:, None]
              * np.exp(rng.normal(0.0, noise_sd, size=(len(loc_pos), n_pairs))))
    fx_close = fx[grid_pos(fx_time)]

    us_cols = [f"XU{i}" for i in range(n_pairs)]
    loc_cols = [f"XL{i}.SY" for i in range(n_pairs)]
    fx_col = "SYUSD=X"
    return AsyncRaw(
        us=pd.DataFrame(us_px, index=dates, columns=us_cols),
        local=pd.DataFrame(loc_px, index=dates[is_local_day], columns=loc_cols),
        fx_daily=pd.DataFrame({fx_col: fx_close}, index=dates),
        fx_grid=pd.Series(fx, index=grid, name=fx_col),
        adrs={u: (loc, fx_col) for u, loc in zip(us_cols, loc_cols)},
        close_time_et={**{u: 16.0 for u in us_cols},
                       **{loc: 16.0 - close_lag_hours for loc in loc_cols},
                       fx_col: 16.0 - close_lag_hours + fx_lag_hours},
        holidays=holidays,
    )


def build_async_panel(raw: AsyncRaw) -> tuple[pd.DataFrame, list[str]]:
    """Legacy panel builder, mirroring `universe_expand.main()`.

    Reference calendar = the US dates; local legs and the daily FX close are
    aligned by date (forward-filled) and converted with the FX close of the
    same date via the repo's `cross_listing` code.
    """
    ref_index = raw.us.index
    us = align_to_ref_calendar(raw.us, ref_index)
    local_usd = build_cross_listed_usd(raw.local, raw.fx_daily, raw.adrs,
                                       ref_index, "usd_per_local")
    closes = pd.concat([us, local_usd], axis=1)
    pairs = [f"{u}-{loc}_USD" for u, (loc, _) in raw.adrs.items()]
    return closes, pairs


def build_async_panel_fixed(raw: AsyncRaw) -> tuple[pd.DataFrame, list[str]]:
    """Corrected panel builder (audit T2), through the repo's `cross_listing` code.

    Each local print, on its own trading dates, is converted with the grid FX
    (`raw.fx_grid` as intraday FX) observed at its own close: the synthetic
    exchange `.SY` (tz America/New_York, close 16:00 − lag).  The panel is the
    union calendar of the US and the local dates, NaN local prints on local
    holidays, nothing filled.
    """
    exchanges = {}
    for loc, _ in raw.adrs.values():
        minutes = round(raw.close_time_et[loc] * 60)
        # overlaps_nyse / region are never read for the synthetic exchange.
        exchanges[loc] = Exchange("SY", ET, time(minutes // 60, minutes % 60), False, "synthetic")
    local_usd, unaligned = build_cross_listed_usd_at_close(
        raw.local, raw.fx_grid.to_frame(), raw.adrs, "usd_per_local", exchanges=exchanges)
    assert not any(unaligned.values())
    closes = assemble_panel(raw.us, local_usd)
    pairs = [f"{u}-{loc}_USD" for u, (loc, _) in raw.adrs.items()]
    return closes, pairs


def ou_positive_control(n_years: int = 10, n_pairs: int = 10,
                        half_life_days: float = 5, stationary_sd: float = 0.02,
                        seed: int = 0, vol: float = 0.015
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Synchronous legs with log p1 = β·log p2 + s, s an AR(1)/OU spread.

    log p2 is a random walk (vol); φ = 0.5**(1/half_life_days), innovation sd
    = stationary_sd·√(1−φ²), s₀ ~ N(0, stationary_sd²); β ~ U(0.5, 1.5) per
    pair; a constant keeps p1 near 100.
    """
    rng = np.random.default_rng(seed)
    idx = business_days(n_years)
    log_p2 = np.log(100.0) + np.cumsum(rng.normal(0.0, vol, size=(len(idx), n_pairs)), axis=0)
    phi = 0.5 ** (1.0 / half_life_days)
    shocks = rng.normal(0.0, stationary_sd * np.sqrt(1.0 - phi ** 2), size=(len(idx), n_pairs))
    shocks[0] = rng.normal(0.0, stationary_sd, size=n_pairs)     # s₀
    s = lfilter([1.0], [1.0, -phi], shocks, axis=0)              # s_t = φ·s_{t−1} + ε_t
    beta = rng.uniform(0.5, 1.5, size=n_pairs)
    log_p1 = beta * log_p2 + s + (1.0 - beta) * np.log(100.0)
    data = {}
    for i in range(n_pairs):
        data[f"OUA{i}"] = np.exp(log_p1[:, i])
        data[f"OUB{i}"] = np.exp(log_p2[:, i])
    closes = pd.DataFrame(data, index=idx)
    return closes, [f"OUA{i}-OUB{i}" for i in range(n_pairs)]
