"""Sanity checks of the synthetic generators in tests/synthetic.py."""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic  # noqa: E402
from placebo_harness import GROSS  # noqa: E402
from wfo_daily_core import run_wfo_daily  # noqa: E402


@pytest.mark.parametrize("gen", [synthetic.independent_rw, synthetic.ou_positive_control])
def test_sync_generators_are_deterministic_by_seed(gen):
    a, pairs_a = gen(n_years=2, seed=3)
    b, pairs_b = gen(n_years=2, seed=3)
    c, _ = gen(n_years=2, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert pairs_a == pairs_b
    assert not a.equals(c)


def test_async_generator_is_deterministic_by_seed():
    a, b, c = (synthetic.async_same_asset(n_years=2, n_pairs=2, seed=s) for s in (3, 3, 4))
    for name in ("us", "local", "fx_daily"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    pd.testing.assert_series_equal(a.fx_grid, b.fx_grid)
    assert a.holidays.equals(b.holidays)
    assert not a.us.equals(c.us)


def test_independent_rw_calendar_names_and_zero_drift():
    closes, pairs = synthetic.independent_rw(seed=0)
    assert closes.index[0] == pd.Timestamp("2005-01-03")
    assert closes.index[-1] == pd.Timestamp("2014-12-31")
    assert (closes.index.dayofweek < 5).all()
    assert len(pairs) == 20 and not any("-" in c for c in closes.columns)
    rets = np.log(closes).diff().iloc[1:].to_numpy().ravel()
    assert abs(rets.mean()) < 4 * 0.015 / np.sqrt(rets.size)
    assert rets.std() == pytest.approx(0.015, rel=0.02)


def test_async_calendars_holidays_and_sample_times():
    raw = synthetic.async_same_asset(n_years=3, n_pairs=2, close_lag_hours=4.5,
                                     holidays_per_year=5, fx_lag_hours=2, seed=1)
    days = synthetic.business_days(3)
    assert raw.us.index.equals(days) and raw.fx_daily.index.equals(days)
    assert len(raw.holidays) == 15 and (raw.holidays.year.value_counts() == 5).all()
    assert raw.holidays.isin(days).all()
    assert not raw.local.index.isin(raw.holidays).any()
    assert raw.local.index.equals(days.difference(raw.holidays))
    assert raw.close_time_et == {"XU0": 16.0, "XU1": 16.0, "XL0.SY": 11.5,
                                 "XL1.SY": 11.5, "SYUSD=X": 13.5}
    assert raw.adrs == {"XU0": ("XL0.SY", "SYUSD=X"), "XU1": ("XL1.SY", "SYUSD=X")}
    # The daily FX close is the grid FX at local close + fx_lag (13:30 ET).
    fx_time = (days + pd.Timedelta(hours=16)).tz_localize(synthetic.ET) - pd.Timedelta(hours=2.5)
    np.testing.assert_array_equal(raw.fx_daily["SYUSD=X"].to_numpy(),
                                  raw.fx_grid.loc[fx_time].to_numpy())


@pytest.mark.parametrize("builder,fx_lag_hours", [
    (synthetic.build_async_panel, 0),
    # The corrected builder converts at the local close itself: the daily FX
    # close time does not matter.
    (synthetic.build_async_panel_fixed, 2),
], ids=["legacy", "fixed"])
def test_async_without_lags_holidays_or_noise_gives_identical_legs_and_no_pnl(builder, fx_lag_hours):
    raw = synthetic.async_same_asset(close_lag_hours=0, holidays_per_year=0,
                                     fx_lag_hours=fx_lag_hours, print_noise_bp=0, seed=0)
    closes, pairs = builder(raw)
    assert pairs == [f"XU{i}-XL{i}.SY_USD" for i in range(10)]
    for pair in pairs:
        us, local_usd = pair.split("-")
        np.testing.assert_allclose(closes[local_usd], closes[us], rtol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_wfo_daily(closes, pairs, GROSS)
    # The brief expects zero trades.  The legs agree only to float precision
    # and the core's OLS β is 1 ± a few ulp, so the z-score of a ~1e-16
    # spread can still cross ±2; such trades must book nothing but float noise.
    assert res.trades.empty or res.trades["pnl"].abs().max() < 1e-12


@pytest.mark.parametrize("lag", [4.5, 15])
def test_fixed_async_panel_converts_at_the_local_close_on_the_union_calendar(lag):
    # Audit T2: each local print is converted with the grid FX at its own close
    # (16:00 − lag ET), on its own trading dates; the panel is the union
    # calendar with NaN local prints on local holidays, nothing filled.
    raw = synthetic.async_same_asset(n_years=3, n_pairs=2, close_lag_hours=lag,
                                     holidays_per_year=5, fx_lag_hours=2, seed=1)
    closes, pairs = synthetic.build_async_panel_fixed(raw)
    days = synthetic.business_days(3)
    assert pairs == ["XU0-XL0.SY_USD", "XU1-XL1.SY_USD"]
    assert closes.index.equals(days) and list(closes.columns) == ["XU0", "XU1", "XL0.SY_USD", "XL1.SY_USD"]
    pd.testing.assert_frame_equal(closes[["XU0", "XU1"]], raw.us, check_names=False)
    local_close = (raw.local.index + pd.Timedelta(hours=16 - lag)).tz_localize(synthetic.ET)
    fx = raw.fx_grid.loc[local_close].to_numpy()
    for i in range(2):
        usd = closes[f"XL{i}.SY_USD"]
        assert usd.isna().to_numpy().tolist() == days.isin(raw.holidays).tolist()
        np.testing.assert_array_equal(usd.dropna().to_numpy(), raw.local[f"XL{i}.SY"].to_numpy() * fx)
    # The legacy panel converts with the daily FX close (2 h later) instead.
    legacy, _ = synthetic.build_async_panel(raw)
    assert not np.allclose(legacy["XL0.SY_USD"].loc[raw.local.index], closes["XL0.SY_USD"].dropna(),
                           rtol=1e-9, atol=0)


def test_ou_spread_half_life_and_sd_match_the_request():
    closes, pairs = synthetic.ou_positive_control(seed=0)
    half_lives, sds = [], []
    for pair in pairs:
        a, b = pair.split("-")
        y, x = np.log(closes[a]).to_numpy(), np.log(closes[b]).to_numpy()
        beta, alpha = np.polyfit(x, y, 1)
        s = y - beta * x - alpha
        phi = np.polyfit(s[:-1], s[1:], 1)[0]
        half_lives.append(np.log(0.5) / np.log(phi))
        sds.append(s.std())
    assert np.mean(half_lives) == pytest.approx(5.0, rel=0.15)
    assert np.mean(sds) == pytest.approx(0.02, rel=0.10)
