"""Leg-level fills and stale prints in the engine (audit task T2).

`backtest_pair` on hand-made 10-day legs with a hand-made z (as in
test_next_close_fills.py): with `leg_prices` the trade PnL comes from the leg
fills, with q1 = 1/P1 and q2 = β/P2 fixed at the entry fill, and never from
the spread series.  `run_wfo_daily` on small synthetic panels: no signal or
fill on a stale print, no entry decision on a zero-return signal day.
"""
import contextlib
import io
import os
import sys
from dataclasses import replace

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
from wfo_daily_core import backtest_pair, run_wfo_daily  # noqa: E402

DAYS = pd.bdate_range("2021-03-01", periods=10)
K = 5   # entry decided on day K (z = 3: short spread), filled on K+1; target
        # (z = 0) decided on K+1, filled on K+2.


def series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=DAYS)


def z_case() -> pd.Series:
    z = np.zeros(len(DAYS))
    z[K] = 3.0
    return series(z)


def leg_case() -> tuple[np.ndarray, np.ndarray]:
    """Leg 1: 50 → 55 (+10%) and leg 2: 20 → 19 (−5%) between the fills."""
    p1, p2 = np.full(len(DAYS), 100.0), np.full(len(DAYS), 40.0)
    p1[K + 1], p1[K + 2] = 50.0, 55.0
    p2[K + 1], p2[K + 2] = 20.0, 19.0
    return p1, p2


def legs(p1, p2) -> pd.DataFrame:
    return pd.DataFrame({"P1": np.asarray(p1, dtype=float), "P2": np.asarray(p2, dtype=float)},
                        index=DAYS)


def run(spread: pd.Series, z: pd.Series, **kw) -> pd.DataFrame:
    """backtest_pair with every bar out-of-sample and zero costs unless given."""
    kw.setdefault("cost_per_trade", 0.0)
    return backtest_pair(spread, DAYS[0], DAYS[-1] + pd.Timedelta(days=1), z_series=z, **kw)


@pytest.mark.parametrize("beta,gross", [(2.0, -0.2), (-1.5, -0.025)])
def test_pnl_comes_from_the_leg_fills_with_quantities_fixed_at_entry(beta, gross):
    # The spread series is flat: it drives no PnL, the legs do.  Short spread
    # (side −1): q1 = 1/50, q2 = β/20, gross = −(q1·(55 − 50) − q2·(19 − 20)):
    # β = 2 → −(0.1 + 0.1); β = −1.5 (q2 < 0: long both legs' risk) → −(0.1 − 0.075).
    p1, p2 = leg_case()
    tr = run(series(np.zeros(len(DAYS))), z_case(), leg_prices=legs(p1, p2), beta=beta)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert (t["entry_signal"], t["entry"]) == (DAYS[K], DAYS[K + 1])
    assert (t["exit_signal"], t["exit"]) == (DAYS[K + 1], DAYS[K + 2])
    assert (t["p1_entry"], t["p2_entry"], t["p1_exit"], t["p2_exit"]) == (50.0, 20.0, 55.0, 19.0)
    assert t["beta"] == beta
    assert t["q1"] == pytest.approx(1 / 50) and t["q2"] == pytest.approx(beta / 20)
    assert t["gross"] == pytest.approx(gross) and t["pnl"] == pytest.approx(gross)


def test_costs_borrow_and_size_apply_to_the_leg_gross():
    p1, p2 = leg_case()
    panic = series(np.zeros(len(DAYS))).astype(bool)
    panic.iloc[K] = True                                   # half size at the entry decision
    t = run(series(np.zeros(len(DAYS))), z_case(), leg_prices=legs(p1, p2), beta=2.0,
            cost_per_trade=0.001, borrow_per_day=0.0002,
            panic_series=panic, panic_size_mult=0.5).iloc[0]
    assert (t["size"], t["held"], t["gross"]) == (0.5, 1, pytest.approx(-0.2))
    assert t["pnl"] == pytest.approx(0.5 * (-0.2 - 0.001 - 1 * 0.0002))


def test_beta_is_the_one_of_the_entry_decision_bar():
    # Kalman path: a β per bar; the trade uses β of its entry decision bar.
    p1, p2 = leg_case()
    beta = series(np.ones(len(DAYS)))
    beta.iloc[K], beta.iloc[K + 1] = 2.0, 3.0
    t = run(series(np.zeros(len(DAYS))), z_case(), leg_prices=legs(p1, p2), beta=beta).iloc[0]
    assert t["beta"] == 2.0 and t["q2"] == pytest.approx(2.0 / 20)
    assert t["gross"] == pytest.approx(-0.2)


def test_without_leg_prices_the_spread_change_is_booked_as_before():
    # Legacy spread PnL: gross = side·Δspread, no leg columns.  With the legs
    # the same trade books the leg PnL instead.
    p1, p2 = leg_case()
    spread = series(np.log(p1) - 2.0 * np.log(p2))
    t = run(spread, z_case(), beta=2.0).iloc[0]
    assert t["gross"] == pytest.approx(-(spread.iloc[K + 2] - spread.iloc[K + 1]))   # −0.1979
    assert t["pnl"] == t["gross"] and t["beta"] == 2.0
    assert t[["p1_entry", "p2_entry", "p1_exit", "p2_exit", "q1", "q2"]].isna().all()
    t = run(spread, z_case(), beta=2.0, leg_prices=legs(p1, p2)).iloc[0]
    assert t["gross"] == pytest.approx(-0.2)


def test_vwap_leg_fills_and_the_per_leg_fallback_to_the_close():
    p1, p2 = leg_case()
    beta = 2.0
    spread = series(np.log(p1) - beta * np.log(p2))
    vw1, vw2 = p1 * 1.01, p2 * 0.99                        # each day's VWAP

    def vwap_args(v1, v2) -> dict:
        vw = legs(v1, v2)                                  # row i = VWAP of bar i+1
        return {"exec_spread": (np.log(vw["P1"]) - beta * np.log(vw["P2"])).shift(-1),
                "exec_leg_prices": vw.shift(-1)}

    t = run(spread, z_case(), leg_prices=legs(p1, p2), beta=beta, **vwap_args(vw1, vw2)).iloc[0]
    assert (t["p1_entry"], t["p2_entry"]) == (vw1[K + 1], vw2[K + 1])
    assert (t["p1_exit"], t["p2_exit"]) == (vw1[K + 2], vw2[K + 2])
    # No leg-2 VWAP on the exit fill day: leg 2 fills at its close, leg 1 at its VWAP.
    v2 = vw2.copy()
    v2[K + 2] = np.nan
    t = run(spread, z_case(), leg_prices=legs(p1, p2), beta=beta, **vwap_args(vw1, v2)).iloc[0]
    assert (t["p1_exit"], t["p2_exit"]) == (vw1[K + 2], p2[K + 2])
    q1, q2 = 1 / vw1[K + 1], beta / vw2[K + 1]
    assert t["gross"] == pytest.approx(-(q1 * (vw1[K + 2] - vw1[K + 1]) - q2 * (p2[K + 2] - vw2[K + 1])))
    # No leg-1 VWAP on the entry fill day: the entry decided on day K is filled
    # all the same (R51: the decision does not read day K+1's VWAP), leg 1 at
    # its close, leg 2 at its VWAP — the exit's per-leg rule.  The legacy
    # same-close path keeps T0's rule (no VWAP spread: not filled).
    v1 = vw1.copy()
    v1[K + 1] = np.nan
    t = run(spread, z_case(), leg_prices=legs(p1, p2), beta=beta, **vwap_args(v1, vw2)).iloc[0]
    assert (t["entry_signal"], t["entry"]) == (DAYS[K], DAYS[K + 1])
    assert (t["p1_entry"], t["p2_entry"]) == (p1[K + 1], vw2[K + 1])
    assert (t["p1_exit"], t["p2_exit"]) == (vw1[K + 2], vw2[K + 2])
    q1, q2 = 1 / p1[K + 1], beta / vw2[K + 1]
    assert t["gross"] == pytest.approx(-(q1 * (vw1[K + 2] - p1[K + 1]) - q2 * (vw2[K + 2] - vw2[K + 1])))
    assert run(spread, z_case(), leg_prices=legs(p1, p2), beta=beta, legacy_same_close=True,
               **vwap_args(v1, vw2)).empty


def test_end_of_data_and_cusum_exits_fill_the_legs_at_the_fill_bar_closes():
    p1 = 100.0 + np.arange(len(DAYS))
    p2 = 50.0 - 0.5 * np.arange(len(DAYS))
    flat = series(np.zeros(len(DAYS)))
    # Entry decided day 7, filled day 8; target decided on day 9 (last bar) → end_of_data at its close.
    t = run(flat, series([0] * 7 + [3, 3, 0]), leg_prices=legs(p1, p2), beta=1.0).iloc[0]
    assert (t["exit"], t["reason"]) == (DAYS[9], "end_of_data")
    assert (t["p1_entry"], t["p1_exit"], t["p2_entry"], t["p2_exit"]) == (p1[8], p1[9], p2[8], p2[9])
    # CUSUM break decided day 3 (S+ = 5.0 > 4.9) → filled on day 4.
    t = run(flat, series([0, 0, 3, 3, 0, 0, 0, 0, 0, 0]), cusum_h=4.9,
            leg_prices=legs(p1, p2), beta=1.0).iloc[0]
    assert (t["exit"], t["reason"]) == (DAYS[4], "cusum_break")
    assert (t["p1_entry"], t["p1_exit"]) == (p1[3], p1[4])
    assert t["gross"] == pytest.approx(-(1 / p1[3] * (p1[4] - p1[3]) - 1 / p2[3] * (p2[4] - p2[3])))


# ── run_wfo_daily: stale prints and the zero-return entry guard ─────────────

FLAT_LEG, FLAT_PAIR = "OUA0", "OUA0-OUB0"
DECISIONS_AND_FILLS = ["entry_signal", "entry", "exit_signal", "exit"]


def quiet(closes, pairs, cfg, **frames):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, pairs, cfg, **frames)


def panel_with_flat_closes():
    """OU panel whose leg OUA0 closes exactly unchanged on every 4th day."""
    closes, pairs = synthetic.ou_positive_control(n_years=4, n_pairs=3, seed=0)
    flat = closes.index[5::4]
    pos = closes.index.get_indexer(flat)
    vals = closes[FLAT_LEG].to_numpy().copy()
    vals[pos] = vals[pos - 1]                  # the day before is never itself flat
    closes[FLAT_LEG] = vals
    volumes = pd.DataFrame(1e6, index=closes.index, columns=closes.columns)
    return closes, pairs, flat, volumes


def pair_dates(trades: pd.DataFrame, cols: list[str]) -> pd.DatetimeIndex:
    g = trades[trades["pair"] == FLAT_PAIR]
    return pd.DatetimeIndex(pd.concat([g[c] for c in cols]))


def test_no_entry_decision_on_a_zero_return_signal_day():
    # The flat closes have volume: genuine prints, so they stay in the panel.
    closes, pairs, flat, volumes = panel_with_flat_closes()
    guarded = quiet(closes, pairs, GROSS, volumes=volumes).trades
    unguarded = quiet(closes, pairs, replace(GROSS, legacy_stale_prints=True), volumes=volumes).trades
    assert pair_dates(unguarded, ["entry_signal"]).isin(flat).any()        # the guard binds here
    assert len(pair_dates(guarded, ["entry_signal"])) > 0
    assert not pair_dates(guarded, ["entry_signal"]).isin(flat).any()
    # Exits are unaffected, and a flat day is still a fill day.
    assert pair_dates(guarded, ["exit_signal", "entry", "exit"]).isin(flat).any()


def test_unchanged_close_without_volume_is_no_print():
    # No signal and no fill on a stale print: the pair's rows skip it.
    closes, pairs, flat, volumes = panel_with_flat_closes()
    volumes.loc[flat, FLAT_LEG] = 0.0
    masked = quiet(closes, pairs, GROSS, volumes=volumes).trades
    unmasked = quiet(closes, pairs, replace(GROSS, legacy_stale_prints=True), volumes=volumes).trades
    assert pair_dates(unmasked, DECISIONS_AND_FILLS).isin(flat).any()
    assert len(pair_dates(masked, DECISIONS_AND_FILLS)) > 0
    assert not pair_dates(masked, DECISIONS_AND_FILLS).isin(flat).any()
