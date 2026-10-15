"""Fills on the next close (audit task T1).

An order decided on bar i of a pair's spread series fills on bar i+1; the
holding period and borrow start at the fill.  Hand-made spreads with a
hand-made z (`z_series`), so no 60-day warm-up is needed.  Zero costs unless
stated.  ENTRY_Z = 2, EXIT_Z = 0, STOP_Z = 3.5: z = 3 opens a short spread
(side -1), z <= 0 is its target.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wfo_daily_core import backtest_pair  # noqa: E402

DAYS = pd.bdate_range("2021-03-01", periods=10)
K = 5           # the spread jumps on day K and fully reverts on day K+1
JUMP = 0.05


def series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=DAYS)


def jump_case() -> tuple[pd.Series, pd.Series]:
    """Spread flat at 0 except +JUMP on day K; z = 3 on day K only, 0 elsewhere."""
    spread, z = np.zeros(len(DAYS)), np.zeros(len(DAYS))
    spread[K], z[K] = JUMP, 3.0
    return series(spread), series(z)


def run(spread: pd.Series, z: pd.Series, **kw) -> pd.DataFrame:
    """backtest_pair with every bar out-of-sample and zero costs."""
    return backtest_pair(spread, DAYS[0], DAYS[-1] + pd.Timedelta(days=1),
                         cost_per_trade=0.0, z_series=z, **kw)


def test_next_close_enters_after_the_jump_and_books_nothing():
    spread, z = jump_case()
    tr = run(spread, z)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["pnl"] == pytest.approx(0.0, abs=1e-12)    # gross ≈ 0: the reversion is not booked
    assert (t["entry_signal"], t["entry"]) == (DAYS[K], DAYS[K + 1])
    assert (t["exit_signal"], t["exit"]) == (DAYS[K + 1], DAYS[K + 2])
    assert (t["side"], t["held"], t["reason"]) == (-1, 1, "target")


def test_legacy_same_close_books_the_reversion():
    spread, z = jump_case()
    tr = run(spread, z, legacy_same_close=True)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["pnl"] == pytest.approx(JUMP)
    assert t["entry"] == t["entry_signal"] == DAYS[K]
    assert t["exit"] == t["exit_signal"] == DAYS[K + 1]


def test_holding_period_and_borrow_start_at_the_fill():
    # Entry decided on day 2, filled on day 3 (held 0); held reaches max_hold = 3
    # on day 6 → time exit decided, filled on day 7: held 4 days, 4 days of borrow.
    z = series([0, 0] + [2.5] * 8)          # above ENTRY_Z, below STOP_Z, never at target
    tr = run(series(np.zeros(10)), z, max_hold=3, borrow_per_day=0.001)
    assert len(tr) == 1                     # the re-entry decided on day 7 is still open at the end
    t = tr.iloc[0]
    assert (t["entry_signal"], t["entry"]) == (DAYS[2], DAYS[3])
    assert (t["exit_signal"], t["exit"]) == (DAYS[6], DAYS[7])
    assert (t["held"], t["reason"]) == (4, "time")
    assert t["pnl"] == pytest.approx(-4 * 0.001)


def test_exit_decided_on_the_last_bar_fills_at_that_close_as_end_of_data():
    spread = series([0] * 7 + [0.04, 0.03, 0.01])
    z = series([0] * 7 + [3, 3, 0])         # entry decided day 7, target decided on day 9 (last bar)
    tr = run(spread, z)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert (t["entry_signal"], t["entry"]) == (DAYS[7], DAYS[8])
    assert t["exit"] == t["exit_signal"] == DAYS[9]
    assert (t["reason"], t["held"]) == ("end_of_data", 1)
    assert t["pnl"] == pytest.approx(-1 * (0.01 - 0.03))


def test_entry_decided_on_the_last_bar_is_not_filled():
    assert run(series(np.zeros(10)), series([0] * 9 + [3])).empty


def test_no_decision_on_bars_after_decision_end():
    # Day 9 is a fill-only bar: decision_end = day 8.
    spread = series([0] * 6 + [0.04, 0.03, 0.02, 0.01])
    # An exit decided on the last decision bar fills on the fill-only bar.
    t = run(spread, series([0] * 6 + [3, 3, 0, 0]), decision_end=DAYS[8]).iloc[0]
    assert (t["exit_signal"], t["exit"], t["reason"]) == (DAYS[8], DAYS[9], "target")
    assert t["pnl"] == pytest.approx(-1 * (0.01 - 0.03))
    # A target on the fill-only bar is no decision: the position stays open (dropped).
    z = series([0] * 7 + [3, 3, 0])
    assert run(spread, z, decision_end=DAYS[8]).empty
    assert run(spread, z)["reason"].tolist() == ["end_of_data"]    # without the cut
    # Nor is an entry signal on it.
    assert run(spread, series([0] * 9 + [3]), decision_end=DAYS[8]).empty
    # Nor a CUSUM break (S+ = 2.5, 3.0, then 5.5 > 4.9 on day 4).
    z = series([0, 0, 3, 1, 3, 0, 0, 0, 0, 0])
    assert run(spread, z, cusum_h=4.9)["reason"].tolist() == ["cusum_break"]
    assert run(spread, z, cusum_h=4.9, decision_end=DAYS[3]).empty


def test_cusum_break_force_close_fills_on_the_next_bar():
    # Entry decided day 2 (S+ = 2.5), filled day 3; S+ = 5.0 > 4.9 on day 3 → close filled day 4.
    spread = series([0, 0, 0.03, 0.04, 0.02, 0, 0, 0, 0, 0])
    tr = run(spread, series([0, 0, 3, 3, 0, 0, 0, 0, 0, 0]), cusum_h=4.9)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert (t["entry_signal"], t["entry"]) == (DAYS[2], DAYS[3])
    assert (t["exit_signal"], t["exit"]) == (DAYS[3], DAYS[4])
    assert (t["reason"], t["held"]) == ("cusum_break", 1)
    assert t["pnl"] == pytest.approx(-1 * (0.02 - 0.04))


def test_vwap_fills_at_the_next_day_vwap_and_falls_back_to_the_next_close():
    spread, z = jump_case()                 # entry decided on day K, exit on day K+1
    vwap = series(0.001 * np.arange(1, 11))  # exec_spread[i] = VWAP spread of day i+1
    t = run(spread, z, exec_spread=vwap).iloc[0]
    assert (t["entry"], t["exit"]) == (DAYS[K + 1], DAYS[K + 2])
    assert t["pnl"] == pytest.approx(-1 * (vwap.iloc[K + 1] - vwap.iloc[K]))
    # No VWAP for the exit day → the close of that day (never an earlier VWAP).
    t = run(spread, z, exec_spread=vwap.where(DAYS != DAYS[K + 1])).iloc[0]
    assert t["exit"] == DAYS[K + 2]
    assert t["pnl"] == pytest.approx(-1 * (spread.iloc[K + 2] - vwap.iloc[K]))
    # No VWAP for the entry fill day → the entry fills at the close of that day
    # (R51: the decision on day K does not read day K+1's VWAP); legacy
    # same-close keeps T0's rule: not filled.
    t = run(spread, z, exec_spread=vwap.where(DAYS != DAYS[K])).iloc[0]
    assert (t["entry_signal"], t["entry"], t["exit"]) == (DAYS[K], DAYS[K + 1], DAYS[K + 2])
    assert t["pnl"] == pytest.approx(-1 * (vwap.iloc[K + 1] - spread.iloc[K + 1]))
    assert run(spread, z, exec_spread=vwap.where(DAYS != DAYS[K]), legacy_same_close=True).empty
