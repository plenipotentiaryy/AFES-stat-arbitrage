"""One continuous position per pair across walk-forward windows (audit task T3).

Run level (`run_wfo_daily` on a hand-made two-leg panel, zero costs): a
trade open at a window cut is kept, its loss after the cut is booked in
`trades` and `daily_pnl`, it keeps the β of its entry window, the pair never
holds a second position, the run closes what is still open at the end of the
last window at the next available close, and β of new entries comes from the
training rows strictly before the OOS start (T9.1).

Engine level (`backtest_pair_continuous` on 32 hand-made rows with a
hand-made z per window): an open trade decides its exits on the z of its own
entry window, whatever window the row is in; the active window gates new
entries only; CUSUM and the Kalman β of the entry decision row.
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

import wfo_daily_core as core  # noqa: E402
from placebo_harness import GROSS  # noqa: E402
from wfo_daily_core import estimate_beta, run_wfo_daily  # noqa: E402

# ── Run level ───────────────────────────────────────────────────────────────

DAYS = pd.bdate_range("2020-01-01", "2021-01-15", name="Date")
PAIR = "PA-PB"
# Windows (6 / 3 / 3 months from 2020-01-01): OOS 1 = [2020-07-01, 2020-10-01),
# OOS 2 = [2020-10-01, 2021-01-01]; decisions up to 2021-01-01 (the last OOS end).
OOS_S = [pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-01")]
LAST_DECISION = pd.Timestamp("2021-01-01")
SIG = 0.005      # sd of the spread noise before the event


def panel(entry_day: str, stop_day: str | None = None, level: float = 2.7,
          ramp: float = 0.15, stop_jump: float = 4.0) -> pd.DataFrame:
    """log PA = 1.5·log PB + s.  s is noise (sd SIG) until `entry_day`, then a
    level of `level`·SIG rising by `ramp`·SIG a day (rolling z stays between
    ENTRY_Z = 2 and STOP_Z = 3.5: a short spread opens, never reaches its
    target and loses), plus a jump of `stop_jump`·SIG on `stop_day` (z > 3.5:
    stop)."""
    rng = np.random.default_rng(3)
    log_pb = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, len(DAYS)))
    s = rng.normal(0.0, SIG, len(DAYS))
    k = DAYS.get_loc(pd.Timestamp(entry_day))
    s[k:] = SIG * (level + ramp * np.arange(len(DAYS) - k))
    if stop_day:
        s[DAYS.get_loc(pd.Timestamp(stop_day)):] += stop_jump * SIG
    log_pa = 1.5 * log_pb + s + np.log(30.0) - 1.5 * np.log(50.0)
    return pd.DataFrame({"PA": np.exp(log_pa), "PB": np.exp(log_pb)}, index=DAYS)


def run(closes: pd.DataFrame, cfg=GROSS):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, [PAIR], cfg)


def train_beta(closes: pd.DataFrame, oos_s: pd.Timestamp, include_oos_start: bool = False) -> float:
    rows = closes.index <= oos_s if include_oos_start else closes.index < oos_s
    return estimate_beta(closes.loc[rows, "PA"], closes.loc[rows, "PB"])


def the_trade(trades: pd.DataFrame, entry_signal: str) -> pd.Series:
    t = trades[trades["entry_signal"] == pd.Timestamp(entry_signal)]
    assert len(t) == 1, f"no trade decided on {entry_signal}"
    return t.iloc[0]


def rolling_z(closes: pd.DataFrame, beta: float) -> pd.Series:
    sp = np.log(closes["PA"]) - beta * np.log(closes["PB"])
    return (sp - sp.rolling(60).mean()) / sp.rolling(60).std()


def test_position_open_at_a_window_cut_keeps_its_loss():
    # Acceptance: entered near the end of window 1 (decided 2020-09-28, filled
    # 09-29), open at the cut 2020-10-01, stopped out in window 2 (decided
    # 10-08, filled 10-09) after the spread kept rising: a loss.  Before T3 the
    # trade was dropped at the cut and window 2 started flat.
    closes = panel("2020-09-28", stop_day="2020-10-08")
    res = run(closes)
    t = the_trade(res.trades, "2020-09-28")
    assert (t["entry"], t["exit_signal"], t["exit"]) == (
        pd.Timestamp("2020-09-29"), pd.Timestamp("2020-10-08"), pd.Timestamp("2020-10-09"))
    assert (t["side"], t["reason"]) == (-1, "stop")
    assert (t["window_entered"], t["window_exited"]) == (1, 2)
    assert (t["window"], t["oos_start"]) == (1, OOS_S[0])
    assert t["pnl"] < -0.02
    # The loss is in daily_pnl, marked to market day by day (T4): the one
    # pair's daily PnL from the entry to the exit fill is this trade's, and
    # most of the loss is booked before the exit date.
    held = res.daily_pnl.loc[t["entry"]:t["exit"]]
    assert held.sum() == pytest.approx(t["pnl"], abs=1e-12)
    assert held.loc[:t["exit_signal"]].sum() < 0.5 * t["pnl"]
    assert res.daily_pnl.sum() == pytest.approx(res.trades["pnl"].sum(), abs=1e-12)
    # PnL from the leg fills, quantities fixed at the entry fill (T2).
    p1e, p2e = closes.loc[t["entry"]]
    p1x, p2x = closes.loc[t["exit"]]
    assert (t["p1_entry"], t["p2_entry"], t["p1_exit"], t["p2_exit"]) == (p1e, p2e, p1x, p2x)
    assert t["pnl"] == pytest.approx(-(1 / p1e * (p1x - p1e) - t["beta"] / p2e * (p2x - p2e)))


def test_open_trade_keeps_its_entry_beta_and_its_own_spread():
    closes = panel("2020-09-28", stop_day="2020-10-08")
    trades = run(closes).trades
    beta1, beta2 = train_beta(closes, OOS_S[0]), train_beta(closes, OOS_S[1])
    assert beta1 != beta2
    t = the_trade(trades, "2020-09-28")
    assert t["beta"] == beta1 and t["q2"] == beta1 / t["p2_entry"]
    # Exit decided on the first row after the fill where the rolling z of
    # log PA − β1·log PB reaches the stop (|z| ≥ 3.5) or the target (z ≤ 0).
    z1 = rolling_z(closes, beta1).loc[t["entry"]:]
    assert t["exit_signal"] == z1.index[((z1.abs() >= 3.5) | (z1 <= 0)).argmax()]
    # New entries of window 2 use β2.
    assert (trades.loc[trades["window_entered"] == 2, "beta"] == beta2).all()
    assert (trades["window_entered"] == 2).any()


def test_never_a_second_position_in_the_pair():
    # Window 2's own z is above ENTRY_Z from its first day while the window-1
    # trade is still open: the per-window reset (legacy) opens a position
    # there, the continuous engine does not.
    closes = panel("2020-09-28", stop_day="2020-10-08")
    trades = run(closes).trades.sort_values("entry_signal")
    t = the_trade(trades, "2020-09-28")
    z2 = rolling_z(closes, train_beta(closes, OOS_S[1])).loc[OOS_S[1]:t["exit_signal"]]
    assert (z2.iloc[:-1] > 2).all()
    assert not trades["entry_signal"].between(t["entry_signal"], t["exit_signal"], inclusive="right").any()
    assert (trades["entry"].iloc[1:].to_numpy() > trades["exit"].iloc[:-1].to_numpy()).all()
    legacy = run(closes, replace(GROSS, legacy_window_reset=True)).trades
    assert legacy["entry_signal"].between(OOS_S[1], t["exit_signal"]).any()
    assert not (legacy["entry_signal"] == pd.Timestamp("2020-09-28")).any()   # dropped at the cut


def test_end_of_data_close_at_the_next_available_close():
    # Entered at the end of window 2 and never closed by a rule: after the last
    # decision row (2021-01-01) the trade is closed at the next row with prints
    # of both legs, reason end_of_data, exit_signal = the last decision row.
    closes = panel("2020-12-28", ramp=0.02)
    t = the_trade(run(closes).trades, "2020-12-28")
    assert (t["reason"], t["exit_signal"], t["exit"]) == (
        "end_of_data", LAST_DECISION, pd.Timestamp("2021-01-04"))
    assert (t["window_entered"], t["window_exited"]) == (2, 2)
    assert (t["p1_exit"], t["p2_exit"]) == tuple(closes.loc["2021-01-04"])
    # No print of PA on 01-04: the next row with both legs, 01-05.
    gap = closes.copy()
    gap.loc["2021-01-04", "PA"] = np.nan
    t = the_trade(run(gap).trades, "2020-12-28")
    assert (t["exit_signal"], t["exit"]) == (LAST_DECISION, pd.Timestamp("2021-01-05"))
    assert (t["p1_exit"], t["p2_exit"]) == tuple(closes.loc["2021-01-05"])
    # No row after the last decision row (cfg.end): at its own close.
    t = the_trade(run(closes, replace(GROSS, end="2021-01-01")).trades, "2020-12-28")
    assert (t["reason"], t["exit_signal"], t["exit"]) == ("end_of_data", LAST_DECISION, LAST_DECISION)


def test_train_slice_ends_strictly_before_the_oos_start():
    # T9.1: β of window w from the rows in [tr_s, oos_s); the per-window reset
    # (legacy) keeps its slice [tr_s, oos_s] that includes the first OOS day.
    closes = panel("2020-09-28", stop_day="2020-10-08")
    trades = run(closes).trades
    legacy = run(closes, replace(GROSS, legacy_window_reset=True)).trades
    for w, oos_s in enumerate(OOS_S, start=1):
        strict, inclusive = train_beta(closes, oos_s), train_beta(closes, oos_s, include_oos_start=True)
        assert strict != inclusive
        assert (trades.loc[trades["window_entered"] == w, "beta"] == strict).all()
        assert (legacy.loc[legacy["window"] == w, "beta"] == inclusive).all()
    assert set(trades["window_entered"]) == set(legacy["window"]) == {1, 2}


@pytest.mark.parametrize("flag", ["legacy_same_close", "legacy_spread_pnl"])
def test_pre_t3_legacy_flags_need_the_window_reset(flag):
    closes = panel("2020-09-28")
    with pytest.raises(ValueError, match="legacy_window_reset"):
        run(closes, replace(GROSS, **{flag: True}))
    # (spread PnL has no leg quantities to mark daily: it also needs the
    # exit-date booking, T4, and the per-trade cost, T8)
    run(closes, replace(GROSS, legacy_window_reset=True, **{flag: True},
                        legacy_exit_date_pnl=flag == "legacy_spread_pnl",
                        legacy_flat_cost=flag == "legacy_spread_pnl"))


# ── Engine level ────────────────────────────────────────────────────────────

ROWS = pd.bdate_range("2021-03-01", periods=32)
# Window 1: OOS rows 10–19, window 2: OOS rows 20–29 (end date = row 30, the
# last decision row); row 31 is the fill-only row.
WINDOWS = [(ROWS[0], ROWS[10], ROWS[10], ROWS[20]), (ROWS[0], ROWS[20], ROWS[20], ROWS[30])]
N_DECIDE = 31
LEGS = np.column_stack([100.0 + np.arange(32), np.full(32, 50.0)])


def zs(**rows) -> np.ndarray:
    """z = 0 everywhere except {row: value}; keys are 'r<row>' or 'r<a>_<b>' (a..b)."""
    z = np.zeros(len(ROWS))
    for key, value in rows.items():
        a, _, b = key[1:].partition("_")
        z[int(a):int(b or a) + 1] = value
    return z


def engine(pw: dict, n_decide: int = N_DECIDE, rows: int = 32, windows=WINDOWS, **kw) -> pd.DataFrame:
    return core.backtest_pair_continuous(ROWS[:rows], LEGS[:rows], n_decide, windows, pw, **kw)


def test_exits_use_the_z_of_the_entry_window():
    # Window 1's z opens a short spread on row 17 and stays at 2.5 (never a
    # target or stop); window 2's z has its target (0) on rows 20-29 and an
    # entry signal (3) on row 22.  The trade ignores window 2's z: it exits by
    # time (max_hold 5: held 5 on row 23, filled on row 24), and no second
    # position opens on row 22.
    pw = {1: core.PairWindow(beta=1.0, z=zs(r17=3.0, r18_31=2.5)),
          2: core.PairWindow(beta=2.0, z=zs(r22=3.0))}
    tr = engine(pw, max_hold=5)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert [t[c] for c in ("entry_signal", "entry", "exit_signal", "exit")] == [ROWS[17], ROWS[18], ROWS[23], ROWS[24]]
    assert (t["side"], t["reason"], t["held"], t["beta"]) == (-1, "time", 6, 1.0)
    assert (t["window_entered"], t["window_exited"]) == (1, 2)
    q1, q2 = 1 / LEGS[18, 0], 1.0 / LEGS[18, 1]
    assert (t["q1"], t["q2"]) == (q1, q2)
    assert t["gross"] == -(q1 * (LEGS[24, 0] - LEGS[18, 0]) - q2 * (LEGS[24, 1] - LEGS[18, 1]))
    # After the exit decision, window 2 enters on its own z with its own β.
    pw[2] = core.PairWindow(beta=2.0, z=zs(r22=3.0, r25=-3.0, r26=-2.5))
    tr = engine(pw, max_hold=5)
    assert len(tr) == 2
    t = tr.iloc[1]
    assert [t[c] for c in ("entry_signal", "exit_signal", "side", "beta", "window_entered")] == [
        ROWS[25], ROWS[27], 1, 2.0, 2]


def test_ineligible_window_and_gaps_allow_no_new_entries_but_open_trades_continue():
    z1 = zs(r17=3.0, r18_31=2.5)
    # Window 2 not eligible for the pair: no entries there; the open trade
    # still exits by its own rules in window 2.
    tr = engine({1: core.PairWindow(beta=1.0, z=z1)}, max_hold=5)
    assert tr["entry_signal"].tolist() == [ROWS[17]] and tr["exit_signal"].tolist() == [ROWS[23]]
    # A gap between the OOS of window 1 (rows 10-14) and window 2 (rows 20-29):
    # the signal on row 17 lies in no OOS, so no entry.
    gap = [(ROWS[0], ROWS[10], ROWS[10], ROWS[15]), WINDOWS[1]]
    assert engine({1: core.PairWindow(beta=1.0, z=z1), 2: core.PairWindow(beta=2.0, z=zs())}, windows=gap).empty
    # Overlapping OOS: the latest-starting window is active.
    overlap = [(ROWS[0], ROWS[10], ROWS[10], ROWS[30]), WINDOWS[1]]
    tr = engine({1: core.PairWindow(beta=1.0, z=zs(r22=3.0)), 2: core.PairWindow(beta=2.0, z=zs(r24=-3.0))},
                windows=overlap)
    assert tr["entry_signal"].tolist() == [ROWS[24]] and tr["beta"].tolist() == [2.0]


def test_end_of_data_closes_after_the_last_decision_row():
    pw = {2: core.PairWindow(beta=1.0, z=zs(r27=3.0, r28_31=2.5))}
    t = engine(pw).iloc[0]
    assert (t["exit_signal"], t["exit"], t["reason"], t["held"]) == (ROWS[30], ROWS[31], "end_of_data", 3)
    assert (t["p1_exit"], t["window_exited"]) == (LEGS[31, 0], 2)
    # No fill-only row: at the close of the last decision row.
    t = engine(pw, rows=31).iloc[0]
    assert (t["exit_signal"], t["exit"], t["reason"]) == (ROWS[30], ROWS[30], "end_of_data")
    # An entry decided on the last decision row fills on the fill-only row and
    # is closed there at once (held 0, no gross; costs are charged).
    windows = [WINDOWS[0], (ROWS[0], ROWS[20], ROWS[20], ROWS[31])]
    t = engine({2: core.PairWindow(beta=1.0, z=zs(r30=3.0), cost=0.001)}, windows=windows).iloc[0]
    assert (t["entry_signal"], t["entry"], t["exit"], t["held"]) == (ROWS[30], ROWS[31], ROWS[31], 0)
    assert (t["gross"], t["pnl"], t["reason"]) == (0.0, -0.001, "end_of_data")


def test_window_exited_of_a_pair_whose_data_end_early():
    # The pair's rows end on row 17 (window 1's OOS) in a two-window run: the
    # trade open then is closed at that row's close (end_of_data) and labelled
    # with the window of that exit decision row, 1, not the run's last window.
    t = engine({1: core.PairWindow(beta=1.0, z=zs(r12=3.0, r13_31=2.5))}, n_decide=18, rows=18).iloc[0]
    assert (t["entry_signal"], t["exit_signal"], t["exit"], t["reason"]) == (
        ROWS[12], ROWS[17], ROWS[17], "end_of_data")
    assert (t["window_entered"], t["window_exited"]) == (1, 1)


def test_cost_and_borrow_of_the_entry_window():
    pw = {1: core.PairWindow(beta=1.0, z=zs(r17=3.0, r18_31=2.5), cost=0.002, borrow=0.0001),
          2: core.PairWindow(beta=1.0, z=zs(), cost=0.5, borrow=0.5)}
    t = engine(pw, max_hold=5).iloc[0]
    assert t["pnl"] == t["size"] * t["gross"] - t["size"] * 0.002 - t["size"] * (6 * 0.0001)


def test_cusum_break_in_the_oos_force_closes_and_blocks_entries():
    # Window 2's CUSUM breaks on row 23 (inside its OOS): the window-1 trade is
    # closed (decided 23, filled 24, cusum_break) and window 2 enters no more.
    z1 = zs(r17=3.0, r18_31=2.5)
    pw = {1: core.PairWindow(beta=1.0, z=z1), 2: core.PairWindow(beta=1.0, z=zs(r26=3.0), cusum_break=23)}
    t = engine(pw).iloc[0]
    assert (t["exit_signal"], t["exit"], t["reason"], t["window_exited"]) == (ROWS[23], ROWS[24], "cusum_break", 2)
    assert len(engine(pw)) == 1
    # A break inside window 2's training rows (row 15) blocks its entries but
    # closes nothing: the trade runs to the end of the data.
    pw[2] = core.PairWindow(beta=1.0, z=zs(r26=3.0), cusum_break=15)
    tr = engine(pw)
    assert tr["reason"].tolist() == ["end_of_data"] and tr["entry_signal"].tolist() == [ROWS[17]]


def test_kalman_beta_of_the_entry_decision_row_and_its_own_exit_z():
    # β_e = the Kalman β of the decision row (2.0 on row 17); the exit z is
    # the one of the spread with β_e: target on row 21 only for β = 2.0.
    kb = np.ones(len(ROWS))
    kb[17] = 2.0

    def exit_z_of(beta: float) -> np.ndarray:
        return zs(r18_31=2.5, r21=0.0) if beta == 2.0 else zs(r18_31=2.5)

    z = zs(r17=3.0, r18_31=2.5)
    pw = {1: core.PairWindow(beta=np.nan, z=z), 2: core.PairWindow(beta=np.nan, z=z)}
    t = engine(pw, kalman_beta=kb, exit_z_of=exit_z_of).iloc[0]
    assert (t["beta"], t["exit_signal"], t["reason"]) == (2.0, ROWS[21], "target")
