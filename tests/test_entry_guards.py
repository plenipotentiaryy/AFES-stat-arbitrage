"""Entry guards (audit task T7).

An entry is decided only if entry_z < |z| < stop_z (ENTRY_Z = 2, STOP_Z = 3.5;
with the vol-tight stop, its tightened stop).  The old rule entered on any
|z| > entry_z, so a 5-10 sigma jump, often from a stale price, was traded and
then "reverted" (`legacy_no_entry_guards`).  The rule lives in one place,
`entry_decision`, shared by both engines: the continuous default and the
per-window `legacy_window_reset` engine.

The two structural guards of T2 are pinned here as well: both legs have a
genuine print on the signal date, and the z-score comes from a trailing window
that excludes no-print days (the pair's rows are the dates on which both legs
printed).

Engine level: hand-made z series (`backtest_pair`, `backtest_pair_continuous`).
Run level (`run_wfo_daily`, zero costs): a two-leg panel whose spread
alternates +-SIG (|z| about 1, never an entry) with one day's jump.
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
from step3j_wfo_daily import build_parser  # noqa: E402
from wfo_daily_core import (ENTRY_Z, EXIT_Z, MAX_HOLD, STOP_Z, Z_WIN, backtest_pair,  # noqa: E402
                            entry_decision, estimate_beta, exit_reason, rolling_z,
                            run_wfo_daily, spread_series, tightened_stop)

BELOW_STOP = float(np.nextafter(STOP_Z, 0.0))      # the largest z below the stop


# ── entry_decision ──────────────────────────────────────────────────────────

def test_entry_decision_enters_only_between_entry_z_and_stop_z():
    for z, side in ((BELOW_STOP, -1), (-BELOW_STOP, +1), (2.5, -1), (-2.5, +1)):
        assert entry_decision(4, z, 10, ENTRY_Z) == (side, 1.0, 5)
    for z in (STOP_Z, -STOP_Z, 3.6, -3.6, 6.0, -6.0, 10.0):            # at or beyond the stop
        assert entry_decision(4, z, 10, ENTRY_Z) is None
    for z in (ENTRY_Z, -ENTRY_Z, 0.0, 1.0, np.nan):                    # not beyond entry_z (strict)
        assert entry_decision(4, z, 10, ENTRY_Z) is None
    assert entry_decision(4, float(np.nextafter(ENTRY_Z, 9.0)), 10, ENTRY_Z) == (-1, 1.0, 5)


def test_entry_decision_stop_z_is_a_value_or_a_function_of_the_bar():
    assert entry_decision(4, 3.0, 10, ENTRY_Z, stop_z=3.0) is None            # |z| == stop
    assert entry_decision(4, 3.0, 10, ENTRY_Z, stop_z=3.01) == (-1, 1.0, 5)
    tight = lambda i: 2.5 if i == 4 else 99.0                                 # noqa: E731
    assert entry_decision(4, 3.0, 10, ENTRY_Z, stop_z=tight) is None
    assert entry_decision(4, 2.4, 10, ENTRY_Z, stop_z=tight) == (-1, 1.0, 5)
    assert entry_decision(5, 3.0, 10, ENTRY_Z, stop_z=tight) == (-1, 1.0, 6)  # asked for its own bar


def test_legacy_flag_restores_the_old_entry_rule():
    for z, side in ((6.0, -1), (-6.0, +1), (STOP_Z, -1), (10.0, -1)):
        assert entry_decision(4, z, 10, ENTRY_Z, legacy_no_entry_guards=True) == (side, 1.0, 5)
    assert entry_decision(4, 2.0, 10, ENTRY_Z, legacy_no_entry_guards=True) is None


def test_an_entry_is_never_stopped_out_on_its_own_bar():
    # For the same z the entry window is the complement of the stop: whatever
    # enters is not past the stop rule that would exit it at once (helper level;
    # on the Kalman continuous path the exit z is another z).
    entries = 0
    for z in (*np.linspace(-8.0, 8.0, 321), STOP_Z, -STOP_Z, BELOW_STOP, -BELOW_STOP):
        entry = entry_decision(4, z, 10, ENTRY_Z)
        if entry is not None:
            entries += 1
            assert exit_reason(entry[0], z, 0, EXIT_Z, STOP_Z, MAX_HOLD) is None
    assert entries > 50                                  # (2, 3.5) both ways, plus BELOW_STOP


# ── Engine level, hand-made z ───────────────────────────────────────────────

DAYS = pd.bdate_range("2021-03-01", periods=10)
K = 5           # the jump day: z is set on day K only


def series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=DAYS)


def one_day_z(value: float) -> pd.Series:
    z = np.zeros(len(DAYS))
    z[K] = value
    return series(z)


def per_window(z: pd.Series, **kw) -> pd.DataFrame:
    """`backtest_pair` (the per-window engine), every bar out-of-sample, zero costs."""
    return backtest_pair(series(np.zeros(len(DAYS))), DAYS[0], DAYS[-1] + pd.Timedelta(days=1),
                         cost_per_trade=0.0, z_series=z, **kw)


ROWS = pd.bdate_range("2021-03-01", periods=32)
WINDOWS = [(ROWS[0], ROWS[10], ROWS[10], ROWS[20]), (ROWS[0], ROWS[20], ROWS[20], ROWS[30])]
LEGS = np.column_stack([100.0 + np.arange(32), np.full(32, 50.0)])


def zs(**rows) -> np.ndarray:
    """z = 0 everywhere except {row: value}; keys are 'r<row>' or 'r<a>_<b>' (a..b)."""
    z = np.zeros(len(ROWS))
    for key, value in rows.items():
        a, _, b = key[1:].partition("_")
        z[int(a):int(b or a) + 1] = value
    return z


def continuous(z: np.ndarray, **kw) -> pd.DataFrame:
    """`backtest_pair_continuous` (the default engine) with one window's z."""
    engine_kw = {k: kw.pop(k) for k in ("stop_z", "vol_tight_stop", "legacy_no_entry_guards")
                 if k in kw}
    pw = {1: core.PairWindow(beta=1.0, z=z, **kw)}
    return core.backtest_pair_continuous(ROWS, LEGS, 31, WINDOWS, pw, **engine_kw)


# (the engine, a function z -> trades; the fill is the next row after the decision)
ENGINES = {
    "per_window": lambda z, **kw: per_window(one_day_z(z), **kw),
    "continuous": lambda z, **kw: continuous(zs(r17=z), **kw),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sign", [+1, -1], ids=["short_spread", "long_spread"])
def test_a_z_at_or_beyond_the_stop_is_no_entry(engine, sign):
    run = ENGINES[engine]
    assert run(sign * 6.0).empty                        # a 6-sigma jump
    assert run(sign * STOP_Z).empty                     # |z| == stop_z
    assert run(sign * 3.6).empty
    tr = run(sign * BELOW_STOP)                         # just below the stop
    assert len(tr) == 1 and tr["side"].iloc[0] == -sign
    tr = run(sign * 2.5)
    assert len(tr) == 1 and tr["side"].iloc[0] == -sign


@pytest.mark.parametrize("engine", ENGINES)
def test_legacy_flag_enters_on_the_jump(engine):
    run = ENGINES[engine]
    for z, side in ((6.0, -1), (-6.0, +1), (STOP_Z, -1)):
        tr = run(z, legacy_no_entry_guards=True)
        assert len(tr) == 1 and tr["side"].iloc[0] == side


@pytest.mark.parametrize("engine", ENGINES)
def test_the_entry_window_follows_the_stop_z_argument(engine):
    run = ENGINES[engine]
    assert run(3.0, stop_z=3.0).empty                    # |z| == its stop
    assert len(run(3.0, stop_z=3.01)) == 1
    assert run(4.0).empty and len(run(4.0, stop_z=4.5)) == 1


def test_the_bar_after_the_jump_is_a_normal_entry_bar():
    # The guard blocks the jump bar only: if z is still in (entry_z, stop_z) on
    # the next bar, the entry is decided then (and the legacy run entered one bar earlier).
    z = zs(r17=6.0, r18=2.5)
    t = continuous(z).iloc[0]
    assert (t["entry_signal"], t["entry"]) == (ROWS[18], ROWS[19])
    t = continuous(z, legacy_no_entry_guards=True).iloc[0]
    assert (t["entry_signal"], t["entry"]) == (ROWS[17], ROWS[18])


# ── Vol-tight stop: the guard uses the tighter stop ─────────────────────────

N_TIGHT = 80            # bars 0-59 quiet, bars 60-79 four times as volatile
I_TIGHT = 75            # decision bar: 20-bar / 60-bar spread std ratio > 1.5
ROWS_TIGHT = pd.bdate_range("2021-03-01", periods=N_TIGHT)
TIGHT_SPREAD = pd.Series(np.random.default_rng(1).normal(0.0, 1.0, N_TIGHT)
                         * np.where(np.arange(N_TIGHT) < 60, 0.01, 0.04), index=ROWS_TIGHT)
TIGHT_STOP_Z = 2.5      # the default of vol_tight_stop_z


def test_vol_tight_stop_precondition():
    assert tightened_stop(TIGHT_SPREAD, I_TIGHT, STOP_Z, 1.5, TIGHT_STOP_Z) == TIGHT_STOP_Z
    assert tightened_stop(TIGHT_SPREAD, 40, STOP_Z, 1.5, TIGHT_STOP_Z) == STOP_Z      # quiet bar


def tight_z(value: float) -> np.ndarray:
    z = np.zeros(N_TIGHT)
    z[I_TIGHT] = value
    return z


def tight_per_window(value: float, **kw) -> pd.DataFrame:
    return backtest_pair(TIGHT_SPREAD, ROWS_TIGHT[0], ROWS_TIGHT[-1] + pd.Timedelta(days=1),
                         cost_per_trade=0.0, z_series=pd.Series(tight_z(value), index=ROWS_TIGHT), **kw)


def tight_continuous(value: float, **kw) -> pd.DataFrame:
    legs = np.column_stack([100.0 + np.arange(N_TIGHT), np.full(N_TIGHT, 50.0)])
    windows = [(ROWS_TIGHT[0], ROWS_TIGHT[10], ROWS_TIGHT[10], ROWS_TIGHT[N_TIGHT - 2])]
    pw = {1: core.PairWindow(beta=1.0, z=tight_z(value), spread=TIGHT_SPREAD)}
    return core.backtest_pair_continuous(ROWS_TIGHT, legs, N_TIGHT - 1, windows, pw, **kw)


@pytest.mark.parametrize("run", [tight_per_window, tight_continuous], ids=["per_window", "continuous"])
def test_with_the_vol_tight_stop_the_entry_window_ends_at_the_tightened_stop(run):
    # z = 3.0 is below STOP_Z but at/above the tightened stop 2.5: no entry
    # (the trade would be stopped out by the stop it carries); z = 2.4 enters.
    assert len(run(3.0)) == 1                              # option off: the plain stop
    assert run(3.0, vol_tight_stop=True).empty
    assert run(TIGHT_STOP_Z, vol_tight_stop=True).empty    # |z| == the tightened stop
    assert len(run(2.4, vol_tight_stop=True)) == 1
    assert run(6.0).empty and run(4.0, vol_tight_stop=True).empty
    assert len(run(3.0, vol_tight_stop=True, legacy_no_entry_guards=True)) == 1


# ── Run level: a single-day jump on a two-leg panel ─────────────────────────

PANEL_DAYS = pd.bdate_range("2020-01-01", "2021-01-15", name="Date")
SIG = 0.005
OOS_START = pd.Timestamp("2020-07-01")       # window 1: OOS [2020-07-01, 2020-10-01)
JUMP_DAY = pd.Timestamp("2020-08-12")


def panel(jump: float | None = None, day=JUMP_DAY) -> pd.DataFrame:
    """log PA = 1.5·log PB + s, s alternating +-SIG (|z| about 1: never an entry)
    plus a jump of `jump`·SIG on one day only."""
    rng = np.random.default_rng(5)
    log_pb = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.003, len(PANEL_DAYS)))
    s = SIG * np.where(np.arange(len(PANEL_DAYS)) % 2 == 0, 1.0, -1.0)
    if jump is not None:
        s[PANEL_DAYS.get_loc(day)] += jump * SIG
    log_pa = 1.5 * log_pb + s + np.log(30.0) - 1.5 * np.log(50.0)
    return pd.DataFrame({"PA": np.exp(log_pa), "PB": np.exp(log_pb)}, index=PANEL_DAYS)


def quiet(closes: pd.DataFrame, cfg=GROSS, pairs=("PA-PB",)):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, list(pairs), cfg)


def jump_z(closes: pd.DataFrame, day=JUMP_DAY, include_oos_start: bool = False) -> float:
    """The rolling z of the jump day, with β from the training rows (before the
    OOS start; the per-window engine's slice includes it)."""
    rows = closes.index <= OOS_START if include_oos_start else closes.index < OOS_START
    beta = estimate_beta(closes.loc[rows, "PA"], closes.loc[rows, "PB"])
    return float(rolling_z(spread_series(closes, "PA", "PB", beta)).loc[day])


ENGINE_CFGS = {"continuous": GROSS, "window_reset": replace(GROSS, legacy_window_reset=True)}


@pytest.mark.parametrize("engine", ENGINE_CFGS)
def test_a_single_day_6_sigma_jump_is_no_entry_with_stop_z_3_5(engine):
    # Acceptance: the jump day's z is beyond STOP_Z = 3.5 and nothing else on
    # the panel is an entry.  Default: no trade; legacy_no_entry_guards: it
    # enters on the jump day (short spread, filled the next day).
    cfg = ENGINE_CFGS[engine]
    closes = panel(jump=6.0)
    assert STOP_Z < jump_z(closes) < 7.7 and STOP_Z < jump_z(closes, include_oos_start=True)
    assert quiet(panel()).trades.empty                                     # the quiet panel
    assert quiet(closes, cfg).trades.empty
    legacy = quiet(closes, replace(cfg, legacy_no_entry_guards=True)).trades
    assert len(legacy) == 1
    t = legacy.iloc[0]
    assert (t["entry_signal"], t["side"]) == (JUMP_DAY, -1)
    assert t["entry"] == PANEL_DAYS[PANEL_DAYS.get_loc(JUMP_DAY) + 1]


@pytest.mark.parametrize("engine", ENGINE_CFGS)
def test_a_jump_to_z_between_entry_z_and_stop_z_still_enters(engine):
    cfg = ENGINE_CFGS[engine]
    closes = panel(jump=2.0)
    assert ENTRY_Z < jump_z(closes) < STOP_Z and ENTRY_Z < jump_z(closes, include_oos_start=True) < STOP_Z
    for flag in (False, True):
        trades = quiet(closes, replace(cfg, legacy_no_entry_guards=flag)).trades
        assert trades["entry_signal"].tolist() == [JUMP_DAY]


def test_a_negative_6_sigma_jump_is_no_entry_either():
    closes = panel(jump=-6.0)
    assert jump_z(closes) < -STOP_Z
    assert quiet(closes).trades.empty
    t = quiet(closes, replace(GROSS, legacy_no_entry_guards=True)).trades.iloc[0]
    assert (t["entry_signal"], t["side"]) == (JUMP_DAY, +1)


def test_the_guard_is_on_in_each_engine_when_only_the_other_legacy_flags_are_on():
    # A run that re-enables one older behaviour (the per-window engine here, or
    # the stale-print rules) still has the guard; only its own flag removes it.
    closes = panel(jump=6.0)
    for cfg in (replace(GROSS, legacy_window_reset=True), replace(GROSS, legacy_stale_prints=True),
                replace(GROSS, legacy_window_reset=True, legacy_stale_prints=True,
                        legacy_exit_date_pnl=True)):
        assert quiet(closes, cfg).trades.empty
        assert len(quiet(closes, replace(cfg, legacy_no_entry_guards=True)).trades) == 1


def test_cli_option():
    assert build_parser().parse_args(["--legacy-no-entry-guards"]).legacy_no_entry_guards is True
    assert build_parser().parse_args([]).legacy_no_entry_guards is False


# ── Structural guards of T2: genuine prints, z window without no-print days ─

def test_z_window_is_the_last_z_win_rows_with_both_prints():
    # No-print days of either leg are not rows of the pair: the 60-row window
    # runs over the rows before them, skipping them (it spans more than 60 days).
    closes = panel(jump=2.0)
    nan_a, nan_b = closes.index[[100, 150, 151]], closes.index[[120, 152]]
    closes.loc[nan_a, "PA"] = np.nan
    closes.loc[nan_b, "PB"] = np.nan
    both = closes.dropna()
    assert len(both) == len(closes) - 5
    sp = spread_series(closes, "PA", "PB", 1.5)
    assert sp.index.equals(both.index)                    # the no-print days are not rows
    assert not sp.index.isin(nan_a.union(nan_b)).any()
    z = rolling_z(sp)
    log_sp = np.log(both["PA"]) - 1.5 * np.log(both["PB"])
    for k in (Z_WIN, 119, 121, 153, 200, len(both) - 1):  # around and after the gaps
        w = log_sp.to_numpy()[k - Z_WIN + 1:k + 1]
        assert len(w) == Z_WIN
        assert z.iloc[k] == pytest.approx((w[-1] - w.mean()) / w.std(ddof=1), rel=1e-12)
    assert z.iloc[:Z_WIN - 1].isna().all() and z.iloc[Z_WIN - 1:].notna().all()
    # the window ending right after a gap reaches back past it: 60 rows of data
    # that span more than 60 rows of the union calendar, the skipped ones being the gaps
    k = both.index.get_loc(closes.index[121])
    span = both.index[k - Z_WIN + 1:k + 1]
    skipped = closes.index[closes.index.get_loc(span[0]):closes.index.get_loc(span[-1]) + 1].difference(span)
    assert len(skipped) >= 2 and skipped.isin(nan_a.union(nan_b)).all()


GAPPY_DAYS = pd.bdate_range("2018-01-01", "2021-01-15", name="Date")
GAPPY_OOS_START = pd.Timestamp("2018-07-01")


def gappy_panel(seed: int = 2) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """One pair, noisy spread (many entries), ~8% no-print days per leg (NaN) —
    including days inside the OOS windows.  Returns (closes, no-print dates)."""
    rng = np.random.default_rng(seed)
    n = len(GAPPY_DAYS)
    log_pb = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    log_pa = 1.5 * log_pb + rng.normal(0.0, SIG, n) + np.log(30.0) - 1.5 * np.log(50.0)
    closes = pd.DataFrame({"PA": np.exp(log_pa), "PB": np.exp(log_pb)}, index=GAPPY_DAYS)
    gaps = {}
    for leg in closes.columns:
        gaps[leg] = closes.index[rng.random(n) < 0.08]
        closes.loc[gaps[leg], leg] = np.nan
    return closes, gaps["PA"].union(gaps["PB"])


@pytest.mark.parametrize("engine", ENGINE_CFGS)
def test_no_print_day_is_not_a_signal_date_a_fill_date_or_a_z_window_row(engine, monkeypatch):
    closes, no_print = gappy_panel()
    printed = closes.dropna().index
    assert len(no_print) > 100 and (no_print >= GAPPY_OOS_START).sum() > 80
    seen = []
    real = core.rolling_z

    def spy(spread, *args, **kw):
        seen.append(spread.index)
        return real(spread, *args, **kw)

    monkeypatch.setattr(core, "rolling_z", spy)
    trades = quiet(closes, ENGINE_CFGS[engine]).trades
    assert len(trades) > 10                                    # the checks below are not vacuous
    for col in ("entry_signal", "entry", "exit_signal", "exit"):
        assert not trades[col].isin(no_print).any(), col
    assert seen
    for idx in seen:                                           # every z window: both-print rows only
        assert idx.isin(printed).all() and not idx.isin(no_print).any()
        assert idx.equals(printed[(printed >= idx[0]) & (printed <= idx[-1])])    # none skipped


def test_run_level_assertion_refuses_an_entry_signalled_on_a_no_print_day(monkeypatch):
    # run_wfo_daily re-checks its final table: both legs print on every
    # entry's signal date and fill date (T2); reused by T7, which adds no
    # second check.  A table that violates it is refused.
    closes = panel(jump=2.0)
    day = closes.index[closes.index.get_loc(JUMP_DAY) - 3]
    closes.loc[day, "PA"] = np.nan
    assert quiet(closes).trades["entry_signal"].tolist() == [JUMP_DAY]        # holds as run

    real = core.continuous_trades

    def corrupt(*args, **kw):
        out = real(*args, **kw)
        out[0]["entry_signal"] = day
        return out

    monkeypatch.setattr(core, "continuous_trades", corrupt)
    with pytest.raises(AssertionError) as exc:
        quiet(closes)
    assert "notna" in str(exc.traceback[-1].statement)
