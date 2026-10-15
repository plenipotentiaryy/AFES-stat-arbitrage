"""The extracted core reproduces the pre-refactor CLI (audit task T0).

tests/fixtures/t0_golden/ holds the trades CSVs that step3j_wfo_daily.py as
of commit 46ffdfb wrote for small synthetic panels (see make_golden.py).
`run_wfo_daily`, given the same inputs and the WfoConfig the CLI builds from
the same arguments, must reproduce every fixture byte for byte.  Later tasks
run these cases with their legacy_* flags on (LEGACY_FLAGS), so the legacy
path keeps reproducing the pre-audit code.

The same inputs also run in the default (corrected) mode (audit T1–T4):
every fill on the pair's next row with genuine prints of both legs after its
decision, nothing after cfg.end, one position per pair across windows, the
booked PnL rebuilt from the fill prices — the legs' prices with quantities
fixed at the entry fill and β from the training rows before the OOS start
(T2, T3), or, with legacy_spread_pnl on the per-window engine, the spread
change between the fill dates (T1) — and daily_pnl the daily mark-to-market
of the final trades (T4).
"""
import contextlib
import dataclasses
import importlib.util
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mtm  # noqa: E402
from cross_listing import mask_stale_prints  # noqa: E402
from step3j_wfo_daily import build_parser  # noqa: E402
from wfo_daily_core import WfoConfig, estimate_beta, make_windows, run_wfo_daily  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "t0_make_golden", os.path.join(_HERE, "fixtures", "t0_golden", "make_golden.py"))
make_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_golden)

# WfoConfig fields that select the pre-audit behaviour; each audit task adds
# its legacy_* flag here.
LEGACY_FLAGS: dict = {"legacy_same_close": True, "legacy_spread_pnl": True,
                      "legacy_stale_prints": True, "legacy_window_reset": True,
                      "legacy_exit_date_pnl": True, "legacy_no_entry_guards": True}
# All of them off: the corrected defaults.
DEFAULTS: dict = dict.fromkeys(LEGACY_FLAGS, False)
# CLI options that are not WfoConfig fields: file IO, and the headline
# universe policy the CLI applies to the pair list before the core (T2).
CLI_ONLY = {"pairs", "closes", "tag", "legacy_cross_listed"}


def as_cli_reads(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """`df` as the CLI reads it back from its CSV (same text, same parser)."""
    if df is None:
        return None
    return pd.read_csv(io.StringIO(df.to_csv()), parse_dates=["Date"]).set_index("Date").sort_index()


def cli_config(args: list[str]) -> WfoConfig:
    """The WfoConfig step3j_wfo_daily.main() builds from these arguments."""
    ns = build_parser().parse_args(args)
    return WfoConfig(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(WfoConfig)})


def run_case(case: str, **overrides):
    inp = make_golden.case_inputs(case)
    cfg = dataclasses.replace(cli_config(inp["args"]), **{**LEGACY_FLAGS, **overrides})
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(as_cli_reads(inp["closes"]), inp["pairs"], cfg,
                             volumes=as_cli_reads(inp["volumes"]),
                             vwap=as_cli_reads(inp["vwap"]))


def fixture_columns(case: str) -> list[str]:
    """Header of a fixture: the trade columns of the pre-refactor CLI."""
    return make_golden.fixture_path(case).read_text().partition("\n")[0].split(",")


@pytest.mark.parametrize("case", list(make_golden.CASES))
def test_core_reproduces_pre_refactor_cli(case):
    res = run_case(case)
    fixture = make_golden.fixture_path(case)
    if fixture.exists():
        # Compared on the fixture's columns: later tasks add columns.
        cols = fixture_columns(case)
        assert res.trades[cols].to_csv(index=False).encode() == fixture.read_bytes()
        # Legacy fills are on the decision dates.
        assert res.trades["entry_signal"].equals(res.trades["entry"])
        assert res.trades["exit_signal"].equals(res.trades["exit"])
    else:                                   # the pre-refactor CLI found no trades
        assert res.trades.empty
        assert res.stats["n_trades"] == 0 and np.isnan(res.stats["sharpe_full"])
        assert len(res.daily_pnl) > 0 and (res.daily_pnl == 0).all()


@pytest.mark.parametrize("case", ["flat", "vwap_exec"])
def test_default_next_bar_fills_do_not_reproduce_the_pre_audit_trades(case):
    # The legacy flags matter: without them the case's trades change.
    res = run_case(case, **DEFAULTS)
    cols = fixture_columns(case)
    assert res.trades[cols].to_csv(index=False).encode() != make_golden.fixture_path(case).read_bytes()


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "default"])
def test_result_daily_pnl_and_stats_match_trades(legacy):
    # default: one exit decided on the last OOS bar fills on the next row.
    res = run_case("flat", **({} if legacy else DEFAULTS))
    tr, daily = res.trades, res.daily_pnl
    assert daily.index.is_monotonic_increasing and daily.index.is_unique
    assert daily.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-12)
    assert set(tr["exit"].dt.normalize()) <= set(daily.index)
    assert res.stats["n_trades"] == len(tr)
    assert res.stats["total_pnl"] == tr["pnl"].sum()
    if legacy:                      # exit-date booking (legacy_exit_date_pnl)
        assert res.stats["sharpe_full"] == daily.mean() / daily.std() * np.sqrt(252)
    else:                           # T4: daily MTM returns on constant capital
        ret = res.daily_returns
        assert res.stats["sharpe"] == ret.mean() / ret.std() * np.sqrt(252)
        assert res.stats["sharpe"] == pytest.approx(daily.mean() / daily.std() * np.sqrt(252))


def test_cli_options_are_the_wfo_config_fields_with_same_defaults():
    defaults = vars(build_parser().parse_args([]))
    fields = {f.name for f in dataclasses.fields(WfoConfig)}
    assert set(defaults) - CLI_ONLY == fields
    assert {k: defaults[k] for k in fields} == dataclasses.asdict(WfoConfig())


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "default"])
@pytest.mark.parametrize("flag", ["metagate", "hrp"])
def test_metagate_and_hrp_keep_pre_existing_unbound_local_error(flag, legacy):
    # Ruling R7: main() referenced cost_log before assignment with these flags;
    # T0 preserves the failure (see AUDIT_FIX_LOG.md, known limitations), in
    # both engines (T3).
    with pytest.raises(UnboundLocalError):
        run_case("flat", **{flag: True, **({} if legacy else DEFAULTS)})


@pytest.mark.parametrize("flags", [{"vwz": True}, {"exec_mode": "vwap"}], ids=["vwz", "vwap"])
def test_flags_needing_aux_frames_raise_without_them(flags):
    inp = make_golden.case_inputs("flat")
    with pytest.raises(ValueError):
        run_wfo_daily(inp["closes"], inp["pairs"], WfoConfig(**flags))


def test_unknown_exec_mode_raises():
    inp = make_golden.case_inputs("flat")
    with pytest.raises(ValueError):
        run_wfo_daily(inp["closes"], inp["pairs"], WfoConfig(exec_mode="same_close"))


# ── The same inputs in the default (corrected) mode (audit T1) ──────────────

def run_quiet(closes: pd.DataFrame, pairs: list[str], cfg: WfoConfig, **frames):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, pairs, cfg, **frames)


@pytest.mark.parametrize("case", list(make_golden.CASES))
def test_default_mode_fills_on_the_pairs_next_row(case):
    # Every option case: each fill is the pair's next row with both legs after
    # its decision (never after cfg.end); only an end_of_data exit without
    # such a row fills on its decision date.  One continuous position per
    # pair (T3): positions never overlap, a trade may exit in a later window
    # than it entered, and what is open at the end of the last window is
    # closed (end_of_data) after the pair's last decision row, on the next
    # row.  daily_pnl holds every trade's PnL.  Rows = genuine prints of both
    # legs (the stale-print mask the engine applies, T2).
    res = run_case(case, **DEFAULTS)
    tr = res.trades
    if tr.empty:                    # only where the pre-refactor CLI had none either
        assert not make_golden.fixture_path(case).exists()
        return
    inp = make_golden.case_inputs(case)
    closes = mask_stale_prints(as_cli_reads(inp["closes"]), as_cli_reads(inp["volumes"]))
    cfg = cli_config(inp["args"])
    data_end = pd.Timestamp(cfg.end) if cfg.end else closes.index.max()
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step, start=cfg.start, end=cfg.end)
    oos_s = pd.DatetimeIndex([w[2] for w in windows])
    oos_e = pd.DatetimeIndex([w[3] for w in windows])
    for pair, g in tr.sort_values("entry_signal").groupby("pair"):
        rows = closes[pair.split("-")].dropna().index
        rows = rows[rows <= data_end]
        nxt_in = rows.searchsorted(g["entry_signal"], side="right")
        nxt_out = rows.searchsorted(g["exit_signal"], side="right")
        eod = (g["reason"] == "end_of_data").to_numpy()
        has_next = nxt_out < len(rows)
        assert (nxt_in < len(rows)).all() and (has_next | eod).all()
        assert (rows[nxt_in] == g["entry"].to_numpy()).all()
        assert (rows[nxt_out[has_next]] == g["exit"].to_numpy()[has_next]).all()
        assert (g["exit"] == g["exit_signal"]).to_numpy()[~has_next].all()
        assert (g["exit_signal"][eod] == rows[rows <= oos_e[-1]][-1]).all()
        prev_exit = g["exit"].shift()
        assert (g["entry"] > prev_exit).iloc[1:].all()
        # Windows: entered in the OOS of window_entered; window_exited = the
        # latest-starting window whose OOS has begun on the exit decision date.
        we, wx = g["window_entered"].to_numpy() - 1, g["window_exited"].to_numpy() - 1
        assert (g["window"] == g["window_entered"]).all()
        assert ((oos_s[we] <= g["entry_signal"]) & (g["entry_signal"] < oos_e[we])).all()
        assert (wx == oos_s.searchsorted(g["exit_signal"], side="right") - 1).all()
        assert (wx >= we).all()
    assert res.daily_pnl.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-9)
    assert set(tr["exit"].dt.normalize()) <= set(res.daily_pnl.index)
    # Daily mark-to-market (T4) of the final trades (after every filter that
    # scales or drops trades): pnl = size·(gross − cost − borrow) with the
    # per-unit columns; per trade the daily gross sums to gross and the daily
    # net to pnl; daily_pnl = their sum over every business day of the span.
    np.testing.assert_allclose(tr["pnl"], tr["size"] * (tr["gross"] - tr["cost"] - tr["borrow"]),
                               rtol=1e-12, atol=1e-15)
    for _, t in tr.iterrows():
        m = mtm.trade_mtm(t, closes)
        assert (m.index[0], m.index[-1]) == (t["entry"], t["exit"])
        assert m["gross"].sum() == pytest.approx(t["gross"], abs=1e-12)
        assert (t["size"] * (m["gross"] - m["cost"] - m["borrow"])).sum() == pytest.approx(t["pnl"], abs=1e-12)
    span = pd.bdate_range(oos_s[0], max(oos_e[-1], tr["exit"].max()), name="Date")
    assert res.daily_pnl.index.equals(span)
    pd.testing.assert_series_equal(res.daily_pnl, mtm.daily_mtm(tr, closes, span))
    # Per pair: every pair of the run on the same dates; the columns, added
    # in order, are daily_pnl exactly.
    by_pair = res.daily_pnl_pairs
    assert set(by_pair.columns) == set(inp["pairs"]) and by_pair.index.equals(span)
    np.testing.assert_array_equal(mtm.portfolio_pnl(by_pair), res.daily_pnl)


@pytest.mark.parametrize("window,days_after", [(3, 0), (9, 0), (15, 0), (21, 0), (20, 1), (20, 3)],
                         ids=["w3", "w9", "w15", "w21", "w20+1d", "w20+3d"])
def test_no_data_after_cfg_end_is_used(window, days_after):
    # Ruling R15: the fill-only bar is never after cfg.end.  With cfg.end set,
    # the run on the full panel equals the run on the panel cut at cfg.end, and
    # no output date is after it.  Ends: OOS ends of windows (no row lies in
    # (oos_e, end], so an exit decided on the last bar is end_of_data) and
    # dates just after one (the fill-only bar lies in (oos_e, end]).
    inp = make_golden.case_inputs("flat")
    closes = as_cli_reads(inp["closes"])
    cfg = cli_config(inp["args"])
    end = (make_windows(closes, cfg.train, cfg.oos, cfg.step)[window - 1][3]
           + pd.Timedelta(days=days_after))
    cfg = dataclasses.replace(cfg, end=str(end.date()))
    full = run_quiet(closes, inp["pairs"], cfg)
    cut = run_quiet(closes.loc[:end], inp["pairs"], cfg)
    pd.testing.assert_frame_equal(full.trades, cut.trades, check_exact=True)
    pd.testing.assert_series_equal(full.daily_pnl, cut.daily_pnl, check_exact=True)
    assert (full.trades[["entry", "exit"]] <= end).all().all()
    assert full.daily_pnl.index.max() <= end


@pytest.mark.parametrize("pnl", ["legs", "spread"])
@pytest.mark.parametrize("exec_mode", ["next_close", "vwap"])
def test_booked_pnl_is_rebuilt_from_the_fill_prices(exec_mode, pnl):
    # Zero costs.  Each fill price belongs to its fill date (the fill-only bar
    # included): the close or (vwap mode) the VWAP — an exit at the end of the
    # data, or a leg without a VWAP, at the close; β = the entry window's train β.
    # legs (default, T2/T3): quantities fixed at the entry fill, q1 = 1/P1 and
    # q2 = β/P2, β from the training rows before oos_s; the trade books
    # size × side × (q1·ΔP1 − q2·ΔP2).
    # spread (legacy_spread_pnl, T1, on the per-window engine whose train
    # includes oos_s): size × side × Δ(log P1 − β·log P2).
    inp = make_golden.case_inputs("vwap_exec")
    closes, vwap = inp["closes"], inp["vwap"]
    cfg = WfoConfig(execution_model="flat", cost=0.0, slip=0.0, borrow=0.0, exec_mode=exec_mode,
                    legacy_spread_pnl=pnl == "spread", legacy_window_reset=pnl == "spread",
                    legacy_exit_date_pnl=pnl == "spread")
    tr = run_quiet(closes, inp["pairs"], cfg, vwap=vwap).trades
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step)
    use_vwap = exec_mode == "vwap"
    rebuilt, exits_after_window = [], 0
    for _, t in tr.iterrows():
        a, b = t["pair"].split("-")
        tr_s, tr_e, oos_s, oos_e = windows[t["window"] - 1]
        train = closes.loc[tr_s:tr_e, [a, b]].dropna()
        if pnl == "legs":       # T3 / T9.1: the training rows strictly before oos_s
            train = train[train.index < oos_s]
        beta = estimate_beta(train[a], train[b])
        assert t["beta"] == beta

        def fill_px(day, leg: str, at_exit: bool) -> float:
            vw = vwap.at[day, leg] if use_vwap and not (at_exit and t["reason"] == "end_of_data") else np.nan
            return closes.at[day, leg] if np.isnan(vw) else vw

        if pnl == "spread":
            def spread(px: pd.DataFrame, day) -> float:
                return np.log(px.at[day, a]) - beta * np.log(px.at[day, b])

            entry = spread(vwap if use_vwap else closes, t["entry"])
            exit_ = spread(vwap, t["exit"]) if use_vwap and t["reason"] != "end_of_data" else np.nan
            if np.isnan(exit_):
                exit_ = spread(closes, t["exit"])
            rebuilt.append(t["size"] * t["side"] * (exit_ - entry))
        else:
            p1e, p2e = fill_px(t["entry"], a, False), fill_px(t["entry"], b, False)
            p1x, p2x = fill_px(t["exit"], a, True), fill_px(t["exit"], b, True)
            q1, q2 = 1 / p1e, beta / p2e
            assert (t["p1_entry"], t["p2_entry"], t["p1_exit"], t["p2_exit"]) == (p1e, p2e, p1x, p2x)
            assert (t["q1"], t["q2"]) == (q1, q2)
            rebuilt.append(t["size"] * t["side"] * (q1 * (p1x - p1e) - q2 * (p2x - p2e)))
        exits_after_window += t["exit"] > oos_e
    assert len(tr) > 100 and exits_after_window > 0
    np.testing.assert_allclose(tr["pnl"].to_numpy(), rebuilt, rtol=0, atol=1e-12)
