"""Daily mark-to-market PnL and date-ordered risk statistics (audit task T4).

Trade level (`mtm.trade_mtm` / `mtm.daily_mtm` on hand-made two-leg trades):
each leg, held in the quantities fixed at its entry fill, is marked from its
entry fill price to its close on every date it prints and ends at its exit
fill price; the round-trip cost is charged half on each fill date, borrow on
each of the pair's held rows; a leg without a print on a date books nothing
that date (its next print carries the move).  The acceptance test: a trade
−5% of capital at mid-life that exits at +1% shows MaxDD −5%.

Run level (`run_wfo_daily`): `daily_pnl` is the MTM of the final trades over
every business day of the OOS span, `daily_returns` = daily_pnl / capital,
`stats` hold the one Sharpe and the date-ordered MaxDD; `legacy_exit_date_pnl`
restores the exit-date booking.

Downstream scripts (smoke test on a small synthetic CLI run, no data/): they
read the MTM file the CLI writes, checked against the trades file (and,
since T10, the run's meta sidecar — tests/test_validation_outputs.py); without
it they stop unless --legacy-exit-date-pnl is given.  The dashboard's
per-pair and sub-block Sharpe ratios come from the per-pair MTM file (none
without it), never from exit-date booking.
"""
import contextlib
import functools
import io
import json
import math
import operator
import os
import shutil
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

import mtm  # noqa: E402
import synthetic  # noqa: E402
import wfo_daily_core as core  # noqa: E402
from cross_listing import mask_stale_prints  # noqa: E402
from wfo_daily_core import WfoConfig, make_windows, run_wfo_daily  # noqa: E402

nan = float("nan")

# ── Trade level ─────────────────────────────────────────────────────────────

DAYS = pd.bdate_range("2021-01-04", periods=8, name="Date")
PA = [100.0, 100.0, 98.0, 95.0, 99.0, 101.0, 101.0, 101.0]
PB = [100.0, 100.0, 102.0, 105.0, 101.0, 99.0, 99.0, 99.0]


def panel(pa=PA, pb=PB) -> pd.DataFrame:
    return pd.DataFrame({"PA": pa, "PB": pb}, index=DAYS)


def trade(px: pd.DataFrame, entry: int, exit_: int, side: int = 1, size: float = 1.0,
          beta: float = 1.0, fills: tuple | None = None, cost: float = 0.0,
          borrow_per_day: float = 0.0) -> dict:
    """One row of the trades table as the engine writes it, filled on rows
    `entry` / `exit_` of `px` (at the closes, or at `fills` = (P1 in, P2 in,
    P1 out, P2 out)); held = the pair's rows with both legs in (entry, exit]."""
    both = px[["PA", "PB"]].notna().all(axis=1).to_numpy()
    held = int(both[entry + 1:exit_ + 1].sum())
    p1e, p2e, p1x, p2x = fills or (px["PA"].iloc[entry], px["PB"].iloc[entry],
                                   px["PA"].iloc[exit_], px["PB"].iloc[exit_])
    q1, q2 = core.entry_quantities(p1e, p2e, beta)
    gross = core.leg_gross(side, q1, q2, p1e, p2e, p1x, p2x)
    return {"pair": "PA-PB", "entry": DAYS[entry], "exit": DAYS[exit_], "held": held,
            "side": side, "size": size, "beta": beta, "q1": q1, "q2": q2,
            "p1_entry": p1e, "p2_entry": p2e, "p1_exit": p1x, "p2_exit": p2x,
            "gross": gross, "cost": cost, "borrow": held * borrow_per_day,
            # The pre-T8 cost schedule (legacy_flat_cost; T8: test_position_costs.py).
            "cost_entry": cost / 2, "cost_exit": cost / 2, "borrow_rate1": nan, "borrow_rate2": nan,
            "pnl": core.net_pnl(size, gross, cost, held * borrow_per_day), "reason": "target"}


def table(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_acceptance_unrealised_loss_at_mid_life_is_the_max_drawdown():
    # Long the spread PA − PB (β = 1: $1 long PA, $1 short PB = 2.0 USD of
    # capital, the capital of one pair).  At mid-life PA is down 5% and PB up
    # 5%: −0.10 USD = −5% of capital; the trade exits at +0.02 = +1%.
    px = panel()
    tr = table(trade(px, 1, 5))
    capital = mtm.CAPITAL_PER_PAIR * 1
    assert tr["pnl"].iloc[0] / capital == pytest.approx(0.01)
    daily = mtm.daily_mtm(tr, px, DAYS)
    np.testing.assert_allclose(daily.to_numpy(), [0, 0, -0.04, -0.06, 0.08, 0.04, 0, 0],
                               rtol=0, atol=1e-12)
    st = mtm.mtm_stats(daily, capital)
    assert st["capital"] == capital
    assert st["max_dd"] == pytest.approx(-0.05, abs=1e-12)          # the −5% at mid-life
    assert (st["max_dd_peak"], st["max_dd_trough"]) == (DAYS[1], DAYS[3])
    # Exit-date booking (legacy) shows no drawdown at all: one winning trade.
    assert core.trade_stats(tr, 1, 1)["max_dd"] == 0.0
    assert (core.exit_date_daily_pnl(tr, DAYS).cumsum() >= 0).all()


def test_drawdown_counts_a_loss_from_the_first_day():
    # The equity curve starts at 0 (capital): a loss on the first day of the
    # series is a drawdown from that start.
    daily = pd.Series([-0.02, -0.01, 0.05], index=DAYS[:3])
    st = mtm.mtm_stats(daily, 1.0)
    assert st["max_dd"] == pytest.approx(-0.03)
    assert st["max_dd_trough"] == DAYS[1] and st["max_dd_peak"] == DAYS[0]
    ret = daily / 1.0
    assert st["sharpe"] == ret.mean() / ret.std() * math.sqrt(252)
    assert st["ann_return"] == ret.mean() * 252 and st["ann_vol"] == ret.std() * math.sqrt(252)


def test_costs_on_the_fill_dates_and_borrow_on_each_held_row():
    px = panel()
    t = trade(px, 1, 5, size=0.5, cost=0.002, borrow_per_day=1e-4)
    m = mtm.trade_mtm(pd.Series(t), px)
    assert list(m.index) == list(DAYS[1:6])
    assert m["cost"].tolist() == [0.001, 0.0, 0.0, 0.0, 0.001]       # half per fill
    np.testing.assert_allclose(m["borrow"], [0.0, 1e-4, 1e-4, 1e-4, 1e-4], rtol=1e-12)
    assert m["gross"].sum() == pytest.approx(t["gross"], abs=1e-15)
    net = t["size"] * (m["gross"] - m["cost"] - m["borrow"])
    assert net.sum() == pytest.approx(t["pnl"], abs=1e-15)
    daily = mtm.daily_mtm(table(t), px, DAYS)
    np.testing.assert_allclose(daily.loc[DAYS[1]:DAYS[5]].to_numpy(), net.to_numpy(), atol=1e-15)
    assert (daily.drop(DAYS[1:6]) == 0).all()


def test_vwap_fills_mark_the_entry_day_to_its_close_and_end_at_the_exit_vwap():
    px = panel()
    t = trade(px, 1, 5, fills=(99.5, 100.4, 100.6, 99.2))
    q1, q2 = t["q1"], t["q2"]
    m = mtm.trade_mtm(pd.Series(t), px)
    expected = [q1 * (100 - 99.5) - q2 * (100 - 100.4),      # entry day: VWAP → close
                q1 * (98 - 100) - q2 * (102 - 100),
                q1 * (95 - 98) - q2 * (105 - 102),
                q1 * (99 - 95) - q2 * (101 - 105),
                q1 * (100.6 - 99) - q2 * (99.2 - 101)]       # exit day: close → VWAP
    np.testing.assert_allclose(m["gross"], expected, rtol=0, atol=1e-15)
    assert m["gross"].sum() == pytest.approx(t["gross"], abs=1e-15)


def test_a_leg_without_a_print_carries_its_move_to_its_next_print():
    # PB has no print on day 3: that day books PA's move only and no borrow
    # (not a row of the pair); PB's next print (day 4) books its move from
    # day 2.  Nothing is filled.
    pb = PB.copy()
    pb[3] = nan
    px = panel(pb=pb)
    t = trade(px, 1, 5, side=-1, cost=0.002, borrow_per_day=1e-4)
    assert t["held"] == 3                                   # the pair's rows: days 2, 4, 5
    q1, q2, s = t["q1"], t["q2"], t["side"]
    m = mtm.trade_mtm(pd.Series(t), px)
    np.testing.assert_allclose(m["gross"], [0.0,
                                            s * (q1 * (98 - 100) - q2 * (102 - 100)),
                                            s * q1 * (95 - 98),
                                            s * (q1 * (99 - 95) - q2 * (101 - 102)),
                                            s * (q1 * (101 - 99) - q2 * (99 - 101))],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(m["borrow"], [0.0, 1e-4, 0.0, 1e-4, 1e-4], rtol=1e-12)
    assert m["gross"].sum() == pytest.approx(t["gross"], abs=1e-15)
    assert (m["gross"] - m["cost"] - m["borrow"]).sum() == pytest.approx(t["pnl"], abs=1e-15)


def test_a_round_trip_on_one_row_books_only_its_costs():
    # An entry decided on the last decision row is closed on its own fill row
    # (held 0, T3): no price move, the whole round-trip cost on that date.
    px = panel()
    t = trade(px, 7, 7, size=0.5, cost=0.002, borrow_per_day=1e-4)
    m = mtm.trade_mtm(pd.Series(t), px)
    assert list(m.index) == [DAYS[7]]
    assert (m.at[DAYS[7], "gross"], m.at[DAYS[7], "cost"], m.at[DAYS[7], "borrow"]) == (0.0, 0.002, 0.0)
    assert mtm.daily_mtm(table(t), px, DAYS).loc[DAYS[7]] == pytest.approx(t["pnl"]) == -0.001


def test_portfolio_sums_the_trades_and_drops_nothing():
    px = panel()
    px["PC"], px["PD"] = np.linspace(50, 57, 8), np.linspace(20, 19, 8)
    t1 = trade(px, 1, 5, cost=0.002, borrow_per_day=1e-4)
    t2 = {**trade(pd.DataFrame({"PA": px["PC"], "PB": px["PD"]}), 2, 6, side=-1, size=0.7,
                  beta=2.0, cost=0.001), "pair": "PC-PD"}
    tr = table(t1, t2)
    daily = mtm.daily_mtm(tr, px, DAYS)
    one = [mtm.daily_mtm(table(t), px, DAYS) for t in (t1, t2)]
    np.testing.assert_allclose(daily.to_numpy(), (one[0] + one[1]).to_numpy(), rtol=0, atol=1e-15)
    assert daily.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-15)
    # A day carrying PnL outside the index would be dropped: refused.
    with pytest.raises(AssertionError):
        mtm.daily_mtm(tr, px, DAYS[:5])


def test_a_panel_whose_rows_do_not_match_the_trade_is_refused():
    # held counts the pair's rows on the engine's panel; marked on another
    # panel (here PB prints on a row the trade did not hold) the borrow
    # schedule would not add up: refused.
    pb = PB.copy()
    pb[3] = nan
    t = trade(panel(pb=pb), 1, 5, borrow_per_day=1e-4)
    with pytest.raises(AssertionError):
        mtm.trade_mtm(pd.Series(t), panel())


# ── Run level ───────────────────────────────────────────────────────────────

def quiet(closes, pairs, cfg, **frames):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, pairs, cfg, **frames)


@pytest.fixture(scope="module")
def ou_runs():
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=3, seed=0)
    cfg = WfoConfig(execution_model="flat")                 # costs and borrow on
    return closes, pairs, cfg, quiet(closes, pairs, cfg)


def test_run_daily_pnl_is_the_mark_to_market_of_the_final_trades(ou_runs):
    closes, pairs, cfg, res = ou_runs
    tr = res.trades
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step)
    span = pd.bdate_range(windows[0][2], max(windows[-1][3], tr["exit"].max()), name="Date")
    assert res.daily_pnl.index.equals(span)
    pd.testing.assert_series_equal(res.daily_pnl, mtm.daily_mtm(tr, mask_stale_prints(closes), span))
    assert res.daily_pnl.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-12)
    np.testing.assert_allclose(tr["pnl"], tr["size"] * (tr["gross"] - tr["cost"] - tr["borrow"]),
                               rtol=1e-12, atol=1e-15)
    assert (tr["cost"] > 0).all() and (tr["borrow"] > 0).any()
    capital = mtm.CAPITAL_PER_PAIR * len(pairs)
    pd.testing.assert_series_equal(res.daily_returns, res.daily_pnl / capital)
    st, ret = res.stats, res.daily_returns
    assert st["capital"] == capital
    assert st["sharpe"] == ret.mean() / ret.std() * math.sqrt(252)
    assert {k: st[k] for k in ("max_dd", "max_dd_peak", "max_dd_trough", "ann_return", "ann_vol")} == \
        {k: v for k, v in mtm.mtm_stats(res.daily_pnl, capital).items() if k not in ("capital", "sharpe")}
    assert st["max_dd"] < 0
    assert not {"sharpe_full", "sharpe_exit"} & set(st)     # rule 4: one Sharpe
    assert (st["n_trades"], st["total_pnl"]) == (len(tr), tr["pnl"].sum())
    # Per pair: every pair of the run, the same dates; the columns, added in
    # order, are daily_pnl exactly, and each sums to its trades' PnL.
    by_pair = res.daily_pnl_pairs
    assert set(by_pair.columns) == set(pairs) and by_pair.index.equals(span)
    np.testing.assert_array_equal(functools.reduce(operator.add, [by_pair[c] for c in by_pair]),
                                  res.daily_pnl)
    for pair, pnl in tr.groupby("pair")["pnl"].sum().items():
        assert by_pair[pair].sum() == pytest.approx(pnl, abs=1e-12)


def test_legacy_exit_date_pnl_restores_the_exit_date_booking(ou_runs):
    closes, pairs, cfg, res = ou_runs
    leg = quiet(closes, pairs, replace(cfg, legacy_exit_date_pnl=True))
    pd.testing.assert_frame_equal(leg.trades, res.trades)
    assert leg.daily_pnl_pairs is None
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step)
    oos_idx = closes.loc[windows[0][2]:max(windows[-1][3], res.trades["exit"].max())].index
    pd.testing.assert_series_equal(leg.daily_pnl, core.exit_date_daily_pnl(res.trades, oos_idx))
    old = core.trade_stats(res.trades, len(windows), len(pairs), oos_idx)
    assert leg.stats.keys() == old.keys() | {"headline"}      # + T5's headline = False
    assert all(leg.stats[k] == old[k] for k in old) and leg.stats["headline"] is False


def test_legacy_spread_pnl_needs_the_exit_date_booking(ou_runs):
    # Spread PnL has no leg quantities to mark.
    closes, pairs, cfg, _ = ou_runs
    legacy = replace(cfg, legacy_window_reset=True, legacy_spread_pnl=True)
    with pytest.raises(ValueError, match="legacy_exit_date_pnl"):
        quiet(closes, pairs, legacy)
    quiet(closes, pairs, replace(legacy, legacy_exit_date_pnl=True, legacy_flat_cost=True))


def test_summary_prints_the_mtm_figures_and_no_exit_day_sharpe(ou_runs):
    closes, pairs, cfg, res = ou_runs
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        core.summarise(res.trades, "x", res.stats["n_windows"], len(pairs), stats=res.stats)
    text = out.getvalue()
    assert f"Sharpe:         {res.stats['sharpe']:+.3f}" in text
    assert f"{res.stats['max_dd'] * 100:+.2f}% of capital" in text
    assert "Sharpe (exit)" not in text and "Sharpe (full)" not in text
    assert "log-spread" not in text and "USD per $1 of leg-1 notional" in text


# ── Downstream scripts (smoke test) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """The CLI on a small synthetic panel in a temp run dir: trades and MTM files."""
    import step3j_wfo_daily
    root = tmp_path_factory.mktemp("t4run")
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=1)
    (root / "data").mkdir()
    (root / "output").mkdir()
    closes.to_csv(root / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(root / "data" / "pairs.csv", index=False)
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(root)
        sys.argv = ["step3j_wfo_daily.py", "--pairs", "pairs.csv", "--closes", "closes.csv",
                    "--execution-model", "flat", "--tag", "t"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            step3j_wfo_daily.main()
    finally:
        os.chdir(cwd)
        sys.argv = argv
    return root, out.getvalue()


def copy_run(data, dst, tag: str, kinds=("wfo_daily_results", "wfo_daily_mtm")) -> str:
    """The fixture run's files under another tag in `dst`, with its meta
    sidecar and closes file (audit T10: the scripts check a run against its
    meta); returns the trades path."""
    for kind in kinds:
        shutil.copy(data / f"{kind}_t.csv", dst / f"{kind}_{tag}.csv")
    meta = json.loads((data / "wfo_daily_meta_t.json").read_text())
    meta["files"] = {k: v and v.replace("_t.csv", f"_{tag}.csv") for k, v in meta["files"].items()}
    (dst / f"wfo_daily_meta_{tag}.json").write_text(json.dumps(meta))
    shutil.copy(data / "closes.csv", dst / "closes.csv")
    return str(dst / f"wfo_daily_results_{tag}.csv")


def run_script(module, args: list[str]) -> str:
    argv = sys.argv
    out = io.StringIO()
    try:
        sys.argv = [module.__name__ + ".py", *args]
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = argv
    return out.getvalue()


def test_cli_writes_the_daily_mtm_file(cli_run):
    root, out = cli_run
    trades = pd.read_csv(root / "data" / "wfo_daily_results_t.csv")
    m = mtm.read_mtm(root / "data" / "wfo_daily_mtm_t.csv").reset_index()       # exact floats
    assert list(m.columns) == ["Date", "pnl", "ret"]
    assert m["pnl"].sum() == pytest.approx(trades["pnl"].sum(), abs=1e-12)
    np.testing.assert_allclose(m["ret"], m["pnl"] / (mtm.CAPITAL_PER_PAIR * 2), rtol=1e-12)
    assert pd.DatetimeIndex(m["Date"]).equals(pd.bdate_range(m["Date"].min(), m["Date"].max()))
    assert "Sharpe (exit)" not in out and "daily mark-to-market" in out
    # The per-pair file: the same dates, one column per pair of the run, and
    # its columns (in file order) sum to the portfolio pnl exactly.
    by_pair = mtm.read_mtm(root / "data" / "wfo_daily_mtm_pairs_t.csv")
    assert by_pair.index.equals(pd.DatetimeIndex(m["Date"]))
    assert set(by_pair.columns) == set(trades["pair"]) == {"OUA0-OUB0", "OUA1-OUB1"}
    np.testing.assert_array_equal(functools.reduce(operator.add, [by_pair[c] for c in by_pair]).to_numpy(),
                                  m["pnl"].to_numpy())


def test_downstream_scripts_read_the_mtm_file(cli_run, tmp_path):
    import bootstrap_sharpe
    import dashboard_daily
    import factor_attribution
    import stress_crisis_analysis
    root, _ = cli_run
    data = root / "data"
    trades, closes = str(data / "wfo_daily_results_t.csv"), str(data / "closes.csv")
    split = ["--split", str(synthetic.START.date())]       # the scripts' default is 2014-12-31
    # The scripts read the pnl column (their PnL units; ret = pnl / capital).
    pnl = mtm.read_mtm(data / "wfo_daily_mtm_t.csv")["pnl"]
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252)

    out = run_script(bootstrap_sharpe, ["--trades", trades, "--closes", closes, *split, "--reps", "20"])
    # (T10: the one Sharpe, sample std — was numpy's ddof = 0)
    assert f"point Sharpe    : {sharpe:+.3f}" in out
    assert "daily mark-to-market" in out and "LEGACY" not in out

    out = run_script(stress_crisis_analysis, ["--trades", trades, "--closes", closes, *split,
                                              "--out", str(tmp_path / "stress.json")])
    rep = json.loads((tmp_path / "stress.json").read_text())
    assert rep["full_sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert rep["var_99"] == pytest.approx(pnl.quantile(0.01), rel=1e-12)
    eq = pnl.cumsum()
    # (T10: only the named windows inside the run's span; none for 2005-2007)
    assert min([c["dd"] for c in rep["crisis_periods"] + rep["calm_periods"]] + [0.0]) >= (eq - eq.cummax()).min()
    assert rep["oos_span"] == [str(pnl.index.min().date()), str(pnl.index.max().date())]
    assert "daily mark-to-market" in out

    out = run_script(factor_attribution, ["--trades", trades, "--closes", closes,
                                          "--factors", "OUA0", "OUB1",
                                          "--out", str(tmp_path / "factor.json"),
                                          "--csv-out", str(tmp_path / "factor.csv")])
    rep = json.loads((tmp_path / "factor.json").read_text())
    daily = pd.read_csv(tmp_path / "factor.csv", index_col=0, parse_dates=True)
    # (T10: the regression's y is the daily MTM return, ret = pnl / capital)
    ret = mtm.read_mtm(data / "wfo_daily_mtm_t.csv")["ret"]
    np.testing.assert_allclose(daily["strategy_ret"], ret.reindex(daily.index), rtol=1e-12)
    assert rep["n_days"] == len(daily) > 100
    assert "daily mark-to-market" in out

    png = tmp_path / "dash.png"
    out = run_script(dashboard_daily, ["--trades", trades, "--closes", closes, *split, "--out", str(png)])
    ret = mtm.read_mtm(data / "wfo_daily_mtm_t.csv")["ret"]
    st = mtm.mtm_stats(ret, 1.0)
    assert png.stat().st_size > 0
    assert st["sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert f"Sharpe={st['sharpe']:+.3f}   maxDD={st['max_dd'] * 100:+.2f}%   (daily mark-to-market)" in out

    # No MTM file: each script stops, unless the legacy exit-date booking is asked for.
    orphan = copy_run(data, tmp_path, "x", kinds=("wfo_daily_results",))
    for module, extra in [(bootstrap_sharpe, [*split, "--reps", "20"]),
                          (stress_crisis_analysis, [*split, "--out", str(tmp_path / "s2.json")]),
                          (factor_attribution, ["--factors", "OUA0", "OUB1",
                                                "--out", str(tmp_path / "f2.json"),
                                                "--csv-out", str(tmp_path / "f2.csv")]),
                          (dashboard_daily, [*split, "--out", str(tmp_path / "d2.png")])]:
        args = ["--trades", orphan, "--closes", str(tmp_path / "closes.csv"), *extra]
        with pytest.raises(SystemExit, match="wfo_daily_mtm_x.csv"):
            run_script(module, args)
        assert "LEGACY exit-date booking" in run_script(module, [*args, "--legacy-exit-date-pnl"])


def test_dashboard_shows_no_sharpe_from_exit_date_booking(cli_run, tmp_path):
    # Every Sharpe on the dashboard is a daily MTM Sharpe: the portfolio from
    # the MTM file, each pair and each sub-block from the per-pair MTM file
    # (block returns on the block's capital).  Without the per-pair file the
    # per-pair and sub-block items show trades and PnL only.
    import dashboard_daily
    root, _ = cli_run
    data = root / "data"
    trades = pd.read_csv(data / "wfo_daily_results_t.csv", parse_dates=["entry", "exit"])
    by_pair = mtm.read_mtm(data / "wfo_daily_mtm_pairs_t.csv")
    ret = mtm.read_mtm(data / "wfo_daily_mtm_t.csv")["ret"]
    split = ["--split", str(synthetic.START.date())]
    args = ["--trades", str(data / "wfo_daily_results_t.csv"), "--closes", str(data / "closes.csv"),
            *split, "--out", str(tmp_path / "d.png")]
    out = run_script(dashboard_daily, args)
    assert "exit-date" not in out and "LEGACY" not in out
    block = mtm.mtm_stats(mtm.portfolio_pnl(by_pair) / (len(by_pair.columns) * mtm.CAPITAL_PER_PAIR), 1.0)
    port = mtm.mtm_stats(ret, 1.0)
    assert f"legend: Non-cross (S={block['sharpe']:+.2f})" in out
    assert (f"table: Non-cross (2) | {len(trades)} | {trades['pnl'].sum():+.2f} | "
            f"{block['sharpe']:+.2f} | {block['max_dd'] * 100:+.2f}%") in out      # T10: the count
    assert (f"table: Combined | {len(trades)} | {trades['pnl'].sum():+.2f} | "
            f"{port['sharpe']:+.2f} | {port['max_dd'] * 100:+.2f}%") in out
    oos = ret.index
    for pair in trades["pair"].unique():
        mine = mtm.mtm_stats(by_pair[pair], mtm.CAPITAL_PER_PAIR)["sharpe"]
        assert f"{pair} {mine:+.3f}" in out
        exit_date = dashboard_daily.sharpe(dashboard_daily.daily_series(trades[trades["pair"] == pair], oos))
        assert f"{pair} {exit_date:+.3f}" not in out
    # Without the per-pair file: trades and PnL only.
    copy_run(data, tmp_path, "y")
    out = run_script(dashboard_daily, ["--trades", str(tmp_path / "wfo_daily_results_y.csv"),
                                       "--closes", str(data / "closes.csv"), *split,
                                       "--out", str(tmp_path / "d2.png")])
    assert "legend: Non-cross" not in out and "no per-pair MTM file, no Sharpe" in out
    assert f"table: Non-cross (2) | {len(trades)} | {trades['pnl'].sum():+.2f} | — | —" in out
    assert f"legend: Combined (S={port['sharpe']:+.2f})" in out
    # Legacy: the old exit-date figures, labelled.
    out = run_script(dashboard_daily, [*args, "--legacy-exit-date-pnl"])
    assert "LEGACY exit-date booking" in out


def test_a_stale_mtm_file_is_refused(cli_run, tmp_path):
    # MTM files that do not belong to the trades file (e.g. left by an
    # earlier run with the same --tag): every script stops.
    import bootstrap_sharpe
    import dashboard_daily
    import factor_attribution
    import stress_crisis_analysis
    root, _ = cli_run
    data = root / "data"
    split = ["--split", str(synthetic.START.date())]
    copy_run(data, tmp_path, "z", kinds=("wfo_daily_results",))
    m = pd.read_csv(data / "wfo_daily_mtm_t.csv")
    m["pnl"] *= 1.1
    m.to_csv(tmp_path / "wfo_daily_mtm_z.csv", index=False)
    common = ["--trades", str(tmp_path / "wfo_daily_results_z.csv"), "--closes", str(data / "closes.csv")]
    for module, extra in [(bootstrap_sharpe, [*split, "--reps", "20"]),
                          (stress_crisis_analysis, [*split, "--out", str(tmp_path / "s.json")]),
                          (factor_attribution, ["--factors", "OUA0", "OUB1",
                                                "--out", str(tmp_path / "f.json"),
                                                "--csv-out", str(tmp_path / "f.csv")]),
                          (dashboard_daily, [*split, "--out", str(tmp_path / "d.png")])]:
        with pytest.raises(SystemExit, match="does not match"):
            run_script(module, [*common, *extra])
    # A per-pair file that does not match the portfolio or the trades.
    shutil.copy(data / "wfo_daily_mtm_t.csv", tmp_path / "wfo_daily_mtm_z.csv")
    by_pair = pd.read_csv(data / "wfo_daily_mtm_pairs_t.csv")
    for broken in (by_pair.assign(**{"OUA0-OUB0": by_pair["OUA0-OUB0"] * 1.1}),
                   by_pair.drop(columns="OUA1-OUB1")):
        broken.to_csv(tmp_path / "wfo_daily_mtm_pairs_z.csv", index=False)
        with pytest.raises(SystemExit, match="does not match"):
            run_script(dashboard_daily, [*common, *split, "--out", str(tmp_path / "d.png")])
