"""Validation outputs that cannot mislead (audit task T10).

The downstream scripts read a daily WFO run through one checked loader
(`mtm.load_mtm`, `mtm.legacy_exit_date_pnl`): its files and date range must
match the run's meta sidecar wfo_daily_meta<tag>.json (audit T8) — the trades,
MTM and closes file names, a span inside the run's start / end and before the
final holdout, every business day of it — else they stop; every script states
the run's span and its HEADLINE / NON-HEADLINE status.  The bootstrap resamples
the one Sharpe (`perf_stats.portfolio_sharpe`) and is labelled "sampling
uncertainty only — does not detect bias"; factor attribution regresses the
daily MTM return and reports its days and dates; the FDR screen reads prices up
to the universe's as-of date (or the development cut) only, labels a list that
is not a frozen point-in-time universe, and writes a screen-only file that the
CLI never takes as headline; legacy exit-date series end at the run's end and
are labelled "not a Sharpe"; step4c's PSR is the per-period PSR.

Fix round 1 (R44–R46): the CLI records the SHA-256 of the trades, MTM,
per-pair MTM and closes files in the sidecar and the scripts verify them (an
edit that keeps Σ pnl and the span is refused; a sidecar without hashes is a
loud warning); a --split / --start inside the run's span is refused; every
legacy exit-date ratio in the CLI and the daily-path research scripts says
"LEGACY exit-date ratio (full days|exit days) — not a Sharpe".

All data are synthetic; nothing reads data/."""
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mtm  # noqa: E402
import perf_stats  # noqa: E402
import synthetic  # noqa: E402

END = "2007-06-29"                       # the synthetic run's --end; its closes go on to 2007-12
START = str(synthetic.START.date())
SAMPLING = "sampling uncertainty only — does not detect bias"
LEGACY = "LEGACY exit-date booking — not a Sharpe"


def run_script(module, args: list[str]) -> str:
    argv = sys.argv
    out = io.StringIO()
    try:
        sys.argv = [module.__name__ + ".py", *map(str, args)]
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = argv
    return out.getvalue()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The CLI on a small synthetic panel (2005-01-03 … 2007-12) with --end
    2007-06-29, in a temp run dir: trades, MTM, per-pair MTM and meta files."""
    import step3j_wfo_daily
    root = tmp_path_factory.mktemp("t10run")
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=1)
    assert closes.index.max() > pd.Timestamp(END) + pd.Timedelta(days=90)
    (root / "data").mkdir()
    (root / "output").mkdir()
    closes.to_csv(root / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(root / "data" / "pairs.csv", index=False)
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(root)
        sys.argv = ["step3j_wfo_daily.py", "--pairs", "pairs.csv", "--closes", "closes.csv",
                    "--end", END, "--tag", "t"]
        with contextlib.redirect_stdout(io.StringIO()):
            step3j_wfo_daily.main()
    finally:
        os.chdir(cwd)
        sys.argv = argv
    return root / "data"


def copy_run(data, dst, tag: str, edit_meta=None, mtm_frame=None, meta: bool = True,
             kinds=("wfo_daily_results", "wfo_daily_mtm", "wfo_daily_mtm_pairs")) -> str:
    """The fixture run's files under another tag in `dst` (and its meta, its
    file names renamed as a CLI run with that tag would write them); returns
    the trades path."""
    dst.mkdir(exist_ok=True)
    for kind in kinds:
        shutil.copy(data / f"{kind}_t.csv", dst / f"{kind}_{tag}.csv")
    if mtm_frame is not None:
        mtm_frame.to_csv(dst / f"wfo_daily_mtm_{tag}.csv")
    if meta:
        m = json.loads((data / "wfo_daily_meta_t.json").read_text())
        m["files"] = {k: v and v.replace("_t.csv", f"_{tag}.csv") for k, v in m["files"].items()}
        if edit_meta:
            edit_meta(m)
        (dst / f"wfo_daily_meta_{tag}.json").write_text(json.dumps(m))
    shutil.copy(data / "closes.csv", dst / "closes.csv")
    return str(dst / f"wfo_daily_results_{tag}.csv")


SCRIPTS = ["bootstrap_sharpe", "stress_crisis_analysis", "factor_attribution", "dashboard_daily",
           "monte_carlo_robustness", "edge_monitor"]


def script_args(name: str, trades: str, out) -> list[str]:
    closes = str(os.path.join(os.path.dirname(trades), "closes.csv"))
    common = ["--trades", trades, "--closes", closes]
    return {
        "bootstrap_sharpe": [*common, "--split", START, "--reps", "20"],
        "stress_crisis_analysis": [*common, "--split", START, "--out", out / "stress.json"],
        "factor_attribution": [*common, "--factors", "OUA0", "OUB1", "--out", out / "factor.json",
                               "--csv-out", out / "factor.csv"],
        "dashboard_daily": [*common, "--split", START, "--out", out / "dash.png"],
        "monte_carlo_robustness": [*common, "--reps", "20", "--out", out / "mc.json",
                                   "--csv-out", out / "mc.csv"],
        "edge_monitor": [*common, "--start", START, "--out", out / "edge.csv",
                         "--alerts", out / "edge.json"],
    }[name]


def module(name: str):
    return __import__(name)


def dates_in(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)]


# ── Every script: the run, its span and its status ──────────────────────────

@pytest.mark.parametrize("name", SCRIPTS)
def test_each_script_states_the_runs_span_and_non_headline_status(run, tmp_path, name):
    trades = str(run / "wfo_daily_results_t.csv")
    m = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")
    out = run_script(module(name), script_args(name, trades, tmp_path))
    assert "daily mark-to-market" in out and "LEGACY" not in out
    assert "NON-HEADLINE" in out
    assert (f"span: {m.index.min().date()} → {m.index.max().date()} "
            f"({len(m)} business days)") in out
    assert max(dates_in(out)) <= pd.Timestamp(END)


@pytest.mark.parametrize("name", SCRIPTS)
def test_each_script_refuses_a_series_whose_dates_or_files_do_not_match_the_meta(run, tmp_path, name):
    mod = module(name)
    m = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")
    # Zero days appended past the run's end: the PnL total still matches the
    # trades, only the date range shows that the file is not the run's.
    pad = pd.bdate_range(m.index.max() + pd.offsets.BDay(1), periods=10, name="Date")
    padded = pd.concat([m, pd.DataFrame(0.0, index=pad, columns=m.columns)])
    trades = copy_run(run, tmp_path / "a", "a", mtm_frame=padded, kinds=("wfo_daily_results",))
    with pytest.raises(SystemExit, match="date range"):
        run_script(mod, script_args(name, trades, tmp_path))
    # A business day missing inside the span (a zero day dropped).
    zero_days = m.index[(m["pnl"] == 0.0).to_numpy()]
    holed = m.drop(index=zero_days[len(zero_days) // 2])
    trades = copy_run(run, tmp_path / "b", "b", mtm_frame=holed, kinds=("wfo_daily_results",))
    with pytest.raises(SystemExit, match="date range"):
        run_script(mod, script_args(name, trades, tmp_path))
    # A span reaching the final holdout, even when the meta's end allows it.
    late = m.copy()                         # moved by whole weeks: still every business day
    late.index = late.index + pd.Timedelta(weeks=(pd.Timestamp("2024-03-29") - late.index.max()).days // 7)
    assert late.index.max() >= pd.Timestamp("2024-01-01")

    def no_end(meta):
        meta["config"]["start"], meta["config"]["end"] = None, None
    trades = copy_run(run, tmp_path / "c", "c", edit_meta=no_end, mtm_frame=late,
                      kinds=("wfo_daily_results",))
    with pytest.raises(SystemExit, match="final holdout"):
        run_script(mod, script_args(name, trades, tmp_path))
    # No meta sidecar; a meta that names another trades file.
    trades = copy_run(run, tmp_path / "d", "d", meta=False)
    with pytest.raises(SystemExit, match="wfo_daily_meta_d.json"):
        run_script(mod, script_args(name, trades, tmp_path))

    def other(meta):
        meta["files"]["trades"] = "wfo_daily_results_other.csv"
    trades = copy_run(run, tmp_path / "e", "e", edit_meta=other)
    with pytest.raises(SystemExit, match="does not match"):
        run_script(mod, script_args(name, trades, tmp_path))
    # R44: one day's PnL split over two days — Σ pnl and the span unchanged,
    # the Sharpe not: only the MTM file's SHA-256 shows it.
    split = m.copy()
    day = split.index[(split["pnl"].abs() > 0).to_numpy()][5]
    nxt = split.index[split.index.get_loc(day) + 1]
    for col in ("pnl", "ret"):
        half = split.loc[day, col] / 2.0
        split.loc[day, col] -= half
        split.loc[nxt, col] += half
    assert math.isclose(split["pnl"].sum(), m["pnl"].sum(), rel_tol=1e-12)
    assert perf_stats.portfolio_sharpe(split["ret"]) != perf_stats.portfolio_sharpe(m["ret"])
    trades = copy_run(run, tmp_path / "f", "f", mtm_frame=split, kinds=("wfo_daily_results",))
    with pytest.raises(SystemExit, match="SHA-256"):
        run_script(mod, script_args(name, trades, tmp_path))
    # R44: two trades' PnL swapped — Σ pnl unchanged, the trades file edited.
    trades = copy_run(run, tmp_path / "g", "g")
    tr = pd.read_csv(trades)
    tr.loc[[0, 1], "pnl"] = tr.loc[[1, 0], "pnl"].to_numpy()
    tr.to_csv(trades, index=False)
    with pytest.raises(SystemExit, match="SHA-256"):
        run_script(mod, script_args(name, trades, tmp_path))
    # R46: a --split / --start inside the run's span (it starts 2005-07-04).
    if name != "factor_attribution":                        # (no such option)
        option = "--start" if name == "edge_monitor" else "--split"
        trades = str(run / "wfo_daily_results_t.csv")
        with pytest.raises(SystemExit, match="inside the run's span"):
            run_script(mod, [*script_args(name, trades, tmp_path), option, "2006-01-03"])
        with pytest.raises(SystemExit, match="inside the run's span"):
            run_script(mod, [*script_args(name, trades, tmp_path), option, "2006-01-03",
                             "--legacy-exit-date-pnl"])


@pytest.mark.parametrize("name", SCRIPTS)
def test_legacy_series_are_labelled_and_end_at_the_runs_end(run, tmp_path, name):
    # The closes run on to 2007-12; the pre-T10 legacy series were zero-padded
    # to the closes' last date.  Now they stop at the run's end (meta config).
    trades = str(run / "wfo_daily_results_t.csv")
    # (Monte Carlo: the OU control's exit-date series has hardly a losing day; a
    # low tail threshold gives the EVT fit its 20 exceedances.)
    extra = ["--tail-quantile", "0"] if name == "monte_carlo_robustness" else []
    out = run_script(module(name), [*script_args(name, trades, tmp_path), "--legacy-exit-date-pnl", *extra])
    assert LEGACY in out and "NON-HEADLINE" in out
    assert max(dates_in(out)) <= pd.Timestamp(END)
    assert not re.search(r"(?<!not a )Sharpe", out.replace(LEGACY, "")), "a legacy ratio called Sharpe"


def test_the_cli_records_the_files_hashes_and_a_sidecar_without_them_warns(run, tmp_path):
    import bootstrap_sharpe
    import factor_attribution
    meta = json.loads((run / "wfo_daily_meta_t.json").read_text())
    assert meta["sha256"] == {k: mtm.file_sha256(run / f) for k, f in [
        ("trades", "wfo_daily_results_t.csv"), ("mtm", "wfo_daily_mtm_t.csv"),
        ("mtm_pairs", "wfo_daily_mtm_pairs_t.csv"), ("closes", "closes.csv")]}
    trades = str(run / "wfo_daily_results_t.csv")
    assert "no file hashes" not in run_script(bootstrap_sharpe, script_args("bootstrap_sharpe", trades, tmp_path))

    def pre_t10(m):
        del m["sha256"]
    trades = copy_run(run, tmp_path / "old", "old", edit_meta=pre_t10)
    for name, mod in [("bootstrap_sharpe", bootstrap_sharpe), ("factor_attribution", factor_attribution)]:
        out = run_script(mod, script_args(name, trades, tmp_path))
        assert "WARNING: no file hashes recorded" in out and "contents not verified" in out


def test_the_closes_file_must_be_the_runs(run, tmp_path):
    import factor_attribution
    trades = copy_run(run, tmp_path, "c")
    args = script_args("factor_attribution", trades, tmp_path)
    other = tmp_path / "closes_other.csv"
    shutil.copy(run / "closes.csv", other)
    with pytest.raises(SystemExit, match="its closes file is closes.csv"):
        run_script(factor_attribution, [*args, "--closes", other])
    closes = pd.read_csv(tmp_path / "closes.csv")                    # same name, edited
    closes.loc[10, "OUA0"] *= 1.01
    closes.to_csv(tmp_path / "closes.csv", index=False)
    with pytest.raises(SystemExit, match="SHA-256"):
        run_script(factor_attribution, args)


@pytest.fixture(scope="module")
def run_2016(tmp_path_factory):
    """A synthetic CLI run dated 2016-01-04 … 2017-12 (--end 2017-06-30), so
    that some named stress windows overlap its span and others do not."""
    import step3j_wfo_daily
    root = tmp_path_factory.mktemp("t10run2016")
    closes, pairs = synthetic.ou_positive_control(n_years=2, n_pairs=2, seed=2)
    closes.index = pd.bdate_range("2016-01-04", periods=len(closes), name="Date")
    (root / "data").mkdir()
    (root / "output").mkdir()
    closes.to_csv(root / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(root / "data" / "pairs.csv", index=False)
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(root)
        sys.argv = ["step3j_wfo_daily.py", "--pairs", "pairs.csv", "--closes", "closes.csv",
                    "--end", "2017-06-30", "--tag", "s"]
        with contextlib.redirect_stdout(io.StringIO()):
            step3j_wfo_daily.main()
    finally:
        os.chdir(cwd)
        sys.argv = argv
    return root / "data"


def test_stress_reports_the_named_windows_that_overlap_the_runs_span(run_2016, tmp_path):
    import stress_crisis_analysis
    trades = str(run_2016 / "wfo_daily_results_s.csv")
    out = run_script(stress_crisis_analysis, ["--trades", trades, "--closes", run_2016 / "closes.csv",
                                              "--out", tmp_path / "stress.json"])
    rep = json.loads((tmp_path / "stress.json").read_text())
    m = mtm.read_mtm(run_2016 / "wfo_daily_mtm_s.csv")
    assert rep["oos_span"] == [str(m.index.min().date()), str(m.index.max().date())]
    assert [w["window"] for w in rep["crisis_periods"]] == ["2016 Brexit"]
    assert [w["window"] for w in rep["calm_periods"]] == ["2017 calm bull"]
    brexit = rep["crisis_periods"][0]
    assert brexit["days"] == len(m.loc["2016-06-01":"2016-07-31"]) > 0       # the overlap only
    assert "2018-Q1 vol-mageddon" in out.split("not reported:")[1].splitlines()[0]


def test_legacy_summary_and_research_scripts_never_call_an_exit_date_ratio_a_sharpe(run):
    import wfo_daily_core as core
    trades = pd.read_csv(run / "wfo_daily_results_t.csv", parse_dates=["entry", "exit"])
    oos = mtm.read_mtm(run / "wfo_daily_mtm_t.csv").index
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        core.summarise(trades, label="L", n_windows=3, n_pairs=2, oos_index=oos, stats=None)
    text = out.getvalue()
    assert "LEGACY exit-date ratio (full days) — not a Sharpe" in text
    assert "LEGACY exit-date ratio (exit days) — not a Sharpe" in text
    assert "Sharpe (" not in text and "correct portfolio Sharpe" not in text
    # Printed / written labels of the daily-path research scripts and the CLI's
    # cross-listed block: "Sharpe" only as "not a Sharpe" / "MTM Sharpe".
    for script in ["step3j_wfo_daily.py", "almgren_chriss_repricing.py", "hmm_modes_proper.py",
                   "audit_baseline.py", "unified_timeline_backtest.py"]:
        src = open(os.path.join(_ROOT, script), encoding="utf-8").read()
        assert "Sharpe(full)" not in src and "full_Sh" not in src and "exit_Sh" not in src, script
        for line in src.splitlines():
            if re.search(r"print\(|\"sharpe", line) and "Sharpe" in line:
                assert re.search(r"not a Sharpe|MTM Sharpe|Sharpe \(daily MTM|DEFLATED SHARPE|Sharpe:  ", line), \
                    (script, line)


def test_almgren_repricing_covers_the_trades_span_only(run, tmp_path, monkeypatch):
    import almgren_chriss_repricing as acr
    closes = pd.read_csv(run / "closes.csv", parse_dates=["Date"]).set_index("Date")
    (tmp_path / "data").mkdir()
    pd.DataFrame(1e6, index=closes.index, columns=closes.columns).to_csv(tmp_path / "volumes.csv")
    monkeypatch.chdir(tmp_path)
    base = ["--trades", run / "wfo_daily_results_t.csv", "--closes", run / "closes.csv",
            "--volumes", tmp_path / "volumes.csv", "--split", START, "--out", tmp_path / "ac.json"]
    out = run_script(acr, base)
    rep = json.loads((tmp_path / "ac.json").read_text())
    trades = pd.read_csv(run / "wfo_daily_results_t.csv", parse_dates=["exit", "oos_start"])
    first = closes.index[closes.index >= trades["oos_start"].min()][0]      # first closes day of the OOS
    assert rep["span"] == [str(first.date()), str(trades["exit"].max().date())]
    assert pd.Timestamp(rep["span"][1]) <= pd.Timestamp(END) < closes.index.max()     # no padding
    assert "LEGACY exit-date ratio (full days) — not a Sharpe" in out
    assert not [k for k in rep if "sharpe" in k]
    late = trades.copy()
    late.loc[0, "exit"] = pd.Timestamp("2024-01-02")
    late.to_csv(tmp_path / "late.csv", index=False)
    with pytest.raises(SystemExit, match="final holdout"):
        run_script(acr, [*base, "--trades", tmp_path / "late.csv"])


def test_audit_baseline_names_its_exit_date_ratio_and_covers_the_trades_span(run, tmp_path):
    import audit_baseline
    base = ["--closes", run / "closes.csv", "--split", START, "--out", tmp_path]
    run_script(audit_baseline, ["--trades", run / "wfo_daily_results_t.csv", *base])
    summary = json.loads((tmp_path / "portfolio_summary_baseline.json").read_text())
    per_pair = pd.read_csv(tmp_path / "pair_summary_baseline.csv")
    assert "legacy_exit_date_ratio_full_days" in summary and not [k for k in summary if "sharpe" in k]
    assert "legacy_exit_date_ratio" in per_pair.columns and "sharpe" not in per_pair.columns
    # R47: the calendar is the trades' own span (first OOS closes day → last
    # exit), not the closes file's end (2007-12-31), and never the holdout.
    trades = pd.read_csv(run / "wfo_daily_results_t.csv", parse_dates=["exit", "oos_start"])
    closes = pd.read_csv(run / "closes.csv", parse_dates=["Date"])["Date"]
    eq = pd.read_csv(tmp_path / "equity_curve_baseline.csv", parse_dates=["date"])
    expected = closes[(closes >= trades["oos_start"].min()) & (closes <= trades["exit"].max())]
    assert list(eq["date"]) == list(expected) and eq["date"].max() < closes.max()
    trades.loc[0, "exit"] = pd.Timestamp("2024-01-02")
    trades.to_csv(tmp_path / "late.csv", index=False)
    with pytest.raises(SystemExit, match="final holdout"):
        run_script(audit_baseline, ["--trades", tmp_path / "late.csv", *base])
    # hmm_modes_proper (a script that runs bias_test at import) uses the same calendar.
    src = open(os.path.join(_ROOT, "hmm_modes_proper.py"), encoding="utf-8").read()
    assert "trades_span_days(OOS_INDEX, tr, SPLIT)" in src and "reindex(OOS_INDEX" not in src


def test_production_universe_reads_the_old_and_the_new_baseline_column(tmp_path, monkeypatch):
    # R47: summaries written by audit_baseline.py before T10 name the ratio
    # `sharpe`, later ones `legacy_exit_date_ratio`; both are read (the new
    # one first) and the output key stays `baseline_sharpe` (owner's decision).
    import build_production_universe as bpu
    rows = {"pair": ["AAA-BBB", "CCC-DDD"], "trades": [10, 5], "win_rate": [0.6, 0.4],
            "total_pnl": [1.0, -0.5], "max_dd": [-0.2, -0.4]}
    path = tmp_path / "pair_summary_baseline.csv"
    monkeypatch.setattr(bpu, "BASELINE_PAIR_SUMMARY", path)
    for cols, expected in [({"sharpe": [0.9, -0.3]}, [0.9, -0.3]),
                           ({"legacy_exit_date_ratio": [0.7, -0.1]}, [0.7, -0.1]),
                           ({"sharpe": [0.9, -0.3], "legacy_exit_date_ratio": [0.7, -0.1]}, [0.7, -0.1])]:
        pd.DataFrame({**rows, **cols}).to_csv(path, index=False)
        out = bpu.load_baseline_pairs()
        assert list(out["pair"]) == ["AAA-BBB", "CCC-DDD"]
        assert list(out["baseline_sharpe"]) == expected, cols
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        bpu.load_baseline_pairs()


# ── bootstrap_sharpe ────────────────────────────────────────────────────────

def test_bootstrap_is_labelled_and_resamples_the_one_sharpe(run):
    import bootstrap_sharpe
    trades = str(run / "wfo_daily_results_t.csv")
    out = run_script(bootstrap_sharpe, ["--trades", trades, "--split", START, "--reps", "20"])
    ret = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")["ret"]
    assert out.count(SAMPLING) >= 2                     # at the top and with the intervals
    point = perf_stats.portfolio_sharpe(ret)
    assert point == pytest.approx(mtm.mtm_stats(ret, 1.0)["sharpe"], rel=1e-12)
    assert f"point Sharpe    : {point:+.3f}" in out
    # The replicates: the same resampled series, the one Sharpe on each.
    rng = np.random.default_rng(42)
    sims = bootstrap_sharpe.stationary_block_bootstrap(ret.to_numpy(), 5.0, 20, rng)
    reps = np.array([perf_stats.portfolio_sharpe(s) for s in sims])
    assert f"mean   = {reps.mean():+.3f}" in out
    assert f"95% CI: [{np.percentile(reps, 2.5):+.3f}, {np.percentile(reps, 97.5):+.3f}]" in out


# ── factor_attribution ──────────────────────────────────────────────────────

def test_factor_attribution_regresses_the_daily_mtm_return_and_reports_its_dates(run, tmp_path):
    import factor_attribution
    trades = str(run / "wfo_daily_results_t.csv")
    out = run_script(factor_attribution, script_args("factor_attribution", trades, tmp_path))
    ret = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")["ret"]
    daily = pd.read_csv(tmp_path / "factor.csv", index_col=0, parse_dates=True)
    rep = json.loads((tmp_path / "factor.json").read_text())
    np.testing.assert_allclose(daily["strategy_ret"], ret.reindex(daily.index), rtol=1e-12)
    assert rep["y"] == "ret" and rep["headline"] is False
    assert rep["n_days"] == len(daily) > 100
    assert rep["aligned_span"] == [str(daily.index.min().date()), str(daily.index.max().date())]
    assert rep["mtm_days"] == len(ret) and rep["mtm_span"] == [str(ret.index.min().date()),
                                                              str(ret.index.max().date())]
    # The one Sharpe on every business day of the MTM span; the factor-neutral
    # ratio (aligned days, alpha + residual) is not called a Sharpe.
    assert rep["sharpe"] == pytest.approx(perf_stats.portfolio_sharpe(ret), rel=1e-12)
    assert [k for k in rep if "sharpe" in k.lower()] == ["sharpe"]
    assert "factor_neutral_ir" in rep
    assert f"aligned days:       {rep['n_days']}  ({rep['aligned_span'][0]} → {rep['aligned_span'][1]})" in out
    assert f"Sharpe (daily MTM, annualised, all {len(ret)} business days): {rep['sharpe']:+.3f}" in out
    assert "raw Sharpe" not in out and "factor-neutral Sh" not in out and "not a Sharpe" in out


# ── fdr_screening ───────────────────────────────────────────────────────────

ASOF = "2007-06-29"
FDR_PAIRS = ["OUA0-OUB0", "OUA1-OUB1", "OUA2-OUB2", "OUA0-OUB1", "OUA1-OUB2"]


@pytest.fixture(scope="module")
def fdr_closes():
    closes, _ = synthetic.ou_positive_control(n_years=5, n_pairs=3, seed=3)
    assert (closes.index <= ASOF).sum() >= 500 and closes.index.max() > pd.Timestamp("2009-06-30")
    return closes


def pit_frame(pairs: list[str], asof: str) -> pd.DataFrame:
    """A file shaped like build_universe_pit.py's (its columns, every value present)."""
    import build_universe_pit as pit
    legs = [p.split("-") for p in pairs]
    return pd.DataFrame({"pair": pairs, "leg1": [a for a, _ in legs], "leg2": [b for _, b in legs],
                         "sector": "S", "corr": 0.5, "beta": 1.0, "hl_days": 5.0,
                         "johansen_stat": 50.0, "asof": asof}, columns=pit.COLUMNS)


def run_fdr(d, candidates: str, frame: pd.DataFrame, closes: pd.DataFrame, *extra) -> tuple[str, dict]:
    import fdr_screening
    d.mkdir(exist_ok=True)
    frame.to_csv(d / candidates, index=False)
    closes.to_csv(d / "closes.csv")
    out = run_script(fdr_screening, ["--candidates", d / candidates, "--closes", d / "closes.csv",
                                     "--out", d / "fdr.json", "--pairs-out", d / "fdr_pairs.csv", *extra])
    return out, json.loads((d / "fdr.json").read_text())


def test_fdr_on_a_frozen_pit_universe_reads_prices_up_to_its_asof_date_only(fdr_closes, tmp_path):
    name = f"universe_pit_{ASOF}.csv"
    out, rep = run_fdr(tmp_path / "full", name, pit_frame(FDR_PAIRS, ASOF), fdr_closes)
    assert rep["pit"] is True and rep["cut"] == ASOF and rep["role"] == "screen_only"
    assert rep["n_candidates"] == len(FDR_PAIRS) and "point-in-time candidate list" in out
    assert "screen only" in out and "NON-HEADLINE / hindsight" not in out
    assert pd.Timestamp(rep["train_span"][1]) <= pd.Timestamp(ASOF)
    # The p-values: Engle-Granger on the log prices up to the as-of date.
    passed = pd.read_csv(tmp_path / "full" / "fdr_pairs.csv")
    assert set(passed["role"]) == {"screen_only"} and "OUA0-OUB0" in set(passed["pair"])
    cut = np.log(fdr_closes.loc[:ASOF])
    expected = coint(cut["OUA0"], cut["OUB0"], trend="c")[1]
    assert passed.set_index("pair").loc["OUA0-OUB0", "p_eg"] == pytest.approx(expected, rel=1e-12)
    # Prices after the as-of date changed (and holes): byte-identical outputs.
    rng = np.random.default_rng(0)
    other = fdr_closes.copy()
    after = other.index > pd.Timestamp(ASOF)
    other.loc[after] = other.loc[after] * np.exp(rng.normal(0, 0.3, size=(after.sum(), other.shape[1])))
    other.loc[other.index[after][::7], "OUA1"] = np.nan
    run_fdr(tmp_path / "other", name, pit_frame(FDR_PAIRS, ASOF), other)
    for f in ("fdr.json", "fdr_pairs.csv"):
        assert (tmp_path / "full" / f).read_bytes() == (tmp_path / "other" / f).read_bytes()
    # One price before the as-of date changed: a different report (the test has power).
    early = fdr_closes.copy()
    early.iloc[100, 0] *= 1.5
    run_fdr(tmp_path / "early", name, pit_frame(FDR_PAIRS, ASOF), early)
    assert (tmp_path / "full" / "fdr.json").read_bytes() != (tmp_path / "early" / "fdr.json").read_bytes()


def test_fdr_refuses_data_after_the_cut_and_labels_a_hindsight_list(fdr_closes, tmp_path):
    name = f"universe_pit_{ASOF}.csv"
    # A frozen universe is screened on its as-of date: another cut is refused.
    with pytest.raises(SystemExit, match="as-of date"):
        run_fdr(tmp_path / "a", name, pit_frame(FDR_PAIRS, ASOF), fdr_closes, "--split", "2008-06-30")
    # Any cut inside the final holdout is refused.
    with pytest.raises(SystemExit, match="final holdout"):
        run_fdr(tmp_path / "b", "pairs.csv", pd.DataFrame({"pair": FDR_PAIRS}), fdr_closes,
                "--split", "2024-01-02")
    # Not a frozen PIT file (a plain list, a builder-shaped file under another
    # name, a PIT name with a hand-appended row): labelled, cut at --split.
    for d, cand, frame in [("c", "pairs.csv", pd.DataFrame({"pair": FDR_PAIRS})),
                           ("d", "candidates.csv", pit_frame(FDR_PAIRS, ASOF)),
                           ("e", name, pd.concat([pit_frame(FDR_PAIRS[:4], ASOF),
                                                  pd.DataFrame({"pair": [FDR_PAIRS[4]]})]))]:
        out, rep = run_fdr(tmp_path / d, cand, frame, fdr_closes, "--split", ASOF)
        assert out.count("NON-HEADLINE / hindsight candidate list") >= 2, d
        assert "WARNING" not in out                       # a cut before 2015: no warning
        assert rep["pit"] is False and rep["label"].startswith("NON-HEADLINE / hindsight candidate list")
        assert rep["cut"] == ASOF and pd.Timestamp(rep["train_span"][1]) <= pd.Timestamp(ASOF)
        assert set(pd.read_csv(tmp_path / d / "fdr_pairs.csv")["role"]) <= {"screen_only"}


def test_fdr_warns_when_a_hindsight_list_is_cut_after_2014(fdr_closes, tmp_path):
    out, rep = run_fdr(tmp_path, "pairs.csv", pd.DataFrame({"pair": FDR_PAIRS}), fdr_closes,
                       "--split", "2015-06-30")
    assert "WARNING: cut 2015-06-30 is after 2014-12-31" in out and rep["cut"] == "2015-06-30"


def test_fdr_output_can_never_be_the_headline_pairs_file(fdr_closes, tmp_path):
    import step3j_wfo_daily as cli
    name = f"universe_pit_{ASOF}.csv"
    run_fdr(tmp_path, name, pit_frame(FDR_PAIRS, ASOF), fdr_closes)
    screened = pd.read_csv(tmp_path / "fdr_pairs.csv")
    first_oos = pd.Timestamp("2008-01-02")                # after the as-of date
    # The screen's output, renamed to a frozen universe's name: not headline.
    reasons = cli.non_headline_reasons(name, screened, first_oos, [])
    assert any("role=screen_only" in r for r in reasons)
    # Even with every builder column present, a role column is refused.
    frozen = pit_frame(FDR_PAIRS[:3], ASOF)
    assert cli.non_headline_reasons(name, frozen, first_oos, []) == []          # (power)
    reasons = cli.non_headline_reasons(name, frozen.assign(role="screen_only"), first_oos, [])
    assert len(reasons) == 1 and "role=screen_only" in reasons[0]
    # The default output never goes into data/ and never matches the PIT name.
    import fdr_screening
    default = fdr_screening.build_parser().parse_args(["--candidates", "x", "--closes", "y"]).pairs_out
    assert not default.startswith("data") and not cli.PIT_UNIVERSE_FILE.fullmatch(os.path.basename(default))


# ── dashboard_daily, stress_crisis_analysis, monte_carlo_robustness ─────────

def test_dashboard_counts_its_pairs_and_labels_legacy_panels(run, tmp_path):
    import dashboard_daily
    trades = str(run / "wfo_daily_results_t.csv")
    out = run_script(dashboard_daily, script_args("dashboard_daily", trades, tmp_path))
    assert "table: Non-cross (2) |" in out and "(26)" not in out
    out = run_script(dashboard_daily, [*script_args("dashboard_daily", trades, tmp_path),
                                       "--legacy-exit-date-pnl"])
    labelled = [line for line in out.splitlines() if line.startswith(("legend:", "panel"))]
    assert len(labelled) >= 5 and all("LEGACY" in line for line in labelled)
    assert "table: Non-cross (2) |" in out


def test_stress_reports_only_windows_inside_the_runs_span(run, tmp_path):
    import stress_crisis_analysis
    trades = str(run / "wfo_daily_results_t.csv")
    run_script(stress_crisis_analysis, script_args("stress_crisis_analysis", trades, tmp_path))
    rep = json.loads((tmp_path / "stress.json").read_text())
    first, last = map(pd.Timestamp, rep["oos_span"])
    windows = rep["crisis_periods"] + rep["calm_periods"]
    assert all(pd.Timestamp(w["start"]) <= last and pd.Timestamp(w["end"]) >= first for w in windows)
    assert windows == []                    # the named windows are 2015-2024; the run is 2005-2007


def test_monte_carlo_reads_the_mtm_series_and_has_no_per_trade_sharpe(run, tmp_path):
    import monte_carlo_robustness
    trades = str(run / "wfo_daily_results_t.csv")
    out = run_script(monte_carlo_robustness, script_args("monte_carlo_robustness", trades, tmp_path))
    rep = json.loads((tmp_path / "mc.json").read_text())
    pnl = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")["pnl"]
    assert rep["observed"]["n_days"] == len(pnl)
    assert rep["observed"]["pnl"] == pytest.approx(pnl.sum(), rel=1e-12)
    assert rep["observed"]["sharpe"] == pytest.approx(perf_stats.portfolio_sharpe(pnl), rel=1e-12)
    assert SAMPLING in out and rep["label"] == SAMPLING
    perm = rep["mc"]["trade_permutation"]
    assert not any("sharpe" in k for k in perm)
    block = out[out.index("trade_permutation"):out.index("evt_tail_injection")]
    assert "Sharpe p05" not in block and "not a Sharpe" in block


# ── compare_results, edge_monitor ───────────────────────────────────────────

def test_compare_results_compares_the_runs_daily_mtm(run, tmp_path):
    import compare_results
    base = copy_run(run, tmp_path, "u")
    cand = str(run / "wfo_daily_results_t.csv")
    args = ["--baseline", base, "--candidate", cand, "--closes", str(run / "closes.csv"),
            "--split", START, "--reps", "20"]
    out = run_script(compare_results, args)
    sharpe = perf_stats.portfolio_sharpe(mtm.read_mtm(run / "wfo_daily_mtm_t.csv")["ret"])
    num = re.escape(f"{sharpe:+.4f}")
    assert re.search(rf"^Sharpe\s+{num}\s+{num}\s+\+0\.0000$", out, re.M)
    assert SAMPLING in out and "NON-HEADLINE" in out and "LEGACY" not in out
    out = run_script(compare_results, [*args, "--legacy-exit-date-pnl"])
    assert LEGACY in out and not re.search(r"^Sharpe", out, re.M)
    assert max(dates_in(out)) <= pd.Timestamp(END)


def test_edge_monitor_rolls_over_the_daily_mtm(run, tmp_path):
    import edge_monitor
    trades = str(run / "wfo_daily_results_t.csv")
    run_script(edge_monitor, script_args("edge_monitor", trades, tmp_path))
    pnl = mtm.read_mtm(run / "wfo_daily_mtm_t.csv")["pnl"]
    ed = pd.read_csv(tmp_path / "edge.csv", parse_dates=["date"], float_precision="round_trip")
    assert pd.DatetimeIndex(ed["date"]).equals(pnl.index)
    np.testing.assert_array_equal(ed["pnl_day"].to_numpy(), pnl.to_numpy())


# ── step4c_bootstrap_sharpe ─────────────────────────────────────────────────

def test_step4c_psr_is_the_per_period_psr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)             # no data/ here: importing the script reads nothing
    import step4c_bootstrap_sharpe as s4c
    ret = pd.Series(np.random.default_rng(7).normal(0.0004, 0.01, 750))
    assert s4c.probabilistic_sharpe(ret, 0.0) == pytest.approx(perf_stats.psr(ret, 0.0), rel=1e-12)
    # PSR(1): P(annualised Sharpe > 1) = P(per-day Sharpe > 1/√252).
    assert s4c.probabilistic_sharpe(ret, 1.0) == pytest.approx(
        perf_stats.psr(ret, 1.0 / math.sqrt(252)), rel=1e-12)


# ── bias_test.py and run_*.sh never reach the final holdout ─────────────────

def test_bias_test_runs_the_cli_on_the_development_period(tmp_path, monkeypatch):
    import bias_test
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=1, seed=0)
    (tmp_path / "data").mkdir()
    closes.to_csv(tmp_path / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(tmp_path / "data" / "cands.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bias_test, "select_pairs_on_train",
                        lambda c, p: pd.DataFrame({"pair": p, "corr": 0.5, "beta": 1.0, "hl_days": 5.0}))
    calls = []
    monkeypatch.setattr(bias_test.subprocess, "run", lambda cmd, check: calls.append(cmd))
    base = ["bias_test.py", "--split", "2005-12-30", "--candidates", "cands.csv", "--closes", "closes.csv"]
    for extra, end in [([], "2023-12-31"), (["--end", "2006-12-29"], "2006-12-29")]:
        monkeypatch.setattr(sys, "argv", base + extra)
        with contextlib.redirect_stdout(io.StringIO()):
            bias_test.main()
        cmd = calls[-1]
        assert cmd[cmd.index("--end") + 1] == end and "--final-holdout" not in cmd


@pytest.mark.parametrize("script", ["run_ablations.sh", "run_final_tests.sh", "run_grid_sharpe.sh",
                                    "run_split.sh"])
def test_run_scripts_pass_an_end_date_before_the_final_holdout(script):
    text = open(os.path.join(_ROOT, script)).read()
    calls = [line for line in text.splitlines()
             if re.search(r"python (step3j_wfo_daily|step3j_wfo|bias_test)\.py", line)]
    assert calls and all("--end" in line for line in calls), script
    assert "--final-holdout" not in text
    assert all(d < pd.Timestamp("2024-01-01") for d in dates_in(text)), script
    assert "full_Sh" not in text and "Sharpe(full)" not in text


# ── Banners ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("doc", ["FULL_DAILY_RUN.md", "PHASE1_QUANT_AUDIT.md"])
def test_superseded_banner_at_the_top(doc):
    lines = open(os.path.join(_ROOT, doc), encoding="utf-8").read().splitlines()
    assert lines[0] == ("> **SUPERSEDED — see RESULTS_v2.md.** The numbers below come from a backtest "
                        "with look-ahead, impossible fills and accounting errors (see AUDIT_FIX_LOG.md); "
                        "do not use them.")
    assert lines[1] == ""
