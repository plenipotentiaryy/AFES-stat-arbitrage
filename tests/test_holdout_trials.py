"""The final holdout and the trials registry (audit task T6).

- `run_wfo_daily` refuses a run that would use a date on or after
  `wfo_daily_core.HOLDOUT_START` (2024-01-01) unless `cfg.final_holdout`; the
  run's last date is min(cfg.end or the panel's last date, the panel's last
  date), which covers the fill-only bar and the panel beyond cfg.end.
- The CLI: without --end on a panel reaching the holdout it stops with a
  message naming --end 2023-12-31 and --final-holdout; with --final-holdout it
  appends one line to HOLDOUT_USES.md BEFORE running; every run whose span lies
  in the development period appends one row to TRIALS.csv and prints the
  annualised Sharpe, PSR(0), DSR and N (= distinct config hashes of TRIALS.csv
  incl. this run; V from one Sharpe per hash, the latest recorded); a legacy
  run is registered without a Sharpe.
- `backfill_trials`: the per-period Sharpe of a historical trades file's
  exit-date daily series, restricted to the development period; one distinct
  config hash per back-filled row.

Synthetic panels only (their dates are moved to 2021-2024 to straddle the
holdout start); the registry files are redirected to tmp (tests/conftest.py)."""
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backfill_trials  # noqa: E402
import step3j_wfo_daily as cli  # noqa: E402
import synthetic  # noqa: E402
import wfo_daily_core as core  # noqa: E402
from perf_stats import deflated_sharpe, psr, sharpe_per_period  # noqa: E402
from wfo_daily_core import WfoConfig, run_wfo_daily  # noqa: E402

HOLDOUT = pd.Timestamp("2024-01-01")
DEV_END = "2023-12-31"
GROSS = dict(execution_model="flat", cost=0.0, slip=0.0, borrow=0.0, fx_cost=0.0, borrow_nonus=0.0)


def panel_into_holdout(start: str = "2021-01-04", n_years: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """An OU panel (2 pairs) on business days from `start` (to ≈ 2024-12)."""
    closes, pairs = synthetic.ou_positive_control(n_years=n_years, n_pairs=2, seed=3)
    closes.index = pd.bdate_range(start, periods=len(closes), name="Date")
    return closes, pairs


def quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(*args, **kwargs)


def all_dates(res) -> pd.Series:
    t = res.trades
    return pd.concat([t["entry_signal"], t["entry"], t["exit_signal"], t["exit"],
                      res.daily_pnl.index.to_series()])


# ── The guard in the core ───────────────────────────────────────────────────

def test_holdout_start_is_2024_01_01():
    assert core.HOLDOUT_START == HOLDOUT
    assert WfoConfig().final_holdout is False


def test_run_that_would_touch_the_holdout_is_refused():
    closes, pairs = panel_into_holdout()
    assert closes.index.max() >= HOLDOUT
    for end in (None, "2024-01-01", "2024-03-28"):
        with pytest.raises(ValueError, match=r"--end 2023-12-31.*--final-holdout"):
            run_wfo_daily(closes, pairs, WfoConfig(end=end, **GROSS))
    dev = quiet(closes, pairs, WfoConfig(end=DEV_END, **GROSS))
    assert len(dev.trades) > 10
    assert all_dates(dev).max() < HOLDOUT
    # The guard is what stopped the run: with the flag it uses holdout dates.
    full = quiet(closes, pairs, WfoConfig(final_holdout=True, **GROSS))
    assert all_dates(full).max() >= HOLDOUT


def test_the_guard_reads_the_one_constant(monkeypatch):
    closes, pairs = panel_into_holdout()
    monkeypatch.setattr(core, "HOLDOUT_START", pd.Timestamp("2023-01-02"))
    with pytest.raises(ValueError, match="2023-01-02"):
        run_wfo_daily(closes, pairs, WfoConfig(end=DEV_END, **GROSS))
    quiet(closes, pairs, WfoConfig(end="2022-12-30", **GROSS))


def test_end_after_a_panel_that_stops_before_the_holdout_is_allowed():
    """The run's last date is the panel's: no data from the holdout exists.
    (Pre-existing, unchanged: make_windows follows cfg.end past the panel, so
    the MTM span carries zero days up to the last window end.)"""
    closes, pairs = panel_into_holdout()
    cut = closes.loc[:"2023-06-30"]
    res = quiet(cut, pairs, WfoConfig(end="2025-06-30", **GROSS))     # last date = the panel's
    trade_dates = all_dates(res).iloc[:4 * len(res.trades)]
    assert len(res.trades) > 10 and trade_dates.max() <= pd.Timestamp("2023-06-30")
    assert (res.daily_pnl.loc["2023-07-01":] == 0.0).all()


def synthetic_volumes(closes: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(rng.uniform(5e5, 2e6, size=closes.shape), index=closes.index,
                        columns=closes.columns)


@pytest.mark.parametrize("case", ["gross", "hmm_block_panic", "almgren_volumes"])
def test_a_development_run_reads_nothing_from_the_holdout(case):
    """A run ending on 2023-12-31 equals the run on the panel (and volumes) cut
    there — also with the HMM gate (fitted on a macro proxy of the whole panel)
    and with Almgren-Chriss costs on volumes (rolling ADV)."""
    closes, pairs = panel_into_holdout()
    cfg = {"gross": WfoConfig(end=DEV_END, **GROSS),
           "hmm_block_panic": WfoConfig(end=DEV_END, hmm="block-panic", **GROSS),
           "almgren_volumes": WfoConfig(end=DEV_END, execution_model="almgren")}[case]
    vol = synthetic_volumes(closes) if case == "almgren_volumes" else None
    full = quiet(closes, pairs, cfg, volumes=vol)
    cut = quiet(closes.loc[:DEV_END], pairs, cfg,
                volumes=None if vol is None else vol.loc[:DEV_END])
    assert len(full.trades) > 10
    pd.testing.assert_frame_equal(full.trades, cut.trades)
    pd.testing.assert_series_equal(full.daily_pnl, cut.daily_pnl)


# ── The CLI ─────────────────────────────────────────────────────────────────

def test_suite_never_writes_the_repo_registry():
    """tests/conftest.py redirects both files for the whole session."""
    root = os.path.realpath(_ROOT)
    assert os.path.realpath(cli.REPO_ROOT) == root
    for f in (cli.TRIALS_FILE, cli.HOLDOUT_USES_FILE):
        assert not os.path.realpath(f).startswith(root + os.sep)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A temp run dir (data/, output/) with the panel; its own registry files."""
    closes, pairs = panel_into_holdout()
    for d in ("data", "output", "registry"):
        (tmp_path / d).mkdir()
    closes.to_csv(tmp_path / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(tmp_path / "data" / "pairs.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TRIALS_FILE", tmp_path / "registry" / "TRIALS.csv")
    monkeypatch.setattr(cli, "HOLDOUT_USES_FILE", tmp_path / "registry" / "HOLDOUT_USES.md")

    def run(*extra, pairs_file="pairs.csv"):
        argv = ["step3j_wfo_daily.py", "--pairs", pairs_file, "--closes", "closes.csv",
                "--execution-model", "flat", "--tag", "t", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = cli.main()
        return result, out.getvalue()
    return tmp_path, run


def read_registry() -> pd.DataFrame:
    return pd.read_csv(cli.TRIALS_FILE, float_precision="round_trip")


def expected_hash(cfg: WfoConfig, pairs_path, closes_name: str, run_pairs=None,
                  volumes: bool = False) -> str:
    fields = dataclasses.asdict(cfg)
    del fields["final_holdout"]
    if run_pairs is None:
        run_pairs = pd.read_csv(pairs_path)["pair"].tolist()
    canonical = json.dumps({"cfg": fields, "run_pairs": run_pairs, "volumes": volumes},
                           sort_keys=True, separators=(",", ":"))
    pairs_sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    return hashlib.sha256("\n".join([canonical, pairs_sha, closes_name]).encode()).hexdigest()


def test_cli_without_end_on_a_panel_into_the_holdout_stops(run_dir):
    root, run = run_dir
    for extra in ([], ["--end", "2024-02-01"]):
        with pytest.raises(SystemExit, match=r"--end 2023-12-31.*--final-holdout"):
            run(*extra)
    assert not cli.TRIALS_FILE.exists() and not cli.HOLDOUT_USES_FILE.exists()
    assert not (root / "data" / "wfo_daily_results_t.csv").exists()


def test_cli_development_run_registers_one_trial_and_prints_sr_psr_dsr_n(run_dir):
    root, run = run_dir
    # A registry with three earlier trials, one without a recorded Sharpe.
    earlier = pd.DataFrame({"config_hash": ["a" * 64, "b" * 64, "c" * 64],
                            "date": ["2026-05-26", "2026-05-27", "2026-05-27"],
                            "source": ["backfill"] * 3, "description": ["x", "y, with a comma", "z"],
                            "n_days": [100, 200, np.nan], "sr_daily": [0.21, -0.03, np.nan],
                            "sr_annual": [0.21 * math.sqrt(252), -0.03 * math.sqrt(252), np.nan]},
                           columns=cli.TRIALS_COLUMNS)
    earlier.to_csv(cli.TRIALS_FILE, index=False)
    res, out = run("--end", DEV_END)

    reg = read_registry()
    assert list(reg.columns) == cli.TRIALS_COLUMNS and len(reg) == 4
    pd.testing.assert_frame_equal(reg.iloc[:3], earlier, check_dtype=False)
    row = reg.iloc[3]
    cfg = WfoConfig(end=DEV_END, execution_model="flat")
    assert row["config_hash"] == expected_hash(cfg, root / "data" / "pairs.csv", "closes.csv")
    assert row["source"] == "cli"
    assert "--end 2023-12-31" in row["description"] and "--tag t" in row["description"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", row["date"])
    ret = res.daily_returns
    assert row["n_days"] == len(ret)
    assert row["sr_daily"] == sharpe_per_period(ret)                     # exact round trip
    assert row["sr_annual"] == pytest.approx(sharpe_per_period(ret) * math.sqrt(252), rel=1e-15)

    sr_trials = [0.21, -0.03, np.nan, sharpe_per_period(ret)]
    assert res.stats["n_trials"] == 4
    assert res.stats["psr0"] == psr(ret)
    assert res.stats["dsr"] == deflated_sharpe(ret, sr_trials)
    assert 0.0 < res.stats["dsr"] < res.stats["psr0"] < 1.0
    assert res.stats["sharpe"] == pytest.approx(sharpe_per_period(ret) * math.sqrt(252), rel=1e-12)
    assert f"Sharpe (daily MTM, annualised): {res.stats['sharpe']:+.3f}" in out
    assert f"PSR(0):  {res.stats['psr0']:.3g}" in out
    assert f"DSR:     {res.stats['dsr']:.3g}" in out
    assert "N = 4 distinct configurations" in out
    assert out.index("DSR:") > out.index("DAILY WFO SUMMARY")
    assert not cli.HOLDOUT_USES_FILE.exists()

    # A second run with another option: another row, another hash, N = 5.
    res2, _ = run("--end", DEV_END, "--maxconc", "1")
    reg = read_registry()
    assert len(reg) == 5 and res2.stats["n_trials"] == 5
    assert reg["config_hash"].iloc[4] != reg["config_hash"].iloc[3]

    # Re-running an identical configuration (another tag: not part of the
    # config) adds a row but not a trial: N stays 5, V and the DSR unchanged.
    res3, out3 = run("--end", DEV_END, "--tag", "again")
    reg = read_registry()
    assert len(reg) == 6 and reg["config_hash"].iloc[5] == reg["config_hash"].iloc[3]
    assert res3.stats["n_trials"] == 5 and "N = 5 distinct configurations" in out3
    sr_trials = [0.21, -0.03, np.nan, sharpe_per_period(ret), sharpe_per_period(res2.daily_returns)]
    assert res3.stats["dsr"] == deflated_sharpe(res3.daily_returns, sr_trials)


def test_registry_keeps_one_sharpe_per_config_hash_the_latest_recorded(run_dir):
    rows = [("x", 0.10), ("y", np.nan), ("x", np.nan), ("z", 0.05), ("x", 0.30), ("x", np.nan),
            ("y", np.nan)]
    pd.DataFrame({"config_hash": [h for h, _ in rows], "date": "d", "source": "s",
                  "description": "e", "n_days": 1, "sr_daily": [s for _, s in rows],
                  "sr_annual": np.nan}, columns=cli.TRIALS_COLUMNS).to_csv(cli.TRIALS_FILE, index=False)
    trials = cli.registry_trials()
    assert list(trials) == ["x", "y", "z"]
    assert trials["x"] == 0.30 and np.isnan(trials["y"]) and trials["z"] == 0.05


def test_legacy_runs_are_registered_without_a_sharpe(run_dir):
    _, run = run_dir
    res, out = run("--end", DEV_END)
    leg, out_leg = run("--end", DEV_END, "--legacy-stale-prints")
    exit_date, _ = run("--end", DEV_END, "--legacy-exit-date-pnl")
    reg = read_registry()
    assert len(reg) == 3 and reg["config_hash"].nunique() == 3
    assert reg["sr_daily"].iloc[0] == sharpe_per_period(res.daily_returns)
    for i, flag in ((1, "--legacy-stale-prints"), (2, "--legacy-exit-date-pnl")):
        assert np.isnan(reg["sr_daily"].iloc[i]) and np.isnan(reg["sr_annual"].iloc[i])
        assert reg["description"].iloc[i].startswith(f"LEGACY ({flag}): step3j_wfo_daily.py")
    for r, n in ((leg, 2), (exit_date, 3)):
        assert r.stats["n_trials"] == n and np.isnan(r.stats["psr0"]) and np.isnan(r.stats["dsr"])
    assert "LEGACY run: registered without a Sharpe" in out_leg


def test_config_hash_covers_everything_that_changes_the_run(run_dir):
    root, _ = run_dir
    pairs = root / "data" / "pairs.csv"
    run_pairs = pd.read_csv(pairs)["pair"].tolist()
    cfg = WfoConfig(end=DEV_END)

    def h(c=cfg, closes="closes.csv", rp=run_pairs, vol=False):
        return cli.config_hash(c, pairs, closes, rp, vol)
    base = h()
    assert base == expected_hash(cfg, pairs, "closes.csv", run_pairs, False)
    assert base == h(WfoConfig(end=DEV_END))
    assert base == h(WfoConfig(end=DEV_END, final_holdout=True))       # not a strategy change
    assert base != h(WfoConfig(end=DEV_END, kalman=True))
    assert base != h(closes="closes_other.csv")
    assert base != h(rp=run_pairs + ["XA-XA.TO_USD"])                  # --legacy-cross-listed
    assert base != h(vol=True)                                          # volumes file used
    pairs.write_text(pairs.read_text() + "X-Y\n")
    assert base != h()


def test_git_revision_ignores_the_two_registries(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("TRIALS.csv", "HOLDOUT_USES.md", "code.py"):
        (repo / name).write_text("x\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(git + ["add", "."], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], cwd=repo, check=True)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    assert re.fullmatch(r"[0-9a-f]{40}", cli.git_revision())
    (repo / "TRIALS.csv").write_text("x\ny\n")
    (repo / "HOLDOUT_USES.md").write_text("x\n- use\n")
    assert re.fullmatch(r"[0-9a-f]{40}", cli.git_revision())
    (repo / "code.py").write_text("y\n")
    assert re.fullmatch(r"[0-9a-f]{40}\+dirty", cli.git_revision())


def test_cli_run_without_trades_still_counts(run_dir):
    root, run = run_dir
    pd.DataFrame({"pair": ["NOPE-NADA"]}).to_csv(root / "data" / "none.csv", index=False)
    res, out = run("--end", DEV_END, pairs_file="none.csv")
    assert res.trades.empty and "No trades." in out
    reg = read_registry()
    assert len(reg) == 1 and np.isnan(reg["sr_daily"].iloc[0])
    assert res.stats["n_trials"] == 1 and "N = 1 distinct configurations" in out


def test_cli_final_holdout_records_the_use_before_running(run_dir, monkeypatch):
    root, run = run_dir
    seen = []

    def stop(*args, **kwargs):          # the core is never reached before the line exists
        seen.append(cli.HOLDOUT_USES_FILE.read_text())
        raise RuntimeError("stopped before the run")
    monkeypatch.setattr(cli, "run_wfo_daily", stop)
    with pytest.raises(RuntimeError, match="stopped"):
        run("--final-holdout")
    assert len(seen) == 1
    lines = [ln for ln in seen[0].splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    cfg = WfoConfig(execution_model="flat", final_holdout=True)
    h = expected_hash(cfg, root / "data" / "pairs.csv", "closes.csv")    # final_holdout excluded
    m = re.fullmatch(r"- (\S+) \| git (\S+) \| (.+) \| config ([0-9a-f]{64})", lines[0])
    assert m, lines[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m.group(1))
    assert re.fullmatch(r"[0-9a-f]{40}(\+dirty)?|unknown", m.group(2))
    assert "step3j_wfo_daily.py" in m.group(3) and "--final-holdout" in m.group(3)
    assert m.group(4) == h
    assert not cli.TRIALS_FILE.exists()

    # A real holdout run on the synthetic panel: a second line; it is no
    # development trial (no TRIALS row), N counts it on top of the registry.
    monkeypatch.setattr(cli, "run_wfo_daily", run_wfo_daily)
    res, out = run("--final-holdout")
    text = cli.HOLDOUT_USES_FILE.read_text()
    assert len([ln for ln in text.splitlines() if ln.startswith("- ")]) == 2
    assert not cli.TRIALS_FILE.exists()
    assert res.daily_pnl.index.max() >= HOLDOUT
    assert res.stats["n_trials"] == 1 and res.stats["dsr"] == res.stats["psr0"]
    assert "FINAL HOLDOUT" in out


def test_a_dev_run_with_the_flag_is_a_trial_and_a_recorded_use(run_dir):
    _, run = run_dir
    run("--final-holdout", "--end", DEV_END)
    assert len(read_registry()) == 1
    assert len([ln for ln in cli.HOLDOUT_USES_FILE.read_text().splitlines()
                if ln.startswith("- ")]) == 1


# ── The back-fill's per-trial Sharpe ────────────────────────────────────────

def test_backfill_sharpe_is_the_exit_date_series_up_to_2023_12_31():
    trades = pd.DataFrame({
        "oos_start": ["2023-12-20", "2023-12-20", "2023-12-20", "2023-12-20", "2024-01-02"],
        "entry": ["2023-12-20", "2023-12-21", "2023-12-26", "2023-12-27", "2024-01-02"],
        "exit": ["2023-12-21", "2023-12-27", "2023-12-27", "2024-01-03", "2024-01-05"],
        "pnl": [0.010, -0.004, 0.002, 0.5, 0.7]})
    sr, n = backfill_trials.exit_date_sharpe(trades)
    days = pd.bdate_range("2023-12-20", "2023-12-29")         # 8 business days
    series = pd.Series(0.0, index=days)
    series["2023-12-21"] = 0.010
    series["2023-12-27"] = -0.004 + 0.002
    assert n == 8 and sr == pytest.approx(series.mean() / series.std(ddof=1), rel=1e-12)
    # no oos_start column: the series starts at the first entry; a pnl column override
    t2 = trades.drop(columns="oos_start").rename(columns={"pnl": "pnl_adj"})
    t2["entry"] = ["2023-12-21", "2023-12-21", "2023-12-26", "2023-12-27", "2024-01-02"]
    sr2, n2 = backfill_trials.exit_date_sharpe(t2, pnl="pnl_adj")
    assert n2 == 7 and sr2 == pytest.approx(series.iloc[1:].mean() / series.iloc[1:].std(ddof=1))
    # an explicit end (a trial whose OOS stopped earlier)
    sr3, n3 = backfill_trials.exit_date_sharpe(trades, end="2023-12-27")
    assert n3 == 6 and sr3 == pytest.approx(series.iloc[:6].mean() / series.iloc[:6].std(ddof=1))


def test_backfill_gives_every_trial_its_own_config_hash(tmp_path, monkeypatch):
    """The back-fill on a fake tree shaped like the owner's data/ and output/
    (incl. two byte-identical trades files and the 16 rows that share a
    source): one distinct config hash per row; it refuses to overwrite."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    trades = pd.DataFrame({"entry": ["2015-07-01", "2016-03-01", "2019-05-02"],
                           "exit": ["2015-07-09", "2016-03-10", "2019-05-13"],
                           "pnl": [0.01, -0.02, 0.015], "pair": ["A-B"] * 3,
                           "oos_start": ["2015-06-30"] * 3})
    tagged = ["", "_base", "_bias", "_cusum", "_fullrun", "_kalman", "_kalman_cusum", "_v2_filter_on"]
    for tag in tagged:
        trades.to_csv(data / f"wfo_daily_results{tag}.csv", index=False)
    trades.to_csv(data / "wfo_v2_filter_off.csv", index=False)
    for name in ("wfo_results.csv", "wfo_folds_L9.csv", "trades_L9.csv"):
        (data / name).write_text("fold,test_pnl\n0,1.0\n")
    for rel in backfill_trials.RUN_FILES:
        (runs / rel).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(runs / rel, index=False)
    logs = list(backfill_trials.RUN_LOGS) + [r for _, _, r in backfill_trials.OTHER if r]
    for rel in logs:
        (runs / rel).parent.mkdir(parents=True, exist_ok=True)
        (runs / rel).write_text("  pairs:      2 from p.csv\n  costs:      x\n")
    out = tmp_path / "TRIALS.csv"
    argv = ["backfill_trials.py", "--data", str(data), "--runs", str(runs), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with contextlib.redirect_stdout(io.StringIO()):
        backfill_trials.main()
    reg = read_registry_at(out)
    n_expected = (len(tagged) + 1 + len(backfill_trials.RUN_FILES) + len(backfill_trials.RUN_LOGS)
                  + len(backfill_trials.OTHER) + 2)
    assert len(reg) == n_expected
    assert reg["config_hash"].nunique() == len(reg)
    assert (reg["source"] == "backfill:FULL_DAILY_RUN.md").sum() == 10
    assert reg["sr_daily"].notna().sum() == len(tagged) + 1 + len(backfill_trials.RUN_FILES)
    with pytest.raises(SystemExit, match="exists"):
        backfill_trials.main()


def read_registry_at(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")
