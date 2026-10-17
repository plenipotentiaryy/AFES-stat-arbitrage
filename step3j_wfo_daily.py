"""step3j_wfo_daily.py — Walk-Forward Optimization on the daily timeframe.

Built on top of the sanity-tested engine in `daily_sanity.py`.  The intraday
`step3j_wfo.py` carries a stack of filters (Johansen on rolling 3y, MetaGate
trained on intraday features, limit-order matching, VW-zscore, grid search
on TRAIN) which all degrade on daily bars — the result on baseline 14 pairs
was Sharpe ≈ 0 with 57 trades vs. sanity's Sharpe ≈ +1.59 with 4564 trades.

This file is the daily replacement: expanding-window WFO with the simple,
robust pieces only.  Filters and gates (HMM / Hurst / MetaGate / sizing /
ticker-overlap) will be layered back in subsequent phases on top of this
backbone, each justified by an A/B comparison vs. this raw baseline.

The engine lives in `wfo_daily_core.py` (`run_wfo_daily`); this file is the
CLI: it reads data/, builds the `WfoConfig`, applies the headline-universe
policy to the pair list (audit T2), prints the summary and the separate
cross-listed table, and writes the trades CSV, the daily mark-to-market
series (wfo_daily_mtm<tag>.csv: Date, pnl, ret — audit T4; the input of
bootstrap_sharpe / dashboard_daily / stress_crisis_analysis /
factor_attribution), the per-pair daily MTM PnL (wfo_daily_mtm_pairs<tag>.csv,
read by dashboard_daily), the run's description wfo_daily_meta<tag>.json (the
cost model, the WfoConfig and its hash, the git revision, the pairs file's
SHA-256 and the headline flag — audit T8) and the equity chart.  A run is
HEADLINE only on a frozen point-in-time universe (audit T5,
`non_headline_reasons`) with costs not below the defaults and the default
execution model (final review R49); every other run prints a NON-HEADLINE
banner in its header and summary, and `WfoResult.stats["headline"]` records
the verdict.
Final holdout (audit T6): a run that would use a date on or after
wfo_daily_core.HOLDOUT_START (2024-01-01) stops unless --final-holdout, and
every --final-holdout run is
recorded in HOLDOUT_USES.md before it starts; every run whose span lies in the
development period is appended to the trials registry TRIALS.csv, and the
summary prints the annualised Sharpe, PSR(0), the Deflated Sharpe Ratio and N
(the number of distinct configurations in TRIALS.csv).

CLI:
    python step3j_wfo_daily.py --pairs pairs_baseline14.csv
    python step3j_wfo_daily.py --pairs pairs_selected.csv --cost 10
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, BAR_TIMEFRAME
from build_universe_pit import COLUMNS as PIT_COLUMNS
from cross_listing import close_time
from mtm import file_sha256
from perf_stats import deflated_sharpe, psr, sharpe_per_period, sr0_threshold
from wfo_daily_core import (
    TRAIN_INIT_MO, OOS_MO, STEP_MO, HOLDOUT_START,
    WfoConfig, cost_model, holdout_violation, make_windows, run_last_date, run_wfo_daily,
    summarise, trade_stats,
)


def save_equity_chart(tr: pd.DataFrame, path: Path, daily_returns: pd.Series | None = None):
    """The date-ordered daily mark-to-market equity (cumulative daily returns
    on capital, audit T4); without daily_returns (legacy exit-date booking)
    the cumulative trade PnL by exit date."""
    if tr.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    if daily_returns is None:
        eq = tr.sort_values("exit")["pnl"].cumsum()
        ax.plot(tr.sort_values("exit")["exit"].values, eq.values)
        ax.set_title("Daily WFO — cumulative log-spread PnL")
    else:
        ax.plot(daily_returns.index, daily_returns.cumsum().to_numpy() * 100)
        ax.set_title("Daily WFO — daily mark-to-market equity (cumulative return, % of capital)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── Headline universe (audit T2) ────────────────────────────────────────────

def headline_exclusions(pairs: list[str], unknown_ok: bool = False) -> pd.DataFrame:
    """Pairs the headline run excludes, with their legs' exchanges and the
    reasons: a leg is fx_unaligned (a "*_USD" leg: a local leg converted
    without an FX rate observed at its own close — this repo has no intraday
    FX), trades on an exchange without NYSE session overlap (ASX, TSE: both
    legs can never fill together), or trades on a European exchange (allowed
    only with synchronous 09:30–11:30 ET snapshots, which this repo lacks).
    A leg with an unknown exchange suffix raises KeyError naming the ticker
    (cross_listing.close_time) unless `unknown_ok` (the --legacy-cross-listed
    run), where the pair is listed with the reason "unknown exchange"."""
    rows = {}
    for pair in pairs:
        legs = pair.split("-")
        names, reasons = [], []
        if any(t.endswith("_USD") for t in legs):
            reasons.append("fx_unaligned")
        for t in legs:
            try:
                ex = close_time(t)
            except KeyError:
                if not unknown_ok:
                    raise
                names.append("unknown")
                reasons.append(f"unknown exchange ({t})")
                continue
            seen = ex.name in names
            names.append(ex.name)
            if seen:                                    # one reason per exchange
                continue
            if not ex.overlaps_nyse:
                reasons.append(f"{ex.name}: no session overlap with NYSE")
            if ex.region == "Europe":
                reasons.append(f"{ex.name}: no synchronous 09:30-11:30 ET snapshots")
        if reasons:
            rows[pair] = {"exchanges": " + ".join(names), "reasons": "; ".join(reasons)}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["exchanges", "reasons"])


def cross_listed_table(excluded: pd.DataFrame, trades: pd.DataFrame | None = None) -> str:
    """The separate table of the pairs the headline excludes.  With `trades`
    (a --legacy-cross-listed run) it adds their numbers, labelled LEGACY —
    NOT VALID."""
    title = f"CROSS-LISTED PAIRS (not in headline): {len(excluded)}"
    table = excluded.copy()
    if trades is not None:
        title += "  — numbers: LEGACY — NOT VALID (--legacy-cross-listed)"
        mine = trades[trades["pair"].isin(excluded.index)].groupby("pair")["pnl"]
        table["trades"] = mine.count().reindex(table.index, fill_value=0)
        table["win_rate"] = (mine.apply(lambda p: (p > 0).mean()).reindex(table.index)
                             .map(lambda w: "—" if pd.isna(w) else f"{w:.1%}"))
        table["total_pnl"] = mine.sum().reindex(table.index, fill_value=0.0).round(4)
    bar = "=" * 70
    return f"{bar}\n{title}\n{bar}\n{table.to_string() if len(table) else '(none)'}"


# ── Headline run (audit T5) ─────────────────────────────────────────────────

PIT_UNIVERSE_FILE = re.compile(r"universe_pit_(\d{4}-\d{2}-\d{2})\.csv")
NON_HEADLINE_BANNER = ("NON-HEADLINE RUN: pairs file is not a frozen point-in-time universe "
                       "(hindsight / survivorship)")
# Audit T9: options no truncation-invariance test clears (tests/
# test_remaining_lookahead.py), with the reason.  A run that would otherwise
# be HEADLINE is refused; any other run names them in its banner.
HEADLINE_FORBIDDEN = {
    "metagate": "crashes before its first window (UnboundLocalError, Ruling R7), so no "
                "truncation-invariance test can run it",
    "hrp": "crashes before its first window (UnboundLocalError, Ruling R7), so no "
           "truncation-invariance test can run it",
}


def frozen_pit_asof(pairs_file: str, pairs_df: pd.DataFrame) -> str | None:
    """The as-of date of a frozen point-in-time universe written by
    build_universe_pit.py — a file named universe_pit_<asof>.csv with the
    builder's columns, a value in each of them on every row, and that one
    as-of date in the `asof` column of every row (a hand-appended pair is not
    frozen); None for any other pairs file.  (Also used by fdr_screening.py.)"""
    name = PIT_UNIVERSE_FILE.fullmatch(Path(pairs_file).name)
    frozen = (name is not None and len(pairs_df) > 0
              and all(c in pairs_df.columns for c in PIT_COLUMNS)
              and bool(pairs_df[PIT_COLUMNS].notna().all().all())
              and bool((pairs_df["asof"].astype(str) == name.group(1)).all()))
    return name.group(1) if frozen else None


SCREEN_ONLY_REASON = ("NON-HEADLINE RUN: pairs file carries a role column (role=screen_only: the "
                      "output of a screen such as fdr_screening.py), never the headline pairs")


# Final review R49: cost parameters a headline run may not lower (higher
# costs, e.g. a stress run, stay HEADLINE); the headline cost model is the
# WfoConfig default (flat).
HEADLINE_COST_FIELDS = ("cost", "slip", "borrow", "borrow_nonus", "fx_cost")
COMPARISON_MODEL_REASON = ("comparison cost model; the headline uses the default (R25: the "
                           "conservative model on the development data)")


def non_headline_reasons(pairs_file: str, pairs_df: pd.DataFrame, first_oos,
                         legacy: list[str], cfg: WfoConfig) -> list[str]:
    """Why a run is not a headline run; empty = HEADLINE.  A headline run
    reads a frozen point-in-time universe (`frozen_pit_asof`) whose as-of date
    is strictly before the run's first OOS date (`first_oos`, None if the run
    has no window), with no `role` column (audit T10: a screen's output, e.g.
    fdr_screening.py's role=screen_only file, is never the headline pairs),
    and uses no legacy option (`legacy`, the --legacy-* options given; hard
    rule 5).  Manual pair lists, step7's OOS-PnL curation and every other
    pairs file are not headline.  Final review R49: the run's costs (`cfg`)
    are not below the WfoConfig defaults (cost, slip, borrow, borrow_nonus,
    fx_cost; higher costs are allowed) and it uses the default execution
    model (flat, R25)."""
    reasons = []
    asof = frozen_pit_asof(pairs_file, pairs_df)
    if asof is None:
        reasons.append(NON_HEADLINE_BANNER)
    elif first_oos is None or not pd.Timestamp(asof) < first_oos:
        first = "none (no window)" if first_oos is None else first_oos.date()
        reasons.append(f"{NON_HEADLINE_BANNER}: as of {asof}, not before the first "
                       f"OOS date {first}")
    if "role" in pairs_df.columns:
        reasons.append(SCREEN_ONLY_REASON)
    if legacy:
        reasons.append("NON-HEADLINE RUN: legacy options " + ", ".join(legacy))
    defaults = {f.name: f.default for f in dataclasses.fields(WfoConfig)}
    for name in HEADLINE_COST_FIELDS:
        value = getattr(cfg, name)
        if not value >= defaults[name]:                      # NaN fails too
            reasons.append(f"NON-HEADLINE RUN: cost parameter below its default: "
                           f"--{name.replace('_', '-')} {value} < {defaults[name]} "
                           "(WfoConfig default)")
    if cfg.execution_model != defaults["execution_model"]:
        reasons.append(f"NON-HEADLINE RUN: --execution-model {cfg.execution_model}: "
                       f"{COMPARISON_MODEL_REASON}")
    return reasons


def headline_block(reasons: list[str], pairs_file: str, first_oos, cfg: WfoConfig) -> str:
    """The banner of a non-headline run, or the headline line (R49: with the
    cost-model label and the run's --start / --end)."""
    if reasons:
        bar = "!" * 70
        return "\n".join([bar, *reasons, bar])
    return (f"HEADLINE RUN: frozen point-in-time universe {Path(pairs_file).name}, "
            f"first OOS date {first_oos.date()}; costs {cost_model(cfg)['label']}; "
            f"--start {cfg.start or 'not set'} --end {cfg.end or 'not set'}")


# ── Final holdout and trials registry (audit T6) ────────────────────────────

# Repo-root files (the CLI runs with cwd = a run dir; DATA_DIR is cwd-relative).
REPO_ROOT = Path(__file__).resolve().parent
TRIALS_FILE = REPO_ROOT / "TRIALS.csv"
HOLDOUT_USES_FILE = REPO_ROOT / "HOLDOUT_USES.md"
TRIALS_COLUMNS = ["config_hash", "date", "source", "description", "n_days", "sr_daily", "sr_annual"]
HOLDOUT_USES_HEADER = """# HOLDOUT_USES

Every run that may use the final holdout (dates from 2024-01-01,
`wfo_daily_core.HOLDOUT_START`) needs `--final-holdout`; `step3j_wfo_daily.py`
appends one line here BEFORE such a run starts: UTC time | git revision
(`+dirty` = uncommitted changes to tracked files other than TRIALS.csv and this
file) | command line | config hash (as in TRIALS.csv).  Only the CLI records:
a direct run_wfo_daily(final_holdout=True) call is not recorded here.

"""


def config_hash(cfg: WfoConfig, pairs_path: Path, closes_name: str,
                run_pairs: list[str], volumes_used: bool) -> str:
    """The identity of a configuration (one trial): SHA-256 of three lines joined
    by "\\n" — the canonical JSON (sorted keys, no spaces) of {"cfg": asdict(cfg)
    without final_holdout (it does not change the strategy), "run_pairs": the
    pairs actually run (after the headline policy, so --legacy-cross-listed
    counts), "volumes": whether a volumes file was used (stale-print mask, ADV)},
    the pairs file's SHA-256 and the closes file name.  The code revision is not
    part of it: the same configuration under another engine version is the
    same trial."""
    fields = dataclasses.asdict(cfg)
    del fields["final_holdout"]
    canonical = json.dumps({"cfg": fields, "run_pairs": list(run_pairs), "volumes": bool(volumes_used)},
                           sort_keys=True, separators=(",", ":"))
    pairs_sha = hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest()
    return hashlib.sha256("\n".join([canonical, pairs_sha, closes_name]).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_revision() -> str:
    """HEAD of the repo holding this file, "+dirty" if tracked files other than
    the two registries (TRIALS.csv, HOLDOUT_USES.md) changed."""
    try:
        def git(*a):
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                    f":(exclude){TRIALS_FILE.name}", f":(exclude){HOLDOUT_USES_FILE.name}")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def record_holdout_use(chash: str) -> None:
    """Append this run's line to HOLDOUT_USES.md (created with its header)."""
    if not HOLDOUT_USES_FILE.exists():
        HOLDOUT_USES_FILE.write_text(HOLDOUT_USES_HEADER)
    with open(HOLDOUT_USES_FILE, "a") as f:
        f.write(f"- {utc_now()} | git {git_revision()} | {shlex.join(sys.argv)} | config {chash}\n")


def sr_daily_or_nan(returns: pd.Series) -> float:
    """Per-period Sharpe of the run's daily returns; NaN if undefined (no trades)."""
    try:
        return sharpe_per_period(returns)
    except ValueError:
        return float("nan")


def append_trial(chash: str, description: str, n_days: int, sr: float) -> None:
    """Append one development-period trial to TRIALS.csv (created with its header)."""
    new = not TRIALS_FILE.exists()
    num = (lambda x: "" if math.isnan(x) else repr(x))
    with open(TRIALS_FILE, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(TRIALS_COLUMNS)
        w.writerow([chash, utc_now(), "cli", description, n_days, num(sr),
                    num(sr * math.sqrt(252))])


def registry_trials() -> dict[str, float]:
    """One per-period Sharpe per distinct config_hash of TRIALS.csv: that of the
    latest row of the hash with an sr_daily (NaN if none has one)."""
    if not TRIALS_FILE.exists():
        return {}
    reg = pd.read_csv(TRIALS_FILE, float_precision="round_trip", dtype={"config_hash": str})
    trials: dict[str, float] = {}
    for h, sr in zip(reg["config_hash"], reg["sr_daily"].astype(float)):
        if h not in trials or not math.isnan(sr):
            trials[h] = sr
    return trials


def deflate(result, chash: str, dev_run: bool, legacy: list[str]) -> str:
    """Register the run (development runs only) and deflate its Sharpe.

    N = the number of distinct config hashes in TRIALS.csv incl. this run
    (re-running a configuration does not change it; a holdout run is counted
    on top of the registry, not added to it); V = sample variance of one
    sr_daily per hash (`registry_trials`).  A legacy run (`legacy`: its
    --legacy-* options) is registered without sr_daily (never an exit-date or
    legacy Sharpe in V) and is not deflated.  Sets stats n_trials, psr0 and
    dsr (NaN when undefined); returns the summary lines."""
    ret = result.daily_returns
    sr_run = sr_daily_or_nan(ret)
    sr = float("nan") if legacy else sr_run
    if dev_run:
        cmd = "step3j_wfo_daily.py " + shlex.join(sys.argv[1:])
        append_trial(chash, f"LEGACY ({', '.join(legacy)}): {cmd}" if legacy else cmd, len(ret), sr)
        trials = registry_trials()
    else:
        trials = registry_trials()
        if chash not in trials or not math.isnan(sr):
            trials[chash] = sr
    values = list(trials.values())
    n = len(values)
    known = [x for x in values if not math.isnan(x)]
    var = float(np.var(known, ddof=1)) if len(known) >= 2 else float("nan")
    psr0 = dsr = float("nan")
    if not math.isnan(sr):
        psr0 = psr(ret)
        if n == 1 or len(known) >= 2:
            dsr = deflated_sharpe(ret, values)
    result.stats.update({"n_trials": n, "psr0": psr0, "dsr": dsr})

    fmt = (lambda x, spec: "n/a" if math.isnan(x) else format(x, spec))
    sr0 = 0.0 if n == 1 else (float("nan") if math.isnan(var) else sr0_threshold(n, var))
    where = (f"config hashes in {TRIALS_FILE.name} incl. this run" if dev_run else
             f"config hashes in {TRIALS_FILE.name} + this holdout run (not added to it)")
    lines = [
        "DEFLATED SHARPE (audit T6; Bailey & López de Prado 2014)",
        f"  Sharpe (daily MTM, annualised): {fmt(result.stats.get('sharpe', float('nan')), '+.3f')}"
        f"   (per day {fmt(sr_run, '+.4f')}, T = {len(ret)} days)",
        f"  PSR(0):  {fmt(psr0, '.3g')}",
        f"  DSR:     {fmt(dsr, '.3g')}   (SR0 = {fmt(sr0, '+.4f')} per day)",
        f"  N = {n} distinct configurations ({where}); V = {fmt(var, '.3g')} from {len(known)} "
        f"recorded Sharpe ratios (one per configuration)",
        f"  config hash {chash}",
    ]
    if legacy:
        lines.insert(1, "  LEGACY run: registered without a Sharpe (not in V); PSR / DSR not computed")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily Walk-Forward Optimization")
    p.add_argument("--pairs",  type=str, default="pairs_selected.csv",
                   help="CSV in data/ with a 'pair' column (e.g. AMT-AWK).")
    p.add_argument("--start",  type=str, default=None)
    p.add_argument("--end",    type=str, default=None)
    p.add_argument("--cost",   type=float, default=5.0,
                   help="Commission + half-spread, bp per leg per fill × the leg's notional "
                        "(flat model; default 5).")
    p.add_argument("--slip",   type=float, default=5.0,
                   help="Slippage, bp per leg per fill × the leg's notional (flat model; "
                        "default 5). Round trip = 2·(cost + slip)·(1 + |β|) per $1 of leg-1 "
                        "notional at unchanged prices.")
    p.add_argument("--borrow", type=float, default=80.0,
                   help="Borrow on a short leg's notional (marked at each held row's close), "
                        "bp/year, US listings (flat model; default 80).")
    p.add_argument("--borrow-nonus", type=float, default=300.0, dest="borrow_nonus",
                   help="Borrow on a short non-US (*_USD) leg's notional, bp/year (flat model; "
                        "default 300 = the highest borrow tier of almgren_chriss_repricing."
                        "TIER_TABLE: the hard-to-borrow / locate cost of foreign lines is "
                        "unknown here). The almgren model uses its tier rates.")
    p.add_argument("--fx-cost", type=float, default=2.0, dest="fx_cost",
                   help="FX conversion cost, bp per fill of a *_USD leg (a local listing "
                        "converted to USD) × its notional; both models (default 2).")
    p.add_argument("--train",  type=int, default=TRAIN_INIT_MO,
                   help=f"Initial train length in months (default {TRAIN_INIT_MO}).")
    p.add_argument("--oos",    type=int, default=OOS_MO,
                   help=f"OOS length in months (default {OOS_MO}).")
    p.add_argument("--step",   type=int, default=STEP_MO,
                   help=f"Walk-forward step in months (default {STEP_MO}).")
    p.add_argument("--kalman", action="store_true",
                   help="Use Kalman dynamic β: a trade is sized with the one-step prior "
                        "β_{t|t−1} of its decision row t and enters on the rolling z of "
                        "log p1 − β_{t|t−1}·log p2, the spread it then exits on (audit T9).")
    p.add_argument("--vwz",    action="store_true",
                   help="Use volume-weighted z + VR≥0.5 liquidity gate.")
    p.add_argument("--metagate", action="store_true",
                   help="Per-window MetaGate ML entry classifier "
                        "(features: abs_z, side, vol_20/60, vol_ratio, hurst).")
    p.add_argument("--hrp", action="store_true",
                   help="Per-window HRP weights scale each pair's trade size. "
                        "Weights renormalized so MEAN weight = 1 (low-corr "
                        "pairs scaled up, high-corr pairs scaled down).")
    p.add_argument("--ivol", action="store_true",
                   help="Inverse-volatility sizing: weight ∝ 1/std(spread) on train, "
                        "mean-normalized to 1. Simpler/more robust than HRP.")
    p.add_argument("--volvol", action="store_true",
                   help="Dynamic vol-targeting: per-entry size ∝ 1/recent_20d_spread_vol, "
                        "normalised by the mean of the trades decided up to its entry decision "
                        "(audit T9). Pro-cyclical.")
    p.add_argument("--pairmom", action="store_true",
                   help="Pair momentum from the rolling per-trade ratio (mean / std of the pair's "
                        "trade PnL, not a Sharpe): hot pair → up-weight, cold → down. "
                        "Causal: only the pair's trades closed by the entry decision (audit T9).")
    p.add_argument("--maxconc", type=int, default=0,
                   help="Hard cap on concurrent open trades across all pairs "
                        "(post-hoc filter, 0 = no cap).")
    p.add_argument("--ticker-overlap", action="store_true",
                   help="Block new entry if either of its tickers is already "
                        "exposed in another open trade.")
    p.add_argument("--per-pair-quarter-cap", type=int, default=0,
                   help="Max trades per pair per calendar quarter (0 = no cap).")
    p.add_argument("--vol-tight-stop", action="store_true",
                   help="Tighten stop_z to 2.5 when entry's 20d/60d vol ratio > 1.5.")
    p.add_argument("--exec", type=str, default="next_close",
                   choices=["next_close", "vwap"], dest="exec_mode",
                   help="Fill of an order decided at close t: next_close = close "
                        "t+1 (default); vwap = next-day VWAP (signal still computed "
                        "on close). vwap requires data/vwap_daily.csv and the static-β "
                        "path (not with --kalman / --vwz, audit T9).")
    p.add_argument("--vwap-exec", action="store_const", const="vwap", dest="exec_mode",
                   help="Alias of --exec vwap.")
    p.add_argument("--legacy-same-close", action="store_true", dest="legacy_same_close",
                   help="LEGACY: restore the pre-audit fills at the close that made "
                        "the signal (impossible fills). For before/after comparison "
                        "only; never use for headline results.")
    p.add_argument("--legacy-spread-pnl", action="store_true", dest="legacy_spread_pnl",
                   help="LEGACY: book trade PnL as the spread change between the fills "
                        "instead of from the legs' fill prices. Comparison only; never "
                        "for headline results.")
    p.add_argument("--legacy-stale-prints", action="store_true", dest="legacy_stale_prints",
                   help="LEGACY: keep exactly unchanged closes with zero/missing volume "
                        "as prints and allow entries on a zero-return signal day. "
                        "Comparison only; never for headline results.")
    p.add_argument("--legacy-window-reset", action="store_true", dest="legacy_window_reset",
                   help="LEGACY: reset each pair's position state per walk-forward window "
                        "(positions still open at a window end are dropped; the training "
                        "slice includes the first OOS day). Needed by --legacy-same-close "
                        "and --legacy-spread-pnl. Comparison only; never for headline results.")
    p.add_argument("--legacy-exit-date-pnl", action="store_true", dest="legacy_exit_date_pnl",
                   help="LEGACY: book each trade's PnL on its exit date (daily PnL; the exit-date "
                        "ratios over full days / exit days, not a Sharpe) and MaxDD over trades "
                        "in table order, instead of the daily "
                        "mark-to-market on constant capital; no MTM file is written. Needed by "
                        "--legacy-spread-pnl. Comparison only; never for headline results.")
    p.add_argument("--legacy-no-entry-guards", action="store_true", dest="legacy_no_entry_guards",
                   help="LEGACY: enter on any |z| > entry_z, as before the audit (no "
                        "|z| < stop_z limit: 5-10 sigma jumps, often stale prices, are "
                        "traded). Comparison only; never for headline results.")
    p.add_argument("--legacy-flat-cost", action="store_true", dest="legacy_flat_cost",
                   help="LEGACY: the pre-T8 cost per trade — one round trip whatever |β| "
                        "(flat: 2·(cost + slip) bp; almgren: 2·(c1 + c2)), no FX cost, borrow "
                        "per unit size per held day (flat: --borrow; almgren: leg 2's tier). "
                        "Needed by --legacy-spread-pnl. Comparison only; never for headline "
                        "results.")
    p.add_argument("--legacy-kalman-posterior", action="store_true", dest="legacy_kalman_posterior",
                   help="LEGACY: the pre-T9 --kalman path — β_t already updated with the day-t "
                        "close (posterior: look-ahead) sizes the trade, and the entry z is the "
                        "z of the dynamic-β spread. Comparison only; never for headline results.")
    p.add_argument("--legacy-hmm-viterbi", action="store_true", dest="legacy_hmm_viterbi",
                   help="LEGACY: the pre-T9 --hmm gate — states from the Viterbi path of the "
                        "whole window (uses later returns), the HMM fitted on rows up to and "
                        "including the OOS start, the EW proxy over every column (*_USD legs "
                        "too). Comparison only; never for headline results.")
    p.add_argument("--legacy-sizing-lookahead", action="store_true", dest="legacy_sizing_lookahead",
                   help="LEGACY: the pre-T9 sizing of --volvol (normalised by the mean over ALL "
                        "trades, later ones too), --pairmom (from same-pair trades not yet closed "
                        "at the entry decision) and --voltarget (leverage at the fill date, from "
                        "exits filled after the decision), and the tie order of --ticker-overlap / "
                        "--per-pair-quarter-cap / --maxconc among entries filled on the same day "
                        "(by their position in the whole trades table). Comparison only; never for "
                        "headline results.")
    p.add_argument("--final-holdout", action="store_true", dest="final_holdout",
                   help="Allow the run to use the final holdout (dates from "
                        f"{HOLDOUT_START.date()}). Recorded in HOLDOUT_USES.md before the "
                        "run starts. Without it, a run whose data reach the holdout "
                        "stops: pass --end 2023-12-31 for the development period.")
    p.add_argument("--legacy-cross-listed", action="store_true", dest="legacy_cross_listed",
                   help="LEGACY: keep the cross-listed pairs (fx_unaligned, ASX/TSE or "
                        "European legs) in the run; their numbers are NOT VALID. "
                        "Comparison only; never for headline results.")
    p.add_argument("--rotation", action="store_true",
                   help="Quarterly per-pair S_perf rescaling (§5): trades sized by "
                        "the pair's rolling per-trade ratio (mean / std of trade PnL × √252, not "
                        "a Sharpe) with Bayesian shrinkage. Drop pairs whose 6-month per-trade "
                        "ratio < 0 for the current quarter.")
    p.add_argument("--voltarget", type=float, default=0.0,
                   help="Portfolio vol target (annualised, e.g. 0.10 = 10%%). "
                        "Applied as causal leverage scaling (the leverage at the entry "
                        "decision, audit T9). 0 = off.")
    p.add_argument("--closes",   type=str, default="closes_daily.csv",
                   help="Closes file in data/ (extended universe support).")
    p.add_argument("--hmm",    type=str, default="off",
                   choices=["off", "block-panic", "block-calm",
                            "size-panic", "extreme-3state"],
                   help="HMM mode: off / block-panic / block-calm (invert) / "
                        "size-panic (50%% size in panic) / extreme-3state "
                        "(3-state, block only highest-vol). States from a causal forward "
                        "filter on SPY or the EW mean of the US closes (audit T9).")
    p.add_argument("--cusum",  type=float, default=None,
                   help="CUSUM break threshold h (e.g. 5.0). If unset, CUSUM off.")
    p.add_argument("--tag",    type=str, default="",
                   help="Suffix for output filenames (e.g. 'kalman_cusum').")
    p.add_argument("--aussie-filter", action="store_true", dest="aussie_filter",
                   help="Conditional gate for Aussie/JP ADR pairs: entry_z>=2.5, "
                        "blackout when 5d realized spread vol > 95th-pct of trailing 252d, "
                        "require rolling 60d AR(1) half-life < 5 days.")
    p.add_argument("--execution-model", type=str, default="flat",
                   choices=["flat", "almgren"], dest="execution_model",
                   help="Cost rate per fill per leg (× the leg's notional at the fill; audit "
                        "T8): flat = --cost + --slip bp, borrow --borrow / --borrow-nonus; "
                        "almgren = the leg's tier half_spread + κ·σ·√(notional/ADV), borrow at "
                        "the short leg's tier rate.  Both add --fx-cost on *_USD legs.  "
                        "Default: flat — the model with the lower development-period Sharpe "
                        "(conservative ruling, AUDIT_FIX_LOG.md T8).")
    p.add_argument("--notional", type=float, default=100_000,
                   dest="notional",
                   help="USD notional per leg, used by Almgren-Chriss impact term. "
                        "Higher notional → larger impact cost.  Default $100k.")
    return p


def main():
    args = build_parser().parse_args()

    if BAR_TIMEFRAME != "daily":
        raise SystemExit(
            f"BAR_TIMEFRAME = {BAR_TIMEFRAME!r}; this script is for daily mode. "
            "Set BAR_TIMEFRAME='daily' in config.py."
        )

    closes_path = DATA_DIR / args.closes
    closes = pd.read_csv(closes_path, parse_dates=["Date"]).set_index("Date").sort_index()
    pairs_df = pd.read_csv(DATA_DIR / args.pairs)
    pairs  = pairs_df["pair"].tolist()

    vwap_daily = None
    if args.exec_mode == "vwap":
        vp = DATA_DIR / "vwap_daily.csv"
        if not vp.exists():
            raise SystemExit("vwap_daily.csv missing — run build_vwap_daily.py")
        vwap_daily = pd.read_csv(vp, parse_dates=["Date"]).set_index("Date").sort_index()

    # volumes_daily.csv: required by --vwz; if present, used for the stale-print
    # mask (audit T2) and by almgren for the $ ADV.
    volumes = None
    vol_path = DATA_DIR / "volumes_daily.csv"
    if args.vwz:
        if not vol_path.exists():
            raise SystemExit("volumes_daily.csv missing — run build_volumes_daily.py")
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()
    elif vol_path.exists():
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # Headline universe (audit T2).  Every *_USD leg is a local leg converted
    # with a daily FX close (no intraday FX at its close time exists here):
    # fx_unaligned.  The legacy run lists unclassifiable legs instead of stopping.
    excluded = headline_exclusions(pairs, unknown_ok=args.legacy_cross_listed)
    run_pairs = pairs if args.legacy_cross_listed else [p for p in pairs if p not in excluded.index]
    print("HEADLINE UNIVERSE POLICY: excludes pairs with an fx_unaligned leg (*_USD), an ASX/TSE leg "
          "(no NYSE overlap) or a European leg (no synchronous 09:30-11:30 ET snapshots)")
    if args.legacy_cross_listed:
        print(f"  LEGACY (--legacy-cross-listed): all {len(pairs)} pairs run; the "
              f"{len(excluded)} cross-listed pairs' numbers are NOT VALID")
    else:
        print(f"  headline pairs: {len(run_pairs)} / {len(pairs)}  "
              f"({len(excluded)} cross-listed pairs excluded, listed at the end)")

    cfg = WfoConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WfoConfig)})

    # Audit T6: the final holdout.  The use is recorded before anything runs.
    holdout = holdout_violation(closes, cfg)
    if holdout:
        raise SystemExit(holdout)

    # Audit T5: HEADLINE only on a frozen point-in-time universe.
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step, start=cfg.start, end=cfg.end)
    first_oos = windows[0][2] if windows else None
    legacy = [f"--{k.replace('_', '-')}" for k, v in vars(args).items() if k.startswith("legacy_") and v]
    reasons = non_headline_reasons(args.pairs, pairs_df, first_oos, legacy, cfg)
    # Audit T9: refused before anything is recorded (holdout use, trial).
    forbidden = {f"--{k}": why for k, why in HEADLINE_FORBIDDEN.items() if getattr(args, k)}
    if forbidden:
        why = "; ".join(f"{k}: {w}" for k, w in forbidden.items())
        if not reasons:
            raise SystemExit(f"{', '.join(forbidden)} refused in a headline run (audit T9): {why}")
        reasons.append(f"NON-HEADLINE RUN: options without a truncation-invariance test "
                       f"(audit T9): {why}")

    chash = config_hash(cfg, DATA_DIR / args.pairs, args.closes, run_pairs, volumes is not None)
    dev_run = run_last_date(closes, cfg) < HOLDOUT_START
    if cfg.final_holdout:
        record_holdout_use(chash)
        print(f"--final-holdout: recorded in {HOLDOUT_USES_FILE.name}; "
              + (f"the run ends on {run_last_date(closes, cfg).date()}, before the holdout"
                 if dev_run else
                 f"FINAL HOLDOUT RUN: uses dates from {HOLDOUT_START.date()}"))
    digest = hashlib.sha256((DATA_DIR / args.pairs).read_bytes()).hexdigest()
    print(f"UNIVERSE: {DATA_DIR / args.pairs}  SHA-256 {digest}")
    banner = headline_block(reasons, args.pairs, first_oos, cfg)
    print(banner)

    result = run_wfo_daily(closes, run_pairs, cfg, volumes=volumes, vwap=vwap_daily)
    result.stats["headline"] = not reasons
    verdict = f"{banner}\nheadline={result.stats['headline']}"
    deflation = deflate(result, chash, dev_run, legacy)
    if result.trades.empty:
        print("\nNo trades.")
        print("\n" + deflation)
        print("\n" + cross_listed_table(excluded))
        print("\n" + verdict)
        return result
    trades = result.trades

    legacy_pnl = args.legacy_exit_date_pnl
    summarise(trades, label=cost_model(cfg)["label"],         # audit T8 (legacy: NET <cost> bps/leg)
              n_windows=result.stats["n_windows"], n_pairs=len(run_pairs),
              oos_index=result.daily_pnl.index, stats=None if legacy_pnl else result.stats)
    if not legacy_pnl:
        print("\n" + deflation)
    if args.legacy_cross_listed:
        print("\n" + cross_listed_table(excluded, trades))
        cross = trades[trades["pair"].isin(excluded.index)]
        if not cross.empty:
            st = trade_stats(cross, result.stats["n_windows"], len(excluded), result.daily_pnl.index)
            # The exit-date ratio only with the legacy booking, never called a Sharpe (rule 4).
            print(f"  cross-listed block (LEGACY — NOT VALID): {st['n_trades']} trades, "
                  f"win {st['win_rate']:.1%}, total PnL {st['total_pnl']:+.4f}"
                  + (f", LEGACY exit-date ratio (full days) — not a Sharpe {st['sharpe_full']:+.3f}"
                     if legacy_pnl else ""))
    else:
        print("\n" + cross_listed_table(excluded))
    print("\n" + verdict)

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = DATA_DIR / f"wfo_daily_results{suffix}.csv"
    trades.to_csv(out_csv, index=False)
    chart = OUTPUT_DIR / f"wfo_daily_equity{suffix}.png"
    save_equity_chart(trades, chart, None if legacy_pnl else result.daily_returns)
    print(f"\nSaved → {out_csv} ({len(trades)} trades)")
    if not legacy_pnl:
        # Audit T4: the daily MTM series, the downstream scripts' input.
        out_mtm = DATA_DIR / f"wfo_daily_mtm{suffix}.csv"
        pd.DataFrame({"pnl": result.daily_pnl, "ret": result.daily_returns}).rename_axis("Date") \
            .to_csv(out_mtm)
        out_pairs = DATA_DIR / f"wfo_daily_mtm_pairs{suffix}.csv"
        result.daily_pnl_pairs.rename_axis("Date").to_csv(out_pairs)
        print(f"MTM    → {out_mtm} ({len(result.daily_pnl)} business days), per pair → {out_pairs}")
    # Audit T8: the run's description next to its results (read by T9–T11).
    out_meta = DATA_DIR / f"wfo_daily_meta{suffix}.json"
    meta = {
        "cost_model": cost_model(cfg),
        "config": dataclasses.asdict(cfg),
        "config_hash": chash,
        "git_revision": git_revision(),
        "pairs_file": args.pairs,
        "pairs_sha256": digest,
        "closes_file": args.closes,
        "headline": result.stats["headline"],
        "legacy_options": legacy,
        "files": {"trades": out_csv.name,
                  "mtm": None if legacy_pnl else f"wfo_daily_mtm{suffix}.csv",
                  "mtm_pairs": None if legacy_pnl else f"wfo_daily_mtm_pairs{suffix}.csv"},
        # Audit T10 (R44): the contents of the files above (as written) and of
        # the closes file, verified by the downstream scripts (mtm.check_run).
        "sha256": {"trades": file_sha256(out_csv),
                   "mtm": None if legacy_pnl else file_sha256(out_mtm),
                   "mtm_pairs": None if legacy_pnl else file_sha256(out_pairs),
                   "closes": file_sha256(closes_path)},
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Meta   → {out_meta}")
    print(f"Chart  → {chart}")
    return result


if __name__ == "__main__":
    main()
