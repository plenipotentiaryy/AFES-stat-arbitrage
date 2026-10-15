"""backfill_trials.py — one-off back-fill of the trials registry TRIALS.csv
(audit task T6).

The 2015–2026 OOS period was used to search the model, so the configurations
tried on it are trials in the sense of the Deflated Sharpe Ratio
(Bailey & López de Prado, 2014).  This script writes one row per distinct
historical trial found in the repo and the owner's checkout (read-only); the
CLI (step3j_wfo_daily.py) appends one row per later development-period run.

    python backfill_trials.py [--data data] [--runs <data's checkout>/output] [--out TRIALS.csv]

Counting rule (conservative: when unsure whether two runs are one
configuration, they are two trials):
  1. every daily WFO trades file in data/: each wfo_daily_results<tag>.csv
     (incl. the untagged one) and wfo_v2_filter_off.csv — one row each, even
     when two files are byte-identical (different tags = separately evaluated
     configurations; the trades SHA-256 is in the description);
  2. every daily WFO run saved under the owner's output/ directories (final,
     optionb, vwap, sizing: trades files; ablations, costs, vwz, hmm,
     hmm_modes, metagate, hrp, sizing: logs) — one row each; a log is merged
     with a data/ trades file only when it is the same run (same tag written
     at the same minute: run_ablations.sh A–D = base / cusum / kalman /
     kalman_cusum, costs/A_realistic.log = realistic, vwz/A_vwz.log = vwz),
     and the audits' trades_baseline.csv copies (cell-for-cell the trades of
     ios_validation / prod_validation / verify) are not new trials;
  3. every configuration of run_ablations.sh (A–D via rule 2, E = one row),
     run_final_tests.sh (its 9 trades files, rule 2), run_grid_sharpe.sh (4)
     and run_split.sh (2) — the last two are the intraday engine step3j_wfo.py;
  4. the other intraday WFO results in data/: wfo_results.csv and each
     wfo_folds_<tag>.csv (trades_<tag>.csv is the same run as its folds file);
  5. FULL_DAILY_RUN.md: the comparison-table rows and the rescue run are files
     of rule 1 (fullrun, cmp_preADR, cmp_prod, v2_filter_on,
     v2_realistic_costs, v2_failed_direct); each sub-block row at each cost
     level (4 blocks × 2) and the two alternatives evaluated in its caveats
     (risk-parity allocation, dropping the Aussie pairs) are one row each;
  6. every flag the audit plan lists (--kalman --hmm --metagate --cusum --ivol
     --pairmom --rotation --voltarget --aussie-filter) must appear in at least
     one row (checked); no extra row is added for a flag that has one.
  Not counted (N is a lower bound): the intraday step4-style backtests outside
  the WFO scripts (data/trades.csv and the data/trades_<tag>.csv without a
  wfo_folds_<tag>.csv — 19 files —, output/backtest_summary_*.csv — 6 files),
  analyses of an existing trial (bootstrap, Monte Carlo, stress, factor
  attribution), and anything not saved to disk.

sr_daily (per-period, not annualised) is filled only for the daily trades files
(rules 1–2): the Sharpe of the exit-date daily PnL series — how those trials
were judged — restricted to the development period: each trade's pnl summed on
its exit date, on every business day from the first OOS start (min oos_start;
min entry when the file has no oos_start) to 2023-12-31 (zeros on days without
an exit; trades exiting later dropped); ios_validation's OOS ended
2014-10-03 (its last window), so its series ends there; the Almgren-Chriss
repricing file is judged on pnl_adj (the column its script evaluated).  The
intraday runs and the log-only rows have no sr_daily: they count in N, not in
V.  config_hash of a back-filled row = SHA-256 of "<source>\\n<description>"
(unique per row, checked: each row is one configuration in N, which counts
distinct config hashes).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from perf_stats import sharpe_per_period
from step3j_wfo_daily import REPO_ROOT, TRIALS_COLUMNS, TRIALS_FILE
from wfo_daily_core import HOLDOUT_START

DEV_END = HOLDOUT_START - pd.Timedelta(days=1)          # 2023-12-31
PLAN_FLAGS = ["--kalman", "--hmm", "--metagate", "--cusum", "--ivol", "--pairmom",
              "--rotation", "--voltarget", "--aussie-filter"]


def exit_date_sharpe(trades: pd.DataFrame, pnl: str = "pnl", end=None) -> tuple[float, int]:
    """(per-period Sharpe, number of days) of the exit-date daily PnL series on
    the business days from the first OOS start to min(end, 2023-12-31)."""
    start = pd.to_datetime(trades["oos_start"] if "oos_start" in trades.columns
                           else trades["entry"]).min().normalize()
    last = min(pd.Timestamp(end), DEV_END) if end is not None else DEV_END
    exits = pd.to_datetime(trades["exit"]).dt.normalize()
    keep = exits <= last
    daily = trades.loc[keep, pnl].groupby(exits[keep]).sum()
    days = pd.bdate_range(start, last)
    assert daily.index.isin(days).all(), "an exit outside the business days of the span"
    series = daily.reindex(days, fill_value=0.0)
    return sharpe_per_period(series), len(series)


# ── The trials ──────────────────────────────────────────────────────────────

# Notes on the daily trades files in data/ (rule 1), by file name.
DATA_NOTES = {
    "wfo_daily_results.csv": "untagged run (the last run without --tag)",
    "wfo_daily_results_base.csv": "run_ablations.sh A: baseline, pairs_selected.csv (+ output/ablations/A_base.log)",
    "wfo_daily_results_cusum.csv": "run_ablations.sh B: --cusum 30.0 (+ output/ablations/B_cusum.log)",
    "wfo_daily_results_kalman.csv": "run_ablations.sh C: --kalman (+ output/ablations/C_kalman.log)",
    "wfo_daily_results_kalman_cusum.csv": "run_ablations.sh D: --kalman --cusum 30.0 (+ output/ablations/D_kalman_cusum.log)",
    "wfo_daily_results_realistic.csv": "realistic costs, pairs_selected.csv (+ output/costs/A_realistic.log)",
    "wfo_daily_results_vwz.csv": "--vwz, pairs_selected.csv (+ output/vwz/A_vwz.log)",
    "wfo_daily_results_fullrun.csv": "FULL_DAILY_RUN.md headline 'v2 full daily' (36 pairs, Sharpe +3.73)",
    "wfo_daily_results_cmp_preADR.csv": "FULL_DAILY_RUN.md 'Pre-ADR baseline' (14 pairs, 691 trades, +0.73)",
    "wfo_daily_results_cmp_prod.csv": "FULL_DAILY_RUN.md 'Production baseline' (110 pairs, 5614 trades, +1.95)",
    "wfo_daily_results_v2_filter_on.csv": "FULL_DAILY_RUN.md 'v2 + Aussie filter' (--aussie-filter, +3.00)",
    "wfo_daily_results_v2_realistic_costs.csv": "FULL_DAILY_RUN.md 'v2 + 3x costs' (15 bps slip + 200 bps borrow, +2.63)",
    "wfo_daily_results_v2_failed_direct.csv": "FULL_DAILY_RUN.md rescue: style ETF + cross-asset candidates run directly (-0.25)",
    "wfo_v2_filter_off.csv": "v2 run with the Aussie filter off",
    "wfo_daily_results_almgren.csv": "almgren_chriss_repricing.py: fullrun trades repriced (judged on pnl_adj)",
    "wfo_daily_results_bias.csv": "bias_test.py output of its last run (overwritten by every bias_test run)",
    "wfo_daily_results_ios_validation.csv": "in-sample validation 2006-2014 (+ output/ios_validation_audit/trades_baseline.csv, same trades)",
    "wfo_daily_results_prod_validation.csv": "production validation (+ output/prod_validation_audit/trades_baseline.csv, same trades)",
    "wfo_daily_results_verify.csv": "verify run (+ output/baseline/trades_baseline.csv, same trades)",
}
DATA_PNL = {"wfo_daily_results_almgren.csv": "pnl_adj"}
DATA_END = {"wfo_daily_results_ios_validation.csv": "2014-10-03"}

# Daily WFO trades files under the owner's output/ (rule 2), with their config.
RUN_FILES = {
    "final/01_combo_plain.csv": "run_final_tests.sh 01: bias_test --split 2014-12-31 --maxconc 3 --pairmom --ivol",
    "final/02_combo_ticker_overlap.csv": "run_final_tests.sh 02: COMBO + --ticker-overlap",
    "final/03_stress_slip15_borrow200.csv": "run_final_tests.sh 03: COMBO + --slip 15 --borrow 200",
    "final/04_split_2010.csv": "run_final_tests.sh 04: COMBO, split 2010-01-01",
    "final/05_split_2018.csv": "run_final_tests.sh 05: COMBO, split 2018-01-01",
    "final/06_quarter_cap_2.csv": "run_final_tests.sh 06: COMBO + --per-pair-quarter-cap 2",
    "final/07_quarter_cap_4.csv": "run_final_tests.sh 07: COMBO + --per-pair-quarter-cap 4",
    "final/08_vol_tight.csv": "run_final_tests.sh 08: COMBO + --vol-tight-stop",
    "final/09_all_combined.csv": "run_final_tests.sh 09: COMBO + --ticker-overlap --per-pair-quarter-cap 3 --vol-tight-stop",
    "optionb/01_combo.csv": "option B 01: COMBO (maxconc 3)",
    "optionb/02_rotation.csv": "option B 02: COMBO + --rotation",
    "optionb/03_voltarget10.csv": "option B 03: COMBO + --voltarget 0.10",
    "optionb/04_voltarget5.csv": "option B 04: COMBO + --voltarget 0.05",
    "optionb/05_both.csv": "option B 05: COMBO + --rotation --voltarget",
    "vwap/E_close.csv": "vwap A/B: close fills (COMBO)",
    "vwap/E_vwap.csv": "vwap A/B: --vwap-exec (COMBO)",
    "vwap/E_stress_close.csv": "vwap A/B: stressed costs, close fills",
    "vwap/E_stress_vwap.csv": "vwap A/B: stressed costs, --vwap-exec",
    "sizing/E_plain.csv": "sizing A/B: plain (log is a failed attempt with --maxconc 10)",
}

# Log-only daily WFO runs under output/ (rule 2) and run_ablations.sh E (rule 3).
RUN_LOGS = {
    "ablations/E_bias.log": "run_ablations.sh E: bias_test.py --split 2014-12-31",
    "costs/E_bias_realistic.log": "bias_test, realistic costs",
    "costs/E_borrow200.log": "bias_test, --borrow 200",
    "costs/E_stress_hi.log": "bias_test, --slip 15 (stress)",
    "vwz/E_vwz.log": "bias_test --vwz",
    "hmm/E_hmm.log": "bias_test --hmm",
    "hmm_modes/E_block-panic.log": "bias_test --hmm block-panic",
    "hmm_modes/E_block-calm.log": "bias_test --hmm block-calm",
    "hmm_modes/E_size-panic.log": "bias_test --hmm size-panic",
    "hmm_modes/E_extreme-3state.log": "bias_test --hmm extreme-3state",
    "metagate/E_metagate.log": "bias_test --metagate",
    "metagate/E_metagate_binary.log": "bias_test --metagate (binary label)",
    "hrp/E_hrp.log": "bias_test --hrp",
    "hrp/E_ivol.log": "bias_test --ivol",
    "sizing/E_combo.log": "sizing A/B: --ivol --maxconc 3 combo",
}

# Intraday engine (step3j_wfo.py) and FULL_DAILY_RUN.md rows without a daily
# trades file: (source, description, file under output/ dating the run or None).
OTHER = [
    *[("run_grid_sharpe.sh", f"intraday step3j_wfo.py --pairs pairs_baseline14.csv, "
       f"WFO_MIN_TRAIN_SHARPE = {s} (output/grid_sharpe/results_s{s}.csv)",
       f"grid_sharpe/log_s{s}.txt") for s in ("0.1", "0.2", "0.3", "0.4")],
    *[("run_split.sh", f"intraday step3j_wfo.py --pairs pairs_baseline14.csv --start {s} --end {e} "
       f"({h}, output/split/results_{h}.csv)", f"split/log_{h}.txt")
      for s, e, h in (("2006-04-03", "2014-12-31", "H1"), ("2015-01-01", "2026-04-03", "H2"))],
    *[("FULL_DAILY_RUN.md", f"sub-block '{b}' of the v2 run at {c}", None)
      for c in ("5/80 costs", "15/200 costs")
      for b in ("Canadian dual-listing", "European ADR", "Aussie ADR", "Non-cross (US single-listed)")],
    ("FULL_DAILY_RUN.md", "caveat 1: risk-parity allocation cross 0.64 / non-cross 0.36 (+4.75)", None),
    ("FULL_DAILY_RUN.md", "caveat 2: drop the Aussie ADR pairs (34 pairs, about +3.0)", None),
]
FULL_DAILY_RUN_DATE = "2026-05-27"          # the document's date


def mtime(path: Path) -> str:
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def log_config(path: Path) -> str:
    """The run header of a WFO log: pairs, costs, flags, filters applied."""
    keep = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if re.match(r"(pairs|costs?|kalman|hmm mode|vwap-exec):", s) or re.match(r"→ \S+", s):
            s = re.sub(r"\s+", " ", s)
            if s not in keep:
                keep.append(s)
    return "; ".join(keep)


def trades_row(path: Path, source: str, note: str, pnl: str = "pnl", end=None) -> dict:
    t = pd.read_csv(path)
    sr, n = exit_date_sharpe(t, pnl=pnl, end=end)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    start = pd.to_datetime(t["oos_start"] if "oos_start" in t.columns else t["entry"]).min().date()
    desc = (f"{note}; {len(t)} trades, {t['pair'].nunique()} pairs, OOS from {start}; "
            f"trades sha256 {sha[:12]}")
    return {"source": source, "date": mtime(path), "description": desc,
            "n_days": n, "sr_daily": sr}


def collect(data: Path, runs: Path) -> list[dict]:
    rows = []
    # Rule 1: the daily trades files in data/.
    files = sorted(data.glob("wfo_daily_results*.csv")) + [data / "wfo_v2_filter_off.csv"]
    for f in files:
        rows.append(trades_row(f, f"backfill:data/{f.name}", DATA_NOTES.get(f.name, "daily WFO trades file"),
                               pnl=DATA_PNL.get(f.name, "pnl"), end=DATA_END.get(f.name)))
    # Rule 2: output/ trades files and logs.
    for rel, note in RUN_FILES.items():
        rows.append(trades_row(runs / rel, f"backfill:output/{rel}", note))
    for rel, note in RUN_LOGS.items():
        p = runs / rel
        rows.append({"source": f"backfill:output/{rel}", "date": mtime(p),
                     "description": f"{note}; log: {log_config(p)}"})
    # Rules 3-5: intraday WFO runs, FULL_DAILY_RUN.md rows without a file.
    for src, note, rel in OTHER:
        rows.append({"source": f"backfill:{src}", "description": note,
                     "date": mtime(runs / rel) if rel else FULL_DAILY_RUN_DATE})
    intraday = [data / "wfo_results.csv"] + sorted(data.glob("wfo_folds_*.csv"))
    for f in intraday:
        if f.name == "wfo_results.csv":
            note = "intraday step3j_wfo.py results of its last run (overwritten by every run)"
        else:
            tag = f.name[len("wfo_folds_"):-len(".csv")]
            note = f"intraday WFO folds, tag {tag}" + (
                f" (+ data/trades_{tag}.csv)" if (data / f"trades_{tag}.csv").exists() else "")
        rows.append({"source": f"backfill:data/{f.name}", "date": mtime(f),
                     "description": f"{note}; sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:12]}"})
    return rows


def row_hash(row: dict) -> str:
    """config_hash of a back-filled trial: SHA-256 of "<source>\\n<description>"."""
    return hashlib.sha256(f"{row['source']}\n{row['description']}".encode()).hexdigest()


def check(rows: list[dict]) -> None:
    """Rule 6 and the run_*.sh coverage."""
    text = "\n".join(r["description"] for r in rows)
    for flag in PLAN_FLAGS:
        assert flag in text, f"plan flag {flag} has no trial"
    for script, n in (("run_ablations.sh", 5), ("run_final_tests.sh", 9),
                      ("run_grid_sharpe.sh", 4), ("run_split.sh", 2)):
        assert sum(script in r["description"] or script in r["source"] for r in rows) == n, script
    assert len({row_hash(r) for r in rows}) == len(rows), "two back-filled trials share a config hash"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--runs", type=Path, default=None,
                   help="the owner's output/ dir (default: next to the resolved data/ dir)")
    p.add_argument("--out", type=Path, default=TRIALS_FILE)
    args = p.parse_args()
    runs = args.runs or args.data.resolve().parent / "output"
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: the back-fill is one-off (it would drop later rows)")
    rows = collect(args.data, runs)
    check(rows)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(TRIALS_COLUMNS)
        for r in rows:
            sr = r.get("sr_daily", float("nan"))
            w.writerow([row_hash(r), r["date"],
                        r["source"], r["description"], r.get("n_days", ""),
                        "" if math.isnan(sr) else repr(sr),
                        "" if math.isnan(sr) else repr(sr * math.sqrt(252))])
    sr = np.array([r.get("sr_daily", np.nan) for r in rows])
    known = sr[np.isfinite(sr)]
    print(f"wrote {args.out}: N = {len(rows)} trials, {len(known)} with sr_daily, "
          f"V = {np.var(known, ddof=1):.6g} (sd {np.std(known, ddof=1):.4f} per day, "
          f"{np.std(known, ddof=1) * math.sqrt(252):.3f} annualised)")


if __name__ == "__main__":
    main()
