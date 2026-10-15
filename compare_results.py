"""compare_results.py — A/B diff two trade ledgers with stat tests (§10).

Computes side-by-side portfolio metrics for two daily WFO runs, and runs a
stationary block-bootstrap on the difference of their daily Sharpe ratios to
test whether the candidate beats the baseline at p < 5% (sampling
uncertainty only — it cannot detect a bias both runs share).

Series (audit T10): each run's daily mark-to-market file (wfo_daily_mtm<tag>.csv
next to its trades file; audit T4), checked against its meta sidecar; Sharpe
and MaxDD (% of capital) are the one definition (`perf_stats.portfolio_sharpe`,
`mtm.mtm_stats`) on each run's own span, the paired bootstrap runs on the union
of the two spans (zeros where a run has no day).  --legacy-exit-date-pnl: the
pre-audit exit-date booking up to each run's end, labelled LEGACY (its ratio is
not a Sharpe, its DD is in trade order).

CLI:
    python compare_results.py --baseline output/baseline/trades_baseline.csv \\
                                --candidate data/wfo_daily_results_bias.csv \\
                                --split 2014-12-31
"""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mtm import LEGACY_RATIO, SAMPLING_LABEL, check_from, legacy_exit_date_pnl, load_mtm, mtm_stats
from perf_stats import portfolio_sharpe, portfolio_sharpe_or_nan as sharpe_or_nan


def metrics(daily: pd.Series, trades: pd.DataFrame) -> dict:
    """LEGACY: exit-date series, trade-order DD; the ratio is not a Sharpe."""
    pnl = trades["pnl"]
    eq = pnl.cumsum()
    dd = float((eq - eq.cummax()).min())
    sd = daily.std()
    sr = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades":   len(trades),
        "win_rate": (pnl > 0).mean(),
        "PnL":      pnl.sum(),
        "DD_trades": dd,
        "PF":       wins / losses if losses > 0 else float("inf"),
        "exit_ratio": sr,
        "DD/PnL":   abs(dd) / pnl.sum() if pnl.sum() != 0 else float("nan"),
        "real_vol": sd * np.sqrt(252),
    }


def mtm_metrics(ret: pd.Series, trades: pd.DataFrame) -> dict:
    """The run's daily MTM returns (its own span): the one Sharpe, the
    date-ordered MaxDD and the annualised vol in % of capital; trade counts
    and PnL from the trades."""
    pnl = trades["pnl"]
    st = mtm_stats(ret, 1.0)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades":   len(trades),
        "win_rate": (pnl > 0).mean(),
        "PnL":      pnl.sum(),
        "MaxDD%":   st["max_dd"] * 100,
        "PF":       wins / losses if losses > 0 else float("inf"),
        "Sharpe":   portfolio_sharpe(ret),
        "ann_vol%": st["ann_vol"] * 100,
    }


def block_bootstrap_sharpe_diff(daily_a: pd.Series, daily_b: pd.Series,
                                  reps: int = 2000, block: float = 5.0,
                                  seed: int = 42) -> tuple[float, float, float]:
    """Return (mean diff, 2.5%, 97.5%) for SR_candidate − SR_baseline."""
    rng = np.random.default_rng(seed)
    a = daily_a.values
    b = daily_b.values
    n = len(a)
    p = 1.0 / block
    diffs = np.empty(reps)
    for r in range(reps):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        coin = rng.random(n)
        for t in range(1, n):
            idx[t] = rng.integers(0, n) if coin[t] < p else (idx[t-1] + 1) % n
        diffs[r] = sharpe_or_nan(b[idx]) - sharpe_or_nan(a[idx])
    return (float(np.nanmean(diffs)), float(np.nanpercentile(diffs, 2.5)),
            float(np.nanpercentile(diffs, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",  required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--closes",    default="data/closes_daily.csv")
    p.add_argument("--split",     default="2014-12-31")
    p.add_argument("--reps",      type=int, default=2000)
    p.add_argument("--legacy-exit-date-pnl", action="store_true", dest="legacy_exit_date_pnl",
                   help="LEGACY: book each trade's PnL on its exit date (the pre-audit series) "
                        "instead of reading the runs' MTM files. Comparison only.")
    args = p.parse_args()
    legacy = args.legacy_exit_date_pnl

    base = pd.read_csv(args.baseline, parse_dates=["entry", "exit"])
    cand = pd.read_csv(args.candidate, parse_dates=["entry", "exit"])

    def run(trades_path):
        return SimpleNamespace(trades=trades_path, mtm=None, meta=None)

    if legacy:
        d_base = legacy_exit_date_pnl(run(args.baseline), args.closes, start=args.split)
        d_cand = legacy_exit_date_pnl(run(args.candidate), args.closes, start=args.split)
    else:
        d_base = load_mtm(run(args.baseline))["ret"]
        d_cand = load_mtm(run(args.candidate))["ret"]
        for d in (d_base, d_cand):
            check_from(d.index, args.split, "--split")   # each whole run (R46)
        m_base, m_cand = mtm_metrics(d_base, base), mtm_metrics(d_cand, cand)
    oos = d_base.index.union(d_cand.index)
    d_base, d_cand = d_base.reindex(oos, fill_value=0.0), d_cand.reindex(oos, fill_value=0.0)
    if legacy:
        m_base, m_cand = metrics(d_base, base), metrics(d_cand, cand)

    print(f"COMPARE  baseline={Path(args.baseline).name}  "
          f"vs  candidate={Path(args.candidate).name}")
    print(f"OOS span: {oos[0].date()} → {oos[-1].date()}  "
          f"({len(oos)} {'closes' if legacy else 'business'} days, union of the two runs)\n")
    if legacy:
        print(f"exit_ratio = {LEGACY_RATIO}; DD_trades = drawdown in trade order\n")

    print(f"{'metric':<12} {'baseline':>14} {'candidate':>14} {'Δ':>14}")
    print("-" * 60)
    for k in m_base:
        bv, cv = m_base[k], m_cand[k]
        try:
            diff = cv - bv
            print(f"{k:<12} {bv:>+14.4f} {cv:>+14.4f} {diff:>+14.4f}")
        except (TypeError, ValueError):
            print(f"{k:<12} {bv:>14} {cv:>14} {'-':>14}")

    stat = "exit_ratio" if legacy else "Sharpe"
    print(f"\nBlock-bootstrap {stat}-difference test (reps={args.reps}, block=5d) — {SAMPLING_LABEL}:")
    mean_d, lo, hi = block_bootstrap_sharpe_diff(d_base, d_cand, reps=args.reps)
    print(f"  mean Δ{stat}        = {mean_d:+.3f}")
    print(f"  95% CI of Δ{stat}   = [{lo:+.3f}, {hi:+.3f}]")
    significant = lo > 0 or hi < 0
    if significant:
        sign = "candidate BEATS baseline" if lo > 0 else "candidate WORSE than baseline"
        print(f"  → Statistically significant: {sign} (95% CI excludes 0).")
    else:
        print(f"  → No significant difference at 95% (CI spans 0).")


if __name__ == "__main__":
    main()
