"""compare_results.py — A/B diff two trade ledgers with stat tests (§10).

Computes side-by-side portfolio metrics for two trade CSVs, and runs a
stationary block-bootstrap on the difference in daily PnL to test whether
the candidate beats the baseline at p < 5%.

CLI:
    python compare_results.py --baseline output/baseline/trades_baseline.csv \\
                                --candidate data/wfo_daily_results_bias.csv \\
                                --split 2014-12-31
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def daily_pnl(trades: pd.DataFrame, oos_index: pd.DatetimeIndex) -> pd.Series:
    d = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    return d.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)


def metrics(daily: pd.Series, trades: pd.DataFrame) -> dict:
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
        "DD":       dd,
        "PF":       wins / losses if losses > 0 else float("inf"),
        "Sharpe":   sr,
        "DD/PnL":   abs(dd) / pnl.sum() if pnl.sum() != 0 else float("nan"),
        "real_vol": sd * np.sqrt(252),
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
        a_s = a[idx]
        b_s = b[idx]
        sr_a = a_s.mean() / a_s.std() * np.sqrt(252) if a_s.std() > 0 else 0
        sr_b = b_s.mean() / b_s.std() * np.sqrt(252) if b_s.std() > 0 else 0
        diffs[r] = sr_b - sr_a
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",  required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--closes",    default="data/closes_daily.csv")
    p.add_argument("--split",     default="2014-12-31")
    p.add_argument("--reps",      type=int, default=2000)
    args = p.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"], usecols=["Date"])
    oos = pd.DatetimeIndex(closes[closes["Date"] >= args.split]["Date"]).normalize()

    base = pd.read_csv(args.baseline, parse_dates=["entry", "exit"])
    cand = pd.read_csv(args.candidate, parse_dates=["entry", "exit"])

    d_base = daily_pnl(base, oos)
    d_cand = daily_pnl(cand, oos)

    m_base = metrics(d_base, base)
    m_cand = metrics(d_cand, cand)

    print(f"COMPARE  baseline={Path(args.baseline).name}  "
          f"vs  candidate={Path(args.candidate).name}")
    print(f"OOS span: {oos[0].date()} → {oos[-1].date()}  "
          f"({len(oos)} trading days)\n")

    print(f"{'metric':<12} {'baseline':>14} {'candidate':>14} {'Δ':>14}")
    print("-" * 60)
    for k in ["trades", "win_rate", "PnL", "DD", "PF", "Sharpe",
              "DD/PnL", "real_vol"]:
        bv, cv = m_base[k], m_cand[k]
        try:
            diff = cv - bv
            print(f"{k:<12} {bv:>+14.4f} {cv:>+14.4f} {diff:>+14.4f}")
        except (TypeError, ValueError):
            print(f"{k:<12} {bv:>14} {cv:>14} {'-':>14}")

    print(f"\nBlock-bootstrap Sharpe-difference test (reps={args.reps}, block=5d):")
    mean_d, lo, hi = block_bootstrap_sharpe_diff(d_base, d_cand, reps=args.reps)
    print(f"  mean ΔSharpe        = {mean_d:+.3f}")
    print(f"  95% CI of ΔSharpe   = [{lo:+.3f}, {hi:+.3f}]")
    significant = lo > 0 or hi < 0
    if significant:
        sign = "candidate BEATS baseline" if lo > 0 else "candidate WORSE than baseline"
        print(f"  → Statistically significant: {sign} (95% CI excludes 0).")
    else:
        print(f"  → No significant difference at 95% (CI spans 0).")


if __name__ == "__main__":
    main()
