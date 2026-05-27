"""bootstrap_sharpe.py — stationary block-bootstrap CI on portfolio Sharpe.

Method: Politis-Romano (1994) stationary block bootstrap on the daily-PnL
series.  Block lengths are i.i.d. geometric with mean L, which preserves
short-range serial dependence (typical for trade-overlap effects) while
still resampling.  Run N replicates, compute Sharpe for each, report
percentiles.

Usage:
    python bootstrap_sharpe.py --trades data/wfo_daily_results_bias.csv \\
        --split 2014-12-31 --reps 10000 --block 5
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def stationary_block_bootstrap(x: np.ndarray, mean_block_len: float,
                                 n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n_reps, len(x)) matrix of resampled series."""
    n = len(x)
    p = 1.0 / mean_block_len           # block continuation prob = 1 - 1/L
    out = np.empty((n_reps, n), dtype=np.float64)
    for rep in range(n_reps):
        # Generate indices via Politis-Romano: start at random pos, with
        # prob (1-p) advance by 1, with prob p jump to new random pos.
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        coin = rng.random(n)
        for t in range(1, n):
            if coin[t] < p:
                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        out[rep] = x[idx]
    return out


def daily_pnl_series(trades_csv: str, split: str, closes_csv: str) -> pd.Series:
    closes = pd.read_csv(closes_csv, parse_dates=["Date"], usecols=["Date"])
    oos = pd.DatetimeIndex(closes[closes["Date"] >= split]["Date"]).normalize()
    tr = pd.read_csv(trades_csv, parse_dates=["entry", "exit"])
    daily = tr.groupby(tr["exit"].dt.normalize())["pnl"].sum()
    return daily.reindex(oos, fill_value=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=str, required=True)
    p.add_argument("--closes", type=str, default="data/closes_daily.csv")
    p.add_argument("--split",  type=str, default="2014-12-31")
    p.add_argument("--reps",   type=int, default=10_000)
    p.add_argument("--block",  type=float, default=5.0,
                   help="Mean block length in trading days (default 5).")
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    series = daily_pnl_series(args.trades, args.split, args.closes)
    x = series.values
    n = len(x)
    print(f"Bootstrap on {args.trades}")
    print(f"  OOS daily series: {n} days, sum PnL = {x.sum():+.3f}")
    print(f"  point Sharpe    : {x.mean() / x.std() * np.sqrt(252):+.3f}")
    print(f"  block bootstrap : reps={args.reps}, mean_block={args.block} days")
    print()

    rng = np.random.default_rng(args.seed)

    # Vectorise for speed: sample reps in batches of 1000.
    batch = 1000
    sharpes = np.empty(args.reps, dtype=np.float64)
    for start in range(0, args.reps, batch):
        end = min(start + batch, args.reps)
        sims = stationary_block_bootstrap(x, args.block, end - start, rng)
        sh = sims.mean(axis=1) / sims.std(axis=1) * np.sqrt(252)
        sharpes[start:end] = sh

    sharpes = sharpes[np.isfinite(sharpes)]
    pcts = np.percentile(sharpes, [2.5, 5, 25, 50, 75, 95, 97.5])
    print(f"  bootstrap Sharpe distribution ({len(sharpes)} reps):")
    print(f"    mean   = {sharpes.mean():+.3f}")
    print(f"    median = {pcts[3]:+.3f}")
    print(f"    std    = {sharpes.std():+.3f}")
    print()
    print(f"    p(Sharpe < 0) = {(sharpes < 0).mean():.3f}")
    print(f"    p(Sharpe > 0.5) = {(sharpes > 0.5).mean():.3f}")
    print(f"    p(Sharpe > 1.0) = {(sharpes > 1.0).mean():.3f}")
    print()
    print(f"    {'percentile':>12}  {'Sharpe':>8}")
    for q, v in zip([2.5, 5, 25, 50, 75, 95, 97.5], pcts):
        print(f"    {q:>10.1f}%   {v:>+8.3f}")
    print()
    print(f"  95% CI: [{pcts[0]:+.3f}, {pcts[-1]:+.3f}]")
    print(f"  90% CI: [{pcts[1]:+.3f}, {pcts[-2]:+.3f}]")
    print(f"  50% CI: [{pcts[2]:+.3f}, {pcts[-3]:+.3f}]")


if __name__ == "__main__":
    main()
