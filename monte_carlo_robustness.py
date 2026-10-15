"""Three Monte Carlo robustness tests for AFES portfolio ledgers.

The goal is not to make the backtest look better.  The goal is to stress the
observed edge under three different uncertainty models:

1. Stationary block bootstrap on daily PnL:
   preserves short-range dependence and overlapping-trade clustering.

2. Trade-order permutation:
   keeps the exact realised trade PnLs but randomises their order, exposing
   sequence-of-returns risk and drawdown sensitivity.

3. EVT / GPD tail-injection Monte Carlo:
   fits a Peaks-over-Threshold model to left-tail daily losses and randomly
   replaces a fraction of simulated days with GPD-sampled tail losses.  This
   avoids Gaussian assumptions and asks how the system behaves if the realised
   future contains more tail days than the historical sample.

Series (audit T10): the run's daily mark-to-market PnL (wfo_daily_mtm<tag>.csv,
written by step3j_wfo_daily.py; audit T4), checked against the run's meta
sidecar; --legacy-exit-date-pnl: the pre-audit exit-date booking, labelled
LEGACY (its ratio is not a Sharpe).  The trade permutation reorders trades,
not days: it reports drawdowns and PnL of the trade sequence, no Sharpe (a
per-trade ratio is not one, and a permutation leaves it unchanged).  The
resampling tests measure sampling uncertainty only — they cannot detect a
bias of the series they resample.

Outputs:
    output/monte_carlo_robustness_report.json
    data/monte_carlo_robustness_sims.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mtm import (LEGACY_LABEL, LEGACY_RATIO, SAMPLING_LABEL, add_series_args, check_dates, check_from,
                 check_run, meta_path, mtm_pnl, run_meta, span_line)
from perf_stats import portfolio_sharpe_or_nan


def max_drawdown(x: np.ndarray) -> float:
    eq = np.cumsum(np.asarray(x, dtype=float))
    return float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0


def profit_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def daily_pnl_series(trades_csv: str, closes_csv: str | None, split: str | None) -> pd.Series:
    trades = pd.read_csv(trades_csv, parse_dates=["entry", "exit"])
    if "pnl" not in trades.columns:
        raise ValueError(f"{trades_csv} must contain a 'pnl' column")
    start = trades["exit"].min().normalize()
    end = trades["exit"].max().normalize()
    if closes_csv:
        closes = pd.read_csv(closes_csv, parse_dates=["Date"], usecols=["Date"])
        idx = pd.DatetimeIndex(closes["Date"]).normalize()
        if split:
            idx = idx[idx >= pd.Timestamp(split)]
        idx = idx[(idx >= start) & (idx <= end)]
    else:
        idx = pd.date_range(start, end, freq="B")
    daily = trades.groupby(trades["exit"].dt.normalize())["pnl"].sum()
    return daily.reindex(idx, fill_value=0.0).sort_index()


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / float(mean_block)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    jumps = rng.random(n)
    for i in range(1, n):
        if jumps[i] < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def summarize_sample(x: np.ndarray) -> dict:
    return {
        "pnl": float(np.sum(x)),
        "sharpe": portfolio_sharpe_or_nan(x),      # the one Sharpe; NaN if undefined
        "max_drawdown": max_drawdown(x),
        "profit_factor": profit_factor(x),
        "worst_day": float(np.min(x)) if len(x) else 0.0,
        "var_1pct": float(np.quantile(x, 0.01)) if len(x) else 0.0,
        "cvar_1pct": float(np.mean(x[x <= np.quantile(x, 0.01)])) if len(x) else 0.0,
    }


def mc_stationary_block(daily: np.ndarray, reps: int, block: float,
                        rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = len(daily)
    for rep in range(reps):
        sample = daily[stationary_indices(n, block, rng)]
        rows.append({"mc_type": "stationary_block", "rep": rep, **summarize_sample(sample)})
    return pd.DataFrame(rows)


def mc_trade_permutation(trade_pnls: np.ndarray, reps: int,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Trade sequences, not days: no Sharpe (rule 4), NaN in its column."""
    rows = []
    n = len(trade_pnls)
    for rep in range(reps):
        sample = trade_pnls[rng.permutation(n)]
        rows.append({"mc_type": "trade_permutation", "rep": rep,
                     **summarize_sample(sample), "sharpe": float("nan")})
    return pd.DataFrame(rows)


def fit_left_tail_gpd(daily: np.ndarray, quantile: float) -> dict:
    losses = -np.asarray(daily, dtype=float)
    losses = losses[np.isfinite(losses)]
    threshold = float(np.quantile(losses, quantile))
    exceed = losses[losses > threshold] - threshold
    exceed = exceed[np.isfinite(exceed) & (exceed > 0)]
    if len(exceed) < 20:
        raise ValueError(f"Need at least 20 left-tail exceedances, got {len(exceed)}")
    xi, _, beta = stats.genpareto.fit(exceed, floc=0.0)
    xi = max(float(xi), 1e-6)
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Invalid GPD beta: {beta}")
    return {
        "threshold": threshold,
        "xi": xi,
        "beta": beta,
        "n_exceedances": int(len(exceed)),
        "tail_frequency": float(len(exceed) / len(losses)),
    }


def mc_evt_tail_injection(
    daily: np.ndarray,
    reps: int,
    block: float,
    tail_quantile: float,
    tail_frequency_mult: float,
    tail_loss_mult: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    fit = fit_left_tail_gpd(daily, tail_quantile)
    n = len(daily)
    base_tail_prob = fit["tail_frequency"]
    sim_tail_prob = min(0.50, base_tail_prob * tail_frequency_mult)
    rows = []
    for rep in range(reps):
        sample = daily[stationary_indices(n, block, rng)].copy()
        inject = rng.random(n) < sim_tail_prob
        if inject.any():
            exceed = stats.genpareto.rvs(
                c=fit["xi"],
                loc=0.0,
                scale=fit["beta"],
                size=int(inject.sum()),
                random_state=rng,
            )
            losses = (fit["threshold"] + exceed) * tail_loss_mult
            sample[inject] = -losses
        rows.append({"mc_type": "evt_tail_injection", "rep": rep, **summarize_sample(sample)})
    fit["sim_tail_probability"] = float(sim_tail_prob)
    fit["tail_frequency_mult"] = float(tail_frequency_mult)
    fit["tail_loss_mult"] = float(tail_loss_mult)
    return pd.DataFrame(rows), fit


def distribution_summary(df: pd.DataFrame, metric: str) -> dict:
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AFES 3-way Monte Carlo robustness.")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--closes", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=5.0)
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--tail-frequency-mult", type=float, default=2.0)
    parser.add_argument("--tail-loss-mult", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="output/monte_carlo_robustness_report.json")
    parser.add_argument("--csv-out", default="data/monte_carlo_robustness_sims.csv")
    add_series_args(parser)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    legacy = args.legacy_exit_date_pnl
    if legacy:
        meta = check_run(args.trades, args.meta, closes=args.closes)
        daily = daily_pnl_series(args.trades, args.closes, None)   # first → last exit
        check_from(daily.index, args.split, "--split")              # the whole run (R46)
        check_dates(daily.index, meta, args.meta or meta_path(args.trades))
        print(f"series: {LEGACY_LABEL}")
        print(span_line(daily.index, "closes days" if args.closes else "business days"))
    else:
        daily = mtm_pnl(args)
        check_from(daily.index, args.split, "--split")              # the whole run (R46)
    sh_name = LEGACY_RATIO if legacy else "Sharpe"
    trades = pd.read_csv(args.trades)
    trade_pnls = pd.to_numeric(trades["pnl"], errors="coerce").dropna().to_numpy(dtype=float)

    observed = summarize_sample(daily.to_numpy(dtype=float))
    observed["n_days"] = int(len(daily))
    observed["n_trades"] = int(len(trade_pnls))

    print(f"AFES MONTE CARLO ROBUSTNESS — {SAMPLING_LABEL}")
    print(f"  trades:       {args.trades}")
    print(f"  days/trades:  {observed['n_days']} / {observed['n_trades']}")
    print(f"  observed PnL: {observed['pnl']:+.4f}")
    print(f"  observed {sh_name}: {observed['sharpe']:+.3f}")
    print(f"  observed DD:  {observed['max_drawdown']:+.4f}")
    print()

    block_df = mc_stationary_block(daily.to_numpy(dtype=float), args.reps, args.block, rng)
    perm_df = mc_trade_permutation(trade_pnls, args.reps, rng)
    evt_df, evt_fit = mc_evt_tail_injection(
        daily.to_numpy(dtype=float),
        args.reps,
        args.block,
        args.tail_quantile,
        args.tail_frequency_mult,
        args.tail_loss_mult,
        rng,
    )
    sims = pd.concat([block_df, perm_df, evt_df], ignore_index=True)

    if legacy:            # the exit-date ratio is not a Sharpe (rule 4)
        observed["legacy_exit_date_ratio"] = observed.pop("sharpe")
    meta = run_meta(args)
    summary = {"label": SAMPLING_LABEL,
               "series": "LEGACY exit-date booking — not a Sharpe" if legacy else "daily mark-to-market",
               "headline": meta.get("headline") is True,
               "observed": observed, "evt_fit": evt_fit, "mc": {}}
    sh_key = "legacy_exit_date_ratio" if legacy else "sharpe"
    for mc_type, grp in sims.groupby("mc_type"):
        daily_series = mc_type != "trade_permutation"
        metrics = ["pnl", *(["sharpe"] if daily_series else []), "max_drawdown", "profit_factor",
                   "worst_day", "cvar_1pct"]
        summary["mc"][mc_type] = {
            (sh_key if metric == "sharpe" else metric): distribution_summary(grp, metric)
            for metric in metrics
        }
        if daily_series:
            summary["mc"][mc_type][f"prob_{sh_key}_lt_0"] = float((grp["sharpe"] < 0).mean())
            summary["mc"][mc_type][f"prob_{sh_key}_lt_1"] = float((grp["sharpe"] < 1.0).mean())
        summary["mc"][mc_type]["prob_pnl_lt_0"] = float((grp["pnl"] < 0).mean())
        summary["mc"][mc_type]["prob_dd_worse_than_observed"] = float(
            (grp["max_drawdown"] < observed["max_drawdown"]).mean()
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    sims.rename(columns={"sharpe": sh_key}).to_csv(args.csv_out, index=False)

    for mc_type in ["stationary_block", "trade_permutation", "evt_tail_injection"]:
        grp = sims[sims["mc_type"].eq(mc_type)]
        dd = distribution_summary(grp, "max_drawdown")
        pnl = distribution_summary(grp, "pnl")
        print(mc_type)
        if mc_type == "trade_permutation":
            print("  ratio: none — trade sequences, not days (a per-trade ratio is not a Sharpe)")
        else:
            sh = distribution_summary(grp, "sharpe")
            print(f"  {sh_name} p05/p50/p95: {sh['p05']:+.3f} / {sh['p50']:+.3f} / {sh['p95']:+.3f}")
        print(f"  PnL    p05/p50/p95: {pnl['p05']:+.3f} / {pnl['p50']:+.3f} / {pnl['p95']:+.3f}")
        print(f"  DD     p05/p50/p95: {dd['p05']:+.3f} / {dd['p50']:+.3f} / {dd['p95']:+.3f}"
              + ("  (trade order)" if mc_type == "trade_permutation" else ""))
        if mc_type != "trade_permutation":
            print(f"  P({'ratio' if legacy else 'Sharpe'}<1): "
                  f"{summary['mc'][mc_type][f'prob_{sh_key}_lt_1']:.1%}", end="  ")
        print(f"P(PnL<0): {summary['mc'][mc_type]['prob_pnl_lt_0']:.1%}")
        print()

    print("EVT fit")
    print(f"  threshold loss: {evt_fit['threshold']:.4f}")
    print(f"  xi/beta:        {evt_fit['xi']:.4f} / {evt_fit['beta']:.4f}")
    print(f"  tail prob:      {evt_fit['tail_frequency']:.2%} -> {evt_fit['sim_tail_probability']:.2%}")
    print(f"\nSaved -> {args.out}")
    print(f"CSV   -> {args.csv_out}")


if __name__ == "__main__":
    main()
