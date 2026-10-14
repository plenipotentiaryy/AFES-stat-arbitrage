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
CLI: it reads data/, builds the `WfoConfig`, prints the summary and writes
the trades CSV and the equity chart.

CLI:
    python step3j_wfo_daily.py --pairs pairs_baseline14.csv
    python step3j_wfo_daily.py --pairs pairs_selected.csv --cost 10
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, BAR_TIMEFRAME
from wfo_daily_core import (
    TRAIN_INIT_MO, OOS_MO, STEP_MO,
    WfoConfig, run_wfo_daily, summarise,
)


def save_equity_chart(tr: pd.DataFrame, path: Path):
    if tr.empty:
        return
    eq = tr.sort_values("exit")["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr.sort_values("exit")["exit"].values, eq.values)
    ax.set_title("Daily WFO — cumulative log-spread PnL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily Walk-Forward Optimization")
    p.add_argument("--pairs",  type=str, default="pairs_selected.csv",
                   help="CSV in data/ with a 'pair' column (e.g. AMT-AWK).")
    p.add_argument("--start",  type=str, default=None)
    p.add_argument("--end",    type=str, default=None)
    p.add_argument("--cost",   type=float, default=5.0,
                   help="Commission + half-spread, bps per leg (default 5).")
    p.add_argument("--slip",   type=float, default=5.0,
                   help="Slippage, bps per leg (default 5).")
    p.add_argument("--borrow", type=float, default=80.0,
                   help="Borrow rate on short leg, bps/year (default 80).")
    p.add_argument("--train",  type=int, default=TRAIN_INIT_MO,
                   help=f"Initial train length in months (default {TRAIN_INIT_MO}).")
    p.add_argument("--oos",    type=int, default=OOS_MO,
                   help=f"OOS length in months (default {OOS_MO}).")
    p.add_argument("--step",   type=int, default=STEP_MO,
                   help=f"Walk-forward step in months (default {STEP_MO}).")
    p.add_argument("--kalman", action="store_true",
                   help="Use Kalman dynamic β (innovations as spread, e/√S as z).")
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
                        "normalised against window mean. Pro-cyclical.")
    p.add_argument("--pairmom", action="store_true",
                   help="Rolling-Sharpe pair momentum: hot pair → up-weight, cold → down. "
                        "Causal: only uses pair's executed OOS trades so far.")
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
                        "on close). vwap requires data/vwap_daily.csv.")
    p.add_argument("--vwap-exec", action="store_const", const="vwap", dest="exec_mode",
                   help="Alias of --exec vwap.")
    p.add_argument("--legacy-same-close", action="store_true", dest="legacy_same_close",
                   help="LEGACY: restore the pre-audit fills at the close that made "
                        "the signal (impossible fills). For before/after comparison "
                        "only; never use for headline results.")
    p.add_argument("--rotation", action="store_true",
                   help="Quarterly per-pair S_perf rescaling (§5): trades sized by "
                        "rolling pair-Sharpe with Bayesian shrinkage. Drop pairs "
                        "with SR_6m < 0 for the current quarter.")
    p.add_argument("--voltarget", type=float, default=0.0,
                   help="Portfolio vol target (annualised, e.g. 0.10 = 10%%). "
                        "Applied as causal leverage scaling. 0 = off.")
    p.add_argument("--closes",   type=str, default="closes_daily.csv",
                   help="Closes file in data/ (extended universe support).")
    p.add_argument("--hmm",    type=str, default="off",
                   choices=["off", "block-panic", "block-calm",
                            "size-panic", "extreme-3state"],
                   help="HMM mode: off / block-panic / block-calm (invert) / "
                        "size-panic (50%% size in panic) / extreme-3state "
                        "(3-state, block only highest-vol).")
    p.add_argument("--cusum",  type=float, default=None,
                   help="CUSUM break threshold h (e.g. 5.0). If unset, CUSUM off.")
    p.add_argument("--tag",    type=str, default="",
                   help="Suffix for output filenames (e.g. 'kalman_cusum').")
    p.add_argument("--aussie-filter", action="store_true", dest="aussie_filter",
                   help="Conditional gate for Aussie/JP ADR pairs: entry_z>=2.5, "
                        "blackout when 5d realized spread vol > 95th-pct of trailing 252d, "
                        "require rolling 60d AR(1) half-life < 5 days.")
    p.add_argument("--execution-model", type=str, default="almgren",
                   choices=["flat", "almgren"], dest="execution_model",
                   help="flat = commission+slip+borrow scalars (--cost --slip --borrow). "
                        "almgren = per-pair tiered cost: half_spread + κ·σ·√(notional/ADV), "
                        "plus tier-specific borrow on short leg.  Default: almgren.")
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
    pairs  = pd.read_csv(DATA_DIR / args.pairs)["pair"].tolist()

    vwap_daily = None
    if args.exec_mode == "vwap":
        vp = DATA_DIR / "vwap_daily.csv"
        if not vp.exists():
            raise SystemExit("vwap_daily.csv missing — run build_vwap_daily.py")
        vwap_daily = pd.read_csv(vp, parse_dates=["Date"]).set_index("Date").sort_index()

    # volumes_daily.csv: required by --vwz; used by almgren for the $ ADV if present.
    volumes = None
    vol_path = DATA_DIR / "volumes_daily.csv"
    if args.vwz:
        if not vol_path.exists():
            raise SystemExit("volumes_daily.csv missing — run build_volumes_daily.py")
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()
    elif args.execution_model == "almgren" and vol_path.exists():
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()

    cfg = WfoConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WfoConfig)})
    result = run_wfo_daily(closes, pairs, cfg, volumes=volumes, vwap=vwap_daily)
    if result.trades.empty:
        print("\nNo trades.")
        return
    trades = result.trades

    summarise(trades, label=f"NET {args.cost} bps/leg",
              n_windows=result.stats["n_windows"], n_pairs=len(pairs),
              oos_index=result.daily_pnl.index)

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = DATA_DIR / f"wfo_daily_results{suffix}.csv"
    trades.to_csv(out_csv, index=False)
    chart = OUTPUT_DIR / f"wfo_daily_equity{suffix}.png"
    save_equity_chart(trades, chart)
    print(f"\nSaved → {out_csv} ({len(trades)} trades)")
    print(f"Chart  → {chart}")


if __name__ == "__main__":
    main()
