"""audit_baseline.py — Build the formal baseline artifact bundle (§1, §11).

Reads a frozen trade ledger and produces the artifact directory required by
the spec.  Also runs anti-bias / data-integrity checks:

    - no pre-OOS entries (entry >= oos_start)
    - no train/test temporal overlap with declared split
    - no duplicate trades (same pair + same entry timestamp)
    - no impossible PnL outliers (|pnl| > 99.9th percentile × 10)
    - exit always after entry
    - no missing timestamps inside the trade rows

Output layout (under --out, default output/baseline/):

    trades_baseline.csv
    pair_summary_baseline.csv
    equity_curve_baseline.csv
    portfolio_summary_baseline.json
    audit_report.md

CLI:
    python audit_baseline.py --trades data/wfo_daily_results_bias.csv \\
                              --split 2014-12-31 --out output/baseline/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def portfolio_metrics(trades: pd.DataFrame, oos_index: pd.DatetimeIndex) -> dict:
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = pnl.cumsum()
    dd = float((eq - eq.cummax()).min())

    daily = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    daily_full = daily.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)
    mu = daily_full.mean()
    sd = daily_full.std()
    sharpe = float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan")
    # Sortino: only downside std
    downside = daily_full[daily_full < 0]
    sortino = (float(mu / downside.std() * np.sqrt(252))
                if not downside.empty and downside.std() > 0 else float("nan"))
    annual_pnl = mu * 252
    calmar = float(annual_pnl / abs(dd)) if dd != 0 else float("nan")

    return {
        "trades":          int(len(trades)),
        "win_rate":        float((pnl > 0).mean()),
        "total_pnl":       float(pnl.sum()),
        "avg_pnl":         float(pnl.mean()),
        "profit_factor":   float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown":    dd,
        "dd_over_pnl":     float(abs(dd) / pnl.sum()) if pnl.sum() != 0 else float("nan"),
        "sharpe_full":     sharpe,
        "sortino":         sortino,
        "calmar":          calmar,
        "avg_holding_bars": float(trades["held"].mean()),
        "median_holding":   float(trades["held"].median()),
        "best_trade":       float(pnl.max()),
        "worst_trade":      float(pnl.min()),
        "n_active_pairs":   int(trades["pair"].nunique()) if "pair" in trades else None,
    }


def per_pair_summary(trades: pd.DataFrame, oos_index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for pair, grp in trades.groupby("pair"):
        pnl = grp["pnl"]
        eq  = pnl.cumsum()
        dd  = float((eq - eq.cummax()).min())
        daily = grp.groupby(pd.to_datetime(grp["exit"]).dt.normalize())["pnl"].sum()
        daily_full = daily.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)
        sr = (daily_full.mean() / daily_full.std() * np.sqrt(252)
               if daily_full.std() > 0 else 0.0)
        rows.append({
            "pair":      pair,
            "trades":    len(grp),
            "win_rate":  (pnl > 0).mean(),
            "total_pnl": pnl.sum(),
            "max_dd":    dd,
            "sharpe":    sr,
            "avg_hold":  grp["held"].mean(),
        })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def equity_curve(trades: pd.DataFrame, oos_index: pd.DatetimeIndex) -> pd.DataFrame:
    daily = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    daily_full = daily.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)
    eq = daily_full.cumsum()
    return pd.DataFrame({
        "date":   daily_full.index,
        "pnl":    daily_full.values,
        "equity": eq.values,
        "drawdown": (eq - eq.cummax()).values,
    })


def run_audit(trades: pd.DataFrame, split: str) -> list[str]:
    """Return list of warnings; empty if all checks pass."""
    warns = []
    split_ts = pd.Timestamp(split)
    entries = pd.to_datetime(trades["entry"])
    exits   = pd.to_datetime(trades["exit"])

    pre = (entries < split_ts).sum()
    if pre:
        warns.append(f"pre_oos_trades: {pre} entries before split {split}")

    bad_order = (exits < entries).sum()
    if bad_order:
        warns.append(f"exit_before_entry: {bad_order} rows")

    missing = trades[["entry", "exit", "pair", "pnl"]].isna().any(axis=1).sum()
    if missing:
        warns.append(f"missing_fields: {missing} rows")

    dups = trades.duplicated(subset=["pair", "entry"]).sum()
    if dups:
        warns.append(f"duplicate_trades: {dups} (pair+entry collisions)")

    if "pnl" in trades:
        thresh = trades["pnl"].abs().quantile(0.999) * 10
        outliers = (trades["pnl"].abs() > thresh).sum()
        if outliers:
            warns.append(f"extreme_pnl_outliers: {outliers} (>10× 99.9th pct)")

    return warns


def write_audit_report(path: Path, trades: pd.DataFrame, metrics: dict,
                        per_pair: pd.DataFrame, warns: list[str], split: str):
    lines = [
        "# Baseline Audit Report",
        "",
        f"- Trade ledger:    {len(trades)} rows",
        f"- OOS split date:  {split}",
        f"- Entry span:      {trades['entry'].min()} → {trades['entry'].max()}",
        f"- Exit span:       {trades['exit'].min()} → {trades['exit'].max()}",
        "",
        "## Anti-bias / integrity checks",
        "",
    ]
    if not warns:
        lines.append("- ✅ All checks passed.")
    else:
        for w in warns:
            lines.append(f"- ⚠ {w}")
    lines += ["", "## Portfolio metrics", "", "```"]
    lines += [f"{k:<20s} {v}" for k, v in metrics.items()]
    lines += ["```", "", "## Top 10 pairs by PnL", "", "```",
               per_pair.head(10).round(4).to_string(index=False),
               "```", "",
               "## Bottom 5 pairs by PnL", "", "```",
               per_pair.tail(5).round(4).to_string(index=False),
               "```"]
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--closes", default="data/closes_daily.csv")
    p.add_argument("--split",  default="2014-12-31")
    p.add_argument("--out",    default="output/baseline/")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    closes = pd.read_csv(args.closes, parse_dates=["Date"], usecols=["Date"])
    oos_index = pd.DatetimeIndex(closes[closes["Date"] >= args.split]["Date"]).normalize()
    trades = pd.read_csv(args.trades, parse_dates=["entry", "exit"])

    print(f"Loaded {len(trades)} trades; OOS calendar = {len(oos_index)} days.")

    metrics  = portfolio_metrics(trades, oos_index)
    per_pair = per_pair_summary(trades, oos_index)
    eqcurve  = equity_curve(trades, oos_index)
    warns    = run_audit(trades, args.split)

    trades.to_csv(out_dir / "trades_baseline.csv", index=False)
    per_pair.to_csv(out_dir / "pair_summary_baseline.csv", index=False)
    eqcurve.to_csv(out_dir / "equity_curve_baseline.csv", index=False)
    with open(out_dir / "portfolio_summary_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    write_audit_report(out_dir / "audit_report.md", trades, metrics, per_pair,
                        warns, args.split)

    print(f"\nPortfolio metrics:")
    for k, v in metrics.items():
        print(f"  {k:<20s} {v}")
    print(f"\nIntegrity checks: {'all passed ✅' if not warns else 'WARNINGS:'}")
    for w in warns:
        print(f"  ⚠ {w}")
    print(f"\nArtifacts → {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
