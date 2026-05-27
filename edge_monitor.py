"""edge_monitor.py — Live-mode rolling Sharpe health monitor (§9 of spec).

Reads a trade ledger, replays it causally, and at each OOS day emits:
    - SR_3m, SR_6m  (rolling realised Sharpe, annualised)
    - DD_3m         (3-month rolling drawdown of cumulative PnL)
    - suggested leverage (0.0 / 0.5 / 0.75 / 1.0) per §9 rules
    - alert event log when leverage level changes

The output is not a backtest layer — it is a runbook for live trading:
the leverage suggestion should be applied to NEW entries from t+1 onward.

Rules (from §9):
    SR_3m < 0.3   →  leverage  0.75   (caution)
    SR_3m < 0     →  leverage  0.50   (deteriorating)
    SR_6m < 0     →  leverage  0.00   (stop new entries until reviewed)

CLI:
    python edge_monitor.py --trades data/wfo_daily_results_bias.csv \\
                            --start 2014-12-31 --out output/edge_monitor.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def daily_pnl(trades: pd.DataFrame, oos_index: pd.DatetimeIndex) -> pd.Series:
    """Sum trade PnL by exit-date, reindexed across the full OOS calendar."""
    daily = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    return daily.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)


def rolling_sharpe(pnl: pd.Series, days: int) -> pd.Series:
    """Annualised Sharpe from rolling window of daily portfolio PnL."""
    win = pnl.rolling(days, min_periods=int(days * 0.6))
    mu  = win.mean()
    sd  = win.std()
    return (mu / sd * np.sqrt(252)).fillna(0.0)


def rolling_dd(pnl: pd.Series, days: int) -> pd.Series:
    """Max drawdown over a rolling window."""
    def _dd(x: np.ndarray) -> float:
        cum = np.cumsum(x)
        return float((cum - np.maximum.accumulate(cum)).min())
    return pnl.rolling(days, min_periods=int(days * 0.6)).apply(_dd, raw=True)


def suggested_leverage(sr_3m: float, sr_6m: float) -> float:
    if not np.isfinite(sr_3m) or not np.isfinite(sr_6m):
        return 1.0
    if sr_6m < 0:
        return 0.0
    if sr_3m < 0:
        return 0.5
    if sr_3m < 0.3:
        return 0.75
    return 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--closes", default="data/closes_daily.csv")
    p.add_argument("--start",  default="2014-12-31")
    p.add_argument("--out",    default="output/edge_monitor.csv")
    p.add_argument("--alerts", default="output/edge_monitor_alerts.json")
    args = p.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"], usecols=["Date"])
    oos = pd.DatetimeIndex(closes[closes["Date"] >= args.start]["Date"]).normalize()
    trades = pd.read_csv(args.trades, parse_dates=["entry", "exit"])

    pnl_daily = daily_pnl(trades, oos)
    sr_3m = rolling_sharpe(pnl_daily, 63)    # ~3 months
    sr_6m = rolling_sharpe(pnl_daily, 126)   # ~6 months
    dd_3m = rolling_dd(pnl_daily, 63)
    lev   = pd.Series([suggested_leverage(s3, s6) for s3, s6 in zip(sr_3m, sr_6m)],
                       index=oos)

    out = pd.DataFrame({
        "date":     oos,
        "pnl_day":  pnl_daily.values,
        "sr_3m":    sr_3m.values,
        "sr_6m":    sr_6m.values,
        "dd_3m":    dd_3m.values,
        "leverage": lev.values,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # Detect leverage transitions → alert events
    alerts = []
    prev_lev = 1.0
    for _, row in out.iterrows():
        if not np.isnan(row["leverage"]) and row["leverage"] != prev_lev:
            alerts.append({
                "date":     row["date"].strftime("%Y-%m-%d"),
                "old_lev":  prev_lev,
                "new_lev":  float(row["leverage"]),
                "sr_3m":    float(row["sr_3m"]),
                "sr_6m":    float(row["sr_6m"]),
                "reason":   ("SR_6m<0 → stop new entries" if row["leverage"] == 0
                              else "SR_3m<0 → cut 50%" if row["leverage"] == 0.5
                              else "SR_3m<0.3 → caution 75%" if row["leverage"] == 0.75
                              else "recover to full"),
            })
            prev_lev = float(row["leverage"])
    Path(args.alerts).parent.mkdir(parents=True, exist_ok=True)
    with open(args.alerts, "w") as f:
        json.dump(alerts, f, indent=2)

    # Summary
    print(f"Edge monitor over {len(oos)} OOS days")
    print(f"  current SR_3m:       {sr_3m.iloc[-1]:+.3f}")
    print(f"  current SR_6m:       {sr_6m.iloc[-1]:+.3f}")
    print(f"  current leverage:    {lev.iloc[-1]:.2f}")
    print(f"  alerts triggered:    {len(alerts)}")
    print(f"  days at full lev:    {(lev == 1.0).sum()}  ({(lev == 1.0).mean()*100:.0f}%)")
    print(f"  days at 75% lev:     {(lev == 0.75).sum()}")
    print(f"  days at 50% lev:     {(lev == 0.5).sum()}")
    print(f"  days at 0% (stop):   {(lev == 0.0).sum()}")
    print(f"\nSaved:\n  {args.out}\n  {args.alerts}")
    if alerts:
        print(f"\nMost recent 5 transitions:")
        for a in alerts[-5:]:
            print(f"  {a['date']}  {a['old_lev']:.2f} → {a['new_lev']:.2f}  ({a['reason']})")


if __name__ == "__main__":
    main()
