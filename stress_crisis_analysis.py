"""stress_crisis_analysis.py — Phase-1 stress and crisis-period analysis.

Slices the daily-WFO output by named crisis episodes, computes VaR/CVaR
on the daily-PnL series, and reports tail behaviour.

Run after step3j_wfo_daily.py:
    python stress_crisis_analysis.py \\
        --trades data/wfo_daily_results_fullrun.csv \\
        --closes data/closes_daily_extended_v2.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────── Crisis windows (named) ─────────────────────────────
CRISIS_WINDOWS = [
    ("2015 China devaluation",   "2015-08-01", "2015-10-15"),
    ("2016 Brexit",              "2016-06-01", "2016-07-31"),
    ("2018-Q1 vol-mageddon",     "2018-01-15", "2018-04-15"),
    ("2018-Q4 selloff",          "2018-10-01", "2018-12-31"),
    ("2020 COVID crash",         "2020-02-15", "2020-04-15"),
    ("2020 COVID recovery",      "2020-04-15", "2020-09-30"),
    ("2022 Fed tightening",      "2022-01-01", "2022-10-31"),
    ("2023 SVB bank crisis",     "2023-03-01", "2023-05-15"),
    ("2024 yen unwind",          "2024-07-15", "2024-08-15"),
]

# Calm-period baseline for relative comparison
CALM_WINDOWS = [
    ("2017 calm bull",           "2017-01-01", "2017-12-31"),
    ("2019 pre-COVID",           "2019-06-01", "2019-12-31"),
    ("2024 H1 bull",             "2024-01-01", "2024-06-30"),
]


def daily_pnl(trades: pd.DataFrame, oos_days: pd.DatetimeIndex) -> pd.Series:
    d = trades.set_index("exit")["pnl"].groupby(level=0).sum()
    return d.reindex(oos_days).fillna(0.0)


def slice_metrics(daily: pd.Series, trades: pd.DataFrame,
                   start: str, end: str) -> dict:
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    d  = daily.loc[s:e]
    tr = trades[(trades["exit"] >= s) & (trades["exit"] <= e)]
    if d.std() == 0 or len(d) == 0:
        return dict(days=len(d), trades=len(tr), pnl=0.0, sharpe=float("nan"),
                     dd=0.0, win=float("nan"), best=0.0, worst=0.0)
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    sh = d.mean() / d.std() * np.sqrt(252)
    return dict(
        days=len(d),
        trades=len(tr),
        pnl=float(d.sum()),
        sharpe=float(sh),
        dd=float(dd),
        win=float((tr.pnl > 0).mean() * 100) if len(tr) else float("nan"),
        best=float(d.max()),
        worst=float(d.min()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--closes", required=True)
    ap.add_argument("--split",  default="2014-12-31")
    ap.add_argument("--out",    default="output/stress_crisis_report.json")
    args = ap.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    oos    = closes.loc[args.split:].index
    t      = pd.read_csv(args.trades, parse_dates=["entry", "exit"])
    d_all  = daily_pnl(t, oos)

    full_sharpe = d_all.mean() / d_all.std() * np.sqrt(252)
    print(f"{'='*92}")
    print(f"OOS span: {oos.min().date()} → {oos.max().date()}   ({len(oos)} days)")
    print(f"Full Sharpe: {full_sharpe:+.3f}    Full PnL: {d_all.sum():+.2f}    Daily σ: {d_all.std():.4f}")
    print(f"{'='*92}")

    # ── Crisis periods ─────────────────────────────────────────────────────
    print("\nCRISIS PERIODS:")
    print(f"  {'window':30s}{'days':>6s}{'trades':>8s}{'PnL':>9s}{'Sharpe':>9s}{'DD':>10s}{'win%':>7s}{'worst':>9s}")
    crisis_results = []
    for name, start, end in CRISIS_WINDOWS:
        m = slice_metrics(d_all, t, start, end)
        crisis_results.append({"window": name, "start": start, "end": end, **m})
        print(f"  {name:30s}{m['days']:>6d}{m['trades']:>8d}{m['pnl']:>+9.2f}"
              f"{m['sharpe']:>+9.2f}{m['dd']:>+10.3f}{m['win']:>6.1f}%{m['worst']:>+9.3f}")

    print("\nCALM PERIODS (benchmark):")
    print(f"  {'window':30s}{'days':>6s}{'trades':>8s}{'PnL':>9s}{'Sharpe':>9s}{'DD':>10s}{'win%':>7s}{'worst':>9s}")
    calm_results = []
    for name, start, end in CALM_WINDOWS:
        m = slice_metrics(d_all, t, start, end)
        calm_results.append({"window": name, "start": start, "end": end, **m})
        print(f"  {name:30s}{m['days']:>6d}{m['trades']:>8d}{m['pnl']:>+9.2f}"
              f"{m['sharpe']:>+9.2f}{m['dd']:>+10.3f}{m['win']:>6.1f}%{m['worst']:>+9.3f}")

    # ── Tail / VaR analysis ────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("TAIL ANALYSIS — daily PnL distribution")
    print(f"{'='*92}")
    pct = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
    print(f"  {'pct':>8s}{'PnL':>10s}")
    for p in pct:
        v = d_all.quantile(p)
        print(f"  {p*100:>7.1f}%{v:>+10.4f}")

    # VaR and CVaR
    print("\n  Risk metrics (per-day, log-spread units):")
    daily_mean, daily_std = d_all.mean(), d_all.std()
    var_99   = d_all.quantile(0.01)
    var_999  = d_all.quantile(0.001)
    cvar_99  = d_all[d_all <= var_99].mean()
    cvar_999 = d_all[d_all <= var_999].mean()
    skew  = ((d_all - daily_mean)**3).mean() / daily_std**3
    kurt  = ((d_all - daily_mean)**4).mean() / daily_std**4 - 3
    print(f"    mean         : {daily_mean:+.5f}")
    print(f"    std          : {daily_std:+.5f}")
    print(f"    skew         : {skew:+.3f}  (negative = left tail heavier)")
    print(f"    excess kurt  : {kurt:+.3f}  (>0 = fat tails)")
    print(f"    VaR  1%      : {var_99:+.4f}   (1-day, historical)")
    print(f"    CVaR 1%      : {cvar_99:+.4f}   (avg of worst 1%)")
    print(f"    VaR  0.1%    : {var_999:+.4f}  (1-day, historical)")
    print(f"    CVaR 0.1%    : {cvar_999:+.4f}  (avg of worst 0.1%)")

    # Cornish-Fisher adjusted VaR (Gaussian + skew/kurt correction)
    from scipy.stats import norm
    z_99 = norm.ppf(0.01); z_999 = norm.ppf(0.001)
    def cf(z, s, k):
        return z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
    z_cf_99  = cf(z_99,  skew, kurt)
    z_cf_999 = cf(z_999, skew, kurt)
    var_cf_99  = daily_mean + z_cf_99  * daily_std
    var_cf_999 = daily_mean + z_cf_999 * daily_std
    print(f"\n  Cornish-Fisher VaR (skew/kurt adjusted):")
    print(f"    VaR  1%      : {var_cf_99:+.4f}   (parametric, tail-aware)")
    print(f"    VaR  0.1%    : {var_cf_999:+.4f}  (parametric, tail-aware)")

    # ── Worst 10 days ──────────────────────────────────────────────────────
    print(f"\nWORST 10 DAYS:")
    worst = d_all.sort_values().head(10)
    for date, pnl in worst.items():
        # Find trades closing that day
        day_trades = t[t["exit"].dt.date == date.date()]
        n_trades = len(day_trades)
        # Which crisis (if any)
        in_crisis = "—"
        for name, s, e in CRISIS_WINDOWS:
            if pd.Timestamp(s) <= date <= pd.Timestamp(e):
                in_crisis = name
                break
        print(f"  {date.date()}  PnL={pnl:+.4f}  ({n_trades:2d} trades closed)   {in_crisis}")

    # ── Stress scenario: 3x and 5x daily-vol shock ─────────────────────────
    print(f"\n{'='*92}")
    print("HYPOTHETICAL TAIL SHOCK SCENARIOS")
    print(f"{'='*92}")
    print("\n  If left-tail volatility were 3x historical:")
    for p in [0.001, 0.01]:
        v = d_all.quantile(p)
        shocked = daily_mean + 3 * (v - daily_mean)
        print(f"    VaR {p*100:.1f}% under 3x shock: {shocked:+.4f}   (vs historical {v:+.4f})")
    print("\n  If left-tail volatility were 5x historical:")
    for p in [0.001, 0.01]:
        v = d_all.quantile(p)
        shocked = daily_mean + 5 * (v - daily_mean)
        print(f"    VaR {p*100:.1f}% under 5x shock: {shocked:+.4f}   (vs historical {v:+.4f})")

    # Survivability: at 5x shock applied to worst day, what's the resulting DD?
    worst_day_pnl = d_all.min()
    shocked_worst = daily_mean + 5 * (worst_day_pnl - daily_mean)
    print(f"\n  Worst historical day:      {worst_day_pnl:+.4f}  (date {d_all.idxmin().date()})")
    print(f"  Same day under 5x tail:    {shocked_worst:+.4f}")
    print(f"  Versus mean+1σ:            {daily_mean + daily_std:+.4f}  (typical good day)")
    print(f"  Survival ratio (5x shock / mean+1σ): "
          f"{abs(shocked_worst) / (daily_mean + daily_std + 1e-12):.1f}x  daily upside lost")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "oos_span": [str(oos.min().date()), str(oos.max().date())],
        "full_sharpe": float(full_sharpe),
        "full_pnl": float(d_all.sum()),
        "daily_mean": float(daily_mean),
        "daily_std": float(daily_std),
        "skew": float(skew),
        "excess_kurt": float(kurt),
        "var_99": float(var_99),
        "cvar_99": float(cvar_99),
        "var_999": float(var_999),
        "cvar_999": float(cvar_999),
        "var_cf_99": float(var_cf_99),
        "var_cf_999": float(var_cf_999),
        "worst_day": {"date": str(d_all.idxmin().date()), "pnl": float(worst_day_pnl)},
        "crisis_periods": crisis_results,
        "calm_periods": calm_results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
