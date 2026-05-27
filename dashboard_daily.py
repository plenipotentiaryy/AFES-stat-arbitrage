"""dashboard_daily.py — visual dashboard for daily WFO output.

Produces a single 6-panel PNG covering equity curve, drawdown, monthly
returns heatmap, per-pair Sharpe ranking, trade-count by year, and
sub-block (cross-listing / non-cross) decomposition.

CLI:
    python dashboard_daily.py --trades data/wfo_daily_results_fullrun.csv \\
        --closes data/closes_daily_extended_v2.csv --split 2014-12-31 \\
        --out output/dashboard_daily.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def daily_series(trades: pd.DataFrame, oos_days: pd.DatetimeIndex) -> pd.Series:
    d = trades.set_index("exit")["pnl"].groupby(level=0).sum()
    return d.reindex(oos_days).fillna(0.0)


def sharpe(d: pd.Series) -> float:
    if d.std() == 0 or np.isnan(d.std()):
        return float("nan")
    return float(d.mean() / d.std() * np.sqrt(252))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--closes", required=True)
    ap.add_argument("--split",  default="2014-12-31")
    ap.add_argument("--out",    default="output/dashboard_daily.png")
    args = ap.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    oos = closes.loc[args.split:].index
    t = pd.read_csv(args.trades, parse_dates=["entry", "exit"])

    # ── Aggregate metrics ─────────────────────────────────────────────────
    d_all = daily_series(t, oos)
    eq = d_all.cumsum()
    dd = eq - eq.cummax()

    sh   = sharpe(d_all)
    pnl  = float(t.pnl.sum())
    n_tr = len(t)
    win  = (t.pnl > 0).mean() * 100
    pf   = float(t[t.pnl > 0].pnl.sum() / abs(t[t.pnl < 0].pnl.sum())) if (t.pnl < 0).any() else float("inf")
    mdd  = float(dd.min())

    # Cross-listing decomposition
    CROSS = {p for p in t.pair.unique() if "_USD" in p}
    t_cr  = t[t.pair.isin(CROSS)]
    t_nc  = t[~t.pair.isin(CROSS)]
    d_cr  = daily_series(t_cr, oos)
    d_nc  = daily_series(t_nc, oos)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.30,
                              left=0.05, right=0.98, top=0.94, bottom=0.06)
    fig.suptitle(
        f"AFES v2 — Daily WFO Dashboard   |   OOS {oos.min().date()} → {oos.max().date()}   |   "
        f"Sharpe={sh:+.3f}   PnL={pnl:+.2f}   trades={n_tr}   win={win:.1f}%   PF={pf:.2f}   maxDD={mdd:+.3f}",
        fontsize=13, y=0.985,
    )

    # ─── Panel 1: Equity curve (combined + sub-blocks)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(eq.index, eq.values, lw=1.6, color="#1f77b4", label=f"Combined (S={sh:+.2f})")
    ax.plot(d_cr.cumsum().index, d_cr.cumsum().values, lw=1.2, color="#2ca02c",
             label=f"Cross-listing (S={sharpe(d_cr):+.2f})", alpha=0.8)
    ax.plot(d_nc.cumsum().index, d_nc.cumsum().values, lw=1.2, color="#ff7f0e",
             label=f"Non-cross (S={sharpe(d_nc):+.2f})", alpha=0.8)
    ax.set_title("Equity curve — combined vs sub-blocks", fontsize=11)
    ax.set_ylabel("Cumulative PnL (log-spread units)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # ─── Panel 2: Drawdown
    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.6)
    ax.set_title(f"Drawdown (max = {mdd:+.3f})", fontsize=11)
    ax.set_ylabel("Underwater (log-units)")
    ax.grid(alpha=0.3)

    # ─── Panel 3: Monthly returns heatmap
    ax = fig.add_subplot(gs[1, 0:2])
    monthly = d_all.resample("ME").sum().to_frame("pnl")
    monthly["year"]  = monthly.index.year
    monthly["month"] = monthly.index.month
    pivot = monthly.pivot(index="year", columns="month", values="pnl")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                    vmin=-pivot.abs().max().max(), vmax=pivot.abs().max().max())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(12):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                         color="black" if abs(v) < pivot.abs().max().max() * 0.6 else "white")
    ax.set_title("Monthly PnL heatmap", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025)

    # ─── Panel 4: Trades by year
    ax = fig.add_subplot(gs[1, 2])
    yr_count = t.groupby(t.exit.dt.year)["pnl"].agg(["count", "sum"])
    bars = ax.bar(yr_count.index, yr_count["count"], color="#1f77b4", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(yr_count.index, yr_count["sum"], color="#d62728", lw=2, marker="o",
              label="PnL per year")
    ax.set_title("Trades & PnL by year", fontsize=11)
    ax.set_ylabel("Trade count", color="#1f77b4")
    ax2.set_ylabel("PnL", color="#d62728")
    ax.grid(alpha=0.3)

    # ─── Panel 5: Per-pair Sharpe ranking
    ax = fig.add_subplot(gs[2, :2])
    pair_sharpes = {}
    for p in t.pair.unique():
        dp = daily_series(t[t.pair == p], oos)
        pair_sharpes[p] = sharpe(dp)
    ps = pd.Series(pair_sharpes).sort_values()
    colors = ["#2ca02c" if p in CROSS else "#1f77b4" for p in ps.index]
    ax.barh(range(len(ps)), ps.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels(ps.index, fontsize=7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Per-pair Sharpe (green = cross-listing, blue = non-cross)", fontsize=11)
    ax.set_xlabel("Sharpe")
    ax.grid(alpha=0.3, axis="x")

    # ─── Panel 6: Sub-block summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    def block_stats(grp, label):
        if len(grp) == 0:
            return [label, "0", "+0.00", "+0.00", "0.0%"]
        ds = daily_series(grp, oos)
        eqs = ds.cumsum()
        return [
            label,
            f"{len(grp)}",
            f"{grp.pnl.sum():+.2f}",
            f"{sharpe(ds):+.2f}",
            f"{(eqs - eqs.cummax()).min():+.3f}",
        ]

    SUB = {
        "Canadian dual": {"RY-RY.TO_USD","TD-TD.TO_USD","BNS-BNS.TO_USD","ENB-ENB.TO_USD","SU-SU.TO_USD"},
        "European ADR":  {"SAP-SAP.DE_USD","SNY-SAN.PA_USD","NVS-NOVN.SW_USD"},
        "Aussie ADR":    {"BHP-BHP.AX_USD","RIO-RIO.AX_USD"},
    }
    rows = [["Block", "Trades", "PnL", "Sharpe", "MaxDD"]]
    for name, ps_set in SUB.items():
        rows.append(block_stats(t[t.pair.isin(ps_set)], name))
    rows.append(block_stats(t_nc, "Non-cross (26)"))
    rows.append(block_stats(t,    "Combined"))

    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor("#cccccc")
        tbl[(0, j)].set_text_props(weight="bold")
    ax.set_title("Sub-block decomposition", fontsize=11, y=0.92)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Dashboard → {out_path}  ({out_path.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
