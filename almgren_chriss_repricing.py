"""almgren_chriss_repricing.py — Replace flat costs with tiered + square-root impact.

Post-trade re-pricing: takes the existing trades_*.csv and substitutes the
embedded flat costs with a per-pair Almgren-Chriss style cost model.

For each leg of a round-trip:
    cost = half_spread_bps + kappa * sigma_daily * sqrt(notional / ADV) * 1e4

Pair tier overrides (see TIER_TABLE below):
    us_large_cap, canadian_dual, european_adr, aussie_adr_local, etc.

Borrow: per-pair annual rate × holding_days / 252.

CLI:
    python almgren_chriss_repricing.py --trades data/wfo_daily_results_fullrun.csv \\
        --closes data/closes_daily_extended_v2.csv --notional 100000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────── Per-leg tier table ──────────────────────────────────
# Tuples: (half_spread_bps, kappa, borrow_annual_bps, comment)
TIER_TABLE = {
    "us_large_cap":      (1.0,  0.5,   50,  "JPM/AAPL/MSFT-class, ADV > $1B"),
    "us_mid_cap":        (5.0,  0.7,  100,  "ADV $100M-$1B"),
    "us_small_cap":      (15.0, 1.0,  300,  "ADV < $100M"),
    "canadian_us_list":  (3.0,  0.6,   80,  "RY/TD/BNS US-listing on NYSE"),
    "canadian_local":    (5.0,  0.8,  120,  "RY.TO/TD.TO local TSX"),
    "european_adr_us":   (4.0,  0.7,  100,  "SAP/NVS/SNY US ADR on NYSE"),
    "european_local":    (8.0,  1.0,  150,  "Local SAP.DE/NOVN.SW in overlap window"),
    "aussie_adr_us":     (5.0,  0.8,  150,  "BHP/RIO US ADR on NYSE"),
    "aussie_local":      (25.0, 2.0,  300,  "BHP.AX/RIO.AX — cross-day exec, FX risk"),
}

# Per-ticker tier assignment (US-listing focus, fallback us_mid_cap).
def classify_ticker(tk: str) -> str:
    tk = tk.upper()
    # Canadian USD-normalized locals
    if tk.endswith(".TO_USD"): return "canadian_local"
    # European USD-normalized locals
    if any(tk.endswith(s) for s in (".DE_USD", ".PA_USD", ".SW_USD", ".L_USD")):
        return "european_local"
    # Aussie/JP locals (worst case)
    if any(tk.endswith(s) for s in (".AX_USD", ".T_USD")):
        return "aussie_local"
    # Canadian US-listings
    if tk in {"RY", "TD", "BNS", "ENB", "SU", "TRP", "CNQ", "MFC", "SLF"}:
        return "canadian_us_list"
    # European ADRs (US-listings)
    if tk in {"SAP", "NVS", "SNY", "SHEL", "BP", "HSBC"}:
        return "european_adr_us"
    # Aussie/JP ADRs (US-listings)
    if tk in {"BHP", "RIO", "TM", "SONY"}:
        return "aussie_adr_us"
    # Default — assume liquid US large/mid blend
    LARGE = {"JPM", "BAC", "WFC", "C", "GS", "MS", "MSFT", "AAPL", "GOOGL", "META",
             "AMZN", "NVDA", "TSLA", "V", "MA", "HD", "LIN", "JNJ", "XOM", "CVX",
             "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLP", "XLY", "XLI",
             "XLU", "XLRE", "XLB"}
    if tk in LARGE:
        return "us_large_cap"
    return "us_mid_cap"


def leg_cost_log(ticker: str, sigma_daily: float, notional: float, adv_usd: float) -> float:
    """Return per-leg one-way cost in LOG-units (price-relative).

    Cost_bps = half_spread + kappa * sigma_daily(%) * sqrt(notional / ADV) * 100
    Cost_log = Cost_bps / 10000

    sigma_daily: as fraction (e.g. 0.015 for 1.5%/day)
    notional: USD notional per leg
    adv_usd: avg daily dollar volume of the ticker
    """
    tier = classify_ticker(ticker)
    half_sp, kappa, _, _ = TIER_TABLE[tier]
    # Square-root impact in bps (Almgren empirical: kappa·σ·sqrt(participation) in PERCENT)
    participation = notional / max(adv_usd, 1e3)
    impact_bps = kappa * (sigma_daily * 100) * np.sqrt(participation) * 100.0  # bps
    total_bps  = half_sp + impact_bps
    return total_bps / 1e4  # log units


def borrow_log_per_day(ticker: str) -> float:
    tier = classify_ticker(ticker)
    _, _, borrow_bps, _ = TIER_TABLE[tier]
    return (borrow_bps / 1e4) / 252.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades",   required=True)
    ap.add_argument("--closes",   required=True)
    ap.add_argument("--volumes",  default="data/volumes_daily.csv")
    ap.add_argument("--notional", type=float, default=100_000,
                    help="USD notional per leg (default $100k).")
    ap.add_argument("--default-cost-bps", type=float, default=20.0,
                    help="Round-trip cost embedded in source trades "
                         "(2*(commission+slip)*2 = 20bps for 5/5).")
    ap.add_argument("--default-borrow-bps", type=float, default=80.0,
                    help="Annual borrow embedded in source trades.")
    ap.add_argument("--split",    default="2014-12-31")
    ap.add_argument("--out",      default="output/almgren_chriss_report.json")
    args = ap.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    oos = closes.loc[args.split:].index
    volumes = pd.read_csv(args.volumes, parse_dates=["Date"]).set_index("Date").sort_index()

    t = pd.read_csv(args.trades, parse_dates=["entry", "exit"])
    t["t1"] = t["pair"].str.split("-").str[0]
    t["t2"] = t["pair"].str.split("-").str[1]

    # Per-ticker rolling 60d sigma and 60d $ADV.
    rets = np.log(closes).diff()
    sigma_60 = rets.rolling(60, min_periods=20).std()
    # Dollar volume: price * volume (approx — volumes file has share volume).
    common = [c for c in volumes.columns if c in closes.columns]
    dollar_vol = (closes[common] * volumes[common]).rolling(20, min_periods=10).mean()

    default_round_log    = args.default_cost_bps / 1e4
    default_borrow_per_d = (args.default_borrow_bps / 1e4) / 252.0

    print(f"Re-pricing {len(t)} trades …")
    print(f"  notional per leg: ${args.notional:,.0f}")
    print(f"  default cost embedded: {args.default_cost_bps:.1f} bps round-trip")
    print(f"  default borrow:        {args.default_borrow_bps:.0f} bps/yr")

    adj_pnls = []
    new_cost_breakdown = []
    fallback_adv = {}
    for _, r in t.iterrows():
        ent_date = r["entry"].normalize()
        # Use σ and ADV at entry date (lookback window-based).
        try:
            sig1 = float(sigma_60.loc[ent_date, r["t1"]])
            sig2 = float(sigma_60.loc[ent_date, r["t2"]])
        except (KeyError, ValueError):
            sig1 = sig2 = 0.015  # 1.5% fallback
        if not np.isfinite(sig1): sig1 = 0.015
        if not np.isfinite(sig2): sig2 = 0.015

        # ADV fallback — most foreign locals not in volumes file.
        def get_adv(tk):
            if tk in dollar_vol.columns:
                try:
                    v = float(dollar_vol.loc[ent_date, tk])
                    if np.isfinite(v) and v > 0:
                        return v
                except KeyError:
                    pass
            # tier-default ADV (rough)
            tier = classify_ticker(tk)
            fallback = {
                "us_large_cap":       2.0e9,
                "us_mid_cap":         3.0e8,
                "us_small_cap":       2.5e7,
                "canadian_us_list":   1.5e8,
                "canadian_local":     1.0e8,
                "european_adr_us":    2.0e8,
                "european_local":     6.0e7,
                "aussie_adr_us":      2.5e8,
                "aussie_local":       3.0e7,
            }
            return fallback[tier]

        adv1 = get_adv(r["t1"])
        adv2 = get_adv(r["t2"])

        # Cost: 2 legs × 2 sides (enter + exit) = 4 per-leg one-way costs.
        c1 = leg_cost_log(r["t1"], sig1, args.notional, adv1)
        c2 = leg_cost_log(r["t2"], sig2, args.notional, adv2)
        new_round_log = 2 * (c1 + c2)

        # Borrow: assumes leg2 is the short side (convention in this codebase).
        b_per_d  = borrow_log_per_day(r["t2"])
        held     = int(r["held"]) if "held" in r else 1
        new_borrow_log = b_per_d * held
        default_borrow_log = default_borrow_per_d * held

        # Adjust: pnl already net of default; add default back, subtract new.
        default_total = default_round_log + default_borrow_log
        new_total     = new_round_log     + new_borrow_log
        delta         = default_total - new_total
        adj_pnls.append(r["pnl"] + delta * (r["size"] if "size" in r else 1.0))
        new_cost_breakdown.append((r["pair"], default_total, new_total))

    t["pnl_adj"] = adj_pnls

    # ── Aggregate ──────────────────────────────────────────────────────────
    d_old = t.set_index("exit")["pnl"].groupby(level=0).sum().reindex(oos).fillna(0.0)
    d_new = t.set_index("exit")["pnl_adj"].groupby(level=0).sum().reindex(oos).fillna(0.0)
    def stats(d, lbl):
        sh = d.mean()/d.std()*np.sqrt(252) if d.std() else float("nan")
        eq = d.cumsum(); dd = (eq - eq.cummax()).min()
        return f"  {lbl:30s}  PnL={d.sum():+6.2f}  Sharpe={sh:+.3f}  DD={dd:+.3f}"
    print()
    print(stats(d_old, "Original (flat 5bps/80bps)"))
    print(stats(d_new, "Almgren-Chriss tiered"))

    # ── Cost breakdown by tier ─────────────────────────────────────────────
    cb = pd.DataFrame(new_cost_breakdown, columns=["pair","default","new"])
    cb["t1"] = cb["pair"].str.split("-").str[0]
    cb["t2"] = cb["pair"].str.split("-").str[1]
    cb["tier1"] = cb["t1"].map(classify_ticker)
    cb["tier2"] = cb["t2"].map(classify_ticker)
    cb["pair_tier"] = cb.apply(lambda r: f"{r['tier1']}+{r['tier2']}", axis=1)
    by_pair_tier = cb.groupby("pair_tier")[["default","new"]].agg(["count","mean","sum"])
    print("\nCost-per-trade by pair-tier (log units):")
    print(by_pair_tier.round(5).to_string())

    # ── Per-pair impact ────────────────────────────────────────────────────
    print("\nPer-pair Sharpe impact (sub-block view):")
    by_pair = (t.groupby("pair")
                 .apply(lambda g: pd.Series({
                     "n": len(g),
                     "pnl_old": g.pnl.sum(),
                     "pnl_new": g.pnl_adj.sum(),
                     "delta":   g.pnl_adj.sum() - g.pnl.sum(),
                 })).sort_values("pnl_new", ascending=False))
    print(by_pair.head(15).round(3).to_string())
    print("...")
    print(by_pair.tail(5).round(3).to_string())

    # ── Save JSON ──────────────────────────────────────────────────────────
    sh_old = d_old.mean()/d_old.std()*np.sqrt(252) if d_old.std() else float("nan")
    sh_new = d_new.mean()/d_new.std()*np.sqrt(252) if d_new.std() else float("nan")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sharpe_original": float(sh_old),
        "sharpe_almgren":  float(sh_new),
        "pnl_original":    float(d_old.sum()),
        "pnl_almgren":     float(d_new.sum()),
        "pnl_delta":       float(d_new.sum() - d_old.sum()),
        "per_pair_delta":  by_pair["delta"].to_dict(),
    }, indent=2))
    print(f"\nSaved → {out}")

    # Save adjusted trade log for downstream re-bootstrap.
    t.to_csv("data/wfo_daily_results_almgren.csv", index=False)
    print("Saved adjusted trades → data/wfo_daily_results_almgren.csv")


if __name__ == "__main__":
    main()
