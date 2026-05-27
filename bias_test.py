"""bias_test.py — Honest out-of-sample pair-selection test.

The current pairs_selected.csv was built by running Johansen on the full
2006-2026 sample.  Any backtest on those 104 pairs is contaminated by
selection bias: we know which pairs cointegrated *with hindsight*.

This script does an honest split:
  1. Re-run pair selection (Johansen + correlation + half-life) using ONLY
     data through SPLIT_DATE.
  2. Take whatever pairs pass that gate.
  3. Run step3j_wfo_daily.py on those pairs, restricted to dates AFTER
     SPLIT_DATE.  This is the un-leaked OOS.

If Sharpe survives this test close to the full-sample number, the edge is
real.  If it collapses, the universe was overfit.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import DATA_DIR

SPLIT_DATE = "2014-12-31"          # first 8.5y → select; last 11+y → test
CORR_MIN   = 0.30                  # min daily-return correlation (relaxed)
HL_MAX     = 90                    # max half-life (days)
JOH_CRIT   = 0.90                  # Johansen crit level (relaxed for half-sample)
N_TOP_BY_CORR = 200                # cap candidate pair list before Johansen


def half_life(spread: pd.Series) -> float:
    a = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    a.columns = ["d", "lag"]
    try:
        import statsmodels.api as sm
        theta = sm.OLS(a["d"], sm.add_constant(a["lag"])).fit().params["lag"]
        return -np.log(2) / theta if theta < 0 else 1e9
    except Exception:
        return 1e9


def johansen_coint(p1: pd.Series, p2: pd.Series, crit=0.95) -> tuple[bool, float]:
    df = pd.concat([p1, p2], axis=1).dropna()
    if len(df) < 100:
        return False, np.nan
    try:
        r = coint_johansen(df, det_order=0, k_ar_diff=1)
        crit_idx = {0.90: 0, 0.95: 1, 0.99: 2}[crit]
        trace = float(r.lr1[0])
        crit_val = float(r.cvt[0, crit_idx])
        if trace <= crit_val:
            return False, np.nan
        v = r.evec[:, 0]
        return True, float(-v[1] / v[0])
    except Exception:
        return False, np.nan


def select_pairs_on_train(closes_train: pd.DataFrame,
                          candidate_pairs: list[str]) -> pd.DataFrame:
    """Run the selection gates on the train slice only.

    NOTE: correlation computed pair-wise (after per-pair dropna).  A global
    dropna on `closes_train` would erase every row because post-2010
    tickers (META, CRWD, …) have NaN before their IPO.
    """
    rows = []
    for pair in candidate_pairs:
        t1, t2 = pair.split("-")
        if t1 not in closes_train.columns or t2 not in closes_train.columns:
            continue
        slc = closes_train[[t1, t2]].dropna()
        if len(slc) < 500:        # need ≥ 2y of daily data
            continue
        rets = np.log(slc).diff().dropna()
        corr = rets[t1].corr(rets[t2])
        if pd.isna(corr) or corr < CORR_MIN:
            continue
        ok, beta = johansen_coint(np.log(slc[t1]), np.log(slc[t2]), crit=JOH_CRIT)
        if not ok or not (0.1 <= abs(beta) <= 15.0):
            continue
        sp = np.log(slc[t1]) - beta * np.log(slc[t2])
        hl = half_life(sp)
        if not (0 < hl <= HL_MAX):
            continue
        rows.append({"pair": pair, "corr": corr, "beta": beta, "hl_days": hl})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("corr", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default=SPLIT_DATE,
                   help=f"Split date (default {SPLIT_DATE}).")
    p.add_argument("--candidates", type=str, default="pairs_selected.csv",
                   help="CSV with candidate pair list (uses 'pair' col).")
    p.add_argument("--cost",   type=float, default=5.0)
    p.add_argument("--slip",   type=float, default=5.0)
    p.add_argument("--borrow", type=float, default=80.0)
    p.add_argument("--kalman", action="store_true")
    p.add_argument("--vwz",    action="store_true")
    p.add_argument("--hmm",    type=str, default="off",
                   choices=["off", "block-panic", "block-calm",
                            "size-panic", "extreme-3state"])
    p.add_argument("--metagate", action="store_true")
    p.add_argument("--hrp",      action="store_true")
    p.add_argument("--ivol",     action="store_true")
    p.add_argument("--volvol",   action="store_true")
    p.add_argument("--pairmom",  action="store_true")
    p.add_argument("--maxconc",  type=int, default=0)
    p.add_argument("--ticker-overlap", action="store_true", dest="ticker_overlap")
    p.add_argument("--per-pair-quarter-cap", type=int, default=0, dest="per_pair_quarter_cap")
    p.add_argument("--vol-tight-stop", action="store_true", dest="vol_tight_stop")
    p.add_argument("--vwap-exec", action="store_true", dest="vwap_exec")
    p.add_argument("--rotation", action="store_true")
    p.add_argument("--voltarget", type=float, default=0.0)
    p.add_argument("--cusum",  type=float, default=None)
    args = p.parse_args()

    print(f"BIAS TEST — split at {args.split}")
    print(f"  selection on:  {{start}} → {args.split}")
    print(f"  trading on:    {args.split} → {{end}}\n")

    closes = (pd.read_csv(DATA_DIR / "closes_daily.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index())
    closes_train = closes.loc[:args.split]
    print(f"Train slice:  {closes_train.index.min().date()} → "
          f"{closes_train.index.max().date()}  ({len(closes_train)} days)")

    candidates = pd.read_csv(DATA_DIR / args.candidates)["pair"].tolist()
    print(f"Candidates:   {len(candidates)} pairs from {args.candidates}")

    sel = select_pairs_on_train(closes_train, candidates)
    out_csv = DATA_DIR / "pairs_bias_test.csv"
    sel.to_csv(out_csv, index=False)
    print(f"\nSelected on train: {len(sel)} pairs")
    if len(sel):
        print(sel.head(15).round(3).to_string(index=False))
    print(f"\nSaved → {out_csv}\n")

    if sel.empty:
        print("No pairs survived selection on train; bias test cannot run.")
        return

    # Now run the daily WFO restricted to post-split dates only.
    cmd = [
        sys.executable, "step3j_wfo_daily.py",
        "--pairs",  "pairs_bias_test.csv",
        "--start",  args.split,
        "--cost",   str(args.cost),
        "--slip",   str(args.slip),
        "--borrow", str(args.borrow),
        "--tag",    "bias",
    ]
    if args.kalman:
        cmd.append("--kalman")
    if args.vwz:
        cmd.append("--vwz")
    if args.hmm != "off":
        cmd += ["--hmm", args.hmm]
    if args.metagate:
        cmd.append("--metagate")
    if args.hrp:
        cmd.append("--hrp")
    if args.ivol:
        cmd.append("--ivol")
    if args.volvol:
        cmd.append("--volvol")
    if args.pairmom:
        cmd.append("--pairmom")
    if args.maxconc > 0:
        cmd += ["--maxconc", str(args.maxconc)]
    if args.ticker_overlap:
        cmd.append("--ticker-overlap")
    if args.per_pair_quarter_cap > 0:
        cmd += ["--per-pair-quarter-cap", str(args.per_pair_quarter_cap)]
    if args.vol_tight_stop:
        cmd.append("--vol-tight-stop")
    if args.vwap_exec:
        cmd.append("--vwap-exec")
    if args.rotation:
        cmd.append("--rotation")
    if args.voltarget > 0:
        cmd += ["--voltarget", str(args.voltarget)]
    if args.cusum is not None:
        cmd += ["--cusum", str(args.cusum)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
