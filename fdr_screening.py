"""fdr_screening.py — Benjamini-Hochberg FDR correction on candidate pair selection.

Runs Engle-Granger ADF + Johansen on each candidate pair using train-only
data (pre-split), collects p-values, and applies FDR correction.

CLI:
    python fdr_screening.py --candidates data/pairs_v2_candidates.csv \\
        --closes data/closes_daily_extended_v2.csv --split 2014-12-31
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Return boolean mask of which p-values pass FDR control at level q."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    thresh = q * ranks / n
    sorted_p = pvals[order]
    passed_sorted = sorted_p <= thresh
    # Largest k for which sorted_p[k] <= q*k/n
    if not passed_sorted.any():
        return np.zeros(n, dtype=bool)
    k_max = np.where(passed_sorted)[0].max()
    mask_sorted = np.zeros(n, dtype=bool)
    mask_sorted[:k_max + 1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = mask_sorted
    return out


def johansen_pvalue(p1: pd.Series, p2: pd.Series) -> float:
    """Approximate Johansen p-value via interpolation on critical values.
    Returns a pseudo p-value in [0, 1]. NaN if numerically unstable."""
    df = pd.concat([p1, p2], axis=1).dropna()
    if len(df) < 100:
        return float("nan")
    try:
        r = coint_johansen(df, det_order=0, k_ar_diff=1)
        trace = float(r.lr1[0])
        # cvt columns: 90%, 95%, 99% critical values for r<=0 hypothesis
        c90, c95, c99 = r.cvt[0, 0], r.cvt[0, 1], r.cvt[0, 2]
        if trace < c90:
            return 0.5  # not rejected at any level
        if trace < c95:
            # interpolate p between 0.10 and 0.05
            frac = (trace - c90) / (c95 - c90)
            return 0.10 - 0.05 * frac
        if trace < c99:
            frac = (trace - c95) / (c99 - c95)
            return 0.05 - 0.04 * frac
        return 0.005  # well past 1% — strongly reject
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--closes",     required=True)
    ap.add_argument("--split",      default="2014-12-31")
    ap.add_argument("--out",        default="output/fdr_report.json")
    args = ap.parse_args()

    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    train = closes.loc[:args.split]
    cands = pd.read_csv(args.candidates)["pair"].tolist()
    print(f"Train slice: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} days)")
    print(f"Candidates:  {len(cands)} pairs")

    rows = []
    for pair in cands:
        t1, t2 = pair.split("-")
        if t1 not in train.columns or t2 not in train.columns:
            rows.append({"pair": pair, "p_eg": np.nan, "p_joh": np.nan, "status": "missing_data"})
            continue
        slc = train[[t1, t2]].dropna()
        if len(slc) < 500:
            rows.append({"pair": pair, "p_eg": np.nan, "p_joh": np.nan,
                          "status": f"insufficient_data ({len(slc)} obs)"})
            continue
        try:
            _, p_eg, _ = coint(np.log(slc[t1]), np.log(slc[t2]), trend="c")
        except Exception:
            p_eg = np.nan
        p_joh = johansen_pvalue(np.log(slc[t1]), np.log(slc[t2]))
        rows.append({"pair": pair, "p_eg": float(p_eg), "p_joh": float(p_joh), "status": "tested"})

    df = pd.DataFrame(rows)
    print(f"\nP-values computed: EG={df.p_eg.notna().sum()}  Joh={df.p_joh.notna().sum()}")

    # ── Apply FDR ──────────────────────────────────────────────────────────
    tested = df[df.p_eg.notna()].copy()
    pvals = tested.p_eg.values

    # Naive uncorrected at α=0.05
    naive_05 = pvals < 0.05
    # FDR-BH at q=0.05 and q=0.10
    bh_05 = benjamini_hochberg(pvals, 0.05)
    bh_10 = benjamini_hochberg(pvals, 0.10)
    # Bonferroni at α=0.05
    bonf  = pvals < (0.05 / len(pvals))

    tested["naive_05"] = naive_05
    tested["bh_05"]    = bh_05
    tested["bh_10"]    = bh_10
    tested["bonf_05"]  = bonf

    print(f"\nMultiple-testing comparison on n={len(pvals)} EG p-values:")
    print(f"  Naive α=0.05:         {naive_05.sum():>3d} pass   (expected false-pos under H0: {0.05*len(pvals):.1f})")
    print(f"  FDR-BH q=0.05:        {bh_05.sum():>3d} pass")
    print(f"  FDR-BH q=0.10:        {bh_10.sum():>3d} pass")
    print(f"  Bonferroni α=0.05:    {bonf.sum():>3d} pass  (very conservative)")

    # ── Compare with bias_test selection ───────────────────────────────────
    bt = pd.read_csv("data/pairs_bias_test.csv")
    bt_selected = set(bt.pair.tolist())
    tested["in_bias_test"] = tested.pair.isin(bt_selected)
    print(f"\nIntersection of selection methods with 36 bias_test selected:")
    for col in ("naive_05", "bh_05", "bh_10", "bonf_05"):
        passed = set(tested[tested[col]].pair)
        overlap = passed & bt_selected
        only_method = passed - bt_selected
        only_bt = bt_selected - passed
        print(f"  {col:14s}  pass={len(passed):3d}  overlap_with_bt={len(overlap):2d}  "
              f"only_method={len(only_method):2d}  only_bt={len(only_bt):2d}")

    # ── Output: FDR-selected pair list ─────────────────────────────────────
    fdr_pairs = tested[tested.bh_05]["pair"].tolist()
    pd.DataFrame({"pair": fdr_pairs}).to_csv("data/pairs_fdr_bh05.csv", index=False)
    print(f"\nSaved FDR (q=0.05) selected → data/pairs_fdr_bh05.csv  ({len(fdr_pairs)} pairs)")

    # Show full table
    print("\nP-value ranking (top 15):")
    print(tested.sort_values("p_eg").head(15)[["pair","p_eg","p_joh","bh_05","bonf_05","in_bias_test"]].round(5).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_candidates":   len(cands),
        "n_tested":       int(df.p_eg.notna().sum()),
        "n_naive_05":     int(naive_05.sum()),
        "n_bh_05":        int(bh_05.sum()),
        "n_bh_10":        int(bh_10.sum()),
        "n_bonf_05":      int(bonf.sum()),
        "n_bias_test":    len(bt_selected),
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
