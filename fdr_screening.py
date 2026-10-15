"""fdr_screening.py — Benjamini-Hochberg FDR correction on candidate pair selection.

Runs Engle-Granger ADF + Johansen on each candidate pair using development
data only (prices up to the cut), collects p-values, and applies FDR
correction.

Audit T10:
- The candidate list is the frozen point-in-time universe of
  build_universe_pit.py (universe_pit_<asof>.csv, as step3j_wfo_daily
  certifies it: `frozen_pit_asof`), every row of it — the pre-registered list,
  not a hand-curated subset.  Its prices are cut at its as-of date; another
  --split is refused.  Any other list is screened on prices up to --split
  (the development cut, default 2014-12-31) and every output says
  "NON-HEADLINE / hindsight candidate list".  A cut inside the final holdout
  is refused.
- The output is a screen, never the headline pairs: the pair file carries
  role=screen_only (step3j_wfo_daily never certifies a pairs file with a
  role column as HEADLINE) and goes to output/, under a name the frozen
  universe pattern cannot match.  Never use it to choose the headline set
  after seeing OOS results.

CLI:
    python fdr_screening.py --candidates data/universe_pit_2014-12-31.csv \\
        --closes data/closes_daily_r3k.csv
    python fdr_screening.py --candidates data/pairs_v2_candidates.csv \\
        --closes data/closes_daily_extended_v2.csv --split 2014-12-31     # NON-HEADLINE
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from step3j_wfo_daily import frozen_pit_asof
from wfo_daily_core import HOLDOUT_START

SCREEN_ONLY = "screen_only"
HINDSIGHT = "NON-HEADLINE / hindsight candidate list"
DEV_CUT = "2014-12-31"


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="The candidate list: the frozen point-in-time universe file "
                         "universe_pit_<asof>.csv (build_universe_pit.py), else NON-HEADLINE.")
    ap.add_argument("--closes",     required=True)
    ap.add_argument("--split",      default=None,
                    help=f"Development cut: p-values on prices up to this date (default {DEV_CUT}). "
                         "A frozen universe is cut at its as-of date; another --split is refused.")
    ap.add_argument("--out",        default="output/fdr_report.json")
    ap.add_argument("--pairs-out",  default="output/fdr_screen_bh05.csv",
                    help="The BH q=0.05 pairs, role=screen_only (never a headline pairs file).")
    ap.add_argument("--compare",    default=None,
                    help="Optional pairs file (e.g. data/pairs_bias_test.csv) to compare with.")
    return ap


def main():
    args = build_parser().parse_args()

    cand_df = pd.read_csv(args.candidates)
    name = Path(args.candidates).name
    asof = frozen_pit_asof(args.candidates, cand_df)
    if asof is not None:
        if args.split is not None and pd.Timestamp(args.split) != pd.Timestamp(asof):
            raise SystemExit(f"--split {args.split}: {name} is a frozen point-in-time universe, "
                             f"screened on prices up to its as-of date {asof} only")
        cut = asof
        label = (f"point-in-time candidate list {name} (frozen, as of {asof}); "
                 "screen only — never the headline pairs file")
    else:
        cut = args.split or DEV_CUT
        label = f"{HINDSIGHT}: {name} is not a frozen point-in-time universe; screen only"
    if pd.Timestamp(cut) >= HOLDOUT_START:
        raise SystemExit(f"cut {cut} is inside the final holdout (from {HOLDOUT_START.date()}): "
                         "the FDR p-values use development data only")
    print(label)
    if asof is None and pd.Timestamp(cut) > pd.Timestamp(DEV_CUT):
        print(f"WARNING: cut {cut} is after {DEV_CUT}: the p-values read years that are "
              "out-of-sample for the WFO runs from --start 2014-12-31 (first OOS 2015-06-30)")

    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    train = closes.loc[:cut]
    assert train.index.max() <= pd.Timestamp(cut)        # nothing after the cut
    cands = cand_df["pair"].tolist()                      # every row: the full list
    print(f"Train slice: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} days; "
          f"cut {cut})")
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

    # ── Compare with another selection (optional) ──────────────────────────
    report_compare = {}
    if args.compare:
        bt_selected = set(pd.read_csv(args.compare).pair.tolist())
        tested["in_compare"] = tested.pair.isin(bt_selected)
        print(f"\nIntersection of selection methods with the {len(bt_selected)} pairs of "
              f"{Path(args.compare).name}:")
        for col in ("naive_05", "bh_05", "bh_10", "bonf_05"):
            passed = set(tested[tested[col]].pair)
            overlap = passed & bt_selected
            only_method = passed - bt_selected
            only_bt = bt_selected - passed
            print(f"  {col:14s}  pass={len(passed):3d}  overlap={len(overlap):2d}  "
                  f"only_method={len(only_method):2d}  only_compare={len(only_bt):2d}")
        report_compare = {"compare_file": Path(args.compare).name, "n_compare": len(bt_selected)}

    # ── Output: the screen's pair list (role=screen_only) ──────────────────
    screened = tested[tested.bh_05][["pair", "p_eg", "p_joh"]].assign(role=SCREEN_ONLY)
    pairs_out = Path(args.pairs_out)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    screened.to_csv(pairs_out, index=False)
    print(f"\nSaved FDR (q=0.05) screen → {pairs_out}  ({len(screened)} pairs, role={SCREEN_ONLY}: "
          "never a headline pairs file)")

    # Show full table
    print("\nP-value ranking (top 15):")
    cols = ["pair", "p_eg", "p_joh", "bh_05", "bonf_05"] + (["in_compare"] if args.compare else [])
    print(tested.sort_values("p_eg").head(15)[cols].round(5).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_span = [str(train.index.min().date()), str(train.index.max().date())]
    out.write_text(json.dumps({
        "label":          label,
        "pit":            asof is not None,
        "role":           SCREEN_ONLY,
        "candidates_file": name,
        "candidates_sha256": hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest(),
        "closes_file":    Path(args.closes).name,
        "cut":            cut,
        "train_span":     train_span,
        "train_days":     int(len(train)),
        "n_candidates":   len(cands),
        "n_tested":       int(df.p_eg.notna().sum()),
        "n_naive_05":     int(naive_05.sum()),
        "n_bh_05":        int(bh_05.sum()),
        "n_bh_10":        int(bh_10.sum()),
        "n_bonf_05":      int(bonf.sum()),
        **report_compare,
        "pvalues":        df.astype(object).where(df.notna(), None).to_dict(orient="records"),
    }, indent=2, allow_nan=False))
    print(f"\nSaved → {out}")
    print(label)


if __name__ == "__main__":
    main()
