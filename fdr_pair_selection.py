"""False-discovery-rate audit for AFES candidate pair selection.

When hundreds or thousands of candidate pairs are screened, uncorrected
5% p-values will produce many false positives.  This script applies the
Benjamini-Hochberg procedure to candidate p-values and reports how many pairs
survive at common FDR levels.

The preferred input is `data/expanded_universe_candidates.csv` from
candidate_scoring.py, containing `eg_p` and component scores.  If a production
universe is also provided, the report flags production pairs whose source
candidate p-value does not survive FDR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: pd.Series, q: float) -> pd.Series:
    """Return boolean mask of discoveries under BH-FDR at level q."""
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna() & np.isfinite(p)
    out = pd.Series(False, index=p.index)
    if valid.sum() == 0:
        return out
    pv = p[valid].sort_values()
    m = len(pv)
    ranks = np.arange(1, m + 1, dtype=float)
    thresholds = q * ranks / m
    passed = pv <= thresholds
    if not passed.any():
        return out
    cutoff = pv[passed].max()
    out.loc[valid] = p.loc[valid] <= cutoff
    return out


def bh_adjusted_q_values(p_values: pd.Series) -> pd.Series:
    """Compute BH-adjusted q-values."""
    p = pd.to_numeric(p_values, errors="coerce")
    qvals = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if valid.sum() == 0:
        return qvals
    pv = p[valid].sort_values()
    m = len(pv)
    ranks = np.arange(1, m + 1, dtype=float)
    raw = pv.values * m / ranks
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    qvals.loc[pv.index] = adj
    return qvals


def normalize_pair(pair: str) -> str:
    parts = str(pair).split("-")
    return "-".join(parts) if len(parts) == 2 else str(pair)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFES FDR pair-selection audit.")
    parser.add_argument("--candidates", default="data/expanded_universe_candidates.csv")
    parser.add_argument("--p-col", default="eg_p",
                        help="Candidate p-value column. Default: eg_p.")
    parser.add_argument("--q", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--production", default="data/production_universe.csv")
    parser.add_argument("--out", default="output/fdr_pair_selection_report.json")
    parser.add_argument("--csv-out", default="data/fdr_candidate_audit.csv")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    if "pair" not in cand.columns:
        if {"t1", "t2"}.issubset(cand.columns):
            cand["pair"] = cand["t1"].astype(str) + "-" + cand["t2"].astype(str)
        else:
            raise ValueError("Candidate file must contain 'pair' or both 't1'/'t2'")
    if args.p_col not in cand.columns:
        raise ValueError(f"{args.candidates} does not contain p-value column {args.p_col!r}")

    cand["pair"] = cand["pair"].map(normalize_pair)
    cand["p_value"] = pd.to_numeric(cand[args.p_col], errors="coerce")
    cand["bh_q_value"] = bh_adjusted_q_values(cand["p_value"])
    for q in args.q:
        cand[f"bh_pass_{q:g}"] = benjamini_hochberg(cand["p_value"], q)

    prod_payload = {}
    prod_path = Path(args.production)
    if prod_path.exists():
        prod = pd.read_csv(prod_path)
        if "status" in prod.columns:
            prod = prod[prod["status"].astype(str).str.lower().eq("production")]
        if "pair" in prod.columns:
            prod_pairs = pd.Series(prod["pair"].map(normalize_pair).unique())
            cand_idx = cand.drop_duplicates("pair").set_index("pair")
            rows = []
            for pair in prod_pairs:
                if pair in cand_idx.index:
                    rec = cand_idx.loc[pair]
                    rows.append({
                        "pair": pair,
                        "candidate_found": True,
                        "p_value": float(rec["p_value"]) if pd.notna(rec["p_value"]) else np.nan,
                        "bh_q_value": float(rec["bh_q_value"]) if pd.notna(rec["bh_q_value"]) else np.nan,
                        **{f"bh_pass_{q:g}": bool(rec[f"bh_pass_{q:g}"]) for q in args.q},
                    })
                else:
                    rows.append({
                        "pair": pair,
                        "candidate_found": False,
                        "p_value": np.nan,
                        "bh_q_value": np.nan,
                        **{f"bh_pass_{q:g}": False for q in args.q},
                    })
            prod_audit = pd.DataFrame(rows)
            prod_payload = {
                "production_pairs": int(len(prod_audit)),
                "production_pairs_found_in_candidates": int(prod_audit["candidate_found"].sum()),
                "production_fdr_pass": {
                    f"q={q:g}": int(prod_audit[f"bh_pass_{q:g}"].sum()) for q in args.q
                },
            }
            prod_audit.to_csv(Path(args.csv_out).with_name("fdr_production_audit.csv"), index=False)

    summary = {
        "candidate_file": args.candidates,
        "p_col": args.p_col,
        "n_candidates": int(len(cand)),
        "n_valid_p_values": int(cand["p_value"].notna().sum()),
        "uncorrected_p_lt_005": int((cand["p_value"] <= 0.05).sum()),
        "fdr_discoveries": {
            f"q={q:g}": int(cand[f"bh_pass_{q:g}"].sum()) for q in args.q
        },
        **prod_payload,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    cand.to_csv(args.csv_out, index=False)

    print("FDR PAIR-SELECTION AUDIT")
    print(f"  candidates:          {summary['n_candidates']}")
    print(f"  valid p-values:      {summary['n_valid_p_values']}")
    print(f"  raw p<=0.05:         {summary['uncorrected_p_lt_005']}")
    for q in args.q:
        print(f"  BH-FDR q={q:g}:       {summary['fdr_discoveries'][f'q={q:g}']}")
    if prod_payload:
        print("\nProduction universe:")
        print(f"  pairs:               {prod_payload['production_pairs']}")
        print(f"  found in candidates: {prod_payload['production_pairs_found_in_candidates']}")
        for q in args.q:
            print(f"  pass q={q:g}:         {prod_payload['production_fdr_pass'][f'q={q:g}']}")
    print(f"\nSaved -> {args.out}")
    print(f"CSV   -> {args.csv_out}")


if __name__ == "__main__":
    main()
