"""step7_curate_universe.py — keeps the pairs with positive OOS PnL (LEGACY).

Audit T5: choosing pairs by their OOS PnL is selection on the test results,
so its output (data/pairs_live.csv) may never be used in a headline run.  It
runs only with --legacy-oos-curation, for comparison; the headline universe is
build_universe_pit.py's frozen point-in-time file.
"""
import argparse

import pandas as pd
from pathlib import Path

GUARD_BANNER = ("LEGACY — step7 curation keeps the pairs with positive OOS PnL (selection on the "
                "test results): its output (data/pairs_live.csv) may never be used in a headline run")


def curate_universe(legacy_oos_curation: bool = False):
    print(GUARD_BANNER)
    if not legacy_oos_curation:
        raise SystemExit("step7_curate_universe.py refused (audit T5): pass --legacy-oos-curation "
                         "to run the legacy OOS-PnL curation, for comparison only")
    params_path = Path("data/wfo_params.csv")
    if not params_path.exists():
        print("Error: data/wfo_params.csv not found. Run WFO first.")
        return

    df = pd.read_csv(params_path)
    
    # Aggregate stats per pair
    summary = df.groupby("pair").agg({
        "oos_pnl": "sum",
        "oos_trades": "sum",
        "oos_sharpe": "mean"
    }).sort_values("oos_pnl", ascending=False)
    
    print("Full Universe OOS Summary:")
    print(summary)
    
    # Filter winners: Positive P&L and at least 5 trades
    winners = summary[(summary["oos_pnl"] > 0) & (summary["oos_trades"] >= 5)]
    
    print("\nSelected Winners:")
    print(winners)
    
    # Load original pairs to get any extra metadata if needed
    original_pairs = pd.read_csv("data/pairs_selected.csv")
    
    # Create the live universe
    live_pairs = original_pairs[original_pairs["pair"].isin(winners.index)]
    
    live_path = Path("data/pairs_live.csv")
    live_pairs.to_csv(live_path, index=False)
    print(f"\nSaved {len(live_pairs)} winners to {live_path}")
    
    # Optional: Update pairs_selected.csv to narrow down future runs
    # live_pairs.to_csv("data/pairs_selected.csv", index=False)


def main():
    p = argparse.ArgumentParser(description="LEGACY OOS-PnL curation of the pairs (never headline)")
    p.add_argument("--legacy-oos-curation", action="store_true", dest="legacy_oos_curation",
                   help="LEGACY: run the curation (pairs with positive OOS PnL and >= 5 trades). "
                        "Comparison only; its output may never be used in a headline run.")
    curate_universe(p.parse_args().legacy_oos_curation)


if __name__ == "__main__":
    main()
