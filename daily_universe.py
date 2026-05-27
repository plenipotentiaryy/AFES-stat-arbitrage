"""Apply the daily-horizon z-strategy to the full 104-pair candidate universe."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import daily_sanity as ds

ds.PAIRS_CSV = Path("data/pairs_selected.csv")


def main():
    df, pairs = ds.load_data()
    print(f"Loaded {len(df)} days × {df.shape[1]} tickers; {len(pairs)} pairs")
    print(f"WFO: TRAIN={ds.TRAIN_YRS}y, OOS={ds.OOS_YRS}y, step={ds.STEP_MO}mo, z_win={ds.Z_WIN}d")

    cost_log = 2 * ds.COST_BPS / 1e4
    tr = ds.run_wfo(df, pairs, cost=cost_log)
    ds.summarise(tr, f"FULL UNIVERSE  ({len(pairs)} pairs, NET {ds.COST_BPS} bps/leg)")

    # Cross-section: which pairs work, which don't
    by_pair = tr.groupby("pair")["pnl"].agg(
        n="count", pnl="sum", mean_pnl="mean",
        win_rate=lambda x: (x > 0).mean(),
    ).sort_values("pnl", ascending=False)
    by_pair["sharpe_per_trade"] = (
        tr.groupby("pair")["pnl"].mean() / tr.groupby("pair")["pnl"].std()
    )

    print(f"\nProfitable pairs: {(by_pair['pnl'] > 0).sum()}/{len(by_pair)}")
    print(f"Mean per-pair PnL: {by_pair['pnl'].mean():+.4f}")
    print(f"Median per-pair PnL: {by_pair['pnl'].median():+.4f}")

    print("\nTop 15:")
    print(by_pair.head(15).round(3).to_string())
    print("\nBottom 10:")
    print(by_pair.tail(10).round(3).to_string())

    by_pair.to_csv("output/daily_universe_perpair.csv")
    tr.to_csv("output/daily_universe_trades.csv", index=False)
    print(f"\nSaved → output/daily_universe_perpair.csv ({len(by_pair)} pairs)")
    print(f"Saved → output/daily_universe_trades.csv ({len(tr)} trades)")


if __name__ == "__main__":
    main()
