"""
step4b_concurrent_cap.py — Apply max-N concurrent positions cap to trades.csv.

The base backtest (step4a) runs each pair independently. This post-processor
walks all trades chronologically by entry_time and keeps only those where the
number of currently-open positions (across all pairs) is below MAX_CONCURRENT.

Rationale: real portfolios can't hold N=95 simultaneous positions on a $10K
account — capital constraints + correlated drawdowns force concentration.
Capping at 7 makes the test resemble real trading.
"""

import pandas as pd
import numpy as np
import heapq
from config import INITIAL_CAPITAL, BARS_PER_DAY, DATA_DIR

MAX_CONCURRENT = 7

t = pd.read_csv(DATA_DIR / "trades.csv")
t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
t["exit_time"]  = pd.to_datetime(t["exit_time"], utc=True)
t = t.sort_values("entry_time").reset_index(drop=True)

kept = []
heap = []   # (exit_time, pair) heap of currently-open positions

for _, r in t.iterrows():
    # Pop positions that have closed before this entry
    while heap and heap[0][0] <= r["entry_time"]:
        heapq.heappop(heap)
    if len(heap) < MAX_CONCURRENT:
        kept.append(r)
        heapq.heappush(heap, (r["exit_time"], r["pair"]))

kept_df = pd.DataFrame(kept).reset_index(drop=True)
skipped = len(t) - len(kept_df)
print(f"Total trades:    {len(t)}")
print(f"Kept (≤{MAX_CONCURRENT} concurrent): {len(kept_df)}")
print(f"Skipped:         {skipped}  ({skipped/len(t)*100:.1f}%)")

# Re-aggregate dollar P&L
kept_df["date"] = kept_df["exit_time"].dt.date
daily = kept_df.groupby("date")["dollar_pnl"].sum()
daily = daily[daily.abs() > 0]
equity = INITIAL_CAPITAL + daily.cumsum()
ret = equity.pct_change().dropna()
ann_ret = ret.mean() * 252
ann_vol = ret.std() * np.sqrt(252)
sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
years   = (daily.index[-1] - daily.index[0]).days / 365.25 if len(daily) > 1 else 1
cagr    = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1/max(years, 0.01)) - 1
dd      = ((equity - equity.cummax()) / equity.cummax()).min()
wr      = (kept_df["dollar_pnl"] > 0).mean() * 100

print(f"\n=== Concurrent-capped portfolio ({MAX_CONCURRENT} max) ===")
print(f"Period:       {daily.index[0]} → {daily.index[-1]}  ({years:.1f}y)")
print(f"Win rate:     {wr:.1f}%")
print(f"Net P&L:      ${kept_df['dollar_pnl'].sum():+.2f}")
print(f"Final bal:    ${equity.iloc[-1]:,.2f}")
print(f"Total return: {(equity.iloc[-1]/INITIAL_CAPITAL - 1)*100:+.2f}%")
print(f"CAGR:         {cagr*100:+.2f}%")
print(f"Annual vol:   {ann_vol*100:.2f}%")
print(f"Sharpe daily: {sharpe:.2f}")
print(f"Max DD:       {dd*100:.2f}%")

# Save the capped trades
kept_df.drop(columns=["date"]).to_csv(DATA_DIR / "trades_capped.csv", index=False)
print(f"\nSaved → {DATA_DIR / 'trades_capped.csv'}")

# Concurrent position distribution
print("\nConcurrent position distribution (kept trades):")
heap2 = []
counts = []
for _, r in kept_df.iterrows():
    while heap2 and heap2[0] <= r["entry_time"]:
        heapq.heappop(heap2)
    heapq.heappush(heap2, r["exit_time"])
    counts.append(len(heap2))
counts = pd.Series(counts)
print(counts.value_counts().sort_index().to_string())
