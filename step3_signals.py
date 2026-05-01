import pandas as pd
import numpy as np
from config import RTH_START, RTH_END, SIGNAL_START, RECENT_BARS, DATA_DIR

closes = pd.read_csv(DATA_DIR / "closes_15min.csv", index_col=0, parse_dates=True)
if closes.index.tz is None:
    closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    closes.index = closes.index.tz_convert("US/Eastern")
closes = closes.between_time(RTH_START, RTH_END).dropna().tail(RECENT_BARS)

pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

best = pairs.iloc[0]
t1, t2 = best["pair"].split("-")
beta = best["beta"]
half_life = int(best["half_life_bars"])

if t1 not in closes.columns or t2 not in closes.columns:
    raise SystemExit(f"Ticker not found in data. Available: {list(closes.columns)}")

print(f"Pair: {t1}-{t2}, beta={beta}, half-life={half_life} bars")

spread = closes[t1] - beta * closes[t2]
window = max(20, min(half_life, 200))

spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()
zscore = (spread - spread_mean) / spread_std

signals = pd.DataFrame({
    f"{t1}_close": closes[t1],
    f"{t2}_close": closes[t2],
    "spread": spread,
    "spread_mean": spread_mean,
    "spread_std": spread_std,
    "zscore": zscore,
}).dropna().between_time(SIGNAL_START, RTH_END)

signals.to_csv(DATA_DIR / "signals.csv")

print(f"Window: {window} bars, Points: {len(signals)}")
print(f"|Z|>2.0: {(signals['zscore'].abs() > 2.0).sum()}")
print(f"|Z|>1.5: {(signals['zscore'].abs() > 1.5).sum()}")
print(f"\nSaved to {DATA_DIR / 'signals.csv'}")
