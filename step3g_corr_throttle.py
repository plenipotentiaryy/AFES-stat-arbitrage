"""
step3g_corr_throttle.py — Layer A: correlated-drawdown throttle.

For each trading day, build the cross-pair correlation matrix of spread
residuals over a rolling 30-day window. When the average absolute
off-diagonal correlation crosses a threshold, the whole book is throttled
(×0.5 sizing) — pairs are no longer independent bets.

Output: data/corr_throttle.csv  (date, mean_abs_corr, throttle_mult)
Consumed by step3e_sizing.py via load_corr_throttle().
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA_DIR, OUTPUT_DIR

CORR_WINDOW_DAYS  = 30
MEAN_CORR_CUTOFF  = 0.60
THROTTLE_ON       = 0.50
THROTTLE_OFF      = 1.00
RESID_ZSCORE_WIN  = 60   # rolling window for spread z-score residual

# ── Load data ────────────────────────────────────────────────────────────────
closes = pd.read_csv(DATA_DIR / "closes_daily.csv", index_col=0, parse_dates=True)
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty")

# ── Build per-pair daily spread residuals ────────────────────────────────────
residuals: dict[str, pd.Series] = {}
for _, row in pairs.iterrows():
    pair = row["pair"]
    t1, t2 = pair.split("-")
    if t1 not in closes.columns or t2 not in closes.columns:
        continue
    beta = float(row.get("beta_daily", row["beta"]) or row["beta"])
    spread = closes[t1] - beta * closes[t2]
    spread = spread.dropna()
    if len(spread) < RESID_ZSCORE_WIN + 5:
        continue
    mu = spread.rolling(RESID_ZSCORE_WIN).mean()
    sd = spread.rolling(RESID_ZSCORE_WIN).std()
    z  = (spread - mu) / sd
    residuals[pair] = z.dropna()

if not residuals:
    raise SystemExit("No pair residuals could be built — check closes_daily.csv columns")

resid_df = pd.DataFrame(residuals).dropna(how="all")
print(f"Built residuals for {resid_df.shape[1]} pairs over {len(resid_df)} days")

# ── Rolling cross-pair mean |corr| ──────────────────────────────────────────
n = resid_df.shape[1]
dates = resid_df.index
mean_abs_corr = pd.Series(index=dates, dtype=float)

for i in range(CORR_WINDOW_DAYS, len(dates)):
    window = resid_df.iloc[i - CORR_WINDOW_DAYS:i]
    # Need at least half the pairs with enough observations
    valid = window.dropna(axis=1, thresh=int(CORR_WINDOW_DAYS * 0.8))
    if valid.shape[1] < 5:
        continue
    C = valid.corr().to_numpy()
    # Mean absolute off-diagonal
    np.fill_diagonal(C, np.nan)
    mean_abs_corr.iloc[i] = float(np.nanmean(np.abs(C)))

mean_abs_corr = mean_abs_corr.dropna()
throttle_mult = np.where(mean_abs_corr > MEAN_CORR_CUTOFF, THROTTLE_ON, THROTTLE_OFF)

out = pd.DataFrame({
    "mean_abs_corr": mean_abs_corr.values,
    "throttle_mult": throttle_mult,
}, index=mean_abs_corr.index)
out.index.name = "date"
out.to_csv(DATA_DIR / "corr_throttle.csv")

n_throttle = int((out["throttle_mult"] < 1.0).sum())
print(f"Throttle days: {n_throttle} / {len(out)}  ({n_throttle/len(out)*100:.1f}%)")
print(f"Mean |corr|: avg={out['mean_abs_corr'].mean():.3f}  "
      f"max={out['mean_abs_corr'].max():.3f}")
print(f"Saved → {DATA_DIR / 'corr_throttle.csv'}")

# ── Chart ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(out.index, out["mean_abs_corr"], color="steelblue", lw=1.2,
        label="Mean |corr| (30d window)")
ax.axhline(MEAN_CORR_CUTOFF, color="red", lw=1, ls="--",
           label=f"Throttle cutoff ({MEAN_CORR_CUTOFF})")
ax.fill_between(out.index, 0, 1,
                where=(out["throttle_mult"] < 1.0),
                color="red", alpha=0.15, transform=ax.get_xaxis_transform(),
                label=f"Throttle ON (×{THROTTLE_ON})")
ax.set_ylabel("Mean |off-diagonal corr|")
ax.set_title("Layer A — correlated-drawdown throttle")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "corr_throttle.png", dpi=140)
print(f"Chart → {OUTPUT_DIR / 'corr_throttle.png'}")
