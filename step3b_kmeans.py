"""
step3b_kmeans.py — K-Means macro regime detection.

Downloads SPY + ^VIX daily data and classifies each trading day into one of
three market regimes using K-Means clustering on realized-vol features:

  0 = Trend   — VIX moderate, directional price action, weak mean-reversion
  1 = Sideways — VIX low-to-moderate, low vol, strong mean-reversion
  2 = Panic    — VIX elevated, vol spikes, spread behavior unreliable

The backtest (step4_backtest.py) uses these labels to:
  - Allow new entries only in Regime 1 (Sideways)
  - Force-close all open positions in Regime 2 (Panic)
  - Hold but block new entries in Regime 0 (Trend)

Output: data/kmeans_regimes.csv  (daily index, 'regime' column: 0/1/2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, OUTPUT_DIR, DAILY_START, KMEANS_N_CLUSTERS, KMEANS_VOL_WINDOW

REGIME_NAMES = {0: "Trend", 1: "Sideways", 2: "Panic"}
FEATURE_TICKERS = ["SPY", "^VIX"]

# ── Walk-forward (leak-free) refit parameters ─────────────────────────────────
MIN_TRAIN_DAYS = 252   # 1 year of trading data for first cluster fit
REFIT_EVERY    = 30    # refit cadence in trading days


# ── Download macro data ───────────────────────────────────────────────────────

def load_macro_data() -> pd.DataFrame:
    raw = yf.download(FEATURE_TICKERS, start=DAILY_START,
                      auto_adjust=True, progress=False, threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.dropna(how="all")
    spy = closes["SPY"].dropna()
    vix = closes["^VIX"].dropna()
    return pd.DataFrame({"spy": spy, "vix": vix}).dropna()


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, window: int = KMEANS_VOL_WINDOW) -> pd.DataFrame:
    spy_ret = df["spy"].pct_change()
    features = pd.DataFrame({
        "spy_rvol":  spy_ret.rolling(window).std(),          # realized vol
        "spy_trend": spy_ret.rolling(window).mean(),         # directional drift
        "vix_level": df["vix"],                              # spot VIX
        "vix_chg":   df["vix"].pct_change(window),          # VIX momentum
        "spy_dd":    (df["spy"] / df["spy"].rolling(window).max() - 1),  # drawdown
    }).dropna()
    return features


# ── Cluster and label regimes ─────────────────────────────────────────────────

def label_clusters(kmeans: KMeans, features: pd.DataFrame) -> pd.Series:
    """
    Assign semantic labels to K-Means clusters based on mean VIX level:
      lowest  VIX → Sideways (1)
      middle  VIX → Trend    (0)
      highest VIX → Panic    (2)
    """
    labels_raw = kmeans.labels_
    cluster_ids = np.arange(KMEANS_N_CLUSTERS)

    vix_by_cluster = {
        c: features["vix_level"].values[labels_raw == c].mean()
        for c in cluster_ids
    }
    sorted_by_vix = sorted(cluster_ids, key=lambda c: vix_by_cluster[c])
    # sorted_by_vix[0] = lowest VIX → Sideways
    # sorted_by_vix[1] = middle VIX → Trend
    # sorted_by_vix[2] = highest VIX → Panic
    cluster_to_regime = {
        sorted_by_vix[0]: 1,   # Sideways
        sorted_by_vix[1]: 0,   # Trend
        sorted_by_vix[2]: 2,   # Panic
    }

    regime_labels = pd.Series(
        [cluster_to_regime[c] for c in labels_raw],
        index=features.index,
        name="regime",
    )

    for c in cluster_ids:
        r = cluster_to_regime[c]
        n = (regime_labels == r).sum()
        pct = n / len(regime_labels) * 100
        avg_vix = vix_by_cluster[c]
        print(f"  Cluster {c} → {REGIME_NAMES[r]:8s}  {n:5d} days ({pct:4.1f}%)  "
              f"avg VIX={avg_vix:.1f}")

    return regime_labels


# ── Main ──────────────────────────────────────────────────────────────────────

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Downloading macro data (SPY + VIX) …", flush=True)
macro = load_macro_data()
print(f"Loaded {len(macro)} trading days  "
      f"({macro.index[0].date()} → {macro.index[-1].date()})")

features = build_features(macro, window=KMEANS_VOL_WINDOW)
print(f"Feature matrix: {features.shape}  (window={KMEANS_VOL_WINDOW}d)")

if len(features) < MIN_TRAIN_DAYS + REFIT_EVERY:
    raise SystemExit(f"Need at least {MIN_TRAIN_DAYS + REFIT_EVERY} days; "
                     f"got {len(features)}")

# ── Walk-forward K-Means: refit on data ≤ t, label next REFIT_EVERY days ──────
print(f"\nWalk-forward K-Means refit  "
      f"(min_train={MIN_TRAIN_DAYS}d, refit_every={REFIT_EVERY}d) …", flush=True)

regime_labels = pd.Series(index=features.index, name="regime", dtype="float")
regime_counts = {0: 0, 1: 0, 2: 0}
n_refits = 0

for i in range(MIN_TRAIN_DAYS, len(features), REFIT_EVERY):
    # Fit on past-only data ≤ index i-1
    scaler_t = StandardScaler()
    X_train  = scaler_t.fit_transform(features.iloc[:i].values)

    km_t = KMeans(n_clusters=KMEANS_N_CLUSTERS, n_init=20, random_state=42)
    km_t.fit(X_train)

    # Stable cluster→regime mapping by avg VIX in TRAIN data
    train_labels = km_t.labels_
    train_vix    = features["vix_level"].iloc[:i].values
    vix_by_cluster = {
        c: train_vix[train_labels == c].mean()
        for c in range(KMEANS_N_CLUSTERS)
    }
    sorted_by_vix = sorted(range(KMEANS_N_CLUSTERS),
                           key=lambda c: vix_by_cluster[c])
    cluster_to_regime = {
        sorted_by_vix[0]: 1,   # lowest VIX → Sideways
        sorted_by_vix[1]: 0,   # middle      → Trend
        sorted_by_vix[2]: 2,   # highest     → Panic
    }

    # Predict next chunk using fitted model
    end = min(i + REFIT_EVERY, len(features))
    X_next = scaler_t.transform(features.iloc[i:end].values)
    next_clusters = km_t.predict(X_next)
    next_regimes  = np.array([cluster_to_regime[c] for c in next_clusters])
    regime_labels.iloc[i:end] = next_regimes
    for r in next_regimes:
        regime_counts[int(r)] += 1
    n_refits += 1

regime_series = regime_labels.dropna().astype(int).rename("regime")
print(f"  Refits: {n_refits}  Labeled days: {len(regime_series)}  "
      f"(first {MIN_TRAIN_DAYS} days unlabeled)")
print(f"  First labeled day: {regime_series.index[0].date()}")
print(f"  Last  labeled day: {regime_series.index[-1].date()}\n")

print("Regime distribution (walk-forward):")
for r in [0, 1, 2]:
    n = regime_counts[r]
    pct = 100 * n / max(len(regime_series), 1)
    print(f"  {REGIME_NAMES[r]:8s}  {n:5d} days ({pct:4.1f}%)")

# Save
out_path = DATA_DIR / "kmeans_regimes.csv"
regime_series.to_csv(out_path, index=True, header=True)
print(f"\nSaved {len(regime_series)} days to {out_path}")

# ── Visualisation ─────────────────────────────────────────────────────────────
colors = {0: "gold", 1: "lightgreen", 2: "salmon"}

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Panel 1: SPY price with regime background
ax = axes[0]
spy_aligned = macro["spy"].reindex(regime_series.index)
ax.plot(spy_aligned.index, spy_aligned.values, color="black", lw=0.8, label="SPY")
for regime_id, color in colors.items():
    mask = (regime_series == regime_id).values
    ax.fill_between(regime_series.index, spy_aligned.min(), spy_aligned.max(),
                    where=mask, color=color, alpha=0.3,
                    label=REGIME_NAMES[regime_id])
ax.set_title("SPY Price — K-Means Macro Regimes")
ax.set_ylabel("Price")
ax.legend(fontsize=8)

# Panel 2: VIX with regime background
ax = axes[1]
vix_aligned = macro["vix"].reindex(regime_series.index)
ax.plot(vix_aligned.index, vix_aligned.values, color="purple", lw=0.8, label="VIX")
for regime_id, color in colors.items():
    mask = (regime_series == regime_id).values
    ax.fill_between(regime_series.index, 0, vix_aligned.max(),
                    where=mask, color=color, alpha=0.3)
ax.axhline(20, color="gray", lw=0.8, linestyle="--", label="VIX=20")
ax.axhline(30, color="red",  lw=0.8, linestyle="--", label="VIX=30")
ax.set_title("VIX — K-Means Macro Regimes")
ax.set_ylabel("VIX")
ax.legend(fontsize=8)

# Panel 3: regime label as a step plot
ax = axes[2]
ax.step(regime_series.index, regime_series.values, color="navy", lw=0.8)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Trend", "Sideways", "Panic"])
ax.set_title("Regime Label Over Time")

plt.suptitle("K-Means Macro Regime Detection", fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = OUTPUT_DIR / "kmeans_regimes.png"
plt.savefig(chart_path, dpi=150)
print(f"Chart saved to {chart_path}")
