import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from config import (
    CLOSES_FILE,
    ENTRY_Z, ENTRY_Z_VOLATILE, STOP_Z,
    COST_PER_SIDE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    BARS_PER_DAY,
    DATA_DIR, OUTPUT_DIR,
)

VOL_WINDOW = 20   # bars for rolling features
N_SEEDS    = 10   # HMM restarts — pick best log-likelihood
EXIT_Z_CMP = 0.0  # exit threshold used in the comparison backtest


# ── Load data ─────────────────────────────────────────────────────────────────
def _data_file():
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        fb = DATA_DIR / "closes_15min.csv"
        if fb.exists():
            return fb
        raise FileNotFoundError(f"No data: {CLOSES_FILE}")
    return p
closes = pd.read_csv(_data_file(), index_col=0, parse_dates=True)
if closes.index.tz is None:
    closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    closes.index = closes.index.tz_convert("US/Eastern")
closes = closes.between_time(RTH_START, RTH_END)

pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
print(f"Pairs: {len(pairs)}\n")


def fit_hmm(X: np.ndarray) -> GaussianHMM | None:
    best_model, best_score = None, -np.inf
    for seed in range(N_SEEDS):
        try:
            m = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, random_state=seed)
            m.fit(X)
            s = m.score(X)
            if s > best_score:
                best_score, best_model = s, m
        except Exception:
            continue
    return best_model


def build_features(spread: pd.Series) -> pd.DataFrame:
    pct = spread.pct_change()
    return pd.DataFrame({
        "vol":     pct.rolling(VOL_WINDOW).std(),
        "ret":     pct.rolling(VOL_WINDOW).mean(),
        "abs_chg": spread.diff().abs().rolling(VOL_WINDOW).mean(),
    }).dropna()


# ── Fit HMM per pair ──────────────────────────────────────────────────────────
all_regimes: dict[str, pd.Series] = {}

for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    beta   = row["beta"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker")
        continue

    spread   = (closes[t1] - beta * closes[t2]).dropna().tail(RECENT_BARS)
    features = build_features(spread)

    if len(features) < 100:
        print(f"  SKIP {row['pair']}: too few bars ({len(features)})")
        continue

    X     = features.values
    model = fit_hmm(X)
    if model is None:
        print(f"  SKIP {row['pair']}: HMM did not converge")
        continue

    states = model.predict(X)

    # Volatile = state with higher mean realized vol
    vol0, vol1 = X[states == 0, 0].mean(), X[states == 1, 0].mean()
    volatile_state = 0 if vol0 > vol1 else 1

    labels = (states == volatile_state).astype(int)   # 1 = volatile, 0 = normal
    regime_series = pd.Series(labels, index=features.index, name=row["pair"])
    all_regimes[row["pair"]] = regime_series

    n_vol  = labels.sum()
    n_calm = len(labels) - n_vol
    ratio  = X[labels == 1, 0].mean() / X[labels == 0, 0].mean()

    print(f"  {row['pair']}:")
    print(f"    Normal   {n_calm:5d} bars ({n_calm/len(labels)*100:.0f}%)")
    print(f"    Volatile {n_vol:5d} bars ({n_vol/len(labels)*100:.0f}%)")
    print(f"    Vol ratio volatile/normal: {ratio:.1f}x\n")


# ── Save regimes (wide format: timestamp × pair) ──────────────────────────────
if all_regimes:
    regime_df = pd.DataFrame(all_regimes)
    regime_df.to_csv(DATA_DIR / "regimes.csv")
    print(f"Saved {DATA_DIR / 'regimes.csv'}  ({regime_df.shape})\n")


# ── Comparison backtest: with vs without regime filter ────────────────────────
def run_backtest(spread, zscore, beta, entry_z_override=None,
                 regime_dict=None, use_regime=False):
    t1_price_proxy = spread  # we use spread units for cost estimation

    position = 0
    entry_spread = entry_t1 = 0.0
    entry_bar = 0
    trades = []

    for i in range(len(zscore)):
        idx = zscore.index[i]
        z   = zscore.iloc[i]
        s   = spread.loc[idx] if idx in spread.index else None
        if s is None or np.isnan(s):
            continue

        if position != 0:
            exit_sig = (position == 1 and z > -EXIT_Z_CMP) or (position == -1 and z < EXIT_Z_CMP)
            stop_sig = (position == 1 and z < -STOP_Z) or (position == -1 and z > STOP_Z)

            if exit_sig or stop_sig:
                gross_pnl   = position * (s - entry_spread)
                holding_days = (i - entry_bar) / BARS_PER_DAY
                # rough cost: notional ~ |entry_spread| * 2 (both legs)
                notional    = abs(entry_spread) * 2
                tx_cost     = 2 * notional * COST_PER_SIDE
                borrow_cost = notional * 0.5 * BORROW_RATE_ANNUAL * holding_days / 252
                trades.append({
                    "net_pnl":     gross_pnl - tx_cost - borrow_cost,
                    "exit_reason": "STOP" if stop_sig else "SIGNAL",
                })
                position = 0

        if position == 0:
            if use_regime and regime_dict is not None:
                is_vol    = regime_dict.get(idx, 0) == 1
                threshold = ENTRY_Z_VOLATILE if is_vol else ENTRY_Z
            else:
                threshold = entry_z_override or ENTRY_Z

            if z < -threshold:
                position = 1; entry_spread = s; entry_bar = i
            elif z > threshold:
                position = -1; entry_spread = s; entry_bar = i

    return pd.DataFrame(trades)


print("=" * 65)
print(f"{'Pair':<12}  {'Mode':<12} {'Trades':>6} {'WR':>6} {'Stops':>6} "
      f"{'Net P&L':>10} {'Sharpe':>7}")
print("=" * 65)

for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    beta, hl = row["beta"], int(row["half_life_bars"])

    if t1 not in closes.columns or t2 not in closes.columns:
        continue

    spread = (closes[t1] - beta * closes[t2]).dropna().tail(RECENT_BARS)
    window = max(20, min(hl, 200))
    zscore = ((spread - spread.rolling(window).mean()) / spread.rolling(window).std()).dropna()
    zscore = zscore.between_time(SIGNAL_START, RTH_END)

    regime_dict = all_regimes.get(row["pair"], pd.Series()).to_dict()

    for label, use_reg in [("no regime", False), ("HMM filter", True)]:
        df = run_backtest(spread, zscore, beta, regime_dict=regime_dict, use_regime=use_reg)
        if df.empty:
            print(f"{row['pair']:<12}  {label:<12} {'—':>6}")
            continue
        p   = df["net_pnl"]
        wr  = (p > 0).mean() * 100
        sh  = p.mean() / p.std() * np.sqrt(len(p)) if p.std() > 0 else 0
        st  = (df["exit_reason"] == "STOP").sum()
        print(f"{row['pair']:<12}  {label:<12} {len(df):>6} {wr:>5.0f}% "
              f"{st:>6} {p.sum():>+10.3f} {sh:>7.2f}")
    print()


# ── Visualisation ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
n_pairs = len(all_regimes)

if n_pairs == 0:
    print("No regimes to plot.")
    raise SystemExit(0)

fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4 * n_pairs), squeeze=False)

for row_idx, (pair_name, regime_series) in enumerate(all_regimes.items()):
    t1, t2 = pair_name.split("-")
    beta   = pairs[pairs["pair"] == pair_name]["beta"].iloc[0]
    spread = (closes[t1] - beta * closes[t2]).dropna().tail(RECENT_BARS)

    features    = build_features(spread)
    is_volatile = regime_series == 1
    common_idx  = regime_series.index

    # Panel left: spread colored by regime
    ax = axes[row_idx, 0]
    spread_aligned = spread.loc[spread.index.isin(common_idx)]
    ax.plot(spread_aligned.index, spread_aligned.values, color="black", lw=0.7)
    for is_vol, color in [(False, "lightgreen"), (True, "salmon")]:
        mask = (is_volatile == is_vol).values
        idx  = common_idx[mask]
        ax.fill_between(common_idx,
                        spread.reindex(common_idx).min(),
                        spread.reindex(common_idx).max(),
                        where=mask, color=color, alpha=0.3, step="post")
    ax.set_title(f"{pair_name} — Spread (green=normal, red=volatile)")
    ax.set_ylabel("Spread")

    # Panel right: realized vol colored by regime
    ax = axes[row_idx, 1]
    vol = features["vol"].reindex(common_idx)
    ax.fill_between(common_idx, vol, where=~is_volatile.values,
                    color="lightgreen", alpha=0.5, label="Normal")
    ax.fill_between(common_idx, vol, where=is_volatile.values,
                    color="salmon", alpha=0.5, label="Volatile")
    ax.plot(common_idx, vol, color="black", lw=0.7)
    ax.set_title(f"{pair_name} — Realized Volatility")
    ax.set_ylabel("Vol")
    ax.legend(fontsize=8)

plt.suptitle("Per-pair HMM Regime Detection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "regimes_pairs.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'regimes_pairs.png'}")
# plt.show()
