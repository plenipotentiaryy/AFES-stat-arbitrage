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

# ── Leak-free training parameters ─────────────────────────────────────────────
# Per-pair HMM:   fit on bars BEFORE test_start_date, predict on test/OOS bars.
# Global SPY HMM: expanding-window refit on past-only daily data.
PER_PAIR_TRAIN_BARS    = 7000   # ~3 months of train-only intraday bars
GLOBAL_MIN_TRAIN_DAYS  = 252    # 1y of SPY daily data for first global fit
GLOBAL_REFIT_EVERY     = 30     # refit cadence in trading days


# ── Load data ─────────────────────────────────────────────────────────────────
def _data_file():
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        fb = DATA_DIR / "closes_15min.csv"
        if fb.exists():
            return fb
        raise FileNotFoundError(f"No data: {CLOSES_FILE}")
    return p
closes = pd.read_csv(_data_file(), index_col=0)
closes.index = pd.to_datetime(closes.index, utc=True)
closes.index = closes.index.tz_convert("US/Eastern")
closes = closes.between_time(RTH_START, RTH_END)

pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
print(f"Pairs: {len(pairs)}\n")

# Test start date — pairs_selected.csv contains this column from step2a.
# Per-pair HMM trains strictly on bars BEFORE this date (no future leak).
if "test_start_date" not in pairs.columns:
    raise SystemExit("pairs_selected.csv lacks 'test_start_date' — re-run step2a")
TEST_START = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
print(f"Train cutoff for per-pair HMM: bars < {TEST_START.date()}")
print(f"Predicting on bars >= {TEST_START.date()}  (OOS labels only)\n")


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
    # Spread is mean-reverting and crosses zero, so pct_change() creates
    # artificial infinities/outliers. Use absolute spread increments instead.
    chg = spread.diff()
    return pd.DataFrame({
        "vol":     chg.rolling(VOL_WINDOW).std(),
        "ret":     chg.rolling(VOL_WINDOW).mean(),
        "abs_chg": chg.abs().rolling(VOL_WINDOW).mean(),
    }).replace([np.inf, -np.inf], np.nan).dropna()


# ── Fit HMM per pair (leak-free) ──────────────────────────────────────────────
# Train on bars strictly BEFORE TEST_START; predict on bars >= TEST_START.
# regimes.csv covers the OOS window only — backtest queries dates beyond.
all_regimes: dict[str, pd.Series] = {}

for _, row in pairs.iterrows():
    t1, t2 = row["pair"].split("-")
    beta   = row["beta"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker")
        continue

    spread_full = (closes[t1] - beta * closes[t2]).dropna()
    feat_full   = build_features(spread_full)   # rolling features look back only

    feat_train = feat_full[feat_full.index < TEST_START].tail(PER_PAIR_TRAIN_BARS)
    feat_test  = feat_full[feat_full.index >= TEST_START]

    if len(feat_train) < 200 or len(feat_test) < 50:
        print(f"  SKIP {row['pair']}: insufficient features "
              f"(train={len(feat_train)}, test={len(feat_test)})")
        continue

    X_train = feat_train.values
    model   = fit_hmm(X_train)
    if model is None:
        print(f"  SKIP {row['pair']}: HMM did not converge on train")
        continue

    # Identify volatile state from TRAIN only (no future leak)
    states_train = model.predict(X_train)
    vol0 = X_train[states_train == 0, 0].mean()
    vol1 = X_train[states_train == 1, 0].mean()
    volatile_state = 0 if vol0 > vol1 else 1

    # Predict on TEST bars using the train-fitted HMM
    X_test = feat_test.values
    states_test = model.predict(X_test)
    labels_test = (states_test == volatile_state).astype(int)

    regime_series = pd.Series(labels_test, index=feat_test.index, name=row["pair"])
    all_regimes[row["pair"]] = regime_series

    n_vol  = int(labels_test.sum())
    n_calm = len(labels_test) - n_vol
    print(f"  {row['pair']:12s}  train={len(X_train)}  test={len(X_test)}  "
          f"Volatile {n_vol} ({n_vol/len(labels_test)*100:.0f}%)  "
          f"Normal {n_calm} ({n_calm/len(labels_test)*100:.0f}%)")

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


# ── Global macro HMM on SPY ───────────────────────────────────────────────────
# Trained on SPY daily returns — single market-wide panic signal.
# When global_hmm = 1, sizing.py cuts ALL pair sizes by HMM_PANIC_MULT (÷3).

import yfinance as yf

print(f"\nFitting global macro HMM on SPY (walk-forward, "
      f"min_train={GLOBAL_MIN_TRAIN_DAYS}d, refit_every={GLOBAL_REFIT_EVERY}d) …")
_spy_raw = yf.download("SPY", start="2005-01-01", interval="1d", progress=False)
_spy     = _spy_raw["Close"].squeeze().dropna()
_spy.index = pd.to_datetime(_spy.index).tz_localize(None)

_spy_ret = _spy.pct_change()
_spy_features = pd.DataFrame({
    "vol":     _spy_ret.rolling(20).std(),
    "ret":     _spy_ret.rolling(20).mean(),
    "abs_chg": _spy_ret.abs().rolling(20).mean(),
}).dropna()

_X_spy = _spy_features.values

if len(_X_spy) < GLOBAL_MIN_TRAIN_DAYS + GLOBAL_REFIT_EVERY:
    print(f"  Insufficient SPY history ({len(_X_spy)} days) — global HMM skipped")
else:
    # Walk-forward: fit on past-only, label next REFIT_EVERY days.
    _wf_labels = np.full(len(_X_spy), -1, dtype=int)
    _n_refits  = 0
    _n_panic   = 0

    for _i in range(GLOBAL_MIN_TRAIN_DAYS, len(_X_spy), GLOBAL_REFIT_EVERY):
        _X_train = _X_spy[:_i]
        _model_t = fit_hmm(_X_train)
        if _model_t is None:
            continue

        # Determine panic state from TRAIN labels only
        _states_train = _model_t.predict(_X_train)
        _vol0 = _X_train[_states_train == 0, 0].mean()
        _vol1 = _X_train[_states_train == 1, 0].mean()
        _panic_s = 0 if _vol0 > _vol1 else 1

        # Predict next REFIT_EVERY bars
        _end = min(_i + GLOBAL_REFIT_EVERY, len(_X_spy))
        _states_test = _model_t.predict(_X_spy[_i:_end])
        _labels_test = (_states_test == _panic_s).astype(int)
        _wf_labels[_i:_end] = _labels_test
        _n_panic += int(_labels_test.sum())
        _n_refits += 1

    # Keep only labeled rows
    _mask = _wf_labels >= 0
    global_hmm_regime = pd.Series(
        _wf_labels[_mask],
        index=_spy_features.index[_mask],
        name="global_hmm",
    )
    global_hmm_regime.index = pd.to_datetime(global_hmm_regime.index).tz_localize(None)
    global_hmm_regime.to_csv(DATA_DIR / "global_hmm_regime.csv", header=True)

    _n = len(global_hmm_regime)
    print(f"  Refits: {_n_refits}  Labeled days: {_n}  "
          f"Panic days: {_n_panic} ({_n_panic/max(_n,1)*100:.0f}%)")
    print(f"  First labeled day: {global_hmm_regime.index[0].date()}")
    print(f"  Last  labeled day: {global_hmm_regime.index[-1].date()}")
    print(f"  Saved {DATA_DIR / 'global_hmm_regime.csv'}")
