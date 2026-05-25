import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from config import (
    REGIME_MULT_NORMAL, REGIME_MULT_VOLATILE, HMM_PANIC_MULT,
    IV_MULT_MAX, IV_MULT_MIN, IV_LOOKBACK,
    MIN_POSITION_SIZE,
    DATA_DIR, OUTPUT_DIR,
)

# load everything

def load_regimes() -> pd.DataFrame | None:
    path = DATA_DIR / "regimes.csv"
    if not path.exists():
        print("  regimes.csv not found — regime_mult = 1.0 for all bars")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df   # wide: timestamp × pair, values 0/1


def load_iv() -> tuple[pd.Series | None, pd.Series | None]:
    path = DATA_DIR / "iv_filter.csv"
    if not path.exists():
        print("  iv_filter.csv not found — iv_mult = 1.0 for all days")
        return None, None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    vix = df["vix"].dropna()
    macro_alert = df["macro_alert"].reindex(df.index).fillna(0).astype(int) \
        if "macro_alert" in df.columns else pd.Series(0, index=df.index)
    return vix, macro_alert


def load_global_hmm() -> pd.Series | None:
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        print("  global_hmm_regime.csv not found — global HMM panic mult = 1.0")
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename("global_hmm")


def load_mc_confidence() -> dict[str, float]:
    path = DATA_DIR / "ou_montecarlo_summary.csv"
    if not path.exists():
        print("  ou_montecarlo_summary.csv not found — mc_conf = 1.0 for all pairs")
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "pair" not in df.columns or "win_rate" not in df.columns:
        return {}
    return {row["pair"]: row["win_rate"] / 100 for _, row in df.iterrows()}


# calc size multipliers from all layers

def regime_multiplier(regimes: pd.DataFrame | None, pair: str, ts) -> float:
    if regimes is None or pair not in regimes.columns:
        return REGIME_MULT_NORMAL
    try:
        val = regimes.loc[ts, pair] if ts in regimes.index else 0
        return REGIME_MULT_VOLATILE if val == 1 else REGIME_MULT_NORMAL
    except Exception:
        return REGIME_MULT_NORMAL


def iv_multiplier_series(vix: pd.Series | None) -> pd.Series:
    """Returns daily iv_mult series (0.5 – 1.0)."""
    if vix is None:
        return pd.Series(dtype=float)
    pct_rank = vix.rolling(IV_LOOKBACK).rank(pct=True)
    iv_mult  = IV_MULT_MAX - (IV_MULT_MAX - IV_MULT_MIN) * pct_rank
    return iv_mult.fillna(IV_MULT_MAX)


def iv_multiplier(iv_mult_series: pd.Series, date) -> float:
    if iv_mult_series.empty:
        return IV_MULT_MAX
    key = pd.Timestamp(date).normalize().tz_localize(None)
    try:
        v = float(iv_mult_series.asof(key))
        return v if not np.isnan(v) else IV_MULT_MAX
    except Exception:
        return IV_MULT_MAX


def global_hmm_multiplier(global_hmm: pd.Series | None, date) -> float:
    if global_hmm is None or global_hmm.empty:
        return 1.0
    key = pd.Timestamp(date).normalize().tz_localize(None)
    try:
        val = global_hmm.asof(key)
        return HMM_PANIC_MULT if (not pd.isna(val) and int(val) == 1) else 1.0
    except Exception:
        return 1.0


def macro_alert_active(macro_alert: pd.Series | None, date) -> bool:
    if macro_alert is None or macro_alert.empty:
        return False
    key = pd.Timestamp(date).normalize().tz_localize(None)
    try:
        val = macro_alert.asof(key)
        return bool(int(val)) if not pd.isna(val) else False
    except Exception:
        return False


def mc_confidence(mc_conf: dict, pair: str) -> float:
    return mc_conf.get(pair, 1.0)


# throttle when pairs are all losing together
def load_corr_throttle() -> pd.Series | None:
    path = DATA_DIR / "corr_throttle.csv"
    if not path.exists():
        print("  corr_throttle.csv not found — corr_throttle = 1.0 for all days")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df["throttle_mult"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def corr_throttle_multiplier(s: pd.Series | None, date) -> float:
    if s is None or s.empty:
        return 1.0
    key = pd.Timestamp(date).normalize().tz_localize(None)
    try:
        v = s.asof(key)
        return float(v) if not pd.isna(v) else 1.0
    except Exception:
        return 1.0


# HRP portfolio weights
def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """Recover leaf order from linkage matrix (Lopez de Prado, Ch.16)."""
    link = link.astype(int)
    n = link.shape[0] + 1
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:
        new_order = []
        for v in order:
            if v < n:
                new_order.append(v)
            else:
                row = link[v - n]
                new_order.extend([row[0], row[1]])
        order = new_order
    return order


def _hrp_recursive_bisect(cov: pd.DataFrame, order: list[str]) -> pd.Series:
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            # inverse-variance allocation between two sub-clusters
            v_left  = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            alpha   = 1 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
            w[left]  *= alpha
            w[right] *= 1 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return w


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items].to_numpy()
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def compute_hrp_weights(closes_train: pd.DataFrame,
                        pairs_df: pd.DataFrame) -> dict[str, float]:
    """
    HRP weights on TRAIN window. Returns dict {pair: weight} normalised
    so that mean weight = 1.0 (multiplier semantics; weights are relative).
    """
    spreads = {}
    for _, row in pairs_df.iterrows():
        pair = row["pair"]
        t1, t2 = pair.split("-")
        if t1 not in closes_train.columns or t2 not in closes_train.columns:
            continue
        beta = float(row.get("beta_daily", row["beta"]) or row["beta"])
        sp = (closes_train[t1] - beta * closes_train[t2]).dropna()
        if len(sp) < 60:
            continue
        # Normalise so all pairs are on the same scale before HRP
        # (raw spread diffs have wildly different magnitudes by price level).
        ret = sp.diff().dropna()
        sd  = ret.std()
        if sd <= 0 or not np.isfinite(sd):
            continue
        spreads[pair] = ret / sd

    if len(spreads) < 3:
        print("  HRP: <3 valid pair spreads — falling back to equal weights")
        return {p: 1.0 for p in pairs_df["pair"]}

    df = pd.DataFrame(spreads).dropna(how="any")
    if len(df) < 60:
        return {p: 1.0 for p in pairs_df["pair"]}

    corr = df.corr()
    # Distance: sqrt(0.5 * (1 - corr)) — Lopez de Prado
    dist = np.sqrt(np.clip(0.5 * (1 - corr.to_numpy()), 0, 1))
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="single")
    order_idx = _hrp_quasi_diag(link)
    order = [corr.index[i] for i in order_idx]

    cov = df.cov()
    w = _hrp_recursive_bisect(cov, order)
    # Clip extremes (HRP can still concentrate on lowest-residual-vol clusters),
    # then normalise to mean = 1.0 so it slots in as a sizing multiplier.
    w = w.clip(lower=w.quantile(0.10), upper=w.quantile(0.90))
    w = w / w.mean()
    out = {p: float(w[p]) if p in w.index else 1.0 for p in pairs_df["pair"]}
    return out


def load_hrp_weights() -> dict[str, float]:
    path = DATA_DIR / "hrp_weights.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r["pair"]: float(r["hrp_weight"]) for _, r in df.iterrows()}


def hrp_multiplier(hrp_w: dict, pair: str) -> float:
    return hrp_w.get(pair, 1.0) if hrp_w else 1.0


def position_size(pair: str, ts, regimes, iv_mult_s, mc_conf: dict,
                  macro_alert_s: pd.Series | None = None,
                  global_hmm_s: pd.Series | None = None,
                  corr_throttle_s: pd.Series | None = None,
                  hrp_w: dict | None = None) -> float:
    date = ts.date() if hasattr(ts, "date") else ts
    alert = macro_alert_active(macro_alert_s, date)
    r = regime_multiplier(regimes, pair, ts)            # per-pair HMM
    g = global_hmm_multiplier(global_hmm_s, date)       # global SPY HMM
    i = iv_multiplier(iv_mult_s, date)                  # VIX percentile
    m = mc_confidence(mc_conf, pair)                    # OU Monte Carlo
    c = corr_throttle_multiplier(corr_throttle_s, date) # Layer A
    h = hrp_multiplier(hrp_w or {}, pair)               # Layer B
    sz = r * g * i * m * c * h
    # VIX9D backwardation: reduce size by 50% but don't block entirely
    if alert:
        sz *= 0.5
    return sz


def main() -> None:
    print("\n" + "="*60)
    print("STEP 3e — DYNAMIC POSITION SIZING")
    print("="*60)
    print("Combines outputs from all previous layers into one number: size.")
    print("Formula: size = regime_mult × IV_mult × MC_confidence × macro_mult")
    print("Any single factor near zero collapses the whole position.")
    print("If size < MIN_POSITION_SIZE the trade is skipped entirely.")
    print()
    print("Loading data sources...\n")

    regimes = load_regimes()
    vix, macro_alert_s = load_iv()
    global_hmm_s = load_global_hmm()
    mc_conf = load_mc_confidence()
    iv_mult_s = iv_multiplier_series(vix)
    corr_throttle_s = load_corr_throttle()

    pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
    pair_names = list(pairs["pair"])

    hrp_w: dict[str, float] = {}
    try:
        closes_daily = pd.read_csv(DATA_DIR / "closes_daily.csv",
                                   index_col=0, parse_dates=True)
        if "test_start_date" in pairs.columns and not pairs.empty:
            test_start = pd.Timestamp(pairs["test_start_date"].iloc[0])
            train_closes = closes_daily[closes_daily.index < test_start]
        else:
            from config import TRAIN_RATIO
            split = int(len(closes_daily) * TRAIN_RATIO)
            train_closes = closes_daily.iloc[:split]
        hrp_w = compute_hrp_weights(train_closes, pairs)
        hrp_df = pd.DataFrame({"pair": list(hrp_w.keys()),
                               "hrp_weight": list(hrp_w.values())})
        hrp_df.to_csv(DATA_DIR / "hrp_weights.csv", index=False)
        print(f"  HRP weights computed on {len(train_closes)} train days  "
              f"→ saved {DATA_DIR / 'hrp_weights.csv'}")
        print(f"  HRP weight range: {min(hrp_w.values()):.2f} – {max(hrp_w.values()):.2f}  "
              f"(mean=1.00 by construction)")
    except Exception as e:
        print(f"  HRP weights skipped: {e}")
        hrp_w = {p: 1.0 for p in pair_names}

    if regimes is not None:
        timestamps = regimes.index
    elif vix is not None:
        timestamps = pd.date_range(vix.index[0], vix.index[-1], freq="B").tz_localize("US/Eastern")
    else:
        print("No time-series data available — cannot build sizing grid.")
        raise SystemExit(0)

    # Convert timestamps to tz-naive normalized dates
    dates_naive = timestamps.normalize().tz_localize(None)

    # Pre-align daily series to timestamps once (outside the loop)
    if macro_alert_s is not None and not macro_alert_s.empty:
        macro_alert_aligned = macro_alert_s.asof(dates_naive).fillna(0).astype(int).to_numpy()
    else:
        macro_alert_aligned = np.zeros(len(timestamps), dtype=int)

    if global_hmm_s is not None and not global_hmm_s.empty:
        global_hmm_aligned = global_hmm_s.asof(dates_naive).fillna(0).astype(int).to_numpy()
        g_mult = np.where(global_hmm_aligned == 1, HMM_PANIC_MULT, 1.0)
    else:
        g_mult = np.ones(len(timestamps))

    if not iv_mult_s.empty:
        i_mult = iv_mult_s.asof(dates_naive).fillna(IV_MULT_MAX).to_numpy()
    else:
        i_mult = np.full(len(timestamps), IV_MULT_MAX)

    if corr_throttle_s is not None and not corr_throttle_s.empty:
        c_mult = corr_throttle_s.asof(dates_naive).fillna(1.0).to_numpy()
    else:
        c_mult = np.ones(len(timestamps))

    sizes = {}
    for pair in pair_names:
        # Per-pair HMM regime multiplier
        if regimes is not None and pair in regimes.columns:
            regime_series = regimes[pair].reindex(timestamps).fillna(0).to_numpy()
            r_mult = np.where(regime_series == 1, REGIME_MULT_VOLATILE, REGIME_MULT_NORMAL)
        else:
            r_mult = np.full(len(timestamps), REGIME_MULT_NORMAL)

        # MC confidence (scalar)
        m = mc_conf.get(pair, 1.0)

        # HRP weight (scalar)
        h = hrp_w.get(pair, 1.0) if hrp_w else 1.0

        # Vectorised multiplication
        sz = r_mult * g_mult * i_mult * m * c_mult * h
        
        # Apply VIX9D backwardation block/multiplier (alert -> size *= 0.5)
        sz = np.where(macro_alert_aligned == 1, sz * 0.5, sz)
        
        sizes[pair] = sz

    df_sizes = pd.DataFrame(sizes, index=timestamps)

    print("=" * 60)
    print("POSITION SIZING SUMMARY")
    print("=" * 60)

    for pair in pair_names:
        if pair not in df_sizes.columns:
            continue
        s = df_sizes[pair]
        mc = mc_conf.get(pair, 1.0)
        pct_skipped = (s < MIN_POSITION_SIZE).mean() * 100

        print(f"\n  {pair}")
        print(f"    MC confidence:   {mc:.2f}  ({mc*100:.0f}% sim win rate)")
        print(f"    Size mean:       {s.mean():.2f}")
        print(f"    Size min / max:  {s.min():.2f} / {s.max():.2f}")
        print(f"    Trades skipped:  {pct_skipped:.0f}%  (size < {MIN_POSITION_SIZE})")

        latest = timestamps[-1]
        r = regime_multiplier(regimes, pair, latest)
        i = iv_multiplier(iv_mult_s, latest.date() if hasattr(latest, "date") else latest)
        date_latest = latest.date() if hasattr(latest, "date") else latest
        alert = macro_alert_active(macro_alert_s, date_latest)
        g = global_hmm_multiplier(global_hmm_s, date_latest)
        print(f"    Latest (current):")
        print(f"      regime_mult={r:.1f}  global_hmm={g:.3f}  iv_mult={i:.2f}  "
              f"mc_conf={mc:.2f}  macro_alert={'BLOCK' if alert else 'ok'}"
              f"  → size={0.0 if alert else r*g*i*mc:.2f}")

    df_sizes.to_csv(DATA_DIR / "position_sizes.csv")
    print(f"\nSaved {DATA_DIR / 'position_sizes.csv'}  {df_sizes.shape}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    n_pairs = len(pair_names)
    MAX_PAIR_PLOTS = 24
    if n_pairs > MAX_PAIR_PLOTS:
        pair_means = df_sizes.mean().sort_values(ascending=False)
        display_pairs = pair_means.head(MAX_PAIR_PLOTS).index.tolist()
        print(f"Plotting compact sizing summary for top {MAX_PAIR_PLOTS} / {n_pairs} pairs by mean size.")
    else:
        display_pairs = pair_names

    fig_height = 6 + 2.2 * len(display_pairs)
    fig = plt.figure(figsize=(18, fig_height))
    gs = gridspec.GridSpec(len(display_pairs) + 2, 2, figure=fig, hspace=0.5, wspace=0.3)

    ax_reg = fig.add_subplot(gs[0, 0])
    ax_iv = fig.add_subplot(gs[0, 1])

    if regimes is not None and pair_names:
        example = pair_names[0]
        if example in regimes.columns:
            reg_vals = regimes[example].map({0: REGIME_MULT_NORMAL, 1: REGIME_MULT_VOLATILE})
            ax_reg.step(reg_vals.index, reg_vals.values, color="steelblue", lw=1, where="post")
            ax_reg.fill_between(reg_vals.index, reg_vals.values,
                                REGIME_MULT_NORMAL, step="post",
                                where=(reg_vals.values < REGIME_MULT_NORMAL),
                                color="salmon", alpha=0.4)
            ax_reg.set_title(f"Regime Multiplier — {example}", fontsize=9, fontweight="bold")
            ax_reg.set_ylim(0, 1.2)
            ax_reg.set_ylabel("Multiplier")

    if not iv_mult_s.empty:
        ax_iv.plot(iv_mult_s.index, iv_mult_s.values, color="darkorange", lw=1.5)
        ax_iv.fill_between(iv_mult_s.index, iv_mult_s.values, IV_MULT_MAX,
                           color="salmon", alpha=0.3)
        ax_iv.set_title("IV Multiplier (VIX-based, smooth)", fontsize=9, fontweight="bold")
        ax_iv.set_ylim(0, 1.1)
        ax_iv.set_ylabel("Multiplier")

    ax_mc = fig.add_subplot(gs[1, :])
    if mc_conf:
        labels = list(mc_conf.keys())
        vals = [mc_conf[p] for p in labels]
        colors = ["#4CAF50" if v >= 0.6 else "#FF9800" if v >= 0.5 else "#F44336"
                  for v in vals]
        bars = ax_mc.bar(labels, vals, color=colors, edgecolor="white")
        ax_mc.axhline(MIN_POSITION_SIZE, color="red", lw=1.5, linestyle="--",
                      label=f"Min size threshold ({MIN_POSITION_SIZE})")
        ax_mc.axhline(1.0, color="gray", lw=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax_mc.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.01, f"{val:.2f}",
                       ha="center", va="bottom", fontsize=9)
        ax_mc.set_title("MC Confidence per Pair (OU Monte Carlo win rate)", fontsize=9, fontweight="bold")
        ax_mc.set_ylabel("Win rate fraction")
        ax_mc.set_ylim(0, 1.15)
        ax_mc.legend(fontsize=8)

    colors_p = plt.cm.tab20(np.linspace(0, 1, max(len(display_pairs), 1)))
    for idx, (pair, color) in enumerate(zip(display_pairs, colors_p)):
        if pair not in df_sizes.columns:
            continue
        ax = fig.add_subplot(gs[2 + idx, :])
        s = df_sizes[pair]

        ax.fill_between(s.index, s.values, 0,
                        where=(s.values >= MIN_POSITION_SIZE),
                        color=color, alpha=0.5, step="post")
        ax.fill_between(s.index, s.values, 0,
                        where=(s.values < MIN_POSITION_SIZE),
                        color="red", alpha=0.3, step="post", label="Skipped (too small)")
        ax.step(s.index, s.values, color=color, lw=1.2, where="post")
        ax.axhline(MIN_POSITION_SIZE, color="red", lw=1, linestyle="--")
        ax.axhline(1.0, color="gray", lw=0.5, linestyle="--")
        ax.set_title(f"{pair}  —  Combined Position Size", fontsize=9, fontweight="bold")
        ax.set_ylabel("Size")
        ax.set_ylim(0, 1.1)
        if (s < MIN_POSITION_SIZE).any():
            ax.legend(fontsize=7)

    plt.suptitle("Dynamic Position Sizing — Layer 5\n"
                 "size = regime_mult × iv_mult × mc_confidence",
                 fontsize=12, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "position_sizing.png", dpi=150, bbox_inches="tight")
    print(f"Chart saved to {OUTPUT_DIR / 'position_sizing.png'}")


if __name__ == "__main__":
    main()
