import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import (
    REGIME_MULT_NORMAL, REGIME_MULT_VOLATILE,
    IV_MULT_MAX, IV_MULT_MIN, IV_LOOKBACK,
    MIN_POSITION_SIZE,
    DATA_DIR, OUTPUT_DIR,
)

# ── Load all sources ──────────────────────────────────────────────────────────

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


def load_iv() -> pd.Series | None:
    path = DATA_DIR / "iv_filter.csv"
    if not path.exists():
        print("  iv_filter.csv not found — iv_mult = 1.0 for all days")
        return None
    df  = pd.read_csv(path, index_col=0, parse_dates=True)
    vix = df["vix"].dropna()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix


def load_mc_confidence() -> dict[str, float]:
    path = DATA_DIR / "ou_montecarlo_summary.csv"
    if not path.exists():
        print("  ou_montecarlo_summary.csv not found — mc_conf = 1.0 for all pairs")
        return {}
    df = pd.read_csv(path)
    return {row["pair"]: row["win_rate"] / 100 for _, row in df.iterrows()}


# ── Compute multipliers ───────────────────────────────────────────────────────

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
        return float(iv_mult_series.asof(key))
    except Exception:
        return IV_MULT_MAX


def mc_confidence(mc_conf: dict, pair: str) -> float:
    return mc_conf.get(pair, 1.0)


def position_size(pair: str, ts, regimes, iv_mult_s, mc_conf: dict) -> float:
    r = regime_multiplier(regimes, pair, ts)
    i = iv_multiplier(iv_mult_s, ts.date() if hasattr(ts, "date") else ts)
    m = mc_confidence(mc_conf, pair)
    return r * i * m


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data sources...\n")

regimes    = load_regimes()
vix        = load_iv()
mc_conf    = load_mc_confidence()
iv_mult_s  = iv_multiplier_series(vix)

pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")
pair_names = list(pairs["pair"])

# ── Build position size grid ──────────────────────────────────────────────────
# Use timestamps from regimes (if available), otherwise from VIX dates
if regimes is not None:
    timestamps = regimes.index
elif vix is not None:
    timestamps = pd.date_range(vix.index[0], vix.index[-1], freq="B").tz_localize("US/Eastern")
else:
    print("No time-series data available — cannot build sizing grid.")
    raise SystemExit(0)

sizes = {}
for pair in pair_names:
    col = []
    for ts in timestamps:
        col.append(position_size(pair, ts, regimes, iv_mult_s, mc_conf))
    sizes[pair] = col

df_sizes = pd.DataFrame(sizes, index=timestamps)

# ── Stats ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("POSITION SIZING SUMMARY")
print("=" * 60)

for pair in pair_names:
    if pair not in df_sizes.columns:
        continue
    s   = df_sizes[pair]
    mc  = mc_conf.get(pair, 1.0)
    pct_skipped = (s < MIN_POSITION_SIZE).mean() * 100

    print(f"\n  {pair}")
    print(f"    MC confidence:   {mc:.2f}  ({mc*100:.0f}% sim win rate)")
    print(f"    Size mean:       {s.mean():.2f}")
    print(f"    Size min / max:  {s.min():.2f} / {s.max():.2f}")
    print(f"    Trades skipped:  {pct_skipped:.0f}%  (size < {MIN_POSITION_SIZE})")

    # Decompose a single bar (latest)
    latest = timestamps[-1]
    r = regime_multiplier(regimes, pair, latest)
    i = iv_multiplier(iv_mult_s, latest.date() if hasattr(latest, "date") else latest)
    print(f"    Latest (current):")
    print(f"      regime_mult={r:.1f}  iv_mult={i:.2f}  mc_conf={mc:.2f}  → size={r*i*mc:.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
df_sizes.to_csv(DATA_DIR / "position_sizes.csv")
print(f"\nSaved {DATA_DIR / 'position_sizes.csv'}  {df_sizes.shape}")

# ── Visualisation ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

n_pairs = len(pair_names)
fig = plt.figure(figsize=(18, 4 + 3 * n_pairs))
gs  = gridspec.GridSpec(n_pairs + 2, 2, figure=fig, hspace=0.5, wspace=0.3)

# Top row: multiplier components
ax_reg = fig.add_subplot(gs[0, 0])
ax_iv  = fig.add_subplot(gs[0, 1])

# Regime multiplier over time (first pair as example)
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

# IV multiplier over time
if not iv_mult_s.empty:
    ax_iv.plot(iv_mult_s.index, iv_mult_s.values, color="darkorange", lw=1.5)
    ax_iv.fill_between(iv_mult_s.index, iv_mult_s.values, IV_MULT_MAX,
                       color="salmon", alpha=0.3)
    ax_iv.set_title("IV Multiplier (VIX-based, smooth)", fontsize=9, fontweight="bold")
    ax_iv.set_ylim(0, 1.1)
    ax_iv.set_ylabel("Multiplier")

# MC confidence bar chart
ax_mc = fig.add_subplot(gs[1, :])
if mc_conf:
    labels = list(mc_conf.keys())
    vals   = [mc_conf[p] for p in labels]
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

# Combined size per pair
colors_p = plt.cm.tab10(np.linspace(0, 1, n_pairs))
for idx, (pair, color) in enumerate(zip(pair_names, colors_p)):
    if pair not in df_sizes.columns:
        continue
    ax = fig.add_subplot(gs[2 + idx, :])
    s  = df_sizes[pair]

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
# plt.show()
