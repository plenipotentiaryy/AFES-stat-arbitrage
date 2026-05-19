"""
MCPT (Monte Carlo Permutation Test) + HMM bootstrap.

Question: is the strategy's Sharpe a result of real mean-reversion signal,
or could random data produce similar results?

Method (per pair):
  1. Build the actual Z-score series (Kalman spread → rolling z).
  2. Run mean-reversion strategy with per-pair optimal_params (entry/exit/stop).
  3. Get real Sharpe.
  4. MCPT: permute the bar-to-bar SPREAD RETURNS (preserves marginal distribution,
     destroys mean-reversion autocorrelation). Re-cumsum to fake spread, recompute z,
     run strategy. Repeat N times → null distribution.
  5. HMM null: fit 2-state HMM on spread returns, sample synthetic returns,
     reconstruct spread, run strategy. Repeat N times.
  6. p-value = P(null_sharpe >= real_sharpe).

Limit to top pairs by P&L for speed. Run on actual trading window (2024-2026).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from kalman import kalman_hedge

DATA_DIR = Path("data")
N_PERM    = 300
N_HMM     = 200
TOP_N     = 15
WINDOW_Z  = 60         # rolling Z-score window (bars)
SEED      = 42

# ── Load data ─────────────────────────────────────────────────────────────────
closes = pd.read_parquet(DATA_DIR / "closes_5min.parquet")
opt    = pd.read_csv(DATA_DIR / "optimal_params.csv").set_index("pair")
trades = pd.read_csv(DATA_DIR / "trades.csv")

# Trading window
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
WIN_START = trades["entry_time"].min().tz_convert("US/Eastern")
WIN_END   = trades["entry_time"].max().tz_convert("US/Eastern")
print(f"MCPT window: {WIN_START.date()} → {WIN_END.date()}")

# Restrict closes to window
if closes.index.tz is None:
    closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
mask = (closes.index >= WIN_START) & (closes.index <= WIN_END)
closes = closes.loc[mask]
print(f"Bars in window: {len(closes)}")

# Pick top pairs
top_pairs = (trades.groupby("pair")["net_pnl"].sum()
                   .sort_values(ascending=False).head(TOP_N).index.tolist())
print(f"Testing top {len(top_pairs)} pairs by P&L\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_spread(closes, t1, t2):
    pc = closes[[t1, t2]].dropna()
    if len(pc) < 200:
        return None
    _, _, spread, _ = kalman_hedge(pc[t1].values, pc[t2].values, delta=1e-5)
    return pd.Series(spread, index=pc.index)


def z_score(spread, window=WINDOW_Z):
    mu  = spread.rolling(window, min_periods=window).mean()
    sd  = spread.rolling(window, min_periods=window).std()
    return ((spread - mu) / sd).dropna()


def simulate(z, entry_z, exit_z, stop_z):
    """Simple mean-reversion sim. Returns array of trade P&L in 'z-units'."""
    z = z.values
    trades = []
    position = 0
    entry_idx = 0
    entry_price = 0.0
    for i in range(len(z)):
        zi = z[i]
        if position == 0:
            if zi >= entry_z:
                position, entry_idx, entry_price = -1, i, zi
            elif zi <= -entry_z:
                position, entry_idx, entry_price = +1, i, zi
        else:
            exit_hit = (position == -1 and zi <= exit_z) or \
                       (position == +1 and zi >= -exit_z)
            stop_hit = abs(zi) >= stop_z and np.sign(zi) == -np.sign(entry_price)*-1
            stop_hit = (position == -1 and zi >= stop_z) or \
                       (position == +1 and zi <= -stop_z)
            if exit_hit or stop_hit:
                pnl = (entry_price - zi) if position == -1 else (zi - entry_price)
                trades.append(pnl)
                position = 0
    return np.array(trades)


def sharpe(pnls, bars_per_year=252*78):  # 5min bars: ~78/day
    if len(pnls) < 3:
        return 0.0
    daily = pnls
    mu, sd = daily.mean(), daily.std()
    if sd == 0:
        return 0.0
    # Approximate annualization assuming ~1 trade/day average
    return mu / sd * np.sqrt(len(pnls))  # simplified: Sharpe-like score


def permute_spread(spread, rng):
    rets = np.diff(spread.values)
    rng.shuffle(rets)
    new = np.empty_like(spread.values)
    new[0] = spread.values[0]
    new[1:] = spread.values[0] + np.cumsum(rets)
    return pd.Series(new, index=spread.index)


def hmm_synthetic_spread(spread, rng, n_states=2):
    """Fit simple 2-regime returns and sample. No external deps."""
    rets = np.diff(spread.values)
    # Simple regime: high-vol vs low-vol (median split by |r|)
    abs_r = np.abs(rets)
    thresh = np.median(abs_r)
    hi = rets[abs_r >= thresh]
    lo = rets[abs_r <  thresh]
    # Transition prob: empirical
    state = (abs_r >= thresh).astype(int)
    P = np.zeros((2, 2))
    for i in range(len(state) - 1):
        P[state[i], state[i+1]] += 1
    P = P / P.sum(axis=1, keepdims=True)
    # Sample
    n = len(rets)
    new_state = np.zeros(n, dtype=int)
    new_state[0] = state[0]
    for i in range(1, n):
        new_state[i] = rng.choice(2, p=P[new_state[i-1]])
    new_rets = np.where(
        new_state == 1,
        rng.choice(hi, size=n),
        rng.choice(lo, size=n),
    )
    new = np.empty(n + 1)
    new[0] = spread.values[0]
    new[1:] = spread.values[0] + np.cumsum(new_rets)
    return pd.Series(new, index=spread.index)


# ── Run MCPT per pair ─────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
results = []

for pair in top_pairs:
    t1, t2 = pair.split("-")
    if pair not in opt.index:
        continue
    entry_z = float(opt.loc[pair, "entry_z"])
    exit_z  = float(opt.loc[pair, "exit_z"])
    stop_z  = float(opt.loc[pair, "stop_z"])

    spread = build_spread(closes, t1, t2)
    if spread is None:
        continue

    z = z_score(spread)
    if len(z) < 200:
        continue
    real_pnls = simulate(z, entry_z, exit_z, stop_z)
    real_sh   = sharpe(real_pnls)

    # MCPT
    perm_shs = []
    for _ in range(N_PERM):
        sp_perm = permute_spread(spread, rng)
        z_p     = z_score(sp_perm)
        pnls    = simulate(z_p, entry_z, exit_z, stop_z)
        perm_shs.append(sharpe(pnls))
    perm_shs = np.array(perm_shs)
    p_mcpt   = (perm_shs >= real_sh).mean()

    # HMM bootstrap
    hmm_shs = []
    for _ in range(N_HMM):
        sp_hmm = hmm_synthetic_spread(spread, rng)
        z_h    = z_score(sp_hmm)
        pnls   = simulate(z_h, entry_z, exit_z, stop_z)
        hmm_shs.append(sharpe(pnls))
    hmm_shs = np.array(hmm_shs)
    p_hmm   = (hmm_shs >= real_sh).mean()

    sig_mcpt = "✓" if p_mcpt < 0.05 else ("~" if p_mcpt < 0.10 else "✗")
    sig_hmm  = "✓" if p_hmm  < 0.05 else ("~" if p_hmm  < 0.10 else "✗")

    print(f"{pair:12s}  trades={len(real_pnls):3d}  real_sh={real_sh:+.2f}  "
          f"MCPT p={p_mcpt:.3f} {sig_mcpt}   HMM p={p_hmm:.3f} {sig_hmm}   "
          f"(perm_med={np.median(perm_shs):+.2f}  hmm_med={np.median(hmm_shs):+.2f})")
    results.append({
        "pair": pair, "trades": len(real_pnls), "real_sh": real_sh,
        "p_mcpt": p_mcpt, "p_hmm": p_hmm,
        "perm_med": float(np.median(perm_shs)),
        "hmm_med":  float(np.median(hmm_shs)),
    })

# Summary
res = pd.DataFrame(results)
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Pairs tested:           {len(res)}")
print(f"Significant MCPT (<5%): {(res['p_mcpt']<0.05).sum()}/{len(res)}")
print(f"Significant HMM (<5%):  {(res['p_hmm']<0.05).sum()}/{len(res)}")
print(f"Median MCPT p-value:    {res['p_mcpt'].median():.3f}")
print(f"Median HMM p-value:     {res['p_hmm'].median():.3f}")
print(f"Combined Fisher χ² (MCPT): ", end="")
from scipy.stats import combine_pvalues
stat, p_combo = combine_pvalues(res['p_mcpt'].clip(1e-4, 1.0), method='fisher')
print(f"p={p_combo:.4e}")
stat, p_combo_h = combine_pvalues(res['p_hmm'].clip(1e-4, 1.0), method='fisher')
print(f"Combined Fisher χ² (HMM):  p={p_combo_h:.4e}")

res.to_csv(DATA_DIR / "mcpt_results.csv", index=False)
print(f"\nSaved → data/mcpt_results.csv")
