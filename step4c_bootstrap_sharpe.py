"""
step4c_bootstrap_sharpe.py — Bootstrap confidence intervals for Sharpe.

Per-trade Sharpe (mean/std × √trades/yr) hides huge uncertainty.
This computes:
  1. Daily-return Sharpe on actual trading days
  2. iid bootstrap CI (resample days with replacement)
  3. Block bootstrap CI (preserve autocorrelation, block_size=5 days)
  4. Probabilistic Sharpe Ratio (PSR) — probability Sharpe > 0
"""

import pandas as pd
import numpy as np
from scipy import stats
from config import INITIAL_CAPITAL, DATA_DIR

N_BOOT = 10_000
BLOCK_SIZE = 5         # trading days per block (captures weekly autocorrelation)
TARGET_SHARPE = 0.0    # benchmark for PSR

def load_daily_returns(path):
    t = pd.read_csv(path)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t["date"] = t["exit_time"].dt.date
    daily = t.groupby("date")["dollar_pnl"].sum()
    # Build equity → returns
    equity = INITIAL_CAPITAL + daily.cumsum()
    ret = equity.pct_change().dropna()
    return ret


def sharpe(returns, annualizer=252):
    if returns.std() == 0: return 0.0
    return returns.mean() / returns.std() * np.sqrt(annualizer)


def iid_bootstrap(returns, n_boot=N_BOOT):
    """Random resample with replacement — ignores autocorrelation."""
    rng = np.random.default_rng(42)
    sharpes = np.empty(n_boot)
    arr = returns.values
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        sharpes[i] = sharpe(pd.Series(sample))
    return sharpes


def block_bootstrap(returns, block_size=BLOCK_SIZE, n_boot=N_BOOT):
    """Moving-block bootstrap — preserves autocorrelation structure."""
    rng = np.random.default_rng(42)
    arr = returns.values
    n   = len(arr)
    n_blocks = n // block_size + 1
    starts_max = n - block_size + 1
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        blocks = [arr[s:s+block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        sharpes[i] = sharpe(pd.Series(sample))
    return sharpes


def probabilistic_sharpe(returns, target=TARGET_SHARPE):
    """Bailey & López de Prado (2014) — P(Sharpe > target | observed)."""
    sh = sharpe(returns)
    n  = len(returns)
    if n < 3: return 0.5
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=False)
    # Asymptotic distribution of Sharpe under null
    se = np.sqrt((1 - skew*sh + (kurt-1)/4 * sh**2) / (n - 1))
    if se == 0: return 1.0 if sh > target else 0.0
    z = (sh - target) / se
    return float(stats.norm.cdf(z))


# ── Main ──────────────────────────────────────────────────────────────────────
for label, path in [("Base (all trades)", DATA_DIR / "trades.csv"),
                    ("Capped (max-7)",   DATA_DIR / "trades_capped.csv")]:
    if not path.exists(): continue
    ret = load_daily_returns(path)
    sh  = sharpe(ret)

    print(f"\n{'='*60}")
    print(f"{label}  —  {path.name}")
    print(f"{'='*60}")
    print(f"Active days:   {len(ret)}")
    print(f"Sharpe (point):  {sh:.3f}")

    # iid bootstrap
    sh_iid = iid_bootstrap(ret)
    iid_lo, iid_md, iid_hi = np.percentile(sh_iid, [5, 50, 95])
    print(f"\n  iid bootstrap (n={N_BOOT}):")
    print(f"    5%   = {iid_lo:+.3f}")
    print(f"    50%  = {iid_md:+.3f}")
    print(f"    95%  = {iid_hi:+.3f}")
    print(f"    P(Sharpe > 0): {(sh_iid > 0).mean()*100:.1f}%")
    print(f"    P(Sharpe > 1): {(sh_iid > 1).mean()*100:.1f}%")

    # Block bootstrap
    sh_blk = block_bootstrap(ret)
    blk_lo, blk_md, blk_hi = np.percentile(sh_blk, [5, 50, 95])
    print(f"\n  block bootstrap (block={BLOCK_SIZE}d):")
    print(f"    5%   = {blk_lo:+.3f}")
    print(f"    50%  = {blk_md:+.3f}")
    print(f"    95%  = {blk_hi:+.3f}")
    print(f"    P(Sharpe > 0): {(sh_blk > 0).mean()*100:.1f}%")
    print(f"    P(Sharpe > 1): {(sh_blk > 1).mean()*100:.1f}%")

    # Probabilistic Sharpe Ratio
    psr = probabilistic_sharpe(ret, target=0.0)
    psr_1 = probabilistic_sharpe(ret, target=1.0)
    print(f"\n  Probabilistic SR (PSR, López de Prado):")
    print(f"    PSR(0): {psr*100:.1f}%   ← prob strategy is not a fluke")
    print(f"    PSR(1): {psr_1*100:.1f}%   ← prob real Sharpe ≥ 1.0")

    # Verdict
    print(f"\n  Verdict:")
    if blk_lo > 1:
        print("    ✓ STRONG — even worst case (5% CI) gives Sharpe > 1")
    elif blk_lo > 0:
        print("    ~ MODERATE — Sharpe > 0 likely, but margin is thin")
    else:
        print("    ✗ WEAK — 5% CI includes negative Sharpe, may be luck")
