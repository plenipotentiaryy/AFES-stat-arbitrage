"""perf_stats.py — the one Sharpe ratio, the Probabilistic Sharpe Ratio and the
Deflated Sharpe Ratio (audit task T6).

- `portfolio_sharpe(daily_returns)`: THE Sharpe of the backtest — daily
  mark-to-market portfolio returns on constant capital, every business day,
  zeros included: mean / std · √252 (sample std, ddof = 1; the same number as
  `mtm.mtm_stats`).
- `psr(returns, sr_benchmark)`: the Probabilistic Sharpe Ratio of Bailey &
  López de Prado (2012/2014) on the per-period (daily, not annualised) Sharpe.
- `deflated_sharpe(returns, sr_trials)`: the Deflated Sharpe Ratio of Bailey &
  López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality" — the PSR against the
  expected maximum Sharpe of N trials with no skill.

Notation (per period, i.e. per day): SR̂ = mean / std of the candidate's
returns, T = their number, γ₃ = their skewness, γ₄ = their kurtosis (raw, not
excess: 3 for a normal distribution; population moments, scipy's defaults),
N = number of trials, V = sample variance of the trials' per-period Sharpe
ratios, Φ = the standard normal CDF:

    PSR(SR*) = Φ( (SR̂ − SR*)·√(T − 1) / √(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²) )
    SR₀      = √V · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))   (N = 1: SR₀ = 0)
    DSR      = PSR(SR₀)

with γ = 0.5772… (Euler–Mascheroni) and e Euler's number.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = float(np.euler_gamma)          # 0.5772156649…
PERIODS_PER_YEAR = 252


def _returns(returns) -> np.ndarray:
    """The returns as a float array: at least 2 values, no NaN, not constant."""
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        raise ValueError(f"a Sharpe ratio needs at least 2 returns, got {len(r)}")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or inf")
    if r.max() == r.min():
        raise ValueError("returns are constant: the Sharpe ratio is undefined")
    return r


def sharpe_per_period(returns) -> float:
    """Per-period Sharpe ratio: mean / sample std (ddof = 1), not annualised."""
    r = _returns(returns)
    return float(r.mean() / r.std(ddof=1))


def portfolio_sharpe(daily_returns) -> float:
    """The one Sharpe definition: daily MTM portfolio returns on constant
    capital over all business days (zeros included), mean / std · √252."""
    return sharpe_per_period(daily_returns) * math.sqrt(PERIODS_PER_YEAR)


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period SR > sr_benchmark) given the
    observed per-period SR̂, T, skewness γ₃ and raw kurtosis γ₄ of `returns`."""
    r = _returns(returns)
    sr = float(r.mean() / r.std(ddof=1))
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    var = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if not var > 0.0:               # ≥ 0 for every distribution; 0 only if degenerate
        raise ValueError(f"PSR undefined: 1 − γ₃·SR + (γ₄ − 1)/4·SR² = {var}")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(len(r) - 1) / math.sqrt(var)))


def sr0_threshold(n_trials: int, var_trials: float) -> float:
    """SR₀, the expected maximum per-period Sharpe of `n_trials` trials with
    no skill whose Sharpe ratios have variance `var_trials`; 0 for one trial."""
    if n_trials < 1:
        raise ValueError(f"the number of trials must be ≥ 1, got {n_trials}")
    if not var_trials >= 0.0:
        raise ValueError(f"the variance of the trials' Sharpe ratios must be ≥ 0, got {var_trials}")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    return math.sqrt(var_trials) * ((1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n)
                                    + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n * math.e)))


def deflated_sharpe(returns, sr_trials) -> float:
    """Deflated Sharpe Ratio of the candidate `returns` (per-period series).

    sr_trials: the per-period Sharpe ratio of every trial INCLUDING the
    candidate; NaN for a trial whose Sharpe is not recorded (it still counts).
    N = len(sr_trials); V = sample variance (ddof = 1) of the recorded ones.
    N = 1 → SR₀ = 0, so DSR = PSR(0)."""
    trials = np.asarray(sr_trials, dtype=float).ravel()
    n = len(trials)
    if n == 0:
        raise ValueError("no trials: N must be ≥ 1 (the candidate itself is a trial)")
    if n == 1:
        return psr(returns, 0.0)
    known = trials[np.isfinite(trials)]
    if len(known) < 2:
        raise ValueError(f"N = {n} trials but {len(known)} recorded Sharpe ratio(s): "
                         "the variance V is undefined")
    return psr(returns, sr0_threshold(n, float(np.var(known, ddof=1))))
