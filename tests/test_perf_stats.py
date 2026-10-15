"""perf_stats: the one Sharpe, the Probabilistic Sharpe Ratio and the Deflated
Sharpe Ratio (Bailey & López de Prado, 2014) — audit task T6.

The expected values are computed by hand here: moments written out with numpy
sums, Φ / Φ⁻¹ from the standard library (statistics.NormalDist), never scipy,
which perf_stats uses."""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtm import mtm_stats  # noqa: E402
from perf_stats import (  # noqa: E402
    deflated_sharpe, portfolio_sharpe, psr, sharpe_per_period, sr0_threshold,
)

N01 = NormalDist()
EULER_GAMMA = 0.5772156649015329          # Euler–Mascheroni

# A short, skewed, fat-tailed daily return series (with zero days).
R = np.array([0.012, -0.004, 0.0, 0.007, -0.021, 0.0, 0.015, 0.003, -0.002, 0.0,
              0.009, 0.031, -0.006, 0.0, 0.004, -0.011, 0.002, 0.0, 0.006, 0.008])


def by_hand(r: np.ndarray) -> tuple[float, int, float, float]:
    """(per-period Sharpe with the sample std, T, skewness, raw kurtosis) —
    population moments m_k = Σ(r − mean)^k / T."""
    t = len(r)
    mean = r.sum() / t
    d = r - mean
    sd = math.sqrt((d ** 2).sum() / (t - 1))
    m2, m3, m4 = (d ** 2).sum() / t, (d ** 3).sum() / t, (d ** 4).sum() / t
    return mean / sd, t, m3 / m2 ** 1.5, m4 / m2 ** 2


def psr_by_hand(r: np.ndarray, sr_star: float) -> float:
    sr, t, g3, g4 = by_hand(r)
    return N01.cdf((sr - sr_star) * math.sqrt(t - 1)
                   / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))


def test_sharpe_is_the_one_definition():
    sr, t, _, _ = by_hand(R)
    assert sharpe_per_period(R) == pytest.approx(sr, rel=1e-12)
    assert portfolio_sharpe(R) == pytest.approx(sr * math.sqrt(252), rel=1e-12)
    # the same number as the daily MTM statistics (mtm.mtm_stats, capital 1)
    s = pd.Series(R, index=pd.bdate_range("2020-01-01", periods=len(R)))
    assert portfolio_sharpe(s) == pytest.approx(mtm_stats(s, 1.0)["sharpe"], rel=1e-12)
    assert portfolio_sharpe(s * 7.0) == pytest.approx(portfolio_sharpe(s), rel=1e-12)


def test_psr_matches_a_hand_computation():
    sr, t, g3, g4 = by_hand(R)
    # the hand numbers are what the series is: skewed, fat-tailed, SR > 0
    assert t == 20 and sr > 0.25 and g3 > 0.4 and g4 > 4.8
    assert psr(R) == pytest.approx(psr_by_hand(R, 0.0), rel=1e-12)
    assert psr(R, sr_benchmark=0.1) == pytest.approx(psr_by_hand(R, 0.1), rel=1e-12)
    # pinned (independent of both implementations above)
    assert psr(R) == pytest.approx(0.8718554, abs=1e-6)


def test_deflated_sharpe_matches_a_hand_computation():
    # Four trials, one without a recorded Sharpe: N = 4, V from the three known.
    trials = [0.05, 0.10, float("nan"), sharpe_per_period(R)]
    known = np.array([0.05, 0.10, sharpe_per_period(R)])
    v = ((known - known.mean()) ** 2).sum() / (len(known) - 1)
    n = 4
    sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * N01.inv_cdf(1 - 1 / n)
                          + EULER_GAMMA * N01.inv_cdf(1 - 1 / (n * math.e)))
    assert sr0_threshold(n, v) == pytest.approx(sr0, rel=1e-12)
    assert deflated_sharpe(R, trials) == pytest.approx(psr_by_hand(R, sr0), rel=1e-12)
    # pinned: SR0 = 0.11193…, DSR = 0.73734…
    assert sr0 == pytest.approx(0.1119259, abs=1e-6)
    assert deflated_sharpe(R, trials) == pytest.approx(0.7373395, abs=1e-6)


def test_dsr_with_one_trial_is_psr_zero():
    assert sr0_threshold(1, 0.0) == 0.0
    assert sr0_threshold(1, 0.04) == 0.0
    assert deflated_sharpe(R, [sharpe_per_period(R)]) == psr(R, 0.0)
    assert deflated_sharpe(R, [float("nan")]) == psr(R, 0.0)


def trials_with_variance(n: int, v: float, centre: float = 0.02) -> np.ndarray:
    """n per-period Sharpe ratios whose sample variance is exactly v."""
    z = np.arange(n, dtype=float)
    z -= z.mean()
    return centre + z * math.sqrt(v / (z ** 2).sum() * (n - 1))


def test_dsr_falls_strictly_as_the_number_of_trials_rises():
    rng = np.random.default_rng(6)
    r = 0.001 + 0.01 * rng.standard_t(5, size=750)      # SR ≈ 0.1 per day, fat tails
    v = 1e-4
    ns = list(range(2, 60)) + [100, 250, 1000, 10_000]
    for n in ns:
        assert np.var(trials_with_variance(n, v), ddof=1) == pytest.approx(v, rel=1e-12)
    sr0 = [sr0_threshold(n, v) for n in [1] + ns]
    dsr = [psr(r, 0.0)] + [deflated_sharpe(r, trials_with_variance(n, v)) for n in ns]
    assert all(np.diff(sr0) > 0)
    assert all(np.diff(dsr) < 0)
    assert 0.0 < dsr[-1] < dsr[0] < 1.0                   # not saturated: the test has power
    # V = 0 (every trial the same Sharpe): SR0 = 0 whatever N
    assert deflated_sharpe(r, [0.05] * 10) == psr(r, 0.0)


def test_edge_cases_raise():
    with pytest.raises(ValueError):
        deflated_sharpe(R, [])                             # N = 0
    with pytest.raises(ValueError):
        sr0_threshold(0, 0.01)
    with pytest.raises(ValueError):
        sr0_threshold(5, -0.01)
    with pytest.raises(ValueError):
        deflated_sharpe(R, [0.1, float("nan"), float("nan")])   # N > 1, V undefined
    for bad in (np.zeros(50), np.full(50, 0.003), np.array([0.01]), np.array([]),
                np.array([0.01, np.nan, 0.02])):
        for f in (sharpe_per_period, portfolio_sharpe, psr):
            with pytest.raises(ValueError):
                f(bad)
        with pytest.raises(ValueError):
            deflated_sharpe(bad, [0.1, 0.2])
