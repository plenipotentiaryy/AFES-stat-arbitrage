from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import TailEVConfig
from .diagnostics import StabilityTest, TailQQPlot, gpd_anderson_darling_test


@dataclass(slots=True)
class GPDFitResult:
    xi: float
    beta: float
    threshold: float
    n_exceedances: int
    log_likelihood: float
    qq_frame: pd.DataFrame
    ad_statistic: float
    ad_pvalue: float
    stability_frame: pd.DataFrame


def gpd_expected_shortfall(
    z: float,
    xi: float,
    beta: float,
    u: float,
    n: int,
    N: int,
    alpha: float = 0.95,
) -> float:
    """
    Compute POT expected shortfall above the current tail level.

    ``z`` is used to raise the effective confidence level whenever the current
    live dislocation already sits deeper in the tail than the base ``alpha``.
    This matters operationally: once the strategy is already inside a severe
    overshoot, risk should be conditioned on that location rather than on a
    generic ES95 anchored closer to the threshold.
    """

    if N <= 0 or n <= 0:
        raise ValueError("n and N must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if xi >= 1:
        raise ValueError("Expected shortfall is undefined for xi >= 1")

    z = float(max(z, u))
    tail_fraction = n / float(N)
    if xi == 0.0:
        tail_prob_at_z = tail_fraction * np.exp(-(z - u) / beta)
        effective_alpha = max(alpha, 1.0 - tail_prob_at_z)
        var_alpha = u - beta * np.log((N / n) * (1.0 - effective_alpha))
        es_alpha = var_alpha + beta
        return float(max(es_alpha, z))

    tail_prob_at_z = tail_fraction * (1.0 + xi * (z - u) / beta) ** (-1.0 / xi)
    effective_alpha = max(alpha, 1.0 - tail_prob_at_z)
    scale_term = (N / n * (1.0 - effective_alpha)) ** (-xi)
    var_alpha = u + beta / xi * (scale_term - 1.0)
    es_alpha = (var_alpha + beta - xi * u) / (1.0 - xi)
    return float(max(es_alpha, z))


class GPDFitter:
    """
    Fit a generalized Pareto tail to POT exceedances.

    MLE is used because it is the standard estimator for POT tails and it gives
    a direct route to VaR/ES analytics. The fitter also runs three diagnostics:
    tail Q-Q, AD goodness-of-fit, and ``xi(u)`` stability over thresholds.
    """

    def __init__(self, config: TailEVConfig):
        self.config = config

    def fit(
        self,
        exceedances: np.ndarray,
        tail_series: pd.Series,
        threshold: float,
    ) -> GPDFitResult:
        """
        Estimate ``xi`` and ``beta`` on threshold exceedances.

        Negative ``xi`` is rejected by default because a bounded tail is rarely
        a credible financial-spread assumption; when MLE drifts below zero that
        usually signals threshold contamination or regime mixing rather than a
        truly bounded support.
        """

        sample = np.asarray(exceedances, dtype=float)
        sample = sample[np.isfinite(sample)]
        if sample.size < self.config.min_exceedances:
            raise ValueError(
                f"need at least {self.config.min_exceedances} exceedances, got {sample.size}"
            )

        xi, _, beta = stats.genpareto.fit(sample, floc=0.0)
        xi = float(xi)
        beta = float(beta)
        if xi < 0.0 and not self.config.allow_negative_xi:
            warnings.warn(
                "Negative xi detected; clamping to xi_floor because AFES treats "
                "bounded financial-spread tails as economically implausible.",
                RuntimeWarning,
            )
            xi = float(self.config.xi_floor)

        log_likelihood = float(
            np.sum(stats.genpareto.logpdf(sample, c=xi, loc=0.0, scale=beta))
        )
        qq_frame = TailQQPlot().compute(sample, xi, beta)
        ad_result = gpd_anderson_darling_test(
            sample,
            xi,
            beta,
            bootstrap_samples=self.config.ad_bootstrap_samples,
            random_state=self.config.random_state,
        )
        stability_frame = StabilityTest(self.config).compute(tail_series)
        return GPDFitResult(
            xi=xi,
            beta=beta,
            threshold=float(threshold),
            n_exceedances=int(sample.size),
            log_likelihood=log_likelihood,
            qq_frame=qq_frame,
            ad_statistic=float(ad_result.statistic),
            ad_pvalue=float(ad_result.p_value),
            stability_frame=stability_frame,
        )
