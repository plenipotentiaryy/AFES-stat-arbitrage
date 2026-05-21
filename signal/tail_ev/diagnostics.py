from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import TailEVConfig


@dataclass(slots=True)
class ADTestResult:
    statistic: float
    p_value: float


class MeanExcessPlot:
    """
    Compute and optionally plot the mean excess curve.

    For a valid GPD tail regime the mean excess curve should become roughly
    linear beyond the threshold. This makes it a practical first diagnostic for
    deciding whether the rolling 90th percentile is entering a stable tail
    region instead of a mixed center+tail sample.
    """

    def compute(
        self,
        series: pd.Series,
        thresholds: np.ndarray | None = None,
    ) -> pd.DataFrame:
        tail = pd.Series(series, dtype=float).dropna().abs()
        if thresholds is None:
            low = float(tail.quantile(0.70))
            high = float(tail.quantile(0.98))
            thresholds = np.linspace(low, high, 25)

        rows: list[dict[str, float]] = []
        for threshold in thresholds:
            exceedances = tail[tail > threshold] - threshold
            if len(exceedances) < 5:
                continue
            rows.append(
                {
                    "threshold": float(threshold),
                    "mean_excess": float(exceedances.mean()),
                    "n_exceedances": float(len(exceedances)),
                }
            )
        return pd.DataFrame(rows)

    def plot(self, summary: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(summary["threshold"], summary["mean_excess"], color="steelblue", lw=2)
        ax.set_title("Mean Excess Plot")
        ax.set_xlabel("Threshold u")
        ax.set_ylabel("E[X - u | X > u]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


class TailQQPlot:
    """
    Compare empirical exceedance quantiles against fitted GPD quantiles.

    Q-Q diagnostics are necessary because the POT fit is only useful if the
    fitted GPD preserves the far-tail shape. In AFES this matters directly for
    sizing loss estimates in the same region where position size is largest.
    """

    def compute(self, exceedances: np.ndarray, xi: float, beta: float) -> pd.DataFrame:
        sample = np.sort(np.asarray(exceedances, dtype=float))
        if sample.size == 0:
            return pd.DataFrame(columns=["probability", "empirical", "theoretical"])
        probs = (np.arange(1, sample.size + 1) - 0.5) / sample.size
        theoretical = stats.genpareto.ppf(probs, c=xi, loc=0.0, scale=beta)
        return pd.DataFrame(
            {
                "probability": probs,
                "empirical": sample,
                "theoretical": theoretical,
            }
        )

    def plot(self, qq_frame: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            qq_frame["theoretical"],
            qq_frame["empirical"],
            s=20,
            color="darkorange",
            alpha=0.8,
        )
        upper = float(
            np.nanmax(
                np.r_[qq_frame["theoretical"].to_numpy(), qq_frame["empirical"].to_numpy()]
            )
        )
        ax.plot([0, upper], [0, upper], color="black", ls="--", lw=1)
        ax.set_title("Tail Q-Q vs GPD")
        ax.set_xlabel("GPD quantile")
        ax.set_ylabel("Empirical quantile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def gpd_anderson_darling_test(
    exceedances: np.ndarray,
    xi: float,
    beta: float,
    bootstrap_samples: int = 200,
    random_state: int = 42,
) -> ADTestResult:
    """
    Estimate an Anderson-Darling p-value for the fitted GPD via PIT bootstrap.

    The exceedances are transformed through the fitted GPD CDF into uniforms.
    Under correct specification those uniforms should look i.i.d. U(0,1). A
    bootstrap is used because the classical AD critical values are not directly
    available for this fitted-composite GPD case.
    """

    sample = np.asarray(exceedances, dtype=float)
    if sample.size < 8:
        return ADTestResult(statistic=np.nan, p_value=np.nan)

    uniforms = np.clip(
        stats.genpareto.cdf(sample, c=xi, loc=0.0, scale=beta),
        1e-10,
        1 - 1e-10,
    )
    observed = _uniform_ad_statistic(uniforms)

    rng = np.random.default_rng(random_state)
    boot = np.empty(bootstrap_samples, dtype=float)
    for i in range(bootstrap_samples):
        sim = stats.genpareto.rvs(
            c=xi,
            loc=0.0,
            scale=beta,
            size=sample.size,
            random_state=rng,
        )
        sim_u = np.clip(
            stats.genpareto.cdf(sim, c=xi, loc=0.0, scale=beta),
            1e-10,
            1 - 1e-10,
        )
        boot[i] = _uniform_ad_statistic(sim_u)

    p_value = float((np.sum(boot >= observed) + 1.0) / (bootstrap_samples + 1.0))
    return ADTestResult(statistic=float(observed), p_value=p_value)


class StabilityTest:
    """
    Re-fit the tail shape over a threshold grid and inspect ``xi(u)``.

    Stable ``xi`` across a reasonable threshold range is the standard POT check
    that the selected threshold sits inside the true tail regime. Strong drift
    in ``xi(u)`` is treated as evidence of a mixed distribution or regime break.
    """

    def __init__(self, config: TailEVConfig):
        self.config = config

    def compute(
        self,
        series: pd.Series,
        threshold_quantiles: np.ndarray | None = None,
    ) -> pd.DataFrame:
        tail = pd.Series(series, dtype=float).dropna().abs()
        if threshold_quantiles is None:
            threshold_quantiles = np.linspace(0.85, 0.97, self.config.stability_points)

        rows: list[dict[str, float]] = []
        for q in threshold_quantiles:
            threshold = float(tail.quantile(float(q)))
            exceedances = (tail[tail > threshold] - threshold).to_numpy(dtype=float)
            if exceedances.size < self.config.min_exceedances:
                continue
            xi, _, beta = stats.genpareto.fit(exceedances, floc=0.0)
            rows.append(
                {
                    "quantile": float(q),
                    "threshold": threshold,
                    "xi": float(xi),
                    "beta": float(beta),
                    "n_exceedances": float(exceedances.size),
                }
            )
        return pd.DataFrame(rows)

    def plot(self, stability_frame: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(stability_frame["threshold"], stability_frame["xi"], marker="o", lw=2)
        ax.set_title("Shape Stability xi(u)")
        ax.set_xlabel("Threshold u")
        ax.set_ylabel("xi")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def _uniform_ad_statistic(uniforms: np.ndarray) -> float:
    uniforms = np.sort(np.asarray(uniforms, dtype=float))
    n = uniforms.size
    i = np.arange(1, n + 1)
    terms = (2 * i - 1) * (np.log(uniforms) + np.log(1 - uniforms[::-1]))
    return float(-n - np.mean(terms))
