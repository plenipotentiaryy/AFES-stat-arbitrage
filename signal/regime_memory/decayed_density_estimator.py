from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import RegimeMemoryConfig
from .regime_classifier import adjacent_regimes

LOGGER = logging.getLogger(__name__)


class DensityProfilerBase(ABC):
    """
    Common profiler interface for legacy and regime-aware estimators.

    The abstract base class exists so the new manager can replace the old
    expanding-window logic without forcing downstream callers to care about the
    internal weighting or memory-bank machinery.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the profiler on historical observations."""

    @abstractmethod
    def density(self, z: float) -> float:
        """Estimate the density at ``z``."""

    @abstractmethod
    def p_revert(self, z: float) -> float:
        """Estimate the conditional reversion probability at ``z``."""

    @abstractmethod
    def ev(self, z: float, gain_fn: Callable, loss_fn: Callable) -> float | None:
        """Estimate expected value at ``z`` using supplied gain/loss functions."""

    @abstractmethod
    def effective_sample_size(self) -> float:
        """Return the effective sample size supporting the current profile."""


class DecayedDensityEstimator(DensityProfilerBase):
    """
    Weighted KDE and weighted reversion estimator conditioned on regime memory.

    The estimator uses two filters at once: time decay for recency and
    half-life distance for regime similarity. This is the core mechanism that
    prevents stale fast-reversion regimes from numerically dominating the
    current slow-reversion profile.
    """

    def __init__(self, config: RegimeMemoryConfig):
        self.config = config
        self.memory_: pd.DataFrame | None = None
        self.weights_: np.ndarray | None = None
        self.bandwidth_: float | None = None
        self.current_regime_: str | None = None
        self.current_hl_: float | None = None
        self.current_timestamp_: pd.Timestamp | None = None
        self.n_eff_: float = 0.0
        self.warning_thin_memory_: bool = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit weighted density and reversion maps from labeled memory.

        The most recent row is treated as the current state because profile
        rebuilds occur after ingesting the newest observation. That gives the
        estimator access to ``HL_now``, current regime, and current timestamp
        without breaking the required ``fit(data)`` interface.
        """

        if data.empty:
            raise ValueError("cannot fit density estimator on empty memory")

        frame = data.copy().sort_values("timestamp").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        latest = frame.iloc[-1]
        self.current_timestamp_ = pd.Timestamp(latest["timestamp"])
        self.current_hl_ = float(latest["hl_estimate"])
        self.current_regime_ = str(latest["regime_label"]).upper()
        self.warning_thin_memory_ = False

        weights = self._compute_weights(frame, allow_adjacent=False)
        if float(weights.sum()) < self.config.min_effective_weight:
            self.warning_thin_memory_ = True
            LOGGER.warning("Thin regime memory — using adjacent regime fallback.")
            weights = self._compute_weights(frame, allow_adjacent=True)

        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(weights) & (weights > 0.0)
        frame = frame.loc[valid].reset_index(drop=True)
        weights = weights[valid]
        if len(frame) == 0 or weights.sum() <= 0.0:
            raise ValueError("no positive weights available after regime weighting")

        self.memory_ = frame
        self.weights_ = weights / weights.sum()
        self.n_eff_ = self._effective_n(self.weights_)
        self.bandwidth_ = self._weighted_silverman_bandwidth(frame["z_score"].to_numpy(dtype=float), self.weights_)

    def density(self, z: float) -> float:
        """
        Evaluate the weighted Gaussian KDE at one z-score.

        Gaussian kernels are kept for continuity with the existing KDE usage in
        AFES, but the weighting now reflects relevance instead of raw count.
        """

        self._require_fit()
        z_values = self.memory_["z_score"].to_numpy(dtype=float)
        u = (float(z) - z_values) / max(self.bandwidth_, 1e-8)
        kernels = norm.pdf(u) / max(self.bandwidth_, 1e-8)
        return float(np.sum(self.weights_ * kernels))

    def p_revert(self, z: float) -> float:
        """
        Estimate local weighted reversion probability near ``z``.

        The primary estimator uses a local neighborhood within the fitted
        bandwidth because the quantity of interest is conditional on similar
        spread stretch. A kernel-weighted fallback is used only if that local
        neighborhood is empty.
        """

        self._require_fit()
        z_values = self.memory_["z_score"].to_numpy(dtype=float)
        reverted = self.memory_["reverted"].to_numpy(dtype=float)
        bandwidth = max(self.bandwidth_, 1e-8)
        mask = np.abs(z_values - float(z)) <= bandwidth
        if mask.any():
            local_w = self.weights_[mask]
            return float(np.sum(local_w * reverted[mask]) / np.sum(local_w))

        kernels = norm.pdf((float(z) - z_values) / bandwidth)
        kernel_w = self.weights_ * kernels
        denom = kernel_w.sum()
        if denom <= 0.0:
            return float(np.average(reverted, weights=self.weights_))
        return float(np.sum(kernel_w * reverted) / denom)

    def ev(self, z: float, gain_fn: Callable, loss_fn: Callable) -> float | None:
        """
        Estimate expected value at ``z`` or refuse when memory quality is weak.

        If effective sample size is below the configured minimum the estimator
        logs a warning and returns ``None``. That converts thin memory into an
        explicit uncertainty state instead of a silently fabricated EV surface.
        """

        self._require_fit()
        if self.current_regime_ == "UNSTABLE":
            LOGGER.warning("Current regime is UNSTABLE — EV surface blocked regardless of sign.")
            return None
        if self.n_eff_ < self.config.min_effective_sample_size:
            LOGGER.warning(
                "Effective sample size %.2f is below minimum %.2f — EV surface blocked.",
                self.n_eff_,
                self.config.min_effective_sample_size,
            )
            return None

        p = self.p_revert(z)
        gain = float(gain_fn(z))
        loss = float(loss_fn(z))
        return float(p * gain - (1.0 - p) * loss)

    def effective_sample_size(self) -> float:
        """Return the fitted effective sample size for diagnostics and gating."""

        return float(self.n_eff_)

    def weight_frame(self) -> pd.DataFrame:
        """
        Expose fitted memory with normalized weights for diagnostics.

        This is used by the contamination audit to inspect how much of the
        active profile is still being driven by old observations.
        """

        self._require_fit()
        frame = self.memory_.copy()
        frame["weight"] = self.weights_
        return frame

    def _compute_weights(self, frame: pd.DataFrame, allow_adjacent: bool) -> np.ndarray:
        current_regime = str(self.current_regime_).upper()
        current_hl = max(float(self.current_hl_), 1e-8)
        now = pd.Timestamp(self.current_timestamp_)

        ages = (now - pd.to_datetime(frame["timestamp"], utc=True)).dt.days.to_numpy(dtype=float)
        hl_values = frame["hl_estimate"].to_numpy(dtype=float)
        regime_values = frame["regime_label"].astype(str).str.upper().to_numpy()

        temporal = np.exp(-self.config.temporal_decay_lambda * np.clip(ages, 0.0, None))
        hl_distance = np.exp(
            -self.config.regime_distance_gamma * np.abs(hl_values - current_hl) / current_hl
        )

        if allow_adjacent:
            adjacent = set(adjacent_regimes(current_regime))
            regime_multiplier = np.where(
                regime_values == current_regime,
                1.0,
                np.where(
                    np.isin(regime_values, list(adjacent)),
                    self.config.adjacent_regime_penalty,
                    0.0,
                ),
            )
        else:
            regime_multiplier = (regime_values == current_regime).astype(float)

        return temporal * hl_distance * regime_multiplier

    @staticmethod
    def _effective_n(weights: np.ndarray) -> float:
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        return float(total**2 / np.sum(np.square(weights)))

    def _weighted_silverman_bandwidth(self, z_values: np.ndarray, weights: np.ndarray) -> float:
        std = self._weighted_std(z_values, weights)
        n_eff = max(self._effective_n(weights), 1.0)
        bandwidth = 0.9 * max(std, 1e-4) * n_eff ** (-1.0 / 5.0)
        return float(max(bandwidth, 0.05))

    @staticmethod
    def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
        mean = float(np.average(values, weights=weights))
        var = float(np.average((values - mean) ** 2, weights=weights))
        return float(np.sqrt(max(var, 1e-12)))

    def _require_fit(self) -> None:
        if self.memory_ is None or self.weights_ is None or self.bandwidth_ is None:
            raise ValueError("density estimator is not fitted")


class LegacyExpandingWindowProfiler(DensityProfilerBase):
    """
    Unweighted expanding-window profiler kept for A/B comparison.

    This wrapper preserves the old logic in one class so the new regime-aware
    estimator can be benchmarked against a direct all-history baseline under the
    same interface contract.
    """

    def __init__(self, config: RegimeMemoryConfig):
        self.config = config
        self.memory_: pd.DataFrame | None = None
        self.bandwidth_: float | None = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the profiler using all historical observations with equal weight.

        This is intentionally the opposite of the regime-aware estimator: every
        row votes equally, which makes it the right control group for A/B tests.
        """

        if data.empty:
            raise ValueError("cannot fit legacy profiler on empty data")
        frame = data.copy().sort_values("timestamp").reset_index(drop=True)
        self.memory_ = frame
        z_values = frame["z_score"].to_numpy(dtype=float)
        n = max(len(z_values), 1)
        std = float(np.std(z_values, ddof=0))
        self.bandwidth_ = max(0.9 * max(std, 1e-4) * n ** (-1.0 / 5.0), 0.05)

    def density(self, z: float) -> float:
        self._require_fit()
        z_values = self.memory_["z_score"].to_numpy(dtype=float)
        u = (float(z) - z_values) / self.bandwidth_
        return float(np.mean(norm.pdf(u) / self.bandwidth_))

    def p_revert(self, z: float) -> float:
        self._require_fit()
        z_values = self.memory_["z_score"].to_numpy(dtype=float)
        reverted = self.memory_["reverted"].to_numpy(dtype=float)
        mask = np.abs(z_values - float(z)) <= self.bandwidth_
        if mask.any():
            return float(np.mean(reverted[mask]))
        kernels = norm.pdf((float(z) - z_values) / self.bandwidth_)
        return float(np.sum(kernels * reverted) / np.sum(kernels))

    def ev(self, z: float, gain_fn: Callable, loss_fn: Callable) -> float | None:
        p = self.p_revert(z)
        return float(p * float(gain_fn(z)) - (1.0 - p) * float(loss_fn(z)))

    def effective_sample_size(self) -> float:
        self._require_fit()
        return float(len(self.memory_))

    def _require_fit(self) -> None:
        if self.memory_ is None or self.bandwidth_ is None:
            raise ValueError("legacy profiler is not fitted")
