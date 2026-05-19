from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BreakDetectorConfig


class CUSUMDetector:
    """
    Recursive CUSUM detector for fast mean-shift recognition.

    CUSUM is used because it accumulates a sequence of small one-sided drifts
    into one statistic. That makes it materially faster than slow structural
    summaries like rolling Hurst when the spread starts walking away from its
    historical mean-reverting regime.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Compute one-sided CUSUM states and alarm flags with explicit resets.

        Reset logic is non-negotiable here:
        1. On zero-cross of the spread, previous break evidence is considered
           stale because the spread has mean-reverted and the prior excursion is
           over.
        2. After an alarm fires, the states are reset so one break does not
           mechanically keep triggering repeated alarms on subsequent bars.
        """

        x = frame["x_t"].astype(float)
        spread = frame["spread"].astype(float)
        mu0 = x.rolling(self.config.mu0_window, min_periods=max(20, self.config.mu0_window // 4)).mean().shift(1)
        sigma_x = x.rolling(self.config.mu0_window, min_periods=max(20, self.config.mu0_window // 4)).std(ddof=0).shift(1)
        mu0 = mu0.fillna(0.0)
        sigma_x = sigma_x.replace(0.0, np.nan).fillna(float(x.std(ddof=0) or 1.0))
        k = self.config.cusum_k_sigma_multiplier * sigma_x
        h = self.config.cusum_h_sigma_multiplier * sigma_x

        s_plus = np.zeros(len(frame), dtype=float)
        s_minus = np.zeros(len(frame), dtype=float)
        flags = np.zeros(len(frame), dtype=int)

        s_p = 0.0
        s_m = 0.0
        spread_values = spread.to_numpy(dtype=float)
        x_values = x.to_numpy(dtype=float)
        mu_values = mu0.to_numpy(dtype=float)
        k_values = k.to_numpy(dtype=float)
        h_values = h.to_numpy(dtype=float)

        for i in range(len(frame)):
            if i > 0 and spread_values[i - 1] * spread_values[i] <= 0.0:
                # Explicit recovery reset: the spread crossed zero, so the old
                # excursion has resolved and we start a new evidence path.
                s_p = 0.0
                s_m = 0.0

            s_p = max(0.0, s_p + x_values[i] - mu_values[i] - k_values[i])
            s_m = min(0.0, s_m + x_values[i] - mu_values[i] + k_values[i])
            fired = int((s_p > h_values[i]) or (abs(s_m) > h_values[i]))

            s_plus[i] = s_p
            s_minus[i] = s_m
            flags[i] = fired

            if fired:
                # Post-alarm reset prevents one structural break from creating
                # repeated duplicate alarms on the same unresolved state.
                s_p = 0.0
                s_m = 0.0

        return pd.DataFrame(
            {
                "mu_0": mu0,
                "sigma_x": sigma_x,
                "cusum_k": k,
                "cusum_h": h,
                "S_plus": s_plus,
                "S_minus": s_minus,
                "cusum_flag": flags,
            },
            index=frame.index,
        )


class KalmanInnovationRatioDetector:
    """
    Detect when the Kalman filter is suddenly surprised by the spread path.

    Structural breaks show up as repeated innovation shocks because the filter's
    local linear state model is no longer adequate. A normalized innovation
    ratio converts that surprise into a scale-free break statistic.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Compare current innovation energy to its recent expected level.

        The denominator is shifted in the data layer so the detector compares
        today's surprise to the variance the model expected before seeing it.
        """

        denom = frame["nu2_rolling_mean"].replace(0.0, np.nan)
        ratio = (frame["nu2_t"] / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        flag = (ratio > self.config.innovation_ratio_threshold).astype(int)
        return pd.DataFrame(
            {
                "innovation_ratio": ratio,
                "innovation_flag": flag,
            },
            index=frame.index,
        )


class LocalHalfLifeExplosionDetector:
    """
    Track local collapse of mean-reversion speed through rolling OU fits.

    A spread can become dangerous before Hurst fully turns because the local OU
    theta starts collapsing first. Monitoring short-window half-life catches
    that loss of mean-reversion speed at the point where risk actually changes.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def compute(self, spread: pd.Series) -> pd.DataFrame:
        """
        Estimate local OU theta, half-life, and explosion flags.

        When theta becomes non-positive the local process is no longer
        mean-reverting. In that case the detector forces an immediate flag,
        because an explosive local fit is qualitatively different from a merely
        slower spread.
        """

        x = pd.Series(spread, dtype=float).dropna()
        theta = pd.Series(np.nan, index=x.index, dtype=float)

        window = self.config.hl_short_window
        for end in range(window, len(x) + 1):
            chunk = x.iloc[end - window : end]
            lag = chunk.shift(1).dropna()
            dx = chunk.diff().dropna()
            aligned = lag.index.intersection(dx.index)
            if len(aligned) < 5:
                continue
            y = dx.loc[aligned].to_numpy(dtype=float)
            x_lag = lag.loc[aligned].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x_lag)), x_lag])
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            theta.iloc[end - 1] = float(-coeffs[1])

        hl_local = pd.Series(np.where(theta > 0.0, np.log(2.0) / theta, 9999.0), index=x.index)
        hl_median = hl_local.shift(1).rolling(
            self.config.hl_median_window,
            min_periods=max(5, self.config.hl_median_window // 2),
        ).median()
        hl_ratio = (hl_local / hl_median.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        hl_ratio = hl_ratio.fillna(1.0)
        explosive = theta <= 0.0
        hl_flag = ((hl_ratio > self.config.hl_ratio_threshold) | explosive).astype(int)

        return pd.DataFrame(
            {
                "theta_local": theta.fillna(0.0),
                "hl_local": hl_local,
                "hl_ratio": hl_ratio,
                "hl_flag": hl_flag,
            },
            index=x.index,
        )


class BetaVelocityMonitor:
    """
    Monitor instability in the rolling hedge ratio.

    Fast beta drift is a leading sign that the cointegration relationship itself
    is deforming. That makes beta velocity a useful early warning even when the
    spread level has not fully expressed the break yet.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize beta velocity against its recent rolling baseline.

        A shifted rolling z-score is used so the detector measures how abnormal
        the latest beta jump is relative to the history that was known before
        the jump arrived.
        """

        velocity = frame["beta_velocity"].astype(float)
        mean = velocity.rolling(
            self.config.beta_velocity_window,
            min_periods=max(5, self.config.beta_velocity_window // 4),
        ).mean().shift(1)
        std = velocity.rolling(
            self.config.beta_velocity_window,
            min_periods=max(5, self.config.beta_velocity_window // 4),
        ).std(ddof=0).shift(1)
        zscore = ((velocity - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        zscore = zscore.fillna(0.0)
        flag = (zscore > self.config.beta_velocity_zscore_threshold).astype(int)
        return pd.DataFrame(
            {
                "beta_velocity_mean": mean.fillna(velocity.expanding().mean()),
                "beta_velocity_std": std.fillna(velocity.expanding().std(ddof=0).replace(0.0, np.nan)).fillna(0.0),
                "beta_velocity_zscore": zscore,
                "beta_flag": flag,
            },
            index=frame.index,
        )
