from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BreakDetectorConfig


class BreakCompositeScorer:
    """
    Aggregate heterogeneous break statistics into one actionable score.

    No single fast statistic is reliable enough to hard-stop risk on its own.
    The composite score combines drift, model surprise, local half-life
    explosion, and hedge-ratio instability into one calibrated risk signal.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the composite break score and categorical break levels.

        The logarithms preserve ordering but compress extreme ratio values so a
        single noisy spike does not dominate the score. The explicit ABORT rule
        requires composite evidence or the special explosive-theta plus CUSUM
        confirmation case mandated by the design.
        """

        innovation_term = np.log(np.maximum(frame["innovation_ratio"].astype(float), 1.0))
        hl_term = np.log(np.maximum(frame["hl_ratio"].astype(float), 1.0))
        beta_term = frame["beta_velocity_zscore"].astype(float).clip(lower=0.0)

        score = (
            self.config.weight_cusum * frame["cusum_flag"].astype(float)
            + self.config.weight_innovation * innovation_term
            + self.config.weight_half_life * hl_term
            + self.config.weight_beta_velocity * beta_term
        )

        abort_special = (frame["theta_local"].astype(float) < 0.0) & (frame["cusum_flag"].astype(int) == 1)
        levels = np.full(len(frame), "NORMAL", dtype=object)
        levels = np.where(score >= self.config.threshold_caution, "CAUTION", levels)
        levels = np.where(score >= self.config.threshold_alarm, "ALARM", levels)
        levels = np.where(score >= self.config.threshold_abort, "ABORT", levels)
        levels = np.where(abort_special, "ABORT", levels)

        return pd.DataFrame(
            {
                "break_score": score,
                "break_level": levels,
                "abort_special": abort_special.astype(int),
            },
            index=frame.index,
        )
