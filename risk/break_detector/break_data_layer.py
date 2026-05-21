from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BreakDetectorConfig


class BreakDetectorDataLayer:
    """
    Prepare aligned inputs for the fast structural-break statistics.

    The layer standardises all raw inputs into a single timestamp-aligned frame.
    This avoids each statistic re-implementing its own alignment logic and
    ensures every detector reacts to the same bar, which matters when the edge
    is speed of regime recognition.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def prepare(
        self,
        spread: pd.Series,
        kalman_innovations: pd.Series,
        beta_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Align spread, innovations, and hedge ratio state into one frame.

        ``x_t`` is the first difference of the spread because a fast break is
        first visible in the spread's incremental motion, not only in its level.
        ``nu_t^2`` and beta velocity are carried forward because they represent
        model surprise and cointegration instability, respectively.
        """

        if not all(isinstance(obj, pd.Series) for obj in (spread, kalman_innovations, beta_series)):
            raise TypeError("spread, kalman_innovations, and beta_series must be pandas Series")

        df = pd.concat(
            [
                spread.rename("spread"),
                kalman_innovations.rename("nu_t"),
                beta_series.rename("beta_t"),
            ],
            axis=1,
            join="inner",
        ).sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        df["x_t"] = df["spread"].diff()
        df["nu2_t"] = df["nu_t"] ** 2
        df["nu2_rolling_mean"] = (
            df["nu2_t"]
            .rolling(self.config.innovation_window, min_periods=max(5, self.config.innovation_window // 4))
            .mean()
            .shift(1)
        )
        df["beta_velocity"] = df["beta_t"].diff().abs()
        return df.dropna().copy()
