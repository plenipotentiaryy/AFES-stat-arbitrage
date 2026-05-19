from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import TailEVConfig


@dataclass(slots=True)
class TailDataset:
    """
    Container with the adaptive threshold path and aligned tail observations.

    Keeping the full threshold series is important because the POT threshold is
    explicitly non-stationary here: each exceedance is measured against the
    threshold that was known at that timestamp, not against a single constant.
    """

    tail_series: pd.Series
    threshold_series: pd.Series
    exceedances: np.ndarray
    exceedance_index: pd.Index
    state_frame: pd.DataFrame

    @property
    def threshold(self) -> float:
        """Return the latest valid threshold for live gating."""
        valid = self.threshold_series.dropna()
        if valid.empty:
            raise ValueError("threshold_series has no valid observations")
        return float(valid.iloc[-1])


class TailDataLayer:
    """
    Extract POT exceedances from normalized spread shocks.

    The layer works on ``abs(z)`` by default because both positive and negative
    spread dislocations represent the same tail-risk magnitude for a
    mean-reversion strategy. Using an adaptive rolling quantile threshold keeps
    the tail sample aligned with local volatility regimes instead of assuming a
    stationary Gaussian scale.
    """

    def __init__(self, config: TailEVConfig):
        self.config = config

    def fit_transform(
        self,
        z_scores: pd.Series,
        context: pd.DataFrame | None = None,
    ) -> TailDataset:
        """
        Build adaptive POT inputs and aligned state vectors.

        The threshold is a shifted rolling quantile so the model only uses
        information available at time ``t``. Exceedances are defined as
        ``Y_t = |Z_t| - u_t`` whenever ``|Z_t| > u_t``.
        """

        if not isinstance(z_scores, pd.Series):
            raise TypeError("z_scores must be a pandas Series")

        tail_series = z_scores.astype(float).abs().rename("tail_z")
        threshold_series = (
            tail_series.rolling(
                window=self.config.rolling_window,
                min_periods=self.config.min_periods,
            )
            .quantile(self.config.threshold_quantile)
            .shift(1)
        )
        threshold_series = threshold_series.fillna(
            tail_series.expanding(min_periods=max(5, self.config.min_periods // 4))
            .quantile(self.config.threshold_quantile)
            .shift(1)
        )

        mask = tail_series.gt(threshold_series)
        exceedance_index = tail_series.index[mask.fillna(False)]
        exceedances = (
            tail_series.loc[exceedance_index] - threshold_series.loc[exceedance_index]
        ).to_numpy(dtype=float)

        state_frame = self.build_state_frame(z_scores, threshold_series, context)
        state_frame = state_frame.loc[exceedance_index]

        return TailDataset(
            tail_series=tail_series,
            threshold_series=threshold_series,
            exceedances=exceedances,
            exceedance_index=exceedance_index,
            state_frame=state_frame,
        )

    def build_state_frame(
        self,
        z_scores: pd.Series,
        threshold_series: pd.Series,
        context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Assemble the conditional state vector for each timestamp.

        Only a few features can be derived directly from ``z``. The remaining
        columns are accepted from external context and default to conservative,
        neutral values when unavailable. This keeps the layer reusable in
        backtests where microstructure features like OFI are absent.
        """

        z_scores = z_scores.astype(float)
        tail_series = z_scores.abs()
        velocity = tail_series.diff().fillna(0.0)
        short_vol = tail_series.rolling(21, min_periods=5).std()
        long_vol = tail_series.rolling(
            self.config.rolling_window,
            min_periods=self.config.min_periods,
        ).std()
        vol_ratio = (short_vol / long_vol.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

        frame = pd.DataFrame(
            {
                "z_score": tail_series,
                "z_velocity": velocity,
                "vol_ratio": vol_ratio.fillna(1.0),
                "OFI": 0.0,
                "macro_regime": 1.0,
                "hurst_exp": 0.5,
                "coint_score": 0.0,
                "threshold": threshold_series,
                "tail_z": tail_series,
            },
            index=z_scores.index,
        )

        if context is not None:
            if not isinstance(context, pd.DataFrame):
                raise TypeError("context must be a pandas DataFrame when provided")
            ctx = context.copy()
            aliases = {
                "hurst": "hurst_exp",
                "Hurst": "hurst_exp",
                "cointegration_score": "coint_score",
                "coint_tstat": "coint_score",
                "ofi": "OFI",
            }
            ctx = ctx.rename(columns=aliases)
            for col in ("OFI", "macro_regime", "hurst_exp", "coint_score", "vol_ratio"):
                if col in ctx.columns:
                    frame[col] = ctx[col].reindex(frame.index)

        frame["vol_ratio"] = frame["vol_ratio"].astype(float).fillna(1.0)
        frame["OFI"] = frame["OFI"].astype(float).fillna(0.0)
        frame["macro_regime"] = frame["macro_regime"].astype(float).fillna(1.0)
        frame["hurst_exp"] = frame["hurst_exp"].astype(float).fillna(0.5)
        frame["coint_score"] = frame["coint_score"].astype(float).fillna(0.0)
        frame["threshold"] = frame["threshold"].ffill().bfill()
        frame["tail_z"] = frame["tail_z"].astype(float)
        return frame
