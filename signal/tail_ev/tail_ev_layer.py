from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import TailEVConfig
from .gpd_fitter import GPDFitResult, GPDFitter, gpd_expected_shortfall
from .revert_survival_model import RevertSurvivalModel, SurvivalFitReport
from .tail_data_layer import TailDataLayer, TailDataset


@dataclass(slots=True)
class TailEVScore:
    """
    Full conditional tail EV decomposition for one live observation.
    """

    z: float
    threshold: float
    p_revert: float
    expected_gain: float
    expected_shortfall: float
    ev: float
    ev_to_es: float


class TailExpectedShortfall:
    """
    Thin wrapper around the analytical POT expected shortfall formula.

    The wrapper stores the fitted tail parameters and exposes a ``compute(z)``
    method so the execution gate can request location-aware loss estimates
    without re-assembling the GPD inputs every time.
    """

    def __init__(self, fit_result: GPDFitResult, total_observations: int, alpha: float):
        self.fit_result = fit_result
        self.total_observations = int(total_observations)
        self.alpha = float(alpha)

    def compute(self, z: float, threshold: float | None = None) -> float:
        threshold = self.fit_result.threshold if threshold is None else float(threshold)
        return gpd_expected_shortfall(
            z=z,
            xi=self.fit_result.xi,
            beta=self.fit_result.beta,
            u=threshold,
            n=self.fit_result.n_exceedances,
            N=self.total_observations,
            alpha=self.alpha,
        )


class TailEVLayer:
    """
    Orchestrate POT fitting, conditional revert modelling, and ES estimation.

    This is the signal layer: it transforms raw tail observations into the
    three quantities the execution gate needs, namely ``P(revert)``, expected
    gain, and tail loss severity via expected shortfall.
    """

    def __init__(
        self,
        config: TailEVConfig,
        gain_model=None,
    ):
        self.config = config
        self.gain_model = gain_model or self._default_gain_model
        self.data_layer = TailDataLayer(config)
        self.gpd_fitter = GPDFitter(config)
        self.survival_model = RevertSurvivalModel(config)
        self.dataset_: TailDataset | None = None
        self.gpd_fit_: GPDFitResult | None = None
        self.tail_es_: TailExpectedShortfall | None = None
        self.survival_report_: SurvivalFitReport | None = None
        self.labels_: pd.Series | None = None

    def fit(self, z_scores: pd.Series, context: pd.DataFrame | None = None) -> "TailEVLayer":
        """
        Fit the full tail EV stack on a z-score history.

        Labels for the revert model are built from first-passage outcomes:
        success means the spread mean-reverted to ``u/2`` before hitting the
        configured stop multiple. This keeps the classifier aligned with the
        economic question the gate actually asks at trade time.
        """

        dataset = self.data_layer.fit_transform(z_scores, context=context)
        labels = self._build_reversion_labels(dataset)
        X = dataset.state_frame.loc[labels.index]

        self.dataset_ = dataset
        self.labels_ = labels
        self.gpd_fit_ = self.gpd_fitter.fit(
            exceedances=dataset.exceedances,
            tail_series=dataset.tail_series,
            threshold=dataset.threshold,
        )
        self.tail_es_ = TailExpectedShortfall(
            fit_result=self.gpd_fit_,
            total_observations=len(dataset.tail_series.dropna()),
            alpha=self.config.alpha,
        )
        self.survival_report_ = self.survival_model.fit(X, labels)
        return self

    def compute(
        self,
        z: float,
        context: dict | pd.Series | pd.DataFrame | None = None,
        threshold: float | None = None,
    ) -> TailEVScore:
        """
        Compute conditional tail EV for a live tail observation.

        The method leaves the non-tail regime to the caller. It assumes
        ``|z| > u`` and returns the decomposed EV terms so execution can inspect
        not only the final decision but also whether the driver was probability,
        gain, or tail-loss severity.
        """

        if self.dataset_ is None or self.tail_es_ is None or self.gpd_fit_ is None:
            raise ValueError("TailEVLayer must be fitted before compute()")

        z_mag = float(abs(z))
        live_threshold = float(threshold if threshold is not None else self.dataset_.threshold)
        state = self.build_state_vector(z, context=context, threshold=live_threshold)
        p_revert = float(self.survival_model.predict_proba(state)[0])
        expected_gain = float(self.gain_model(z_mag, live_threshold))
        expected_shortfall = float(self.tail_es_.compute(z_mag, threshold=live_threshold))
        ev = p_revert * expected_gain - (1.0 - p_revert) * expected_shortfall
        ratio = ev / expected_shortfall if expected_shortfall > 0 else -np.inf
        return TailEVScore(
            z=float(z),
            threshold=live_threshold,
            p_revert=p_revert,
            expected_gain=expected_gain,
            expected_shortfall=expected_shortfall,
            ev=float(ev),
            ev_to_es=float(ratio),
        )

    def build_state_vector(
        self,
        z: float,
        context: dict | pd.Series | pd.DataFrame | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Build a single-row live state vector in the training feature schema.

        The method tolerates sparse live context and fills absent features with
        neutral values so the gate can still operate in environments where only
        ``z`` and volatility information are available.
        """

        row = {col: 0.0 for col in self.config.feature_columns}
        row["z_score"] = float(abs(z))
        row["z_velocity"] = float(context["z_velocity"]) if isinstance(context, dict) and "z_velocity" in context else 0.0
        row["vol_ratio"] = 1.0
        row["macro_regime"] = 1.0
        row["hurst_exp"] = 0.5

        if isinstance(context, pd.Series):
            row.update({k: float(v) for k, v in context.to_dict().items() if k in row})
        elif isinstance(context, pd.DataFrame):
            if len(context) != 1:
                raise ValueError("context DataFrame must have exactly one row for live scoring")
            row.update({k: float(v) for k, v in context.iloc[0].to_dict().items() if k in row})
        elif isinstance(context, dict):
            row.update({k: float(v) for k, v in context.items() if k in row})

        if threshold is not None:
            row.setdefault("threshold", float(threshold))
        return pd.DataFrame([row], columns=self.config.feature_columns).fillna(0.0)

    def _build_reversion_labels(self, dataset: TailDataset) -> pd.Series:
        tail = dataset.tail_series
        thresholds = dataset.threshold_series
        labels: dict[pd.Timestamp, int] = {}
        for idx in dataset.exceedance_index:
            loc = tail.index.get_loc(idx)
            entry = float(tail.iloc[loc])
            threshold = float(thresholds.iloc[loc])
            target = threshold * self.config.revert_target_fraction
            stop_level = max(
                entry * self.config.stop_multiple,
                threshold + self.config.min_stop_excess,
            )
            future = tail.iloc[loc + 1 : loc + 1 + self.config.max_holding_period]
            if future.empty:
                labels[idx] = 0
                continue

            future_values = future.to_numpy(dtype=float)
            hit_revert = np.where(future_values <= target)[0]
            hit_stop = np.where(future_values >= stop_level)[0]
            first_revert = hit_revert[0] if hit_revert.size else np.inf
            first_stop = hit_stop[0] if hit_stop.size else np.inf
            labels[idx] = int(first_revert < first_stop)
        return pd.Series(labels, dtype=int).sort_index()

    @staticmethod
    def _default_gain_model(z_mag: float, threshold: float) -> float:
        return float(max(z_mag - threshold * 0.50, 0.0))
