from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import RegimeMemoryConfig

LOGGER = logging.getLogger(__name__)

REGIME_ORDER = ["FAST", "MEDIUM", "SLOW", "UNSTABLE"]


def adjacent_regimes(regime: str) -> tuple[str, ...]:
    """
    Return the ordered adjacent regimes for thin-memory fallback.

    The adjacency map encodes economic neighborhood: FAST is closest to MEDIUM,
    MEDIUM bridges FAST and SLOW, and SLOW is the closest stable analogue to
    UNSTABLE even though UNSTABLE should not be traded on directly.
    """

    regime = regime.upper()
    mapping = {
        "FAST": ("MEDIUM",),
        "MEDIUM": ("FAST", "SLOW"),
        "SLOW": ("MEDIUM", "UNSTABLE"),
        "UNSTABLE": ("SLOW",),
    }
    return mapping.get(regime, tuple())


class RegimeClassifier:
    """
    Assign canonical regime labels with rule-based fallback and optional K-Means.

    The rule-based classifier is always available and fully interpretable. The
    weekly K-Means path is an adaptive overlay that refines stable-regime
    boundaries once enough labeled history exists, but it never replaces the
    explicit UNSTABLE rules.
    """

    FEATURE_COLUMNS = ["hl_estimate", "hurst", "vol_ratio", "macro_regime"]

    def __init__(self, config: RegimeMemoryConfig):
        self.config = config
        self.scaler_: StandardScaler | None = None
        self.model_: KMeans | None = None
        self.cluster_label_map_: dict[int, str] = {}
        self.last_calibrated_timestamp_: pd.Timestamp | None = None
        self._insufficient_samples_logged_ = False

    def classify(self, observation: pd.Series | dict) -> str:
        """
        Classify one observation into FAST / MEDIUM / SLOW / UNSTABLE.

        Rule-based classification is the cold-start and safety path. It remains
        the fallback whenever K-Means has not been calibrated yet, which keeps
        the system functional and interpretable even with sparse memory.
        """

        row = observation if isinstance(observation, pd.Series) else pd.Series(observation)
        rule_label = self.classify_rule_based(row)
        if rule_label == "UNSTABLE":
            return rule_label
        if self.model_ is None or self.scaler_ is None:
            return rule_label

        features = self._feature_frame(pd.DataFrame([row]))
        cluster = int(self.model_.predict(self.scaler_.transform(features))[0])
        return self.cluster_label_map_.get(cluster, rule_label)

    def classify_rule_based(self, observation: pd.Series | dict) -> str:
        """
        Apply the explicit fast-path regime rules.

        These rules are kept simple on purpose: half-life, Hurst, and break
        score already encode the main stability dimensions and give a reliable
        label when clustering is unavailable or suspicious.
        """

        row = observation if isinstance(observation, pd.Series) else pd.Series(observation)
        hl = float(row.get("hl_estimate", np.inf))
        hurst = float(row.get("hurst", 0.5))
        break_score = float(row.get("break_score", 0.0))

        if break_score > self.config.break_abort_threshold:
            return "UNSTABLE"
        if hurst > self.config.hurst_unstable_threshold:
            return "UNSTABLE"
        if hl < self.config.hl_fast_threshold:
            return "FAST"
        if hl < self.config.hl_slow_threshold:
            return "MEDIUM"
        if hl < self.config.hl_break_threshold:
            return "SLOW"
        return "UNSTABLE"

    def maybe_recalibrate(
        self,
        memory: pd.DataFrame,
        now: pd.Timestamp | None = None,
        force: bool = False,
    ) -> bool:
        """
        Recalibrate the K-Means regime map weekly or on forced invalidation.

        Weekly recalibration is frequent enough to adapt to slow regime drift but
        not so frequent that the cluster map becomes noisy and hard to audit.
        """

        if memory.empty:
            return False
        if now is None:
            now = pd.Timestamp(memory["timestamp"].max())

        if not force and self.last_calibrated_timestamp_ is not None:
            days_since = (pd.Timestamp(now) - self.last_calibrated_timestamp_).days
            if days_since < self.config.kmeans_recalibration_days:
                return False

        fit_frame = memory.copy()
        features = self._feature_frame(fit_frame)
        if len(features) < max(40, self.config.min_regime_samples):
            if not self._insufficient_samples_logged_:
                LOGGER.info(
                    "K-Means regime calibration skipped during cold start: only %d samples available.",
                    len(features),
                )
                self._insufficient_samples_logged_ = True
            return False

        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        model = KMeans(
            n_clusters=4,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.kmeans_random_state,
        )
        clusters = model.fit_predict(X)
        centroids = pd.DataFrame(
            scaler.inverse_transform(model.cluster_centers_),
            columns=self.FEATURE_COLUMNS,
        )
        order = centroids["hl_estimate"].sort_values().index.tolist()
        label_map = {
            int(order[0]): "FAST",
            int(order[1]): "MEDIUM",
            int(order[2]): "SLOW",
            int(order[3]): "UNSTABLE",
        }

        self.scaler_ = scaler
        self.model_ = model
        self.cluster_label_map_ = label_map
        self.last_calibrated_timestamp_ = pd.Timestamp(now)
        self._insufficient_samples_logged_ = False
        self.config.cluster_centroids = {
            label_map[idx]: tuple(float(v) for v in centroids.loc[idx, self.FEATURE_COLUMNS].to_numpy())
            for idx in centroids.index
        }
        LOGGER.info("K-Means regime classifier recalibrated on %d samples.", len(features))
        return True

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.FEATURE_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0
        out = out[self.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
