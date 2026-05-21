from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import RegimeMemoryConfig
from .decayed_density_estimator import DecayedDensityEstimator, DensityProfilerBase
from .regime_classifier import RegimeClassifier
from .regime_memory_bank import InsufficientMemoryError, ObservationRecord, RegimeMemoryBank

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProfileUpdateResult:
    """
    Outcome of one profile manager update step.

    Returning explicit rebuild metadata makes it easy to audit when the memory
    bank switched regimes or invalidated stale profiles during live trading or
    backtests.
    """

    regime: str
    n_eff: float
    profile_rebuilt: bool
    ev_surface: Callable | None


class RegimeAwareProfileManager(DensityProfilerBase):
    """
    Orchestrate the regime-aware memory bank and active density estimator.

    The manager owns profile lifecycle rather than just density estimation. That
    is necessary because the edge comes from rebuilding the profile at the right
    time, not only from the weighting formula itself.
    """

    def __init__(self, config: RegimeMemoryConfig):
        self.config = config
        self.classifier = RegimeClassifier(config)
        self.bank = RegimeMemoryBank(config, classifier=self.classifier)
        self.active_estimator = DecayedDensityEstimator(config)
        self.current_regime: str | None = None
        self.profile_dirty = True
        self.last_rebuild_timestamp: pd.Timestamp | None = None
        self._hl_drift_alarm_active = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the manager from a historical observation table.

        This method allows the manager to satisfy the same interface as the
        legacy profiler while still supporting incremental updates through the
        memory bank.
        """

        self.bank.memory = pd.DataFrame(columns=self.bank.COLUMNS)
        for _, row in data.iterrows():
            record = ObservationRecord(
                timestamp=pd.Timestamp(row["timestamp"]),
                z_score=float(row["z_score"]),
                hl_estimate=float(row["hl_estimate"]),
                vol_ratio=float(row["vol_ratio"]),
                hurst=float(row["hurst"]),
                macro_regime=int(row["macro_regime"]),
                outcome_pnl=float(row["outcome_pnl"]),
                reverted=int(row["reverted"]),
                break_score=float(row.get("break_score", 0.0)),
                regime_label=str(row["regime_label"]) if "regime_label" in row and pd.notna(row["regime_label"]) else None,
            )
            self.bank.add_observation(record)

        if self.bank.memory.empty:
            raise ValueError("cannot fit profile manager on empty data")

        now = pd.Timestamp(self.bank.memory["timestamp"].max())
        self.classifier.maybe_recalibrate(self.bank.memory, now=now, force=True)
        self.current_regime = str(self.bank.memory.iloc[-1]["regime_label"]).upper()
        self._rebuild_profile(now=now, lookback_days=None)

    def update(self, new_observation: ObservationRecord) -> ProfileUpdateResult:
        """
        Ingest one observation, refresh regime state, and rebuild when needed.

        The rebuild trigger combines three conditions: hard invalidation on
        half-life drift, explicit regime change, and periodic weekly refresh.
        That is the operational layer that replaces the raw expanding window.
        """

        now = pd.Timestamp(new_observation.timestamp)
        self.bank.add_observation(new_observation)
        self.classifier.maybe_recalibrate(self.bank.memory, now=now, force=False)

        latest = self.bank.memory.iloc[-1]
        regime = str(latest["regime_label"]).upper()
        regime_changed = regime != self.current_regime
        self.current_regime = regime

        rebuilt = False
        if self.check_invalidation(float(latest["hl_estimate"])):
            self.rebuild_only_on_recent(self.config.recent_rebuild_lookback_days, now=now)
            rebuilt = True
        else:
            interval_elapsed = (
                self.last_rebuild_timestamp is None
                or (now - self.last_rebuild_timestamp).days >= self.config.rebuild_interval_days
            )
            if self.profile_dirty or regime_changed or interval_elapsed:
                self._rebuild_profile(now=now, lookback_days=None)
                rebuilt = True

        return ProfileUpdateResult(
            regime=regime,
            n_eff=self.effective_sample_size(),
            profile_rebuilt=rebuilt,
            ev_surface=self.ev,
        )

    def check_invalidation(self, hl_now: float, q: float | None = None) -> bool:
        """
        Check whether the current half-life invalidates the active profile.

        This method is separated for testability because invalidation speed is a
        key validation target: it must react within a few bars of a genuine
        half-life regime jump.
        """

        stale = self.bank.invalidate_stale_profile(hl_now, q=q)
        if not stale:
            self._hl_drift_alarm_active = False
            return False

        if self._hl_drift_alarm_active:
            return False

        if stale and not self.bank.memory.empty:
            recent = self.bank.memory.tail(self.config.stale_lookback_observations)
            hl_median = float(pd.to_numeric(recent["hl_estimate"], errors="coerce").median())
            LOGGER.warning(
                "Half-life drift detected: HL_now=%.1f > %.2fx median HL=%.1f. "
                "Profile invalidated and rebuilt on last %d days only.",
                hl_now,
                self.config.invalidation_q() if q is None else float(q),
                hl_median,
                self.config.recent_rebuild_lookback_days,
            )
            self.profile_dirty = True
            self._hl_drift_alarm_active = True
        return True

    def rebuild_only_on_recent(self, lookback_days: int, now: pd.Timestamp | None = None) -> None:
        """
        Rebuild the active profile using only recent observations.

        This is the hard-reset path after severe half-life drift. It discards
        old analogs entirely because once half-life has moved far enough, old
        data is more misleading than helpful.
        """

        self._rebuild_profile(now=now, lookback_days=lookback_days)
        self.profile_dirty = False

    def density(self, z: float) -> float:
        return self.active_estimator.density(z)

    def p_revert(self, z: float) -> float:
        return self.active_estimator.p_revert(z)

    def ev(self, z: float, gain_fn: Callable, loss_fn: Callable) -> float | None:
        return self.active_estimator.ev(z, gain_fn, loss_fn)

    def effective_sample_size(self) -> float:
        return self.active_estimator.effective_sample_size()

    def _rebuild_profile(self, now: pd.Timestamp | None, lookback_days: int | None) -> None:
        if self.current_regime is None:
            raise ValueError("cannot rebuild profile before current regime is known")

        candidate = self.bank.get_weightable_memory(self.current_regime, lookback_days=lookback_days)
        if candidate.empty:
            raise InsufficientMemoryError(f"no weightable memory for regime={self.current_regime}")

        latest = self.bank.memory.iloc[-1].copy()
        if candidate.empty or pd.Timestamp(candidate["timestamp"].max()) != pd.Timestamp(latest["timestamp"]):
            candidate = pd.concat([candidate, pd.DataFrame([latest])], ignore_index=True)
            candidate = candidate.drop_duplicates(subset=["timestamp", "z_score"], keep="last")

        self.classifier.maybe_recalibrate(candidate, now=now, force=self.bank.cache_dirty)
        self.active_estimator.fit(candidate)
        self.profile_dirty = False
        self.bank.cache_dirty = False
        self.last_rebuild_timestamp = pd.Timestamp(now if now is not None else candidate["timestamp"].max())
