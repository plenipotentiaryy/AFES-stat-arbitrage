from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

import numpy as np
import pandas as pd

from .config import RegimeMemoryConfig
from .regime_classifier import RegimeClassifier, adjacent_regimes

LOGGER = logging.getLogger(__name__)


class InsufficientMemoryError(RuntimeError):
    """Raised when the memory bank cannot supply enough relevant observations."""


@dataclass(slots=True)
class ObservationRecord:
    """
    One historical observation stored in the regime-aware memory bank.

    Each record carries the market state, its realized outcome, and the regime
    label assigned at insert time so the profiler can later retrieve only the
    relevant analogs instead of querying raw expanding history.
    """

    timestamp: pd.Timestamp
    z_score: float
    hl_estimate: float
    vol_ratio: float
    hurst: float
    macro_regime: int
    outcome_pnl: float
    reverted: int
    break_score: float = 0.0
    regime_label: str | None = None


class RegimeMemoryBank:
    """
    Persistent store of regime-labeled observations for one spread or pair.

    The bank is deliberately tabular instead of object-heavy because the core
    use case is repeated filtering and weighted estimation over historical rows.
    A DataFrame keeps those operations simple, explicit, and testable.
    """

    COLUMNS = [
        "timestamp",
        "z_score",
        "hl_estimate",
        "vol_ratio",
        "hurst",
        "macro_regime",
        "regime_label",
        "outcome_pnl",
        "reverted",
        "break_score",
    ]

    def __init__(
        self,
        config: RegimeMemoryConfig,
        classifier: RegimeClassifier | None = None,
    ):
        self.config = config
        self.classifier = classifier or RegimeClassifier(config)
        self.memory = pd.DataFrame(columns=self.COLUMNS)
        self.cache_dirty = False

    def add_observation(self, obs: ObservationRecord) -> None:
        """
        Append one observation and assign its regime label on insert.

        Labeling at write time keeps the memory bank self-describing. Downstream
        profilers can then filter memory directly without having to re-run the
        regime logic on every historical row during each rebuild.
        """

        record = asdict(obs)
        if record["regime_label"] is None:
            record["regime_label"] = self.classifier.classify(record)
        row = pd.DataFrame([record], columns=self.COLUMNS)
        row["timestamp"] = pd.to_datetime(row["timestamp"], utc=True)
        if self.memory.empty:
            self.memory = row.reset_index(drop=True)
        else:
            self.memory = pd.concat([self.memory, row], ignore_index=True)
        self.memory = self.memory.sort_values("timestamp").reset_index(drop=True)
        self.prune_old_observations(self.config.max_age_days)

    def get_regime_memory(self, regime: str, min_samples: int = 50) -> pd.DataFrame:
        """
        Return only observations labeled with the requested regime.

        This method intentionally stays strict: it either returns exact
        regime-matched memory, falls back to MEDIUM when the requested regime is
        too thin, or raises an error. Silent all-regime fallback is forbidden
        because it reintroduces the exact contamination this module is designed
        to remove.
        """

        regime = regime.upper()
        frame = self.memory[self.memory["regime_label"] == regime].copy()
        if len(frame) >= min_samples:
            return frame.sort_values("timestamp").reset_index(drop=True)

        if regime != "MEDIUM":
            medium = self.memory[self.memory["regime_label"] == "MEDIUM"].copy()
            if len(medium) >= min_samples:
                LOGGER.warning(
                    "Insufficient %s memory (%d rows). Falling back to MEDIUM regime memory.",
                    regime,
                    len(frame),
                )
                return medium.sort_values("timestamp").reset_index(drop=True)

        raise InsufficientMemoryError(
            f"insufficient memory for regime={regime}: {len(frame)} rows available"
        )

    def get_weightable_memory(
        self,
        regime: str,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """
        Return regime memory plus adjacent regimes for weighted fallback logic.

        The estimator needs access to adjacent regimes when exact-match memory is
        too thin. The bank centralizes that retrieval so the profile manager can
        rebuild from one consistent candidate set.
        """

        regimes = {regime.upper(), *adjacent_regimes(regime)}
        frame = self.memory[self.memory["regime_label"].isin(regimes)].copy()
        if lookback_days is not None and not frame.empty:
            cutoff = pd.Timestamp(frame["timestamp"].max()) - pd.Timedelta(days=lookback_days)
            frame = frame[frame["timestamp"] >= cutoff]
        return frame.sort_values("timestamp").reset_index(drop=True)

    def invalidate_stale_profile(self, hl_now: float, q: float | None = None) -> bool:
        """
        Flag the active profile as stale when half-life jumps too far.

        Half-life drift is the cleanest profile-invalidating event in this
        module because the density profile and EV surface are both dominated by
        reversion speed assumptions. Once current half-life is a large multiple
        of recent median half-life, old memory is statistically poisonous.
        """

        if self.memory.empty:
            return False

        q_use = self.config.invalidation_q() if q is None else float(q)
        recent = self.memory.tail(self.config.stale_lookback_observations)
        hl_median = float(pd.to_numeric(recent["hl_estimate"], errors="coerce").dropna().median())
        if not np.isfinite(hl_median) or hl_median <= 0:
            return False

        stale = float(hl_now) > q_use * hl_median
        if stale:
            self.cache_dirty = True
        return stale

    def prune_old_observations(self, max_age_days: int = 756) -> None:
        """
        Hard-cap memory length by dropping observations older than ``max_age_days``.

        The bank is designed to remember regime-relevant history, not every tick
        forever. A three-year cap is a practical balance between enough analogs
        and avoiding unbounded stale-memory growth.
        """

        if self.memory.empty:
            return
        latest = pd.Timestamp(self.memory["timestamp"].max())
        cutoff = latest - pd.Timedelta(days=max_age_days)
        self.memory = self.memory[self.memory["timestamp"] >= cutoff].reset_index(drop=True)
