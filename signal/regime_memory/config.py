from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RegimeMemoryConfig:
    """
    Central configuration for the regime-aware memory bank and profiler stack.

    Every decay parameter, regime threshold, and invalidation rule is explicit
    here because this module is fundamentally about calibration speed. Hidden
    constants would make it impossible to reason about why the profile is using
    one slice of history instead of another.
    """

    pair_name: str | None = None

    hl_fast_threshold: float = 15.0
    # Below 15 days the spread is treated as a genuinely fast regime where
    # mean-reversion assumptions should be calibrated from only the quickest
    # historical analogs.
    hl_slow_threshold: float = 30.0
    # 15-30 days is the transition zone where many pairs still revert, but more
    # slowly than the classic high-frequency mean-reversion regime.
    hl_break_threshold: float = 60.0
    # Above 60 days the spread behaves too slowly for the old fast-memory
    # samples to remain trustworthy.
    hurst_unstable_threshold: float = 0.55
    break_caution_threshold: float = 0.75
    break_abort_threshold: float = 2.75

    temporal_decay_lambda: float = 0.0125
    # 0.0125 implies a decay half-life of roughly 55 days. That is short
    # enough to force >80% of profile weight into the last 90 days during a
    # sustained slow regime, which is the contamination-control target for this
    # module.
    regime_distance_gamma: float = 2.0
    # Gamma=2 penalizes half-life mismatch aggressively so nearby but not
    # identical regimes only contribute when memory is genuinely thin.
    min_effective_weight: float = 10.0
    min_effective_sample_size: float = 30.0
    min_regime_samples: int = 50
    max_age_days: int = 756
    adjacent_regime_penalty: float = 0.65
    # Adjacent regimes are allowed only as a thin-memory fallback and receive a
    # further multiplicative discount to make the fallback explicit rather than
    # silently broad.

    default_invalidation_q: float = 2.5
    # The profile is invalidated once current half-life exceeds 2.5x recent
    # median half-life. This captures genuine structural drift instead of normal
    # local noise.
    pair_invalidation_q: dict[str, float] = field(default_factory=dict)
    stale_lookback_observations: int = 60
    recent_rebuild_lookback_days: int = 60

    rebuild_interval_days: int = 7
    kmeans_recalibration_days: int = 7
    kmeans_random_state: int = 42
    kmeans_n_init: int = 20
    cluster_centroids: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    local_half_life_window: int = 20
    z_near_threshold: float = 1.0
    outcome_horizon_bars: int = 20
    stop_multiple: float = 1.77
    profile_density_grid_size: int = 200

    validation_pairs: int = 6
    validation_output_file: str = "reports/regime_memory_validation.csv"

    def invalidation_q(self) -> float:
        """
        Return the per-pair invalidation threshold.

        Different spreads have different half-life volatility. A single global
        ``q`` would either reset calm pairs too often or react too slowly on
        unstable pairs, so the config supports per-pair overrides.
        """

        if self.pair_name and self.pair_name in self.pair_invalidation_q:
            return float(self.pair_invalidation_q[self.pair_name])
        return float(self.default_invalidation_q)
