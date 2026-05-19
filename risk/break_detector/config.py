from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BreakDetectorConfig:
    """
    Central configuration for the fast structural-break detector.

    The detector is a risk-control layer, so every threshold and action knob is
    explicit here. This keeps calibration auditable and prevents hidden magic
    constants from changing how aggressively AFES de-risks during spread breaks.
    """

    cusum_window: int = 1
    hl_short_window: int = 20
    hl_median_window: int = 20
    innovation_window: int = 60
    beta_velocity_window: int = 60
    mu0_window: int = 252
    min_history: int = 20

    cusum_k_sigma_multiplier: float = 0.50
    # Half a sigma is the classic allowance term: small noise is ignored,
    # sustained drift is accumulated into the CUSUM state.
    cusum_h_sigma_multiplier: float = 4.00
    # Four sigma is intentionally conservative to target long average run length
    # on stationary spreads while still reacting quickly after a genuine break.

    innovation_ratio_threshold: float = 4.00
    # Default target is <2% false positives on OOS stationary regimes.
    hl_ratio_threshold: float = 3.00
    # Fires when local half-life triples relative to recent behavior.
    beta_velocity_zscore_threshold: float = 2.50
    # Two and a half sigmas is high enough to avoid normal beta wiggle noise.

    weight_cusum: float = 0.35
    weight_innovation: float = 0.25
    weight_half_life: float = 0.25
    weight_beta_velocity: float = 0.15

    threshold_caution: float = 1.00
    # Blocks fresh risk when early evidence appears but before forced de-risking.
    threshold_alarm: float = 2.80
    # Requires composite confirmation strong enough to justify cutting size.
    threshold_abort: float = 3.00
    # High bar so ABORT is not triggered by one noisy statistic in isolation.

    caution_cancel_entries: bool = True
    caution_reduce_factor: float = 0.0
    alarm_cancel_entries: bool = True
    alarm_reduce_factor: float = 0.50
    abort_cancel_entries: bool = True
    abort_reduce_factor: float = 1.00
    abort_tighten_stop_or_derisk: bool = True

    hurst_threshold: float = 0.65
    hurst_window: int = 60
    hurst_lag: int = 10

    validation_pairs: int = 6
    validation_z_window: int = 60
    validation_time_stop: int = 30
    validation_output_file: str = "reports/break_detector_validation.csv"
