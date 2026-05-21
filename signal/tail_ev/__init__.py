"""
Tail EV package for POT-based tail-aware expected value estimation.

The package lives under ``signal/tail_ev`` to match the AFES layout, but it is
intended to be imported by adding ``AFES/signal`` to ``sys.path`` and then
importing ``tail_ev``. This avoids collisions with Python's stdlib ``signal``
module while keeping the requested directory structure intact.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class TailEVConfig:
    """
    Central configuration for the three-layer tail EV stack.

    The parameters are grouped here so that thresholding, POT calibration,
    survival modelling and execution gating share one source of truth. This
    keeps the system testable and prevents hidden constants from changing the
    economic meaning of the gate.
    """

    threshold_quantile: float = 0.90
    rolling_window: int = 252
    min_periods: int = 126
    min_exceedances: int = 40
    alpha: float = 0.95
    risk_appetite_k: float = 0.15
    revert_target_fraction: float = 0.50
    stop_multiple: float = 1.77
    min_stop_excess: float = 0.50
    max_holding_period: int = 78
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    min_train_size: int = 80
    walk_forward_splits: int = 4
    ad_bootstrap_samples: int = 200
    stability_points: int = 8
    random_state: int = 42
    allow_negative_xi: bool = False
    xi_floor: float = 0.0
    feature_columns: tuple[str, ...] = (
        "z_score",
        "z_velocity",
        "vol_ratio",
        "OFI",
        "macro_regime",
        "hurst_exp",
        "coint_score",
    )


__all__ = ["TailEVConfig"]
