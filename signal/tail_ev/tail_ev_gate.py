from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import TailEVConfig
from .tail_ev_layer import TailEVLayer, TailEVScore


@dataclass(slots=True)
class TailEVDecision:
    """
    Binary execution decision together with the tail-risk decomposition.
    """

    signal: bool
    regime: str
    ev: float
    ratio: float
    threshold: float
    p_revert: float | None = None
    expected_shortfall: float | None = None
    expected_gain: float | None = None
    empirical_ev: float | None = None


class TailEVDecisionGate:
    """
    Binary entry gate that preserves the old empirical logic below the threshold.

    The gate is intentionally thin: it decides whether to route a candidate
    trade through the old empirical EV logic or through the new tail-aware POT
    stack, which makes backward compatibility explicit and independently
    testable.
    """

    def __init__(
        self,
        tail_layer: TailEVLayer,
        empirical_ev_fn,
        config: TailEVConfig,
    ):
        self.tail_layer = tail_layer
        self.empirical_ev_fn = empirical_ev_fn
        self.config = config

    def evaluate(
        self,
        z: float,
        context=None,
        threshold: float | None = None,
    ) -> TailEVDecision:
        """
        Apply the two-condition tail gate above ``u`` and old logic below ``u``.

        Below the adaptive threshold the existing empirical EV remains the
        source of truth. Above the threshold the decision requires both a
        positive tail-adjusted EV and a minimum EV/ES ratio to prevent small
        positive edges from masking catastrophic loss asymmetry.
        """

        if self.tail_layer.dataset_ is None:
            raise ValueError("tail_layer must be fitted before the gate is used")

        z_mag = float(abs(z))
        live_threshold = float(
            threshold if threshold is not None else self.tail_layer.dataset_.threshold
        )
        if z_mag <= live_threshold:
            empirical_ev = float(self.empirical_ev_fn(z))
            return TailEVDecision(
                signal=bool(empirical_ev > 0.0),
                regime="empirical",
                ev=empirical_ev,
                ratio=np.nan,
                threshold=live_threshold,
                empirical_ev=empirical_ev,
            )

        score: TailEVScore = self.tail_layer.compute(z, context=context, threshold=live_threshold)
        signal = bool(score.ev > 0.0 and score.ev_to_es > self.config.risk_appetite_k)
        return TailEVDecision(
            signal=signal,
            regime="tail",
            ev=score.ev,
            ratio=score.ev_to_es,
            threshold=live_threshold,
            p_revert=score.p_revert,
            expected_shortfall=score.expected_shortfall,
            expected_gain=score.expected_gain,
        )
