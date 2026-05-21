"""Structural break speed detector for AFES."""

from .config import BreakDetectorConfig
from .break_data_layer import BreakDetectorDataLayer
from .break_statistics import (
    CUSUMDetector,
    KalmanInnovationRatioDetector,
    LocalHalfLifeExplosionDetector,
    BetaVelocityMonitor,
)
from .break_composite_scorer import BreakCompositeScorer
from .break_risk_gate import BreakRiskGate, RiskAction

__all__ = [
    "BreakDetectorConfig",
    "BreakDetectorDataLayer",
    "CUSUMDetector",
    "KalmanInnovationRatioDetector",
    "LocalHalfLifeExplosionDetector",
    "BetaVelocityMonitor",
    "BreakCompositeScorer",
    "BreakRiskGate",
    "RiskAction",
]
