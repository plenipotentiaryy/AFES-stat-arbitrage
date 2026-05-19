from __future__ import annotations

from dataclasses import dataclass

from .config import BreakDetectorConfig


@dataclass(slots=True)
class RiskAction:
    """
    Stateless action instruction returned by the break risk gate.

    Returning a data object instead of mutating positions directly keeps the
    gate pure and makes it easy for the current AFES backtest loop to consume
    the instructions without introducing hidden side effects.
    """

    break_level: str
    pass_signal_through: bool
    cancel_entries: bool
    reduce_existing_size_factor: float
    tighten_stop_or_derisk: bool


class BreakRiskGate:
    """
    Translate break levels into deterministic portfolio actions.

    The gate is deliberately a pure function of ``break_level`` and config, so
    it can be used by any caller that manages positions differently, including
    the current AFES backtest loop which does not expose a PositionManager
    object yet.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config

    def action_for_level(self, break_level: str) -> RiskAction:
        """
        Return the configured action map for one break level.

        The factors are not hardcoded because the right response is
        strategy-dependent: some portfolios should halve risk on ALARM, others
        should exit more aggressively.
        """

        level = str(break_level).upper()
        if level == "NORMAL":
            return RiskAction(level, True, False, 0.0, False)
        if level == "CAUTION":
            return RiskAction(
                level,
                False,
                self.config.caution_cancel_entries,
                self.config.caution_reduce_factor,
                False,
            )
        if level == "ALARM":
            return RiskAction(
                level,
                False,
                self.config.alarm_cancel_entries,
                self.config.alarm_reduce_factor,
                False,
            )
        if level == "ABORT":
            return RiskAction(
                level,
                False,
                self.config.abort_cancel_entries,
                self.config.abort_reduce_factor,
                self.config.abort_tighten_stop_or_derisk,
            )
        raise ValueError(f"unknown break level: {break_level}")
