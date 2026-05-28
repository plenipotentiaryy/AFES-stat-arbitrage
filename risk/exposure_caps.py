"""Hierarchical exposure caps for AFES.

The controller is intentionally deterministic and state-free: callers pass the
current open-position book plus a candidate entry, and the controller returns a
hard allow/block decision with an audit reason.  It enforces gross exposure
limits, not net beta exposure, because the purpose of this layer is catastrophic
concentration prevention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def split_pair_name(pair_name: str) -> tuple[str, str]:
    """Split a standard AFES pair key into two ticker legs.

    AFES pair names are normally `LEG1-LEG2`; synthetic FX-normalized legs use
    dots/underscores (`SAP.DE_USD`), so the separator remains unambiguous for
    the current production universe.  If a future ticker itself contains a
    hyphen, callers should pass explicit `legs` in the position dictionaries.
    """
    parts = str(pair_name).split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Cannot infer pair legs from {pair_name!r}; expected exactly one '-'. "
            "Pass explicit legs in the position dictionary for non-standard tickers."
        )
    return parts[0], parts[1]


def position_pair_name(position: dict[str, Any]) -> str:
    """Return a normalized pair name from an open-position dictionary."""
    for key in ("pair_name", "pair"):
        value = position.get(key)
        if value:
            return str(value)
    raise KeyError(f"Open position missing 'pair_name'/'pair': {position}")


def position_legs(position: dict[str, Any]) -> tuple[str, str]:
    """Return the two ticker legs for an open position."""
    legs = position.get("legs")
    if legs is not None:
        if len(legs) != 2:
            raise ValueError(f"Position legs must contain two tickers: {position}")
        return str(legs[0]), str(legs[1])
    return split_pair_name(position_pair_name(position))


def position_nominal(position: dict[str, Any]) -> float:
    """Extract absolute gross nominal exposure from an open-position dictionary."""
    for key in ("nominal_size_usd", "nominal_usd", "notional", "gross_notional"):
        if key in position:
            value = float(position[key])
            if value < 0:
                value = abs(value)
            return value
    raise KeyError(f"Open position missing nominal exposure: {position}")


@dataclass
class ExposureCapsController:
    """Hard hierarchical exposure caps for candidate entries.

    Parameters
    ----------
    max_ticker_exposure_pct:
        Maximum gross exposure associated with a single ticker, expressed as a
        fraction of current capital.
    max_sector_exposure_pct:
        Maximum gross exposure associated with a single pair sector, expressed
        as a fraction of current capital.
    max_portfolio_leverage:
        Maximum total gross nominal exposure divided by current capital.
    """

    max_ticker_exposure_pct: float = 0.15
    max_sector_exposure_pct: float = 0.30
    max_portfolio_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.max_ticker_exposure_pct <= 0:
            raise ValueError("max_ticker_exposure_pct must be positive")
        if self.max_sector_exposure_pct <= 0:
            raise ValueError("max_sector_exposure_pct must be positive")
        if self.max_portfolio_leverage <= 0:
            raise ValueError("max_portfolio_leverage must be positive")

    def current_ticker_exposures(self, open_positions: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate gross nominal exposure by ticker across open pairs."""
        exposures: dict[str, float] = {}
        for pos in open_positions:
            nominal = position_nominal(pos)
            for ticker in position_legs(pos):
                exposures[ticker] = exposures.get(ticker, 0.0) + nominal
        return exposures

    def current_sector_exposures(
        self,
        open_positions: list[dict[str, Any]],
        pair_sector_map: dict[str, str],
    ) -> dict[str, float]:
        """Aggregate gross nominal exposure by pair sector."""
        exposures: dict[str, float] = {}
        for pos in open_positions:
            pair = position_pair_name(pos)
            sector = str(pos.get("sector") or pair_sector_map.get(pair, "UNKNOWN"))
            exposures[sector] = exposures.get(sector, 0.0) + position_nominal(pos)
        return exposures

    def current_gross_exposure(self, open_positions: list[dict[str, Any]]) -> float:
        """Return total gross nominal exposure across all open pairs."""
        return float(sum(position_nominal(pos) for pos in open_positions))

    def check_entry_allowed(
        self,
        pair_name: str,
        direction: str,
        nominal_size_usd: float,
        open_positions: list[dict[str, Any]],
        current_capital: float,
        pair_sector_map: dict[str, str],
    ) -> tuple[bool, str]:
        """Validate whether a candidate entry violates exposure caps.

        Returns
        -------
        tuple[bool, str]
            `(True, "OK")` if the candidate is allowed.  Otherwise `(False,
            reason)`, where reason is a stable machine-readable block code.
        """
        del direction  # Gross caps are direction-agnostic by design.
        if current_capital <= 0:
            return False, "BLOCKED_INVALID_CAPITAL"
        nominal = abs(float(nominal_size_usd))
        if not nominal > 0:
            return False, "BLOCKED_INVALID_NOMINAL"

        try:
            leg1, leg2 = split_pair_name(pair_name)
        except ValueError as exc:
            return False, f"BLOCKED_BAD_PAIR_NAME: {exc}"

        ticker_cap = self.max_ticker_exposure_pct * current_capital
        ticker_exp = self.current_ticker_exposures(open_positions)
        for ticker in (leg1, leg2):
            projected = ticker_exp.get(ticker, 0.0) + nominal
            if projected > ticker_cap + 1e-12:
                return (
                    False,
                    "BLOCKED_TICKER_EXPOSURE "
                    f"{ticker} projected={projected:.2f} cap={ticker_cap:.2f}",
                )

        sector_cap = self.max_sector_exposure_pct * current_capital
        sector = str(pair_sector_map.get(pair_name, "UNKNOWN"))
        sector_exp = self.current_sector_exposures(open_positions, pair_sector_map)
        projected_sector = sector_exp.get(sector, 0.0) + nominal
        if projected_sector > sector_cap + 1e-12:
            return (
                False,
                "BLOCKED_SECTOR_EXPOSURE "
                f"{sector} projected={projected_sector:.2f} cap={sector_cap:.2f}",
            )

        leverage_cap = self.max_portfolio_leverage * current_capital
        projected_gross = self.current_gross_exposure(open_positions) + nominal
        if projected_gross > leverage_cap + 1e-12:
            return (
                False,
                "BLOCKED_PORTFOLIO_LEVERAGE "
                f"projected={projected_gross:.2f} cap={leverage_cap:.2f}",
            )

        return True, "OK"

    def to_dict(self) -> dict[str, float]:
        """Serialize the controller configuration."""
        return {
            "max_ticker_exposure_pct": float(self.max_ticker_exposure_pct),
            "max_sector_exposure_pct": float(self.max_sector_exposure_pct),
            "max_portfolio_leverage": float(self.max_portfolio_leverage),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ExposureCapsController":
        """Restore a controller from a dictionary."""
        return cls(
            max_ticker_exposure_pct=float(state.get("max_ticker_exposure_pct", 0.15)),
            max_sector_exposure_pct=float(state.get("max_sector_exposure_pct", 0.30)),
            max_portfolio_leverage=float(state.get("max_portfolio_leverage", 2.0)),
        )
