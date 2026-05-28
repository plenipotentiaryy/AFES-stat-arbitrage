"""Tail EV Risk Governor for AFES.

This module converts statistical tail risk in spread Z-score space into dollar
expected shortfall and uses that value as a hard portfolio-level safety layer.
It deliberately uses EVT/POT with a Generalized Pareto tail rather than a
Gaussian approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass(slots=True)
class GPDParameters:
    """Fitted GPD tail parameters for one pair."""

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_observations: int
    source: str = "mle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": float(self.shape),
            "scale": float(self.scale),
            "threshold": float(self.threshold),
            "n_exceedances": int(self.n_exceedances),
            "n_observations": int(self.n_observations),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "GPDParameters":
        return cls(
            shape=float(state["shape"]),
            scale=float(state["scale"]),
            threshold=float(state["threshold"]),
            n_exceedances=int(state["n_exceedances"]),
            n_observations=int(state["n_observations"]),
            source=str(state.get("source", "mle")),
        )


@dataclass(slots=True)
class TailRiskBreakdown:
    """Dollar tail-risk decomposition for one open position."""

    pair_name: str
    current_z: float
    es_z: float
    spread_std_usd: float
    position_size_units: float
    es_usd: float
    p_revert: float
    expected_gain_usd: float
    tail_ev_usd: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "pair_name": self.pair_name,
            "current_z": float(self.current_z),
            "es_z": float(self.es_z),
            "spread_std_usd": float(self.spread_std_usd),
            "position_size_units": float(self.position_size_units),
            "es_usd": float(self.es_usd),
            "p_revert": float(self.p_revert),
            "expected_gain_usd": float(self.expected_gain_usd),
            "tail_ev_usd": float(self.tail_ev_usd),
        }


class TailEVRiskGovernor:
    """Portfolio tail expected-shortfall governor using GPD/POT tails.

    The class is intentionally pickle-safe and does not own market data.  Fit
    calls must be supplied with historical Z-score observations that end at
    `T-1` relative to the candidate entry bar.  This keeps look-ahead prevention
    explicit at the call site.
    """

    def __init__(
        self,
        u_threshold: float = 3.0,
        max_portfolio_es_pct: float = 0.02,
        max_pair_tail_contribution: float = 0.25,
        contribution_mode: str = "BUDGET_RELATIVE",
        prior_shape: float = 0.1,
        prior_scale: float = 1.0,
        min_exceedances: int = 20,
        min_observations: int = 100,
    ):
        if u_threshold <= 0:
            raise ValueError("u_threshold must be positive")
        if not (0 < max_portfolio_es_pct <= 1.0):
            raise ValueError("max_portfolio_es_pct must be in (0, 1]")
        if not (0 < max_pair_tail_contribution <= 1.0):
            raise ValueError("max_pair_tail_contribution must be in (0, 1]")
        mode = str(contribution_mode).upper()
        if mode not in {"BUDGET_RELATIVE", "ACTIVE_COUNT_CONDITIONAL"}:
            raise ValueError(
                "contribution_mode must be 'BUDGET_RELATIVE' or "
                "'ACTIVE_COUNT_CONDITIONAL'"
            )
        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive")
        if min_exceedances < 1:
            raise ValueError("min_exceedances must be positive")
        if min_observations < 1:
            raise ValueError("min_observations must be positive")

        self.u_threshold = float(u_threshold)
        self.max_portfolio_es_pct = float(max_portfolio_es_pct)
        self.max_pair_tail_contribution = float(max_pair_tail_contribution)
        self.contribution_mode = mode
        self.prior_shape = float(prior_shape)
        self.prior_scale = float(prior_scale)
        self.min_exceedances = int(min_exceedances)
        self.min_observations = int(min_observations)
        self.gpd_parameters: dict[str, GPDParameters] = {}
        self.last_breakdowns: list[TailRiskBreakdown] = []
        self.last_portfolio_es_usd: float = 0.0

    def _prior_parameters(self, pair_name: str, n_observations: int = 0) -> GPDParameters:
        """Return conservative prior parameters when MLE is unavailable."""
        del pair_name
        shape = min(max(self.prior_shape, 1e-9), 0.99)
        return GPDParameters(
            shape=shape,
            scale=self.prior_scale,
            threshold=self.u_threshold,
            n_exceedances=0,
            n_observations=int(n_observations),
            source="prior",
        )

    def fit_pair_gpd(self, pair_name: str, historical_zscores: np.ndarray | pd.Series) -> None:
        """Fit GPD parameters to historical absolute-Z exceedances.

        Parameters
        ----------
        pair_name:
            AFES pair key.
        historical_zscores:
            Causal Z-score history ending at `T-1`.  The method never shifts or
            expands the data internally; callers are responsible for passing the
            correct time slice.
        """
        z = np.asarray(historical_zscores, dtype=float)
        z = z[np.isfinite(z)]
        if z.size < self.min_observations:
            self.gpd_parameters[pair_name] = self._prior_parameters(pair_name, z.size)
            return

        tail = np.abs(z)
        exceedances = tail[tail > self.u_threshold] - self.u_threshold
        exceedances = exceedances[np.isfinite(exceedances) & (exceedances > 0)]
        if exceedances.size < self.min_exceedances:
            self.gpd_parameters[pair_name] = self._prior_parameters(pair_name, z.size)
            return

        try:
            shape, _, scale = genpareto.fit(exceedances, floc=0.0)
            shape = float(shape)
            scale = float(scale)
        except Exception:
            self.gpd_parameters[pair_name] = self._prior_parameters(pair_name, z.size)
            return

        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            self.gpd_parameters[pair_name] = self._prior_parameters(pair_name, z.size)
            return

        # Negative xi implies bounded support; for financial spread tails that
        # is usually threshold contamination, so clamp to a very light positive
        # tail instead of pretending the tail cannot extend further.
        shape = max(shape, 1e-9)
        self.gpd_parameters[pair_name] = GPDParameters(
            shape=shape,
            scale=scale,
            threshold=self.u_threshold,
            n_exceedances=int(exceedances.size),
            n_observations=int(z.size),
            source="mle",
        )

    def get_pair_parameters(self, pair_name: str) -> GPDParameters:
        """Return fitted pair parameters or conservative prior fallback."""
        return self.gpd_parameters.get(pair_name, self._prior_parameters(pair_name))

    @staticmethod
    def expected_shortfall_z(current_z: float, params: GPDParameters) -> float:
        """Compute GPD expected shortfall in Z-score units.

        Implements the specification exactly:

            ES_z = u + (beta + xi * (|z_t| - u)) / (1 - xi), xi < 1
            ES_z = |z_t| * 2.0, xi >= 1

        For `|z| < u`, the exceedance term is floored at zero; the result is the
        expected tail severity conditional on crossing the threshold.
        """
        z_abs = float(abs(current_z))
        u = float(params.threshold)
        xi = float(params.shape)
        beta = float(params.scale)
        if xi >= 1.0:
            return float(max(z_abs, u) * 2.0)
        x = max(0.0, z_abs - u)
        es_z = u + (beta + xi * x) / (1.0 - xi)
        return float(max(es_z, z_abs, u))

    def calculate_pair_es_usd(
        self,
        pair_name: str,
        current_z: float,
        spread_std_usd: float,
        position_size_units: float,
    ) -> float:
        """Calculate expected tail loss in dollars for a single pair."""
        spread_std = abs(float(spread_std_usd))
        units = abs(float(position_size_units))
        if not np.isfinite(spread_std) or spread_std <= 0:
            raise ValueError("spread_std_usd must be finite and positive")
        if not np.isfinite(units) or units <= 0:
            raise ValueError("position_size_units must be finite and positive")
        params = self.get_pair_parameters(pair_name)
        es_z = self.expected_shortfall_z(current_z, params)
        return float(es_z * spread_std * units)

    def position_breakdown(self, position: dict[str, Any]) -> TailRiskBreakdown:
        """Compute full tail-risk decomposition for one position dictionary."""
        pair_name = str(position.get("pair_name") or position.get("pair"))
        if not pair_name or pair_name == "None":
            raise KeyError(f"Position missing pair_name/pair: {position}")
        current_z = float(position.get("current_z", position.get("zscore", 0.0)))
        spread_std = float(position["spread_std_usd"])
        units = float(position["position_size_units"])
        es_usd = self.calculate_pair_es_usd(pair_name, current_z, spread_std, units)

        p_revert = float(position.get("p_revert", 0.50))
        p_revert = float(np.clip(p_revert, 0.0, 1.0))
        expected_gain = position.get("expected_gain_usd")
        if expected_gain is None:
            threshold = self.get_pair_parameters(pair_name).threshold
            expected_gain = max(abs(current_z) - threshold, 0.0) * spread_std * units
        expected_gain = float(max(expected_gain, 0.0))
        tail_ev = p_revert * expected_gain - (1.0 - p_revert) * es_usd

        return TailRiskBreakdown(
            pair_name=pair_name,
            current_z=current_z,
            es_z=self.expected_shortfall_z(current_z, self.get_pair_parameters(pair_name)),
            spread_std_usd=spread_std,
            position_size_units=abs(units),
            es_usd=es_usd,
            p_revert=p_revert,
            expected_gain_usd=expected_gain,
            tail_ev_usd=float(tail_ev),
        )

    def calculate_position_breakdowns(
        self,
        positions: list[dict[str, Any]],
    ) -> list[TailRiskBreakdown]:
        """Return tail-risk breakdowns without mutating governor diagnostics."""
        return [self.position_breakdown(pos) for pos in positions]

    @staticmethod
    def _sum_es_by_pair(
        breakdowns: list[TailRiskBreakdown],
    ) -> dict[str, float]:
        """Aggregate expected shortfall by pair name.

        Live engines normally hold at most one position per pair, but summing
        makes the contribution calculation robust to partial fills, duplicated
        state rows, or future multi-sleeve deployment.
        """
        out: dict[str, float] = {}
        for b in breakdowns:
            out[b.pair_name] = out.get(b.pair_name, 0.0) + float(b.es_usd)
        return out

    def calculate_portfolio_es_usd(
        self,
        open_positions: list[dict[str, Any]],
    ) -> float:
        """Return aggregate dollar expected shortfall without side effects."""
        return float(sum(b.es_usd for b in self.calculate_position_breakdowns(open_positions)))

    def portfolio_es_usd(self, open_positions: list[dict[str, Any]]) -> float:
        """Return aggregate dollar expected shortfall and update diagnostics.

        This method preserves the legacy diagnostic side effects used by older
        AFES live/reporting code.  New decision methods call the pure
        calculation helpers directly.
        """
        breakdowns = self.calculate_position_breakdowns(open_positions)
        self.last_breakdowns = breakdowns
        self.last_portfolio_es_usd = float(sum(b.es_usd for b in breakdowns))
        return self.last_portfolio_es_usd

    def calculate_tail_contributions(
        self,
        positions: list[dict[str, Any]],
        current_capital: float,
    ) -> dict[str, float]:
        """Calculate pair-level tail contribution ratios.

        Two modes are supported:

        ``BUDGET_RELATIVE``
            ``Tail_ES_i / (max_portfolio_es_pct * current_capital)``.  This is
            equivalent to an absolute per-pair tail-risk budget and avoids the
            small-portfolio paradox where one of two open positions must exceed
            a 25% share of actual realised portfolio ES.

        ``ACTIVE_COUNT_CONDITIONAL``
            Bypass the contribution constraint until the projected active count
            reaches ``ceil(1 / max_pair_tail_contribution)``.  Once active,
            contribution is measured against actual portfolio ES.
        """
        if not positions or current_capital <= 0:
            return {}

        breakdowns = self.calculate_position_breakdowns(positions)
        es_by_pair = self._sum_es_by_pair(breakdowns)
        if not es_by_pair:
            return {}

        if self.contribution_mode == "BUDGET_RELATIVE":
            budget_es = self.max_portfolio_es_pct * float(current_capital)
            denom = max(budget_es, 1e-9)
            return {pair: float(es_usd / denom) for pair, es_usd in es_by_pair.items()}

        min_active = int(np.ceil(1.0 / self.max_pair_tail_contribution))
        if len(es_by_pair) < min_active:
            return {pair: 0.0 for pair in es_by_pair}

        total_actual_es = max(float(sum(es_by_pair.values())), 1e-9)
        return {pair: float(es_usd / total_actual_es) for pair, es_usd in es_by_pair.items()}

    def evaluate_portfolio_risk(
        self,
        open_positions: list[dict[str, Any]],
        current_capital: float,
    ) -> tuple[bool, list[str]]:
        """Evaluate portfolio tail ES and return de-risk order if unsafe.

        Returns
        -------
        tuple[bool, list[str]]
            `(True, [])` if aggregate ES is within cap.  If unsafe, returns
            `(False, pairs_to_de_risk_in_order)`, ranked from most negative
            Tail EV to least.
        """
        if current_capital <= 0:
            raise ValueError("current_capital must be positive")
        cap = self.max_portfolio_es_pct * float(current_capital)
        breakdowns = self.calculate_position_breakdowns(open_positions)
        portfolio_es = float(sum(b.es_usd for b in breakdowns))

        derisk_order: list[str] = []
        if portfolio_es > cap + 1e-12:
            ranked = sorted(breakdowns, key=lambda b: (b.tail_ev_usd, -b.es_usd))
            remaining_es = portfolio_es
            for b in ranked:
                if remaining_es <= cap + 1e-12:
                    break
                derisk_order.append(b.pair_name)
                remaining_es -= b.es_usd

        contributions = self.calculate_tail_contributions(open_positions, current_capital)
        unsafe_contrib_pairs = {
            pair for pair, contrib in contributions.items()
            if contrib > self.max_pair_tail_contribution + 1e-12
        }
        if unsafe_contrib_pairs:
            violators = sorted(
                (b for b in breakdowns if b.pair_name in unsafe_contrib_pairs),
                key=lambda b: -b.es_usd,
            )
            for b in reversed(violators):
                if b.pair_name in derisk_order:
                    derisk_order.remove(b.pair_name)
                derisk_order.insert(0, b.pair_name)

        if not derisk_order:
            return True, []
        return False, derisk_order

    def check_candidate_entry(
        self,
        candidate_position: dict[str, Any],
        open_positions: list[dict[str, Any]],
        current_capital: float,
    ) -> tuple[bool, str, float]:
        """Check whether adding a candidate would breach portfolio ES cap."""
        if current_capital <= 0:
            return False, "BLOCKED_INVALID_CAPITAL", float("nan")
        projected = list(open_positions) + [candidate_position]
        portfolio_es = self.calculate_portfolio_es_usd(projected)
        cap = self.max_portfolio_es_pct * float(current_capital)
        if portfolio_es > cap + 1e-12:
            return (
                False,
                f"BLOCKED_TAIL_GOVERNOR projected_es={portfolio_es:.2f} cap={cap:.2f}",
                portfolio_es,
            )

        contributions = self.calculate_tail_contributions(projected, current_capital)
        for pair_name, contrib in contributions.items():
            if contrib > self.max_pair_tail_contribution + 1e-12:
                return (
                    False,
                    "BLOCKED_TAIL_CONTRIBUTION "
                    f"{pair_name} contrib={contrib:.3f} "
                    f"limit={self.max_pair_tail_contribution:.3f}",
                    portfolio_es,
                )
        return True, "OK", portfolio_es

    @staticmethod
    def rolling_spread_std_usd(
        spread: pd.Series,
        window: int = 60,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Causal rolling spread std using only bars up to T-1.

        The returned value at timestamp `t` is `spread.shift(1).rolling(...).std()`,
        so it is safe to use for an entry decision on bar `t`.
        """
        if min_periods is None:
            min_periods = max(5, window // 3)
        return spread.astype(float).shift(1).rolling(window, min_periods=min_periods).std()

    def to_dict(self) -> dict[str, Any]:
        """Serialize governor configuration and fitted GPD parameters."""
        return {
            "u_threshold": self.u_threshold,
            "max_portfolio_es_pct": self.max_portfolio_es_pct,
            "max_pair_tail_contribution": self.max_pair_tail_contribution,
            "contribution_mode": self.contribution_mode,
            "prior_shape": self.prior_shape,
            "prior_scale": self.prior_scale,
            "min_exceedances": self.min_exceedances,
            "min_observations": self.min_observations,
            "gpd_parameters": {
                pair: params.to_dict() for pair, params in self.gpd_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TailEVRiskGovernor":
        """Restore governor from serialized state."""
        obj = cls(
            u_threshold=float(state.get("u_threshold", 3.0)),
            max_portfolio_es_pct=float(state.get("max_portfolio_es_pct", 0.02)),
            max_pair_tail_contribution=float(
                state.get("max_pair_tail_contribution", 0.25)
            ),
            contribution_mode=str(state.get("contribution_mode", "BUDGET_RELATIVE")),
            prior_shape=float(state.get("prior_shape", 0.1)),
            prior_scale=float(state.get("prior_scale", 1.0)),
            min_exceedances=int(state.get("min_exceedances", 20)),
            min_observations=int(state.get("min_observations", 100)),
        )
        params = state.get("gpd_parameters", {})
        obj.gpd_parameters = {
            str(pair): GPDParameters.from_dict(pstate) for pair, pstate in params.items()
        }
        return obj
