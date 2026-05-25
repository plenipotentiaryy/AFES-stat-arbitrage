"""
feedback.py — Pair-Level Performance Feedback Loop for AFES.

Single-instance tracker keyed by pair name. Each entry holds a FIFO queue
of the last N_window Return-on-Notional values (R_i = Net PnL / Notional).
The tracker translates the recent track record of a pair into a smooth
position-size multiplier S_perf so underperforming pairs are *damped*
(rather than hard-killed) and outperformers receive a moderate boost.

Mathematical specification (all formulas exactly per design doc):

    R_i        = Net PnL_i / Notional_i
    SR_trade   = mean(R_buf) / (std(R_buf, ddof=1) + ε)
    SR_eff     = n/(n+K) · SR_trade + K/(n+K) · SR_prior
    S_perf     = piecewise linear from S_min at SR_floor
                                          to S_max at SR_target
                 clipped to [S_min, S_max]

Cold start: with n = 0 in the buffer, SR_trade is treated as SR_prior
exactly, so S_perf = 1.0 (neutral). As trades accumulate, the shrinkage
factor n/(n+K) drifts toward 1.

The tracker is JSON-serialisable via `to_dict()` / `from_dict()` so it
can live inside `live_state.pkl` without depending on this module's
class identity.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Any

import numpy as np

from config import (
    PFB_WINDOW, PFB_K_PRIOR, PFB_SR_PRIOR,
    PFB_S_MIN, PFB_S_MAX, PFB_SR_FLOOR, PFB_SR_TARGET, PFB_EPSILON,
)


class PerformanceFeedbackTracker:
    """
    Bayesian rolling Sharpe tracker → smooth position-size scaler.

    Construction parameters mirror the design-doc symbols; defaults are
    pulled from `config.py`. The tracker holds one FIFO deque per pair
    and updates it on every executed trade exit.

    Public API
    ----------
    add_trade(pair_name, net_pnl, notional)
        Append a single Return-on-Notional to `pair_name`'s buffer.
        notional ≤ 0 is silently rejected (cannot compute RoN).

    get_multiplier(pair_name) -> float
        Compute SR_eff via Bayesian shrinkage and map to S_perf.
        Always returns a finite value in [S_min, S_max].

    get_diagnostics(pair_name) -> dict
        Return {n, mean_R, std_R, sr_trade, sr_effective, s_perf}
        for telemetry / CSV writers.

    to_dict() / from_dict(state)
        Round-trip serialisation. Class identity is reconstructed but
        the deques are stored as plain lists in the dict.
    """

    def __init__(self,
                 window:    int   = PFB_WINDOW,
                 k_prior:   float = PFB_K_PRIOR,
                 sr_prior:  float = PFB_SR_PRIOR,
                 s_min:     float = PFB_S_MIN,
                 s_max:     float = PFB_S_MAX,
                 sr_floor:  float = PFB_SR_FLOOR,
                 sr_target: float = PFB_SR_TARGET,
                 epsilon:   float = PFB_EPSILON):
        if window <= 0:
            raise ValueError("window must be positive")
        if sr_target <= sr_floor:
            raise ValueError("sr_target must be > sr_floor")
        if s_max < s_min:
            raise ValueError("s_max must be >= s_min")
        if k_prior < 0.0:
            raise ValueError("k_prior must be >= 0")
        self.window    = int(window)
        self.k_prior   = float(k_prior)
        self.sr_prior  = float(sr_prior)
        self.s_min     = float(s_min)
        self.s_max     = float(s_max)
        self.sr_floor  = float(sr_floor)
        self.sr_target = float(sr_target)
        self.epsilon   = float(epsilon)
        self._buffers: dict[str, deque[float]] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def add_trade(self, pair_name: str, net_pnl: float, notional: float) -> None:
        """
        Push R_i = Net PnL / Notional onto pair's FIFO buffer.

        Silently ignores trades with non-positive or non-finite notional
        (cannot compute Return on Notional). Also ignores non-finite
        net_pnl. This keeps the loop robust to upstream data glitches.
        """
        try:
            net_pnl_f  = float(net_pnl)
            notional_f = float(notional)
        except (TypeError, ValueError):
            return
        if not (isfinite(net_pnl_f) and isfinite(notional_f)) or notional_f <= 0.0:
            return
        ron = net_pnl_f / notional_f
        if not isfinite(ron):
            return
        if pair_name not in self._buffers:
            self._buffers[pair_name] = deque(maxlen=self.window)
        self._buffers[pair_name].append(ron)

    def get_multiplier(self, pair_name: str) -> float:
        """Return the current S_perf for `pair_name` (1.0 if unknown)."""
        return self._compute(pair_name)["s_perf"]

    def get_diagnostics(self, pair_name: str) -> dict[str, Any]:
        """Full breakdown: n, mean_R, std_R, sr_trade, sr_effective, s_perf."""
        return self._compute(pair_name)

    def reset(self, pair_name: str | None = None) -> None:
        """Clear all buffers (`pair_name=None`) or just one pair's."""
        if pair_name is None:
            self._buffers.clear()
        elif pair_name in self._buffers:
            self._buffers[pair_name].clear()

    @property
    def known_pairs(self) -> list[str]:
        return list(self._buffers.keys())

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot for live_state.pkl."""
        return {
            "version":   1,
            "window":    self.window,
            "k_prior":   self.k_prior,
            "sr_prior":  self.sr_prior,
            "s_min":     self.s_min,
            "s_max":     self.s_max,
            "sr_floor":  self.sr_floor,
            "sr_target": self.sr_target,
            "epsilon":   self.epsilon,
            "buffers":   {p: list(q) for p, q in self._buffers.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "PerformanceFeedbackTracker":
        """Rebuild a tracker from a `to_dict()` snapshot."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dict produced by to_dict()")
        obj = cls(
            window    = state.get("window",    PFB_WINDOW),
            k_prior   = state.get("k_prior",   PFB_K_PRIOR),
            sr_prior  = state.get("sr_prior",  PFB_SR_PRIOR),
            s_min     = state.get("s_min",     PFB_S_MIN),
            s_max     = state.get("s_max",     PFB_S_MAX),
            sr_floor  = state.get("sr_floor",  PFB_SR_FLOOR),
            sr_target = state.get("sr_target", PFB_SR_TARGET),
            epsilon   = state.get("epsilon",   PFB_EPSILON),
        )
        for pair, vals in (state.get("buffers") or {}).items():
            dq = deque(maxlen=obj.window)
            for v in vals:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if isfinite(fv):
                    dq.append(fv)
            obj._buffers[str(pair)] = dq
        return obj

    # ── Internal math ───────────────────────────────────────────────────

    def _compute(self, pair_name: str) -> dict[str, Any]:
        """
        Core calculation used by both `get_multiplier` and `get_diagnostics`.

        Returns a self-consistent dict containing every intermediate value.
        For pairs with zero history the result is exactly the neutral case:
            mean_R = 0, std_R = 0, sr_trade = sr_prior, sr_effective = sr_prior,
            s_perf = piecewise(sr_prior).
        """
        buf = self._buffers.get(pair_name)
        n   = 0 if buf is None else len(buf)

        if n == 0:
            mean_R = 0.0
            std_R  = 0.0
            sr_trade = self.sr_prior     # neutral fallback
        else:
            arr = np.asarray(buf, dtype=np.float64)
            mean_R = float(arr.mean())
            # Sample standard deviation (ddof=1). Falls back to 0.0 when
            # only a single observation exists.
            std_R = float(arr.std(ddof=1)) if n >= 2 else 0.0
            sr_trade = mean_R / (std_R + self.epsilon)

        # Bayesian shrinkage toward sr_prior. With n=0 the weight on the
        # empirical SR is 0, so sr_effective = sr_prior exactly.
        denom = n + self.k_prior
        if denom <= 0:
            sr_effective = sr_trade  # degenerate (k_prior=0 and n=0)
        else:
            sr_effective = (n / denom) * sr_trade + (self.k_prior / denom) * self.sr_prior

        s_perf = self._map_to_s_perf(sr_effective)
        return {
            "pair":         pair_name,
            "n":            n,
            "mean_R":       mean_R,
            "std_R":        std_R,
            "sr_trade":     sr_trade,
            "sr_effective": sr_effective,
            "s_perf":       s_perf,
        }

    def _map_to_s_perf(self, sr_eff: float) -> float:
        """Piecewise-linear map from SR_eff to S_perf, clipped to [S_min, S_max]."""
        if not isfinite(sr_eff):
            return 1.0  # degrade gracefully on numerical pathology
        if sr_eff <= self.sr_floor:
            return self.s_min
        if sr_eff >= self.sr_target:
            return self.s_max
        span = self.sr_target - self.sr_floor
        frac = (sr_eff - self.sr_floor) / span
        s = self.s_min + (self.s_max - self.s_min) * frac
        if s < self.s_min:
            return self.s_min
        if s > self.s_max:
            return self.s_max
        return float(s)
