"""
post_trade.py — Post-Trade Learning components for AFES.

Two classes operate during OOS / live execution:

    ShadowBuffer
        Maintains a FIFO of (features, net_pnl, label) records — one per
        primary z-trigger — INCLUDING trades MetaGate would have rejected.
        Eliminates selection bias when retraining the meta-classifier.

        Every K_retrain new records: refit LogisticRegression on the buffer
        and grid-search θ ∈ [θ_lo, θ_hi] to maximise the specified
        drawdown-adjusted utility:

            U(θ) = sum(PnL_i · I(P_i >= θ)) / (MaxDD(θ) + ε)

    PairPerformanceTracker
        Rolling buffer of the last M_perf EXECUTED trade PnLs (only those
        actually opened by the strategy). Computes annualised SR_M and
        translates it into a smooth size scaler S_perf ∈ [S_floor, 1.0].

Both classes are pickle-safe so they can live inside `live_state.pkl` for
the paper-trading loop.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# sklearn 1.10 deprecates the `penalty` kwarg syntax we use; the call still
# works correctly and L2 is the LogisticRegression default anyway.
warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)

from config import (
    POSTTRADE_SHADOW_N,
    POSTTRADE_RETRAIN_K,
    POSTTRADE_MIN_BUFFER,
    POSTTRADE_THETA_GRID_LO,
    POSTTRADE_THETA_GRID_HI,
    POSTTRADE_THETA_GRID_STEP,
    POSTTRADE_PERF_M,
    POSTTRADE_SR_FLOOR,
    POSTTRADE_SR_TARGET,
    POSTTRADE_SPERF_FLOOR,
    POSTTRADE_BARS_PER_YEAR,
    METAGATE_THETA_ENTRY,
)


@dataclass
class ShadowRecord:
    """Single virtual-trade outcome."""
    features: list   # RegimeState-backed MetaGate feature vector
    net_pnl:  float  # realised net PnL of the simulated round-trip
    label:    int    # 1 if net_pnl > 0 else 0


# ── ShadowBuffer ─────────────────────────────────────────────────────────────

class ShadowBuffer:
    """
    Per-pair virtual-trade buffer + rolling MetaGate retrainer.

    Records are inserted via `push(features, net_pnl)`. Every
    `retrain_every` insertions the buffer:
      - refits a LogisticRegression(L2) classifier on all current records;
      - grid-searches θ to maximise the in-buffer Sharpe of accepted trades.

    Public attributes the caller reads each bar:
      - `theta`            — current decision threshold (defaults to base)
      - `classifier`       — current fitted sklearn pipeline (or None)
      - `n_records`        — len(buffer)
      - `last_train_size`  — N at last successful retrain

    Pickle-safe (deque + native types).
    """

    def __init__(self,
                 size:             int = POSTTRADE_SHADOW_N,
                 retrain_every:    int = POSTTRADE_RETRAIN_K,
                 min_buffer:       int = POSTTRADE_MIN_BUFFER,
                 theta_base:       float = METAGATE_THETA_ENTRY,
                 theta_lo:         float = POSTTRADE_THETA_GRID_LO,
                 theta_hi:         float = POSTTRADE_THETA_GRID_HI,
                 theta_step:       float = POSTTRADE_THETA_GRID_STEP):
        self.size           = int(size)
        self.retrain_every  = int(retrain_every)
        self.min_buffer     = int(min_buffer)
        self.theta_base     = float(theta_base)
        self.theta          = float(theta_base)
        self.theta_lo       = float(theta_lo)
        self.theta_hi       = float(theta_hi)
        self.theta_step     = float(theta_step)
        self._buf: deque[ShadowRecord] = deque(maxlen=self.size)
        self._since_retrain = 0
        self.classifier      = None
        self.last_train_size = 0

    @property
    def n_records(self) -> int:
        return len(self._buf)

    def push(self, features: Sequence[float], net_pnl: float) -> None:
        """Insert a completed virtual trade; trigger retrain if cadence met."""
        rec = ShadowRecord(
            features=self._normalise_features(features),
            net_pnl=float(net_pnl),
            label=1 if float(net_pnl) > 0.0 else 0,
        )
        self._buf.append(rec)
        self._since_retrain += 1
        if (self._since_retrain >= self.retrain_every
                and self.n_records >= self.min_buffer):
            self._retrain_and_retune()
            self._since_retrain = 0

    def predict_proba(self, features: Sequence[float]) -> float:
        """P(Win | features). 0.5 if no classifier is fitted yet."""
        if self.classifier is None:
            return 0.5
        arr = np.asarray(self._normalise_features(features), dtype=np.float64).reshape(1, -1)
        try:
            return float(self.classifier.predict_proba(arr)[0, 1])
        except Exception:
            return 0.5

    def size_multiplier(self, p_win: float) -> float:
        """Probabilistic sizing using the CURRENT (possibly retuned) θ."""
        if p_win < self.theta:
            return 0.0
        denom = 1.0 - self.theta
        if denom <= 0:
            return 1.0
        return float(max(0.0, min(1.0, (p_win - self.theta) / denom)))

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_features(features: Sequence[float]) -> list[float]:
        """
        Align shadow-buffer records to the current MetaGate feature schema.

        Old live states may contain seven-feature records.  The first seven
        columns are intentionally unchanged in the extended schema, so legacy
        records can be padded with neutral RegimeState defaults and mixed with
        new records without creating ragged arrays.
        """
        from metagate import FEATURE_NAMES

        neutral_tail = [1.0, 0.5, 1.0, 1.0, 0.0, 0.0]
        vals: list[float] = []
        for value in features:
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = 0.0
            vals.append(v if np.isfinite(v) else 0.0)
        while len(vals) < len(FEATURE_NAMES):
            vals.extend(neutral_tail)
        if len(vals) > len(FEATURE_NAMES):
            vals = vals[:len(FEATURE_NAMES)]
        return vals

    def _retrain_and_retune(self) -> None:
        """Refit logit on all buffered records then grid-search θ."""
        X = np.asarray([self._normalise_features(r.features) for r in self._buf], dtype=np.float64)
        y = np.asarray([r.label    for r in self._buf], dtype=np.int64)
        pnls = np.asarray([r.net_pnl for r in self._buf], dtype=np.float64)
        if len(np.unique(y)) < 2:
            # Degenerate label distribution — keep prior classifier/θ.
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l2", C=1.0,
                                   solver="lbfgs", max_iter=500),
            )
            pipe.fit(X, y)
            self.classifier = pipe
            self.last_train_size = int(len(self._buf))
        except Exception:
            return

        # Predict P(Win) for every buffered record (cheap, N≤100).
        try:
            p_win = self.classifier.predict_proba(X)[:, 1]
        except Exception:
            return

        # Grid-search θ to maximise U(θ) from the master specification.
        try:
            from metagate import search_best_threshold
            result = search_best_threshold(
                p_win,
                pnls,
                theta_lo=self.theta_lo,
                theta_hi=self.theta_hi,
                theta_step=self.theta_step,
                min_accepted=5,
            )
            if np.isfinite(result["utility"]):
                self.theta = float(result["theta"])
            else:
                self.theta = self.theta_base
        except Exception:
            self.theta = self.theta_base

    # Allow restoring from a snapshot inside live_state.pkl.
    def __getstate__(self):
        return {
            "size":            self.size,
            "retrain_every":   self.retrain_every,
            "min_buffer":      self.min_buffer,
            "theta_base":      self.theta_base,
            "theta":           self.theta,
            "theta_lo":        self.theta_lo,
            "theta_hi":        self.theta_hi,
            "theta_step":      self.theta_step,
            "buf":             list(self._buf),
            "since_retrain":   self._since_retrain,
            "classifier":      self.classifier,
            "last_train_size": self.last_train_size,
        }

    def __setstate__(self, state):
        self.size           = state["size"]
        self.retrain_every  = state["retrain_every"]
        self.min_buffer     = state["min_buffer"]
        self.theta_base     = state["theta_base"]
        self.theta          = state["theta"]
        self.theta_lo       = state["theta_lo"]
        self.theta_hi       = state["theta_hi"]
        self.theta_step     = state["theta_step"]
        self._buf           = deque(state["buf"], maxlen=self.size)
        self._since_retrain = state["since_retrain"]
        self.classifier     = state["classifier"]
        self.last_train_size = state["last_train_size"]


# ── PairPerformanceTracker ───────────────────────────────────────────────────

class PairPerformanceTracker:
    """
    Rolling SR_M over the last M_perf EXECUTED trade PnLs (NOT shadow).

    Exact master-spec formulas:

        SR_M = mean(PnL_executed) / (std(PnL_executed) + ε) * sqrt(252)

        S_perf = max(0.2, min(1.2,
                   0.2 + 1.0 * (SR_M - SR_floor)/(SR_target - SR_floor)))

    For n < 3 executed trades, SR_M is Bayesian-shrunk toward a prior Sharpe
    of 1.0:

        SR_eff = (n/3) * SR_empirical + (1 - n/3) * 1.0

    The public `sr_m` stores the shrunk/effective Sharpe used for sizing.
    """

    def __init__(self,
                 m:         int = POSTTRADE_PERF_M,
                 sr_floor:  float = POSTTRADE_SR_FLOOR,
                 sr_target: float = POSTTRADE_SR_TARGET,
                 s_floor:   float = POSTTRADE_SPERF_FLOOR,
                 ann_factor: int = POSTTRADE_BARS_PER_YEAR,
                 s_ceiling: float = 1.2,
                 sr_prior: float = 1.0,
                 shrink_n: int = 3,
                 epsilon: float = 1e-9):
        if sr_target <= sr_floor:
            raise ValueError("sr_target must be > sr_floor")
        self.m          = int(m)
        self.sr_floor   = float(sr_floor)
        self.sr_target  = float(sr_target)
        self.s_floor    = float(s_floor)
        self.s_ceiling  = float(s_ceiling)
        self.sr_prior   = float(sr_prior)
        self.shrink_n   = int(shrink_n)
        self.epsilon    = float(epsilon)
        self.ann_factor = int(ann_factor)
        self._pnls: deque[float] = deque(maxlen=self.m)
        self._sr_m: float = float("nan")
        self._s_perf: float = 1.0

    def record_trade(self, net_pnl: float) -> None:
        """Push an EXECUTED trade's net PnL; recompute SR_M and S_perf."""
        self._pnls.append(float(net_pnl))
        self._recompute()

    @property
    def sr_m(self) -> float:
        return self._sr_m

    @property
    def s_perf(self) -> float:
        return self._s_perf

    @property
    def n_trades(self) -> int:
        return len(self._pnls)

    def _recompute(self) -> None:
        n = len(self._pnls)
        if n == 0:
            self._sr_m = self.sr_prior
            self._s_perf = self._map_to_s_perf(self._sr_m)
            return
        arr = np.asarray(self._pnls, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=0))
        sr_emp = mean / (std + self.epsilon) * np.sqrt(self.ann_factor)
        if n < self.shrink_n:
            w = n / float(self.shrink_n)
            self._sr_m = w * sr_emp + (1.0 - w) * self.sr_prior
        else:
            self._sr_m = sr_emp
        self._s_perf = self._map_to_s_perf(self._sr_m)

    def _map_to_s_perf(self, sr_value: float) -> float:
        span = self.sr_target - self.sr_floor
        raw = self.s_floor + 1.0 * (sr_value - self.sr_floor) / span
        return float(max(self.s_floor, min(self.s_ceiling, raw)))

    def __getstate__(self):
        return {
            "m":         self.m,
            "sr_floor":  self.sr_floor,
            "sr_target": self.sr_target,
            "s_floor":   self.s_floor,
            "s_ceiling": self.s_ceiling,
            "sr_prior":  self.sr_prior,
            "shrink_n":  self.shrink_n,
            "epsilon":   self.epsilon,
            "ann_factor": self.ann_factor,
            "pnls":      list(self._pnls),
            "sr_m":      self._sr_m,
            "s_perf":    self._s_perf,
        }

    def __setstate__(self, state):
        self.m          = state["m"]
        self.sr_floor   = state["sr_floor"]
        self.sr_target  = state["sr_target"]
        self.s_floor    = state["s_floor"]
        self.s_ceiling  = state.get("s_ceiling", 1.2)
        self.sr_prior   = state.get("sr_prior", 1.0)
        self.shrink_n   = state.get("shrink_n", 3)
        self.epsilon    = state.get("epsilon", 1e-9)
        self.ann_factor = state["ann_factor"]
        self._pnls      = deque(state["pnls"], maxlen=self.m)
        self._sr_m      = state["sr_m"]
        self._s_perf    = state["s_perf"]
