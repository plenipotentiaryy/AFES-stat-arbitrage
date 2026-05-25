"""
regime.py — Unified bar-level regime state for AFES.

The AFES execution stack historically consumed macro regime labels,
cointegration diagnostics, Hurst values, ADF p-values, break scores and
MetaGate features through separate objects and series.  RegimeState makes
one immutable object the single source of truth for a bar ``t``.

The class deliberately contains only deterministic transformations of raw
signals.  It does not fit models, cache market data, or make I/O calls.  That
keeps it safe to construct in tight backtest loops and safe to pickle inside
live state objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import (
    ADAPT_BREAK_BETA,
    ADAPT_ENTRY_Z_CAP,
    ADAPT_GAMMA,
    METAGATE_BREAK_HI,
    METAGATE_CUSUM_H,
    METAGATE_HURST_HI,
    METAGATE_HURST_LO,
)


_KMEANS_TREND = 0
_KMEANS_SIDEWAYS = 1
_KMEANS_PANIC = 2
_HMM_NORMAL = 0
_HMM_PANIC = 1


def _finite_float(value: Any, default: float) -> float:
    """
    Convert ``value`` to a finite float, otherwise return ``default``.

    Pandas, NumPy and CSV-loaded values routinely arrive as ``NaN``, ``None``
    or scalar extension types.  Centralising the coercion keeps every
    RegimeState property mathematically well-defined.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if isfinite(out) else float(default)


def _finite_int(value: Any, default: int) -> int:
    """
    Convert ``value`` to an integer label, otherwise return ``default``.

    The function rounds through ``float`` first because regime labels often
    come out of pandas as ``0.0`` / ``1.0`` rather than Python ``int``.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return int(default)
    if not isfinite(out):
        return int(default)
    return int(out)


def _clip01(value: float) -> float:
    """Return ``value`` clipped to the closed unit interval [0, 1]."""
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class RegimeState:
    """
    Immutable bar-level regime and spread-quality state.

    Parameters
    ----------
    timestamp:
        Timestamp or timestamp-like object identifying the bar.

    hmm_regime:
        Global SPY HMM regime label, where ``0`` means normal and ``1`` means
        panic.  Missing / NaN values are sanitised to ``0`` (normal), matching
        the existing AFES fallback where absent HMM data does not block entry.

    kmeans_regime:
        Macro K-Means label, where ``0`` means trend, ``1`` means sideways and
        ``2`` means panic.  Missing / NaN values are sanitised to ``1``
        (sideways), matching the existing MacroFilter fallback.

    break_score:
        Composite structural break score ``B_t``.  The mathematical domain is
        non-negative; negative numeric glitches are floored to zero inside
        ``s_break``.

    cusum_val:
        Active two-sided CUSUM statistic
        ``C_t = max(S_t^+, |S_t^-|)``.  The mathematical domain is
        non-negative; negative numeric glitches are floored to zero inside
        ``s_cusum``.

    hurst_val:
        Rolling Hurst exponent ``H_t``.

    adf_p_value:
        ADF p-value for the current spread window.

    rcdp_score:
        Regime-Conditioned Dynamic Profiling quality score.  AFES does not
        always have a scalar RCDP score available in legacy runs; missing
        values are neutralised to ``0.5``.

    vol_ratio:
        Fast-to-slow realised volatility ratio.  Missing values are neutralised
        to ``1.0``.

    time_since_break:
        Number of bars elapsed since the latest structural-break event
        ``B_t > B_max``.  Missing values are set to ``0``.

    Mathematical indicators
    -----------------------
    Structural Break Risk is implemented exactly as:

        SBR_t = 1.0 - (s_Break,t * s_CUSUM,t)

    Reversion Quality Index is implemented exactly as:

        RQI_t = 0.4*s_Hurst,t + 0.4*s_ADF,t + 0.2*(1.0 - SBR_t)

    Both are guaranteed finite and in [0, 1] after input sanitisation.
    """

    timestamp: object
    hmm_regime: int = _HMM_NORMAL
    kmeans_regime: int = _KMEANS_SIDEWAYS
    break_score: float = 0.0
    cusum_val: float = 0.0
    hurst_val: float = METAGATE_HURST_LO
    adf_p_value: float = 0.0
    rcdp_score: float = 0.5
    vol_ratio: float = 1.0
    time_since_break: int = 0

    def __post_init__(self) -> None:
        """
        Sanitise all fields while preserving immutability.

        ``frozen=True`` prevents accidental mutation after construction.  The
        only legal place to normalise values is therefore ``__post_init__``,
        using ``object.__setattr__``.
        """
        object.__setattr__(self, "hmm_regime",
                           _finite_int(self.hmm_regime, _HMM_NORMAL))
        object.__setattr__(self, "kmeans_regime",
                           _finite_int(self.kmeans_regime, _KMEANS_SIDEWAYS))
        object.__setattr__(self, "break_score",
                           _finite_float(self.break_score, 0.0))
        object.__setattr__(self, "cusum_val",
                           _finite_float(self.cusum_val, 0.0))
        object.__setattr__(self, "hurst_val",
                           _finite_float(self.hurst_val, METAGATE_HURST_LO))
        object.__setattr__(self, "adf_p_value",
                           _finite_float(self.adf_p_value, 0.0))
        object.__setattr__(self, "rcdp_score",
                           _clip01(_finite_float(self.rcdp_score, 0.5)))
        object.__setattr__(self, "vol_ratio",
                           max(0.0, _finite_float(self.vol_ratio, 1.0)))
        object.__setattr__(self, "time_since_break",
                           max(0, _finite_int(self.time_since_break, 0)))

    @property
    def s_hmm(self) -> float:
        """HMM stability confidence: 1.0 for normal, 0.0 for panic."""
        return 1.0 if self.hmm_regime == _HMM_NORMAL else 0.0

    @property
    def s_hurst(self) -> float:
        """
        Hurst mean-reversion confidence.

        Implemented exactly as:

            max(0, min(1, (0.65 - H_t) / (0.65 - 0.50)))

        with configurable AFES constants that currently equal 0.50 and 0.65.
        """
        h = self.hurst_val
        if not np.isfinite(h):
            return 0.5
        denom = METAGATE_HURST_HI - METAGATE_HURST_LO
        if denom <= 0.0:
            return 0.5
        return _clip01((METAGATE_HURST_HI - h) / denom)

    @property
    def s_adf(self) -> float:
        """
        ADF cointegration confidence.

        Implemented exactly as ``1.0 - p_ADF,t`` and clipped to [0, 1] for
        robustness against malformed p-values outside the probability domain.
        """
        p = self.adf_p_value
        if not np.isfinite(p):
            return 0.5
        return _clip01(1.0 - p)

    @property
    def s_break(self) -> float:
        """
        Continuous break confidence.

        Implemented on the non-negative score domain as:

            max(0.0, 1.0 - B_t / B_max)

        where ``B_max`` defaults to ``4.5``.  Because ``B_t`` is mathematically
        non-negative, negative data glitches are floored to zero before the
        formula is evaluated.
        """
        b = self.break_score
        if not np.isfinite(b):
            return 0.5
        b = max(0.0, b)
        return float(max(0.0, 1.0 - b / METAGATE_BREAK_HI))

    @property
    def s_cusum(self) -> float:
        """
        Continuous CUSUM stability confidence.

        Implemented on the non-negative statistic domain as:

            max(0.0, 1.0 - C_t / h_cusum)

        where ``h_cusum`` defaults to ``5.0``.  Because ``C_t`` is
        mathematically non-negative, negative data glitches are floored to
        zero before the formula is evaluated.
        """
        c = self.cusum_val
        if not np.isfinite(c):
            return 0.5
        c = max(0.0, c)
        return float(max(0.0, 1.0 - c / METAGATE_CUSUM_H))

    @property
    def sbr(self) -> float:
        """
        Structural Break Risk ``SBR_t`` in [0, 1].

        Exact formula:

            SBR_t = 1.0 - (s_Break,t * s_CUSUM,t)

        A value of 0.0 means maximum structural stability.  A value of 1.0
        means the break / CUSUM evidence has fully invalidated the relationship.
        """
        return _clip01(1.0 - (self.s_break * self.s_cusum))

    @property
    def rqi(self) -> float:
        """
        Reversion Quality Index ``RQI_t`` in [0, 1].

        Exact formula:

            RQI_t = 0.4*s_Hurst,t + 0.4*s_ADF,t + 0.2*(1.0 - SBR_t)

        Scores near 1.0 represent high-quality, stationary mean-reversion with
        low structural risk.  Scores below roughly 0.4 indicate poor trading
        conditions.
        """
        return _clip01(
            0.4 * self.s_hurst
            + 0.4 * self.s_adf
            + 0.2 * (1.0 - self.sbr)
        )

    def is_panic(self) -> bool:
        """Hard entry block: K-Means Panic or global HMM Panic."""
        return self.kmeans_regime == _KMEANS_PANIC or self.hmm_regime == _HMM_PANIC

    def to_features(self, abs_z: float, vr: float) -> list[float]:
        """
        Export the extended MetaGate feature vector.

        The returned order is:

            s_HMM, s_Hurst, s_ADF, s_Break, s_CUSUM, abs_z, vr,
            kmeans_regime, rcdp_score, vol_ratio, rqi, sbr,
            time_since_break

        ``abs_z`` and ``vr`` are passed explicitly because they are execution
        features rather than regime-state fields in some AFES call sites.
        """
        abs_z_f = max(0.0, _finite_float(abs_z, 0.0))
        vr_f = max(0.0, _finite_float(vr, 1.0))
        return [
            self.s_hmm,
            self.s_hurst,
            self.s_adf,
            self.s_break,
            self.s_cusum,
            abs_z_f,
            vr_f,
            float(self.kmeans_regime),
            self.rcdp_score,
            self.vol_ratio,
            self.rqi,
            self.sbr,
            float(self.time_since_break),
        ]

    @classmethod
    def from_score_row(cls,
                       timestamp: object,
                       row: Any,
                       *,
                       default_hmm: int = _HMM_NORMAL,
                       default_kmeans: int = _KMEANS_SIDEWAYS) -> "RegimeState":
        """
        Construct from a row returned by ``metagate.build_score_frame``.

        The method accepts either a pandas Series or any mapping-like object.
        Missing keys fall back to neutral defaults, so callers can use the same
        construction path with legacy score frames.
        """
        getter = row.get if hasattr(row, "get") else lambda key, default=None: default
        return cls(
            timestamp=timestamp,
            hmm_regime=getter("hmm_regime", default_hmm),
            kmeans_regime=getter("kmeans_regime", default_kmeans),
            break_score=getter("B_t", 0.0),
            cusum_val=getter("C_t", 0.0),
            hurst_val=getter("hurst_val", METAGATE_HURST_LO),
            adf_p_value=getter("adf_p_value", 0.0),
            rcdp_score=getter("rcdp_score", 0.5),
            vol_ratio=getter("vol_ratio", 1.0),
            time_since_break=getter("time_since_break", 0),
        )

    @classmethod
    def neutral(cls, timestamp: object = None) -> "RegimeState":
        """
        Return a non-blocking, structurally stable default state.

        This is used when an upstream data layer is missing entirely.  It
        preserves backward compatibility with the historical behaviour where
        absent optional filters did not block trading.
        """
        return cls(timestamp=timestamp)


def shift_daily_to_t1(series: "pd.Series | None") -> "pd.Series | None":
    """
    Shift a daily-frequency series by one day so consumers receive only T-1
    information when reindexing or as-of-querying at intraday timestamp T.

    A value indexed by date D historically reflects D's closing observation
    (HMM regime label, daily IV multiplier, macro alert flag, KMeans cluster,
    correlation throttle).  Forward-filling that value onto intraday bars of
    the same date D leaks the day's close into the day's open / mid-session
    bars, biasing every downstream gate.

    The fix is mathematically identical to relabelling the series as
    "value as known at the close of D-1" and then ffill'ing it onto bars of
    D.  ``pandas.Series.shift(1)`` produces that relabelling without touching
    the index, which keeps every downstream ``.asof(d)`` / ``.reindex(...)``
    call working unchanged.

    Idempotency is intentionally NOT enforced — shifting twice would push the
    series to T-2.  Each consumer must apply the shift exactly once, at the
    boundary between data ingestion and trading-loop access.
    """
    if series is None:
        return None
    try:
        if hasattr(series, "empty") and series.empty:
            return series
    except Exception:
        return series
    return series.shift(1)


def time_since_break_series(break_scores: Sequence[float],
                            threshold: float = METAGATE_BREAK_HI) -> np.ndarray:
    """
    Compute bars elapsed since the most recent structural break.

    A break event is defined exactly as ``B_t > threshold``.  On a break bar the
    output is ``0``.  Before the first break the count increases from ``0`` so
    that the feature is finite from the first row onward.
    """
    b = np.asarray(break_scores, dtype=np.float64)
    out = np.zeros(len(b), dtype=np.int64)
    last_break = -1
    for i, score in enumerate(b):
        if np.isfinite(score) and score > threshold:
            last_break = i
            out[i] = 0
        elif last_break >= 0:
            out[i] = i - last_break
        else:
            out[i] = i
    return out


def apply_sbr_guard(regime: "RegimeState",
                    entry_z_base: float,
                    current_stop_z: float,
                    threshold: float = 0.60,
                    stop_mult: float = 1.5,
                    enabled: bool = True) -> float:
    """
    Single-Bullet Entry Guard stop tightener (Section 7.4).

    When ``regime.sbr`` exceeds ``threshold`` the function returns
    ``min(current_stop_z, stop_mult * entry_z_base)``, otherwise it returns
    ``current_stop_z`` unchanged.  The min() keeps the guard from ever making
    the stop *looser* than what the upstream adaptive logic already chose;
    Section 7.4 explicitly frames the override as a tightening.

    The other two Section 7.4 effects — disallowing grid averaging and
    capping concurrent entries at one — are structurally enforced by the
    current AFES backtest loops (each ``backtest_oos`` / ``backtest_pair``
    runs at most one open position per pair).  This helper therefore
    encapsulates the only behavioural lever exposed by the guard today, and
    can be unit-tested without standing up a full backtest.

    Parameters
    ----------
    regime:
        Bar-level RegimeState (must expose ``.sbr`` in [0, 1]).
    entry_z_base:
        Base entry z-score from the WFO grid / config (not the per-bar
        adaptive value).
    current_stop_z:
        Whatever stop_z the adaptive logic would otherwise have used.
    threshold:
        Minimum SBR for the guard to fire (spec default 0.60).
    stop_mult:
        Multiplier on ``entry_z_base`` defining the tightened stop level
        (spec default 1.5).
    enabled:
        When False, the function is an identity pass-through on
        ``current_stop_z``.  Lets the caller honour a kill-switch flag from
        config without branching at every call site.
    """
    if not enabled:
        return float(current_stop_z)
    sbr = getattr(regime, "sbr", None)
    if sbr is None or not isfinite(float(sbr)):
        return float(current_stop_z)
    if float(sbr) <= float(threshold):
        return float(current_stop_z)
    tightened = float(stop_mult) * float(entry_z_base)
    return float(min(float(current_stop_z), tightened))


@dataclass(frozen=True)
class EffectiveTradeKnobs:
    """
    Per-bar execution knobs produced by the feedback controller.

    These are derived from the base WFO parameters and the current
    ``RegimeState``; they are not fitted parameters and carry no hidden state.
    """

    entry_z_eff: float
    stop_z_eff: float
    max_hold_eff: int


def compute_effective_trade_knobs(regime: RegimeState,
                                  entry_z_base: float,
                                  stop_z_base: float,
                                  max_hold_base: int,
                                  gamma: float = ADAPT_GAMMA,
                                  break_beta: float = ADAPT_BREAK_BETA,
                                  entry_z_cap: float = ADAPT_ENTRY_Z_CAP,
                                  h_cusum: float = METAGATE_CUSUM_H) -> EffectiveTradeKnobs:
    """
    Compute effective entry, stop and max-hold knobs exactly from the spec.

    Entry threshold:

        entry_z_eff = entry_z_base *
            [1 + γ * max(0, σ_fast/σ_slow - 1)] + β * B_t

        clipped to [entry_z_base, 5.0].

    Stop threshold:

        stop_z_eff = stop_z_base * max(0.6, 1 - 0.4*C_t/h_cusum)

    Max holding bars:

        max_hold_eff = max_hold_base *
            max(0.3, 1 - 0.7*max(0, (H_t - 0.50)/(0.65 - 0.50)))

    The returned max-hold is rounded to the nearest integer and floored at one
    bar so the trading loop never receives a zero holding horizon.
    """
    entry_base = max(0.0, _finite_float(entry_z_base, 0.0))
    stop_base = max(0.0, _finite_float(stop_z_base, 0.0))
    hold_base = max(1, _finite_int(max_hold_base, 1))

    vol_ratio = max(0.0, _finite_float(regime.vol_ratio, 1.0))
    vol_excess = max(0.0, vol_ratio - 1.0)
    break_score = max(0.0, _finite_float(regime.break_score, 0.0))
    entry_eff = entry_base * (1.0 + float(gamma) * vol_excess) + float(break_beta) * break_score
    entry_eff = float(max(entry_base, min(float(entry_z_cap), entry_eff)))

    cusum = max(0.0, _finite_float(regime.cusum_val, 0.0))
    stop_scale = max(0.6, 1.0 - 0.4 * (cusum / float(h_cusum)))
    stop_eff = float(stop_base * stop_scale)

    hurst = _finite_float(regime.hurst_val, METAGATE_HURST_LO)
    denom = METAGATE_HURST_HI - METAGATE_HURST_LO
    if denom <= 0.0:
        trend_component = 0.0
    else:
        trend_component = max(0.0, (hurst - METAGATE_HURST_LO) / denom)
    hold_scale = max(0.3, 1.0 - 0.7 * trend_component)
    hold_eff = max(1, int(round(hold_base * hold_scale)))

    return EffectiveTradeKnobs(
        entry_z_eff=entry_eff,
        stop_z_eff=stop_eff,
        max_hold_eff=hold_eff,
    )
