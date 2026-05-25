"""
metagate.py — Bayesian / Ensemble Gate Aggregator for AFES.

Replaces the rigid AND-chain of binary risk filters
(HMM, Hurst, ADF, Break, CUSUM) with a probabilistic meta-model:

    1. Each filter emits a continuous confidence score s_j ∈ [0, 1].
    2. A meta-classifier (logit / shallow GBM) is fit on the UNFILTERED
       virtual-trade pool from the WFO train window and outputs
       P(Win | RegimeState.to_features(|z|, VR)).
    3. Live execution gates entries by P ≥ θ_entry and scales position
       size by the probability margin (P - θ) / (1 - θ).

Design constraints:
  - Per-pair models (sample size is small) — keep the classifier shallow.
  - Avoid selection bias: train on virtual trades simulated with ALL
    risk gates disabled (every |z| ≥ entry_z trigger becomes a trade).
  - Safe fallback: < METAGATE_MIN_TRAIN_N training trades or unable to fit
    → revert to the legacy AND-gate (no model is stored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import (
    METAGATE_HURST_LO,
    METAGATE_HURST_HI,
    METAGATE_BREAK_HI,
    METAGATE_CUSUM_H,
    METAGATE_THETA_ENTRY,
    METAGATE_MIN_TRAIN_N,
    METAGATE_GBM_MIN_N,
    POSTTRADE_THETA_GRID_LO,
    POSTTRADE_THETA_GRID_HI,
    POSTTRADE_THETA_GRID_STEP,
)
from regime import RegimeState, time_since_break_series, shift_daily_to_t1


# Order of features in the feature vector x_t.
FEATURE_NAMES: tuple[str, ...] = (
    "s_HMM",
    "s_Hurst",
    "s_ADF",
    "s_Break",
    "s_CUSUM",
    "abs_z",
    "vr",
    "kmeans_regime",
    "rcdp_score",
    "vol_ratio",
    "rqi",
    "sbr",
    "time_since_break",
)


_LEGACY_FEATURE_COUNT = 7
_NEUTRAL_EXTENDED_TAIL: tuple[float, ...] = (
    1.0,  # kmeans_regime: Sideways, matching MacroFilter's no-block default.
    0.5,  # rcdp_score: neutral profile quality.
    1.0,  # vol_ratio: fast volatility equals slow volatility.
    1.0,  # rqi: high-quality neutral when optional filters are disabled.
    0.0,  # sbr: no structural break risk when optional filters are disabled.
    0.0,  # time_since_break: finite cold-start value.
)


# ── Confidence-score primitives ──────────────────────────────────────────────

def hurst_confidence(h_val: float,
                     lo: float = METAGATE_HURST_LO,
                     hi: float = METAGATE_HURST_HI) -> float:
    """s_Hurst = clip((hi - H) / (hi - lo), 0, 1)."""
    if not np.isfinite(h_val):
        return 0.5
    if h_val <= lo:
        return 1.0
    if h_val >= hi:
        return 0.0
    return float((hi - h_val) / (hi - lo))


def adf_confidence(p_value: float) -> float:
    """s_ADF = 1 - p. Highly cointegrated (p≈0) → 1, unit-root (p≈1) → 0."""
    if not np.isfinite(p_value):
        return 0.5
    return float(np.clip(1.0 - p_value, 0.0, 1.0))


def break_confidence(b_t: float,
                     hi: float = METAGATE_BREAK_HI) -> float:
    """s_Break = clip(1 - Bt/hi, 0, 1)."""
    if not np.isfinite(b_t):
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - b_t / hi)))


def cusum_confidence(c_t: float,
                     h_cusum: float = METAGATE_CUSUM_H) -> float:
    """s_CUSUM = clip(1 - Ct/h_cusum, 0, 1).  Ct = max(S+, |S-|)."""
    if not np.isfinite(c_t):
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - c_t / h_cusum)))


def hmm_confidence(regime_or_prob, panic_state: int = 1) -> float:
    """
    Accepts either:
      - discrete regime int/bool  → 1.0 if Normal, 0.0 if Panic
      - posterior P(Panic) float  → 1.0 - P
    """
    if regime_or_prob is None:
        return 1.0
    try:
        v = float(regime_or_prob)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(v):
        return 1.0
    if 0.0 <= v <= 1.0 and not float(v).is_integer():
        return float(1.0 - v)
    return 0.0 if int(v) == panic_state else 1.0


# ── Bar-level feature builder ────────────────────────────────────────────────

def build_feature_vector(s_hmm: float,
                         s_hurst: float,
                         s_adf: float,
                         s_break: float,
                         s_cusum: float,
                         abs_z: float,
                         vr: float,
                         kmeans_regime: float = 1.0,
                         rcdp_score: float = 0.5,
                         vol_ratio: float = 1.0,
                         rqi: float | None = None,
                         sbr: float | None = None,
                         time_since_break: float = 0.0) -> list[float]:
    """
    Return a MetaGate feature vector aligned with ``FEATURE_NAMES``.

    The first seven arguments preserve the legacy AFES API.  When the extended
    fields are omitted, the function derives neutral and mathematically
    consistent defaults:

      - ``sbr = 1 - s_Break*s_CUSUM``
      - ``rqi = 0.4*s_Hurst + 0.4*s_ADF + 0.2*(1 - sbr)``
      - K-Means = Sideways, RCDP = 0.5, vol_ratio = 1.0

    New code should generally prefer ``RegimeState.to_features(abs_z, vr)``.
    This helper remains for backward-compatible call sites and tests.
    """
    s_hmm_f = float(np.clip(s_hmm, 0.0, 1.0))
    s_hurst_f = float(np.clip(s_hurst, 0.0, 1.0))
    s_adf_f = float(np.clip(s_adf, 0.0, 1.0))
    s_break_f = float(np.clip(s_break, 0.0, 1.0))
    s_cusum_f = float(np.clip(s_cusum, 0.0, 1.0))
    if sbr is None:
        sbr_f = float(np.clip(1.0 - (s_break_f * s_cusum_f), 0.0, 1.0))
    else:
        sbr_f = float(np.clip(sbr, 0.0, 1.0))
    if rqi is None:
        rqi_f = float(np.clip(
            0.4 * s_hurst_f + 0.4 * s_adf_f + 0.2 * (1.0 - sbr_f),
            0.0,
            1.0,
        ))
    else:
        rqi_f = float(np.clip(rqi, 0.0, 1.0))
    try:
        kmeans_f = float(kmeans_regime)
    except (TypeError, ValueError):
        kmeans_f = 1.0
    if not np.isfinite(kmeans_f):
        kmeans_f = 1.0
    try:
        rcdp_f = float(rcdp_score)
    except (TypeError, ValueError):
        rcdp_f = 0.5
    if not np.isfinite(rcdp_f):
        rcdp_f = 0.5
    try:
        vol_ratio_f = float(vol_ratio)
    except (TypeError, ValueError):
        vol_ratio_f = 1.0
    if not np.isfinite(vol_ratio_f):
        vol_ratio_f = 1.0
    try:
        tsb_f = float(time_since_break)
    except (TypeError, ValueError):
        tsb_f = 0.0
    if not np.isfinite(tsb_f):
        tsb_f = 0.0
    return [
        s_hmm_f,
        s_hurst_f,
        s_adf_f,
        s_break_f,
        s_cusum_f,
        float(max(0.0, abs_z)),
        float(max(0.0, vr)),
        kmeans_f,
        float(np.clip(rcdp_f, 0.0, 1.0)),
        float(max(0.0, vol_ratio_f)),
        rqi_f,
        sbr_f,
        float(max(0.0, tsb_f)),
    ]


# ── Rolling CUSUM precomputation on a z-score series ─────────────────────────

def rolling_cusum(z: np.ndarray,
                  k: float = 0.5,
                  h: float = METAGATE_CUSUM_H) -> np.ndarray:
    """
    One-sided CUSUM that resets on threshold breach. Returns C_t = max(S+, |S-|).

    Mirrors the reset behaviour of filters._compute_break_scores so a single
    per-bar value is available for live MetaGate scoring without re-running
    the Numba kernel.
    """
    n = len(z)
    out = np.empty(n, dtype=np.float64)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        zi = z[i] if np.isfinite(z[i]) else 0.0
        s_pos = max(0.0, s_pos + zi - k)
        s_neg = min(0.0, s_neg + zi + k)
        c = max(s_pos, abs(s_neg))
        out[i] = c
        if c > h:
            s_pos = 0.0
            s_neg = 0.0
    return out


def simple_break_score(z: np.ndarray,
                       cusum: np.ndarray,
                       h_cusum: float = METAGATE_CUSUM_H,
                       win: int = 60) -> np.ndarray:
    """
    Lightweight Bt proxy for the WFO context (fixed beta, fixed HL):
        Bt = 0.5 * log1p(z^2 / var_z) + 0.5 * (C_t / h_cusum)

    Uses rolling variance of z over `win` bars and the precomputed CUSUM.
    Matches the original `_compute_break_scores` shape (innovation shock +
    CUSUM term) under the simplifications valid for fixed-β backtests.
    """
    n = len(z)
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    sq = 0.0
    buf = np.zeros(win, dtype=np.float64)
    head = 0
    filled = 0
    for i in range(n):
        zi = z[i] if np.isfinite(z[i]) else 0.0
        old = buf[head]
        buf[head] = zi
        head = (head + 1) % win
        if filled < win:
            filled += 1
            s += zi
            sq += zi * zi
        else:
            s += zi - old
            sq += zi * zi - old * old
        m = s / filled
        var = max(sq / filled - m * m, 1e-9)
        shock = (zi * zi) / var
        s1 = np.log1p(shock)
        s3 = cusum[i] / h_cusum
        out[i] = 0.5 * s1 + 0.5 * s3
    return out


def _normalise_daily_index(series: pd.Series) -> pd.Series:
    """
    Return ``series`` indexed by timezone-naive calendar dates.

    AFES regime files are produced by several scripts; some are UTC-aware,
    some are already naive, and some are daily while others are intraday.  The
    MetaGate frame uses timezone-naive normalised bar dates for daily regime
    alignment, so this helper gives every optional daily source the same index
    convention before ``reindex(..., method='ffill')``.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    out = series.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    return out.sort_index()


# ── Vectorised score frame builder ───────────────────────────────────────────

def build_score_frame(
    sig: pd.DataFrame,
    hurst_series: pd.Series | None,
    adf_pvalue_series: pd.Series | None,
    hmm_series: pd.Series | None,
    kmeans_series: pd.Series | None = None,
    rcdp_series: pd.Series | None = None,
    vol_ratio_series: pd.Series | None = None,
    cusum_k: float = 0.5,
    cusum_h: float = METAGATE_CUSUM_H,
) -> pd.DataFrame:
    """
    Build a bar-aligned RegimeState / MetaGate feature frame.

    For each bar in ``sig``, the function computes raw ``B_t`` and ``C_t``,
    aligns optional daily regime sources, constructs the exact
    ``RegimeState`` formulas, and returns a DataFrame indexed like ``sig``.

    Output columns include every name in ``FEATURE_NAMES`` plus raw inputs
    useful for diagnostics and later RegimeState reconstruction:
    ``B_t``, ``C_t``, ``hurst_val``, ``adf_p_value``, ``hmm_regime``.

    Backward compatibility:
      - all newly added series are optional;
      - missing HMM defaults to normal (0);
      - missing K-Means defaults to Sideways (1);
      - missing RCDP defaults to 0.5;
      - missing vol_ratio defaults to 1.0.
    """
    z = sig["zscore"].to_numpy(dtype=np.float64)
    c_arr = rolling_cusum(z, k=cusum_k, h=cusum_h)
    b_arr = simple_break_score(z, c_arr, h_cusum=cusum_h)
    t_since = time_since_break_series(b_arr, threshold=METAGATE_BREAK_HI)

    # Hurst confidence — linearly interpolate from precomputed Series.
    if hurst_series is not None and not hurst_series.empty:
        h_aligned = hurst_series.reindex(sig.index, method="ffill").fillna(METAGATE_HURST_LO)
    else:
        h_aligned = pd.Series(METAGATE_HURST_LO, index=sig.index)

    # ADF confidence — daily p-values forward-filled into intraday.
    if adf_pvalue_series is not None and not adf_pvalue_series.empty:
        p_aligned = adf_pvalue_series.reindex(sig.index, method="ffill").fillna(0.0)
    else:
        p_aligned = pd.Series(0.0, index=sig.index)

    # HMM confidence — daily regime/posterior reindexed onto bar timestamps.
    idx = sig.index
    try:
        day_idx = idx.normalize().tz_localize(None)
    except (TypeError, AttributeError):
        day_idx = pd.DatetimeIndex(idx).normalize()

    # Daily regime labels are computed on day D's close, so forward-filling
    # them onto intraday bars of D would leak the close into the open / mid
    # session.  Shift by one day to expose only T-1 information.
    if hmm_series is not None and not hmm_series.empty:
        hmm_shifted = shift_daily_to_t1(_normalise_daily_index(hmm_series))
        hmm_aligned = hmm_shifted.reindex(day_idx, method="ffill").fillna(0)
    else:
        hmm_aligned = pd.Series(0, index=day_idx)

    if kmeans_series is not None and not kmeans_series.empty:
        kmeans_shifted = shift_daily_to_t1(_normalise_daily_index(kmeans_series))
        kmeans_aligned = kmeans_shifted.reindex(day_idx, method="ffill").fillna(1)
    else:
        kmeans_aligned = pd.Series(1, index=day_idx)

    if rcdp_series is not None and not rcdp_series.empty:
        rcdp_aligned = rcdp_series.reindex(sig.index, method="ffill").fillna(0.5)
    else:
        rcdp_aligned = pd.Series(0.5, index=sig.index)

    if vol_ratio_series is not None and not vol_ratio_series.empty:
        vol_ratio_aligned = vol_ratio_series.reindex(sig.index, method="ffill").fillna(1.0)
    elif "vol_ratio" in sig.columns:
        vol_ratio_aligned = sig["vol_ratio"].fillna(1.0)
    else:
        vol_ratio_aligned = pd.Series(1.0, index=sig.index)

    vr = sig.get("vr", pd.Series(1.0, index=sig.index)).to_numpy(dtype=np.float64)
    abs_z = np.abs(z)

    states = [
        RegimeState(
            timestamp=sig.index[i],
            hmm_regime=hmm_aligned.iloc[i],
            kmeans_regime=kmeans_aligned.iloc[i],
            break_score=b_arr[i],
            cusum_val=c_arr[i],
            hurst_val=h_aligned.iloc[i],
            adf_p_value=p_aligned.iloc[i],
            rcdp_score=rcdp_aligned.iloc[i],
            vol_ratio=vol_ratio_aligned.iloc[i],
            time_since_break=int(t_since[i]),
        )
        for i in range(len(sig))
    ]

    feature_rows = [state.to_features(abs_z[i], vr[i]) for i, state in enumerate(states)]
    feature_df = pd.DataFrame(feature_rows, index=sig.index, columns=FEATURE_NAMES)

    raw_df = pd.DataFrame(
        {
            "B_t":              b_arr,
            "C_t":              c_arr,
            "hurst_val":        h_aligned.to_numpy(dtype=np.float64),
            "adf_p_value":      p_aligned.to_numpy(dtype=np.float64),
            "hmm_regime":       hmm_aligned.to_numpy(dtype=np.float64),
            "kmeans_regime":    kmeans_aligned.to_numpy(dtype=np.float64),
            "rcdp_score":       rcdp_aligned.to_numpy(dtype=np.float64),
            "vol_ratio":        vol_ratio_aligned.to_numpy(dtype=np.float64),
            "time_since_break": t_since,
        },
        index=sig.index,
    )
    # FEATURE_NAMES includes several columns also present in raw_df.  Keep the
    # feature version first and only add genuinely raw / diagnostic columns.
    return pd.concat(
        [feature_df, raw_df[[c for c in raw_df.columns if c not in feature_df.columns]]],
        axis=1,
    )


# ── MetaGate model wrapper ───────────────────────────────────────────────────

@dataclass
class MetaGateModel:
    """
    Thin wrapper over a fitted sklearn classifier. Stores the classifier,
    its kind ('logit' | 'gbm' | 'fallback'), and training-sample count so
    callers can decide whether to apply or revert to the AND-gate.
    """
    clf: object | None
    kind: str
    n_train: int
    theta_entry: float = METAGATE_THETA_ENTRY
    feature_names: tuple[str, ...] = FEATURE_NAMES
    theta_calibrated: float | None = None
    calibration_utility: float | None = None
    calibration_n_oof: int = 0

    @property
    def usable(self) -> bool:
        return self.clf is not None and self.kind != "fallback"

    def _expected_feature_count(self) -> int:
        """
        Infer the feature count expected by the wrapped classifier.

        New models are trained on the extended RegimeState vector.  Older
        persisted AFES states may still contain a seven-feature classifier.
        Rather than failing hard in live trading, prediction aligns the vector
        to the classifier's declared ``n_features_in_`` when available.
        """
        if self.clf is None:
            return len(getattr(self, "feature_names", FEATURE_NAMES))
        direct = getattr(self.clf, "n_features_in_", None)
        if direct is not None:
            return int(direct)
        steps = getattr(self.clf, "steps", None)
        if steps:
            for _, step in steps:
                n_features = getattr(step, "n_features_in_", None)
                if n_features is not None:
                    return int(n_features)
        return len(getattr(self, "feature_names", FEATURE_NAMES))

    @staticmethod
    def _align_feature_array(arr: np.ndarray, expected_n: int) -> np.ndarray:
        """
        Trim or pad a one-row feature array to ``expected_n`` columns.

        Padding uses neutral extended-feature defaults.  Trimming is safe for
        legacy seven-feature models because the first seven columns preserve
        the old MetaGate order exactly.
        """
        actual_n = int(arr.shape[1])
        if actual_n == expected_n:
            return arr
        if actual_n > expected_n:
            return arr[:, :expected_n]
        pad_values = list(_NEUTRAL_EXTENDED_TAIL)
        while actual_n + len(pad_values) < expected_n:
            pad_values.extend(_NEUTRAL_EXTENDED_TAIL)
        pad_row = np.asarray(pad_values[:expected_n - actual_n], dtype=np.float64)
        pad = np.tile(pad_row.reshape(1, -1), (arr.shape[0], 1))
        return np.concatenate([arr, pad], axis=1)

    def predict_proba(self, x_row: Sequence[float]) -> float:
        """Return P(Win | x). 0.5 if unusable so callers can detect fallback."""
        if not self.usable:
            return 0.5
        arr = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        arr = self._align_feature_array(arr, self._expected_feature_count())
        try:
            return float(self.clf.predict_proba(arr)[0, 1])
        except Exception:
            return 0.5

    def size_multiplier(self, p_win: float) -> float:
        """Probabilistic sizing: (P - θ) / (1 - θ), clipped to [0, 1]."""
        if p_win < self.theta_entry:
            return 0.0
        denom = 1.0 - self.theta_entry
        if denom <= 0:
            return 1.0
        return float(max(0.0, min(1.0, (p_win - self.theta_entry) / denom)))


def fit_metagate(features: np.ndarray,
                 labels: np.ndarray,
                 theta_entry: float = METAGATE_THETA_ENTRY,
                 pnl_values: np.ndarray | Sequence[float] | None = None,
                 calibrate: bool = True,
                 n_splits: int = 5) -> MetaGateModel:
    """
    Fit a meta-classifier on the per-trade feature matrix `features`
    (rows = trades, cols = FEATURE_NAMES) and binary labels `labels`
    (1 = winning trade, 0 = losing/breakeven).

    When `pnl_values` is supplied, the function first performs temporal
    out-of-fold probability calibration and searches θ ∈ [0.45, 0.65]:

        U(θ) = sum(PnL_i * I(P_i >= θ)) / (MaxDD(θ) + ε)

    The selected θ* becomes the model's `theta_entry`; the final classifier is
    still fitted on the full training pool after calibration.

    Selection rules:
      - N < METAGATE_MIN_TRAIN_N           → fallback (no model fit)
      - Single class in labels              → fallback
      - METAGATE_MIN_TRAIN_N ≤ N < GBM_MIN  → LogisticRegression(L2)
      - N ≥ METAGATE_GBM_MIN_N              → HistGradientBoostingClassifier
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()

    n = int(features.shape[0]) if features.ndim == 2 else 0
    if n > 0:
        features = MetaGateModel._align_feature_array(features, len(FEATURE_NAMES))
    theta_eff = float(theta_entry)
    calibration_utility = None
    calibration_n_oof = 0
    if calibrate and pnl_values is not None and n >= METAGATE_MIN_TRAIN_N:
        pnl_arr = np.asarray(pnl_values, dtype=np.float64).ravel()
        if len(pnl_arr) == n:
            calib = calibrate_threshold_oof(
                features,
                labels,
                pnl_arr,
                n_splits=n_splits,
                theta_lo=POSTTRADE_THETA_GRID_LO,
                theta_hi=POSTTRADE_THETA_GRID_HI,
                theta_step=POSTTRADE_THETA_GRID_STEP,
            )
            if calib["n_oof"] > 0 and np.isfinite(calib["theta"]):
                theta_eff = float(calib["theta"])
                calibration_utility = float(calib["utility"])
                calibration_n_oof = int(calib["n_oof"])

    if n < METAGATE_MIN_TRAIN_N or len(np.unique(labels)) < 2:
        return MetaGateModel(clf=None, kind="fallback", n_train=n,
                             theta_entry=theta_eff,
                             feature_names=FEATURE_NAMES,
                             theta_calibrated=theta_eff,
                             calibration_utility=calibration_utility,
                             calibration_n_oof=calibration_n_oof)

    try:
        if n >= METAGATE_GBM_MIN_N:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_depth=2, max_iter=30, min_samples_leaf=10,
                learning_rate=0.05,
            )
            clf.fit(features, labels)
            return MetaGateModel(clf=clf, kind="gbm", n_train=n,
                                 theta_entry=theta_eff,
                                 feature_names=FEATURE_NAMES,
                                 theta_calibrated=theta_eff,
                                 calibration_utility=calibration_utility,
                                 calibration_n_oof=calibration_n_oof)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=500),
            )
            pipe.fit(features, labels)
            return MetaGateModel(clf=pipe, kind="logit", n_train=n,
                                 theta_entry=theta_eff,
                                 feature_names=FEATURE_NAMES,
                                 theta_calibrated=theta_eff,
                                 calibration_utility=calibration_utility,
                                 calibration_n_oof=calibration_n_oof)
    except Exception:
        return MetaGateModel(clf=None, kind="fallback", n_train=n,
                             theta_entry=theta_eff,
                             feature_names=FEATURE_NAMES,
                             theta_calibrated=theta_eff,
                             calibration_utility=calibration_utility,
                             calibration_n_oof=calibration_n_oof)


def max_drawdown_from_pnl(pnl_values: Sequence[float]) -> float:
    """
    Compute maximum drawdown from an ordered sequence of trade PnLs.

    Equity starts at zero.  MaxDD is the maximum peak-to-trough decline of the
    cumulative PnL curve and is therefore always non-negative.
    """
    pnl = np.asarray(pnl_values, dtype=np.float64).ravel()
    if pnl.size == 0:
        return 0.0
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = running_peak - equity
    return float(max(0.0, np.max(drawdown) if drawdown.size else 0.0))


def threshold_utility(probabilities: Sequence[float],
                      pnl_values: Sequence[float],
                      theta: float,
                      epsilon: float = 1e-9,
                      min_accepted: int = 1) -> float:
    """
    Evaluate U(θ) exactly:

        U(θ) = sum(PnL_i * I(P_i >= θ)) / (MaxDD(θ) + ε)

    Trades are kept in their original temporal order before MaxDD is computed.
    If fewer than `min_accepted` trades pass θ, utility is -∞ so the threshold
    cannot be selected.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    pnl = np.asarray(pnl_values, dtype=np.float64).ravel()
    if p.size != pnl.size or p.size == 0:
        return -np.inf
    mask = np.isfinite(p) & np.isfinite(pnl) & (p >= float(theta))
    if int(mask.sum()) < int(min_accepted):
        return -np.inf
    accepted_pnl = pnl[mask]
    total_pnl = float(np.sum(accepted_pnl))
    max_dd = max_drawdown_from_pnl(accepted_pnl)
    return float(total_pnl / (max_dd + epsilon))


def search_best_threshold(probabilities: Sequence[float],
                          pnl_values: Sequence[float],
                          theta_lo: float = POSTTRADE_THETA_GRID_LO,
                          theta_hi: float = POSTTRADE_THETA_GRID_HI,
                          theta_step: float = POSTTRADE_THETA_GRID_STEP,
                          min_accepted: int = 1) -> dict:
    """
    Grid-search θ and return the best utility point.

    Ties are resolved toward the lower θ so the controller does not become
    unnecessarily selective when utility is numerically identical.
    """
    best_theta = float(theta_lo)
    best_utility = -np.inf
    best_accepted = 0
    thetas = np.arange(theta_lo, theta_hi + theta_step * 0.5, theta_step)
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    pnl = np.asarray(pnl_values, dtype=np.float64).ravel()
    for theta in thetas:
        utility = threshold_utility(p, pnl, float(theta), min_accepted=min_accepted)
        accepted = int((np.isfinite(p) & np.isfinite(pnl) & (p >= theta)).sum())
        if utility > best_utility:
            best_theta = float(theta)
            best_utility = float(utility)
            best_accepted = accepted
    if not np.isfinite(best_utility):
        best_theta = float(METAGATE_THETA_ENTRY)
        best_utility = -np.inf
        best_accepted = 0
    return {
        "theta": best_theta,
        "utility": best_utility,
        "accepted": best_accepted,
    }


def _fit_oof_classifier(features: np.ndarray,
                        labels: np.ndarray):
    """
    Fit the same model family as `fit_metagate` for one OOF split.

    Small samples use LogisticRegression; larger samples use the shallow GBM.
    This helper intentionally returns None on any numerical issue so the
    calibration loop can skip that fold rather than poisoning θ selection.
    """
    if len(features) < METAGATE_MIN_TRAIN_N or len(np.unique(labels)) < 2:
        return None
    try:
        if len(features) >= METAGATE_GBM_MIN_N:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_depth=2,
                max_iter=30,
                min_samples_leaf=10,
                learning_rate=0.05,
            )
            clf.fit(features, labels)
            return clf
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500),
        )
        pipe.fit(features, labels)
        return pipe
    except Exception:
        return None


def calibrate_threshold_oof(features: np.ndarray,
                            labels: np.ndarray,
                            pnl_values: np.ndarray,
                            n_splits: int = 5,
                            theta_lo: float = POSTTRADE_THETA_GRID_LO,
                            theta_hi: float = POSTTRADE_THETA_GRID_HI,
                            theta_step: float = POSTTRADE_THETA_GRID_STEP) -> dict:
    """
    Temporal out-of-fold calibration for MetaGate θ.

    The validation folds are contiguous and ordered.  For fold k, the model is
    trained only on observations strictly before that validation block.  This
    preserves temporal causality and prevents future training samples from
    contributing to earlier OOF predictions.

    Returns a dict with:
      - theta: selected θ*
      - utility: U(θ*) on OOF trades
      - n_oof: number of OOF predictions used
      - accepted: number of OOF trades accepted at θ*
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    pnl = np.asarray(pnl_values, dtype=np.float64).ravel()
    n = int(X.shape[0]) if X.ndim == 2 else 0
    if n == 0 or len(y) != n or len(pnl) != n:
        return {"theta": float(METAGATE_THETA_ENTRY), "utility": -np.inf, "n_oof": 0, "accepted": 0}
    X = MetaGateModel._align_feature_array(X, len(FEATURE_NAMES))
    n_splits = int(max(2, min(n_splits, n)))
    fold_edges = np.linspace(0, n, n_splits + 1, dtype=int)
    p_oof = np.full(n, np.nan, dtype=np.float64)

    for fold_idx in range(n_splits):
        start = int(fold_edges[fold_idx])
        end = int(fold_edges[fold_idx + 1])
        if end <= start:
            continue
        if start < METAGATE_MIN_TRAIN_N:
            continue
        train_idx = np.arange(0, start)
        val_idx = np.arange(start, end)
        clf = _fit_oof_classifier(X[train_idx], y[train_idx])
        if clf is None:
            continue
        try:
            p_oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        except Exception:
            p_oof[val_idx] = np.nan

    valid = np.isfinite(p_oof) & np.isfinite(pnl)
    if int(valid.sum()) == 0:
        return {"theta": float(METAGATE_THETA_ENTRY), "utility": -np.inf, "n_oof": 0, "accepted": 0}

    search = search_best_threshold(
        p_oof[valid],
        pnl[valid],
        theta_lo=theta_lo,
        theta_hi=theta_hi,
        theta_step=theta_step,
        min_accepted=1,
    )
    search["n_oof"] = int(valid.sum())
    return search


# ── Diagnostics writer ───────────────────────────────────────────────────────

def append_diagnostics(path: Path,
                       rows: list[dict]) -> None:
    """
    Append rows (each containing features, P, decision, fallback flag)
    to a CSV. Creates the file with a header if it does not yet exist.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    hdr = not Path(path).exists()
    df.to_csv(path, mode="a", header=hdr, index=False)
