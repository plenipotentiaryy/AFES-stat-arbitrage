"""
filters.py — CointegrationFilter and MacroFilter for backtest_pair.

CointegrationFilter
    Tier 1 (daily):   pre-computed EG p-value on daily closes, sampled every
                      COINT_RECHECK_DAYS days and forward-filled. O(1) lookup.
    Tier 2 (lazy):    ADF on recent intraday spread, triggered only when
                      |z| >= entry threshold. Cached per calendar day —
                      at most one ADF call per pair per day.

MacroFilter
    is_entry_blocked: VIX9D backwardation (macro_alert) OR K-Means != Sideways
    is_force_close:   K-Means Panic (regime 2)  OR  global SPY HMM panic
    force_close_reason: label written to trade record
"""

import hashlib
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import date as _date
from statsmodels.tsa.stattools import coint, adfuller
from numba import njit

from collections import deque

from config import (
    COINT_WINDOW_DAYS, COINT_BREAK_P, COINT_RECHECK_DAYS,
    BARS_PER_DAY, DATA_DIR,
    HURST_ENTRY_WINDOW, HURST_ENTRY_MAX,
    ADAPT_ALPHA, ADAPT_GAMMA, ADAPT_ETA, ADAPT_DELTA,
    ADAPT_W_MIN, ADAPT_W_MAX, ADAPT_B_MAX,
    ADAPT_CUSUM_K, ADAPT_CUSUM_H,
    ADAPT_SIGMA_FAST_BARS, ADAPT_SIGMA_SLOW_BARS,
    ADAPT_ENTRY_Z_CAP, ADAPT_STOP_Z_FLOOR,
    ADAPT_EPSILON,
)

# Intraday window for lazy ADF: same calendar span as daily pre-compute
_LAZY_WINDOW_BARS = COINT_WINDOW_DAYS * BARS_PER_DAY   # 90d × 26 bars = 2 340


# ── Numba-accelerated CUSUM + break score ────────────────────────────────────

@njit(cache=True)
def _compute_break_scores(nu, var_nu, hl, hl_med, d_beta,
                          z, k_cusum, h_cusum, threshold, half_life_val):
    """Vectorized break score over all bars using AR(1) pre-whitened innovations.

    CUSUM resets to 0 after each trigger so it does not accumulate
    permanently across the full history and falsely flag the entire OOS.
    """
    n = len(nu)
    scores     = np.empty(n, dtype=np.float64)
    broken_pos = np.empty(n, dtype=np.bool_)
    broken_neg = np.empty(n, dtype=np.bool_)
    
    # 1. Compute AR(1) parameter phi based on half-life
    phi = np.exp(-np.log(2.0) / max(half_life_val, 1.0))
    
    # 2. Compute raw AR(1) residuals: raw_resid[i] = z[i] - phi * z[i-1]
    raw_resid = np.empty(n, dtype=np.float64)
    raw_resid[0] = 0.0
    for i in range(1, n):
        raw_resid[i] = z[i] - phi * z[i-1]
        
    # 3. Compute rolling standard deviation of residuals (window=60)
    rolling_std = np.empty(n, dtype=np.float64)
    W = 60
    for i in range(n):
        start = max(0, i - W + 1)
        count = i - start + 1
        mean_val = 0.0
        for j in range(start, i + 1):
            mean_val += raw_resid[j]
        mean_val /= count
        
        var_val = 0.0
        for j in range(start, i + 1):
            var_val += (raw_resid[j] - mean_val) ** 2
        var_val /= count
        
        rolling_std[i] = np.sqrt(max(var_val, 1e-8))

    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        innov_shock = (nu[i] ** 2) / max(var_nu[i], 1e-9)
        s1 = np.log1p(innov_shock)
        hl_ratio = hl[i] / max(hl_med[i], 1e-9)
        s2 = np.log1p(max(0.0, hl_ratio - 1.0))
        
        # 4. Standardized pre-whitened innovation
        eta = raw_resid[i] / rolling_std[i]
        
        s_pos = max(0.0, s_pos + eta - k_cusum)
        s_neg = min(0.0, s_neg + eta + k_cusum)
        cusum_val = max(s_pos, abs(s_neg))
        s3 = cusum_val / h_cusum
        s4 = abs(d_beta[i]) * 100.0
        bt = 0.3 * s1 + 0.3 * s2 + 0.3 * s3 + 0.1 * s4
        triggered = bt > threshold or cusum_val > h_cusum
        scores[i] = bt
        broken_pos[i] = triggered and (s_pos >= 0.5 * h_cusum) and (s_pos >= abs(s_neg))
        broken_neg[i] = triggered and (abs(s_neg) >= 0.5 * h_cusum) and (abs(s_neg) > s_pos)
        # Standard CUSUM: reset after detection to prevent infinite accumulation
        if triggered:
            s_pos = 0.0
            s_neg = 0.0
    return scores, broken_pos, broken_neg


def compute_break_scores_vectorized(df: pd.DataFrame,
                                    var_nu: pd.Series,
                                    hl_median: pd.Series,
                                    d_beta_dt: pd.Series,
                                    half_life: float,
                                    threshold: float = 4.5,
                                    k_cusum: float = 0.5,
                                    h_cusum: float = 5.0):
    """Drop-in replacement for the iterrows loop in build_signals."""
    scores, broken_pos, broken_neg = _compute_break_scores(
        df["innov"].values.astype(np.float64),
        var_nu.values.astype(np.float64),
        df["half_life_bars"].values.astype(np.float64),
        hl_median.values.astype(np.float64),
        d_beta_dt.values.astype(np.float64),
        df["zscore"].values.astype(np.float64),
        k_cusum, h_cusum, threshold,
        float(half_life),
    )
    return scores, broken_pos, broken_neg


# ── CointegrationFilter ───────────────────────────────────────────────────────

_COINT_CACHE_DIR = Path(DATA_DIR) / "coint_cache"
_COINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _coint_cache_key(t1: str, t2: str, window: int,
                     p_thresh: float, step: int, n_rows: int) -> str:
    raw = f"{t1}-{t2}-{window}-{p_thresh}-{step}-{n_rows}"
    return hashlib.md5(raw.encode()).hexdigest()


class CointegrationFilter:
    """
    Usage:
        cf = CointegrationFilter(daily_df, "JPM", "BAC")
        if not cf.is_valid(today):          # daily pre-computed check
            ...
        if not cf.lazy_check(spread_tail, today):   # intraday ADF on trigger
            ...
    """

    def __init__(self, daily_df: pd.DataFrame | None, t1: str, t2: str,
                 window: int = COINT_WINDOW_DAYS,
                 p_thresh: float = COINT_BREAK_P,
                 step: int = COINT_RECHECK_DAYS):
        self._p_thresh   = p_thresh
        self._daily_dict: dict[_date, bool] = {}
        self._lazy_cache: dict[_date, bool] = {}

        if daily_df is None:
            return
        if t1 not in daily_df.columns or t2 not in daily_df.columns:
            return

        pc = daily_df[[t1, t2]].dropna()
        if len(pc) < window:
            return

        cache_key  = _coint_cache_key(t1, t2, window, p_thresh, step, len(pc))
        cache_file = _COINT_CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                sampled = pickle.load(f)
        else:
            sampled: dict = {}
            for i in range(window, len(pc) + 1, step):
                chunk = pc.iloc[i - window:i]
                try:
                    _, pval, _ = coint(chunk[t1], chunk[t2])
                    sampled[pc.index[i - 1]] = pval < p_thresh
                except Exception:
                    sampled[pc.index[i - 1]] = False
            with open(cache_file, "wb") as f:
                pickle.dump(sampled, f)

        if sampled:
            s = pd.Series(sampled).reindex(pc.index).ffill().fillna(True)
            self._daily_dict = {
                (ts.date() if hasattr(ts, "date") else ts): bool(v)
                for ts, v in s.items()
            }

    # ── Tier 1: O(1) daily lookup ─────────────────────────────────────────
    def is_valid(self, d) -> bool:
        """True = cointegrated per daily pre-compute. Defaults to True if no data."""
        key = d if isinstance(d, _date) else pd.Timestamp(d).date()
        return self._daily_dict.get(key, True)

    # ── MetaGate hook: continuous confidence score ────────────────────────
    def get_pvalue(self, spread_tail: pd.Series, d) -> float:
        """
        Return the most recent intraday ADF p-value for the spread tail.
        Cached per calendar day. Used by MetaGate to compute s_ADF = 1 - p.
        """
        key = d if isinstance(d, _date) else pd.Timestamp(d).date()
        cache_attr = "_pval_cache"
        if not hasattr(self, cache_attr):
            self._pval_cache: dict[_date, float] = {}
        if key in self._pval_cache:
            return self._pval_cache[key]
        s = spread_tail.dropna()
        if len(s) < 60:
            self._pval_cache[key] = 0.5
            return 0.5
        try:
            _, pval, _, _, _, _ = adfuller(s.values, maxlag=1,
                                           regression="c", autolag=None)
        except Exception:
            pval = 0.5
        self._pval_cache[key] = float(pval)
        return float(pval)

    def get_confidence_score(self, spread_tail: pd.Series, ts) -> float:
        """MetaGate score s_ADF = 1 - p_ADF, clipped to [0, 1]."""
        from metagate import adf_confidence
        return adf_confidence(self.get_pvalue(spread_tail, ts))

    # ── Tier 2: lazy intraday ADF on Z-trigger ────────────────────────────
    def lazy_check(self, spread_tail: pd.Series, d) -> bool:
        """
        Run ADF on the most recent intraday spread window.
        Returns True = spread still stationary (OK to enter).
        Cached per calendar day — runs at most once per pair per day.
        Call only when |z| >= entry threshold to keep the hot loop fast.
        """
        key = d if isinstance(d, _date) else pd.Timestamp(d).date()
        if key in self._lazy_cache:
            return self._lazy_cache[key]

        s = spread_tail.dropna()
        if len(s) < 60:
            self._lazy_cache[key] = True
            return True

        try:
            _, pval, _, _, _, _ = adfuller(s.values, maxlag=1,
                                           regression="c", autolag=None)
            ok = pval < self._p_thresh
        except Exception:
            ok = True   # be permissive on numerical failure

        self._lazy_cache[key] = ok
        return ok


# ── MacroFilter ───────────────────────────────────────────────────────────────

class MacroFilter:
    """
    Unified macro regime decision gate.

    Entry blocking (is_entry_blocked):
        — VIX9D > VIX (backwardation) from iv.py  → macro_alert_s
        — K-Means regime != 1 (Sideways)           → kmeans_series

    Force-close (is_force_close):
        — K-Means regime == 2 (Panic)
        — Global SPY HMM panic                     → global_hmm_s

    All three input Series are optional; missing → conservative default
    (no block, no force-close).
    """

    def __init__(self,
                 macro_alert_s: pd.Series | None,   # VIX9D backwardation (0/1)
                 global_hmm_s:  pd.Series | None,   # global HMM on SPY (0/1)
                 kmeans_series: pd.Series | None):  # K-Means regime (0/1/2)
        self._alert: dict[_date, bool] = {}
        self._hmm:   dict[_date, int]  = {}
        self._km:    dict[_date, int]  = {}

        if macro_alert_s is not None and not macro_alert_s.empty:
            for ts, v in macro_alert_s.items():
                self._alert[_to_date(ts)] = bool(v)

        if global_hmm_s is not None and not global_hmm_s.empty:
            for ts, v in global_hmm_s.items():
                self._hmm[_to_date(ts)] = int(v)

        if kmeans_series is not None and not kmeans_series.empty:
            for ts, v in kmeans_series.items():
                self._km[_to_date(ts)] = int(v)

    # ── Public API ────────────────────────────────────────────────────────
    def get_hmm_regime(self, ts) -> int:
        """Return raw global HMM label for ``ts``; missing data defaults to normal."""
        return int(self._hmm.get(_to_date(ts), 0))

    def get_kmeans_regime(self, ts) -> int:
        """Return raw K-Means label for ``ts``; missing data defaults to Sideways."""
        return int(self._km.get(_to_date(ts), 1))

    def is_entry_blocked(self, ts) -> bool:
        """True → do not open new positions on this bar.

        VIX9D backwardation is handled via size reduction (position_size),
        not as a hard entry block. Only K-Means Panic is a hard block.
        """
        d = _to_date(ts)
        if self._km and self._km.get(d, 1) == 2:  # K-Means Panic only
            return True
        return False

    def is_force_close(self, ts) -> bool:
        """True → immediately close any open position on this bar."""
        d = _to_date(ts)
        if self._km.get(d, 1) == 2:   # K-Means Panic
            return True
        if self._hmm.get(d, 0) == 1:  # global SPY HMM panic (new)
            return True
        return False

    def force_close_reason(self, ts) -> str:
        """Human-readable exit reason for the trade record."""
        d = _to_date(ts)
        if self._km.get(d, 1) == 2:
            return "PANIC"
        if self._hmm.get(d, 0) == 1:
            return "HMM_PANIC"
        return "FORCE_CLOSE"


# ── HurstFilter ───────────────────────────────────────────────────────────────

def get_hurst_multiplier(h_val: float) -> float:
    """
    Tier-1 Soft Sizing.
    H <= 0.50: 1.0 (Full size)
    H = 0.65:  0.3 (Reduced size)
    """
    if h_val <= 0.50: return 1.0
    if h_val >= 0.65: return 0.3
    
    # Linear scale-down
    penalty = (h_val - 0.50) / (0.65 - 0.50)
    return max(0.3, 1.0 - penalty * 0.7)

class HurstFilter:
    """
    Fast Vectorized Hurst Exponent (Variance Ratio Proxy).
    H \approx 0.5 * [log(Var(P_t - P_{t-tau})) / log(tau)]

    Purpose: catch structural drift where the spread is trending.
    """

    def __init__(self, h_max: float = 0.65, # Relaxed to 0.65
                 window: int = HURST_ENTRY_WINDOW,
                 hurst_lag: int = 10):
        self._h_max  = h_max
        self._window = window
        self._hurst_lag = hurst_lag
        self._cache: dict[_date, tuple[bool, float]] = {}

    def compute_hurst(self, spread_series: pd.Series) -> pd.Series:
        if len(spread_series) < self._window + self._hurst_lag:
            return pd.Series(0.5, index=spread_series.index)
            
        diff_1 = spread_series.diff(1)
        diff_tau = spread_series.diff(self._hurst_lag)
        
        var_1 = diff_1.rolling(window=self._window).var()
        var_tau = diff_tau.rolling(window=self._window).var()
        
        var_1 = var_1.replace(0, np.nan)
        hurst = 0.5 * (np.log(var_tau / var_1) / np.log(self._hurst_lag))
        return hurst.fillna(0.5)

    def should_block(self, spread_tail: pd.Series, ts) -> tuple[bool, float]:
        """Returns (should_block, hurst_value). Cached per calendar day."""
        key = _to_date(ts)
        if key in self._cache:
            return self._cache[key]

        s = spread_tail.dropna()
        if len(s) < self._window + self._hurst_lag:
            self._cache[key] = (False, 0.5)
            return False, 0.5

        # Calculate only the last value for performance in hot loop
        diff_1 = s.diff(1).tail(self._window)
        diff_tau = s.diff(self._hurst_lag).tail(self._window)
        
        v1 = diff_1.var()
        vt = diff_tau.var()
        
        if v1 == 0 or np.isnan(v1):
            h = 0.5
        else:
            h = 0.5 * (np.log(vt / v1) / np.log(self._hurst_lag))
            
        blocked = h > self._h_max
        self._cache[key] = (blocked, h)
        return blocked, h

    def get_confidence_score(self, spread_tail: pd.Series, ts) -> float:
        """MetaGate score s_Hurst = clip((h_max - H) / (h_max - 0.5), 0, 1)."""
        from metagate import hurst_confidence
        _, h_val = self.should_block(spread_tail, ts)
        return hurst_confidence(h_val)


# ── BreakVelocityDetector ─────────────────────────────────────────────────────

class BreakVelocityDetector:
    """
    Structural Break Velocity Layer (Criticality 10 upgrade).
    Detects early-stage structural breaks before Hurst/Cointegration filters react.
    
    Indicators:
    1. Kalman Innovation Shock: r_t = (nu_t^2) / E[nu^2]_rolling
    2. Half-Life Ratio: HL_now / HL_median_rolling
    3. CUSUM (Cumulative Sum): Detects cumulative mean drift.
    """

    def __init__(self, 
                 window_fast: int = 20, 
                 window_slow: int = 120,
                 threshold: float = 4.5):
        self.window_fast = window_fast
        self.window_slow = window_slow
        self.threshold   = threshold
        
        # CUSUM params
        self.k_cusum = 0.5   # Slack parameter (0.5 sigma)
        self.h_cusum = 5.0   # CUSUM threshold (5 sigma)
        self.s_pos = 0.0
        self.s_neg = 0.0

    def update_cusum(self, z_score: float):
        """Standard CUSUM algorithm for mean drift detection."""
        self.s_pos = max(0, self.s_pos + z_score - self.k_cusum)
        self.s_neg = min(0, self.s_neg + z_score + self.k_cusum)
        return max(self.s_pos, abs(self.s_neg))

    def get_break_score(self, 
                        nu_t: float, 
                        rolling_var_nu: float,
                        hl_t: float, 
                        hl_median: float,
                        d_beta_dt: float,
                        z_t: float) -> tuple[float, bool]:
        """
        Computes composite Break Score (Bt).
        Returns: (score, is_broken)
        """
        # 1. Innovation Shock (log scale to dampen outliers)
        innovation_shock = (nu_t**2) / max(rolling_var_nu, 1e-9)
        s1 = np.log1p(innovation_shock)
        
        # 2. HL Explosion ratio
        hl_ratio = hl_t / max(hl_median, 1e-9)
        s2 = np.log1p(max(0, hl_ratio - 1))
        
        # 3. CUSUM drift
        cusum_val = self.update_cusum(z_t)
        s3 = cusum_val / self.h_cusum
        
        # 4. Beta Instability
        s4 = abs(d_beta_dt) * 100.0  # Scale beta velocity
        
        # Composite score: Bt = w1*s1 + w2*s2 + w3*s3 + w4*s4
        bt = 0.3 * s1 + 0.3 * s2 + 0.3 * s3 + 0.1 * s4
        
        is_broken = bt > self.threshold or cusum_val > self.h_cusum
        return float(bt), bool(is_broken)

    def get_confidence_score(self,
                             nu_t: float,
                             rolling_var_nu: float,
                             hl_t: float,
                             hl_median: float,
                             d_beta_dt: float,
                             z_t: float) -> tuple[float, float]:
        """
        MetaGate scores (s_Break, s_CUSUM) at the current bar. Uses the
        same composite Bt and CUSUM Ct as `get_break_score`, but mapped
        to [0, 1] confidences. Mutates internal CUSUM state — call once
        per bar in chronological order.
        """
        from metagate import break_confidence, cusum_confidence
        bt, _ = self.get_break_score(nu_t, rolling_var_nu, hl_t, hl_median,
                                     d_beta_dt, z_t)
        c_t = max(self.s_pos, abs(self.s_neg))
        return break_confidence(bt, hi=self.threshold), cusum_confidence(c_t, h_cusum=self.h_cusum)


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_date(ts) -> _date:
    if isinstance(ts, _date) and not isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts).date()

# ── Online Parameter Adaptation (between WFO refits) ─────────────────────────
#
# Implements three coupled live-adjustment layers:
#
#   Layer 1 — Adaptive EWMA window:
#       W_t  = clip( W_base · (1 - α · min(1, Bt/B_max)),  [W_min, W_max] )
#       λ_t  = 2 / (W_t + 1)
#       μ_t  = λ_t·s_t + (1-λ_t)·μ_{t-1}
#       σ²_t = λ_t·(s_t - μ_t)² + (1-λ_t)·σ²_{t-1}
#       z_t  = (s_t - μ_t) / sqrt(σ²_t + ε)
#
#   Layer 2 — Volatility-adjusted entry threshold:
#       entry_z_t = entry_z_base · (1 + γ · max(0, σ_fast/σ_slow - 1))
#                   clipped to  [entry_z_base, ENTRY_Z_CAP]
#
#   Layer 3 — Drift-tightened stop threshold:
#       stop_z_t  = stop_z_base · max( δ, 1 - η · (Ct/h_cusum) )
#                   clipped to  [STOP_Z_FLOOR, stop_z_base]
#
# Two interfaces are exposed:
#   - OnlineParameterAdapter (state-machine for live / paper trading).
#   - compute_adaptive_arrays() (numba kernel for vectorised backtest engines).


@njit(cache=True)
def _adaptive_kernel(spread, B_arr, C_arr, sigma_fast_arr, sigma_slow_arr,
                     w_base, entry_z_base, stop_z_base,
                     alpha, gamma, eta, delta,
                     w_min, w_max, b_max, cusum_h,
                     entry_z_cap, stop_z_floor, epsilon):
    """
    Vectorised computation of (W_t, entry_z_t, stop_z_t, μ_t, σ_t, z_t) arrays.

    Pure-numpy/numba so it can be called once per pair-window before the inner
    bar loop, eliminating Python overhead. Inputs are precomputed numpy
    arrays of identical length; outputs share that length. The kernel uses
    exclusively the math specified in the docstring above — no hidden
    smoothing, look-ahead, or special-casing.
    """
    n = spread.shape[0]
    W_t_arr     = np.empty(n, dtype=np.int64)
    entry_z_arr = np.empty(n, dtype=np.float64)
    stop_z_arr  = np.empty(n, dtype=np.float64)
    mu_arr      = np.empty(n, dtype=np.float64)
    sigma_arr   = np.empty(n, dtype=np.float64)
    z_arr       = np.empty(n, dtype=np.float64)

    mu  = 0.0
    var = 1.0
    initialised = False

    for i in range(n):
        # ── Layer 1: adaptive window ─────────────────────────────────────
        bt = B_arr[i]
        if not np.isfinite(bt) or bt < 0.0:
            bt = 0.0
        ratio = bt / b_max
        if ratio > 1.0:
            ratio = 1.0
        w_continuous = w_base * (1.0 - alpha * ratio)
        if w_continuous < w_min:
            w_continuous = w_min
        elif w_continuous > w_max:
            w_continuous = w_max
        # integer-rounded for symmetry with classical rolling windows
        W_t = int(round(w_continuous))
        if W_t < w_min:
            W_t = w_min
        elif W_t > w_max:
            W_t = w_max
        W_t_arr[i] = W_t

        # ── EWMA recursive update ───────────────────────────────────────
        s = spread[i]
        if not np.isfinite(s):
            s = mu  # absorb NaN to prior mean (no-op update)
        if not initialised:
            mu  = s
            var = 1.0
            initialised = True
        else:
            lam = 2.0 / (W_t + 1.0)
            mu  = lam * s + (1.0 - lam) * mu
            var = lam * (s - mu) * (s - mu) + (1.0 - lam) * var
        sigma = (var + epsilon) ** 0.5
        mu_arr[i]    = mu
        sigma_arr[i] = sigma
        z_arr[i]     = (s - mu) / sigma

        # ── Layer 2: volatility-adjusted entry threshold ────────────────
        sf = sigma_fast_arr[i]
        ss = sigma_slow_arr[i]
        if (not np.isfinite(sf)) or (not np.isfinite(ss)) or ss <= 1e-12:
            r = 0.0
        else:
            r = sf / ss - 1.0
            if r < 0.0:
                r = 0.0
        ez = entry_z_base * (1.0 + gamma * r)
        if ez < entry_z_base:
            ez = entry_z_base
        elif ez > entry_z_cap:
            ez = entry_z_cap
        entry_z_arr[i] = ez

        # ── Layer 3: drift-tightened stop threshold ─────────────────────
        ct = C_arr[i]
        if not np.isfinite(ct) or ct < 0.0:
            ct = 0.0
        c_ratio = ct / cusum_h
        if c_ratio > 1.0:
            c_ratio = 1.0
        scale = 1.0 - eta * c_ratio
        if scale < delta:
            scale = delta
        sz = stop_z_base * scale
        if sz < stop_z_floor:
            sz = stop_z_floor
        elif sz > stop_z_base:
            sz = stop_z_base
        stop_z_arr[i] = sz

    return W_t_arr, entry_z_arr, stop_z_arr, mu_arr, sigma_arr, z_arr


def compute_adaptive_arrays(spread: np.ndarray,
                            B_arr:   np.ndarray,
                            C_arr:   np.ndarray,
                            sigma_fast_arr: np.ndarray,
                            sigma_slow_arr: np.ndarray,
                            w_base: int,
                            entry_z_base: float,
                            stop_z_base:  float,
                            alpha:        float = ADAPT_ALPHA,
                            gamma:        float = ADAPT_GAMMA,
                            eta:          float = ADAPT_ETA,
                            delta:        float = ADAPT_DELTA,
                            w_min:        int   = ADAPT_W_MIN,
                            w_max:        int   = ADAPT_W_MAX,
                            b_max:        float = ADAPT_B_MAX,
                            cusum_h:      float = ADAPT_CUSUM_H,
                            entry_z_cap:  float = ADAPT_ENTRY_Z_CAP,
                            stop_z_floor: float = ADAPT_STOP_Z_FLOOR,
                            epsilon:      float = ADAPT_EPSILON):
    """
    Public wrapper around `_adaptive_kernel`. Casts inputs to float64 (numba
    is strict about dtypes) and returns six aligned arrays.

    Returns
    -------
    W_t_arr      : int64[n]   — per-bar adaptive EWMA window
    entry_z_arr  : float64[n] — per-bar adaptive entry threshold
    stop_z_arr   : float64[n] — per-bar adaptive stop threshold
    mu_arr       : float64[n] — recursive EWMA mean of the spread
    sigma_arr    : float64[n] — recursive EWMA std-dev (with ε floor)
    z_arr        : float64[n] — adaptive z-score (spread - μ) / σ
    """
    spread_f         = np.ascontiguousarray(spread,         dtype=np.float64)
    B_arr_f          = np.ascontiguousarray(B_arr,          dtype=np.float64)
    C_arr_f          = np.ascontiguousarray(C_arr,          dtype=np.float64)
    sigma_fast_arr_f = np.ascontiguousarray(sigma_fast_arr, dtype=np.float64)
    sigma_slow_arr_f = np.ascontiguousarray(sigma_slow_arr, dtype=np.float64)
    return _adaptive_kernel(
        spread_f, B_arr_f, C_arr_f, sigma_fast_arr_f, sigma_slow_arr_f,
        int(w_base), float(entry_z_base), float(stop_z_base),
        float(alpha), float(gamma), float(eta), float(delta),
        int(w_min), int(w_max), float(b_max), float(cusum_h),
        float(entry_z_cap), float(stop_z_floor), float(epsilon),
    )


class OnlineParameterAdapter:
    """
    Bar-by-bar parameter adaptation state machine for live / paper trading.

    Construction
    ------------
    Pass the baseline (frozen-WFO) parameters: W_base, entry_z_base,
    stop_z_base. Adaptation knobs (α, γ, η, δ, …) default to values in
    config.py — override per-pair if needed.

    Usage
    -----
    For every incoming bar:
        adapter.update(ts, spread_val, break_score, cusum_val)
        entry_z = adapter.get_adaptive_entry_z()
        stop_z  = adapter.get_adaptive_stop_z()
        z       = adapter.get_ewma_zscore()   # or pass a fresh spread

    The adapter maintains a rolling buffer of spread *differences* for
    σ_fast / σ_slow, plus the running EWMA μ_t / σ²_t.

    Backward-compatibility note
    ---------------------------
    Setting α=γ=η=0 freezes all three layers (W_t = W_base, entry_z_t =
    entry_z_base, stop_z_t = stop_z_base) so the adapter degrades to a
    constant-window EWMA. Combined with the `adaptive=False` kill-switch
    in the backtest callers, the rolling-window baseline runs untouched.
    """

    __slots__ = (
        "w_base", "entry_z_base", "stop_z_base",
        "alpha", "gamma", "eta", "delta",
        "w_min", "w_max", "b_max", "cusum_h", "cusum_k",
        "entry_z_cap", "stop_z_floor",
        "sigma_fast_bars", "sigma_slow_bars",
        "epsilon",
        "_spread_diffs", "_prev_spread",
        "_mu", "_var",
        "_W_t", "_entry_z_t", "_stop_z_t", "_z_t",
        "_initialised", "_last_ts",
    )

    def __init__(self,
                 w_base:        int,
                 entry_z_base:  float,
                 stop_z_base:   float,
                 alpha:        float = ADAPT_ALPHA,
                 gamma:        float = ADAPT_GAMMA,
                 eta:          float = ADAPT_ETA,
                 delta:        float = ADAPT_DELTA,
                 w_min:        int   = ADAPT_W_MIN,
                 w_max:        int   = ADAPT_W_MAX,
                 b_max:        float = ADAPT_B_MAX,
                 cusum_k:      float = ADAPT_CUSUM_K,
                 cusum_h:      float = ADAPT_CUSUM_H,
                 entry_z_cap:  float = ADAPT_ENTRY_Z_CAP,
                 stop_z_floor: float = ADAPT_STOP_Z_FLOOR,
                 sigma_fast_bars: int = ADAPT_SIGMA_FAST_BARS,
                 sigma_slow_bars: int = ADAPT_SIGMA_SLOW_BARS,
                 epsilon:      float = ADAPT_EPSILON):
        if w_base < w_min:
            w_base = w_min
        if w_base > w_max:
            w_base = w_max
        if sigma_slow_bars < sigma_fast_bars:
            raise ValueError("sigma_slow_bars must be >= sigma_fast_bars")
        self.w_base          = int(w_base)
        self.entry_z_base    = float(entry_z_base)
        self.stop_z_base     = float(stop_z_base)
        self.alpha           = float(alpha)
        self.gamma           = float(gamma)
        self.eta             = float(eta)
        self.delta           = float(delta)
        self.w_min           = int(w_min)
        self.w_max           = int(w_max)
        self.b_max           = float(b_max)
        self.cusum_h         = float(cusum_h)
        self.cusum_k         = float(cusum_k)
        self.entry_z_cap     = float(entry_z_cap)
        self.stop_z_floor    = float(stop_z_floor)
        self.sigma_fast_bars = int(sigma_fast_bars)
        self.sigma_slow_bars = int(sigma_slow_bars)
        self.epsilon         = float(epsilon)

        # Rolling buffer for spread DIFFERENCES (Δs_t = s_t − s_{t-1}).
        # Capped at sigma_slow_bars; σ_fast just slices the tail.
        self._spread_diffs: deque = deque(maxlen=self.sigma_slow_bars)
        self._prev_spread: float | None = None

        # EWMA state
        self._mu:  float | None = None
        self._var: float | None = None
        self._initialised = False

        # Adaptive params (initialise to baseline so first-bar callers
        # always see safe values before any update() call).
        self._W_t        = self.w_base
        self._entry_z_t  = self.entry_z_base
        self._stop_z_t   = self.stop_z_base
        self._z_t        = 0.0
        self._last_ts    = None

    # ── Public API ───────────────────────────────────────────────────────

    def update(self, ts, spread_val: float,
               break_score: float, cusum_val: float) -> None:
        """
        Ingest one new bar. Updates EWMA state and recomputes the three
        adaptive parameters in order: W_t → entry_z_t → stop_z_t.
        """
        self._last_ts = ts
        s = float(spread_val) if np.isfinite(spread_val) else (self._mu or 0.0)

        # Roll the spread-diff buffer (skipped on the very first bar).
        if self._prev_spread is not None:
            self._spread_diffs.append(s - self._prev_spread)
        self._prev_spread = s

        # σ_fast / σ_slow from buffered diffs.
        n_buf = len(self._spread_diffs)
        sigma_fast = float("nan")
        sigma_slow = float("nan")
        if n_buf >= self.sigma_fast_bars:
            tail_fast = list(self._spread_diffs)[-self.sigma_fast_bars:]
            sigma_fast = float(np.std(tail_fast, ddof=0))
        if n_buf >= max(2, min(self.sigma_slow_bars, self.sigma_fast_bars + 1)):
            # Use as much history as we have, up to slow_bars.
            tail_slow = list(self._spread_diffs)
            sigma_slow = float(np.std(tail_slow, ddof=0))

        # ── Layer 1: adaptive window ────────────────────────────────────
        bt = float(break_score) if np.isfinite(break_score) else 0.0
        if bt < 0.0:
            bt = 0.0
        ratio = min(1.0, bt / self.b_max)
        w_cont = self.w_base * (1.0 - self.alpha * ratio)
        w_cont = max(self.w_min, min(self.w_max, w_cont))
        self._W_t = int(round(w_cont))
        self._W_t = max(self.w_min, min(self.w_max, self._W_t))

        # ── EWMA recursive update ───────────────────────────────────────
        if not self._initialised:
            self._mu = s
            self._var = 1.0
            self._initialised = True
        else:
            lam = 2.0 / (self._W_t + 1.0)
            self._mu  = lam * s + (1.0 - lam) * self._mu
            self._var = lam * (s - self._mu) ** 2 + (1.0 - lam) * self._var
        sigma = (self._var + self.epsilon) ** 0.5
        self._z_t = (s - self._mu) / sigma

        # ── Layer 2: entry_z adaptation ─────────────────────────────────
        if (np.isfinite(sigma_fast) and np.isfinite(sigma_slow)
                and sigma_slow > 1e-12):
            r = max(0.0, sigma_fast / sigma_slow - 1.0)
        else:
            r = 0.0
        ez = self.entry_z_base * (1.0 + self.gamma * r)
        self._entry_z_t = max(self.entry_z_base, min(self.entry_z_cap, ez))

        # ── Layer 3: stop_z adaptation ──────────────────────────────────
        ct = float(cusum_val) if np.isfinite(cusum_val) else 0.0
        if ct < 0.0:
            ct = 0.0
        c_ratio = min(1.0, ct / self.cusum_h)
        scale = max(self.delta, 1.0 - self.eta * c_ratio)
        sz = self.stop_z_base * scale
        self._stop_z_t = max(self.stop_z_floor, min(self.stop_z_base, sz))

    def get_adaptive_window(self) -> int:
        """Current adaptive EWMA window W_t."""
        return int(self._W_t)

    def get_adaptive_entry_z(self) -> float:
        """Current adaptive entry threshold entry_z_t."""
        return float(self._entry_z_t)

    def get_adaptive_stop_z(self) -> float:
        """Current adaptive stop threshold stop_z_t."""
        return float(self._stop_z_t)

    def get_ewma_zscore(self, spread_val: float | None = None) -> float:
        """
        Latest EWMA z-score. If `spread_val` is supplied (and EWMA state
        is initialised) computes (s - μ_t)/σ_t for that fresh spread —
        useful for "what-if" probes without mutating state.
        """
        if spread_val is None or not self._initialised:
            return float(self._z_t)
        sigma = (self._var + self.epsilon) ** 0.5
        return float((float(spread_val) - self._mu) / sigma)

    def state_snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of the current state."""
        return {
            "ts":               self._last_ts,
            "W_t":              int(self._W_t),
            "entry_z_t":        float(self._entry_z_t),
            "stop_z_t":         float(self._stop_z_t),
            "ewma_z":           float(self._z_t),
            "ewma_mu":          None if self._mu is None else float(self._mu),
            "ewma_var":         None if self._var is None else float(self._var),
            "buffer_size":      len(self._spread_diffs),
            "initialised":      bool(self._initialised),
        }


from scipy.stats import gaussian_kde

def validate_kde_density(z_series: pd.Series, entry_z: float, threshold_ratio: float = 0.5) -> bool:
    """
    KDE Structural Filter.
    Checks if the historical density at entry_z is at least `threshold_ratio` of the 
    theoretical Gaussian density at entry_z.
    If KDE(entry_z) < Gaussian(entry_z) * threshold_ratio, it's a Low Density Node (void).
    Returns True if valid (HVN or normal), False if invalid (LDN / void).
    """
    z = z_series.dropna().values
    if len(z) < 100:
        return True # Not enough data
        
    try:
        kde = gaussian_kde(z)
        
        # We evaluate empirical density at entry_z and -entry_z
        d_pos = kde(entry_z)[0]
        d_neg = kde(-entry_z)[0]
        d_empirical = (d_pos + d_neg) / 2.0
        
        # Theoretical Gaussian PDF at entry_z
        # phi(z) = (1 / sqrt(2*pi)) * e^(-0.5 * z^2)
        d_theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (entry_z ** 2))
        
        return bool(d_empirical >= (d_theoretical * threshold_ratio))
    except Exception:
        # e.g., singular matrix if variance is zero
        return True
