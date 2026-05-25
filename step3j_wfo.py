"""
wfo.py — Walk-Forward Optimization (WFO) Engine.

Implements the Gatev, Goetzmann, Rouwenhorst (2006) baseline framework:
  - Formation (train) window:  12 months
  - Trading (OOS) window:       6 months
  - Step:                       6 months (non-overlapping OOS windows)

Workflow per window:
  1. Build signals on TRAIN slice (rolling zscore)
  2. Grid search (entry_z, exit_z, stop_z) → pick best by Sharpe
  3. Trade those params on the NEXT 6-month OOS slice (zero look-ahead)
  4. Accumulate OOS trades into a continuous equity curve

Output:
  data/wfo_results.csv      — OOS trades from all windows
  data/wfo_params.csv       — best params per pair per window
  output/wfo_equity.png     — continuous 18-year OOS equity curve
  output/wfo_stability.png  — train vs test Sharpe per window per pair

Pipeline position: step 3i (optional, after grid.py)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import numpy as np
import pandas as pd
from numba import njit
import scipy.optimize as opt
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import (
    TAIL_HEDGE_DRAG_ANNUAL, TAIL_HEDGE_PAYOUT_MULT, INITIAL_CAPITAL,
    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z,
    CLOSES_FILE, VOLUMES_FILE, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR,
    WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, WFO_STEP_MONTHS, WFO_MIN_TRADES,
    WFO_EXPANDING, HMM_PANIC_MULT,
    HURST_ENTRY_WINDOW,
)
from filters import (
    HurstFilter, CointegrationFilter,
    get_hurst_multiplier, validate_kde_density,
    OnlineParameterAdapter, compute_adaptive_arrays,
)
from metagate import (
    FEATURE_NAMES,
    build_score_frame,
    fit_metagate,
    append_diagnostics,
    MetaGateModel,
)
from regime import (
    RegimeState, compute_effective_trade_knobs,
    shift_daily_to_t1, apply_sbr_guard,
)
from post_trade import ShadowBuffer, PairPerformanceTracker
from feedback import PerformanceFeedbackTracker
from config import (
    METAGATE_THETA_ENTRY, METAGATE_DIAG_CSV,
    ADAPT_ALPHA, ADAPT_GAMMA, ADAPT_ETA, ADAPT_DELTA,
    ADAPT_W_MIN, ADAPT_W_MAX, ADAPT_B_MAX,
    ADAPT_CUSUM_K, ADAPT_CUSUM_H,
    ADAPT_SIGMA_FAST_BARS, ADAPT_SIGMA_SLOW_BARS,
    ADAPT_ENTRY_Z_CAP, ADAPT_STOP_Z_FLOOR,
    ADAPT_EPSILON, ADAPT_DIAG_CSV, ADAPT_ENABLED,
    POSTTRADE_ENABLED, PFB_ENABLED, METAGATE_REQUIRE_USABLE,
    ALLOCATION_METHOD, COVARIANCE_SHRINKAGE,
    CORR_BLOCK_MODE, CORR_BLOCK_THRESHOLD,
    CORR_BLOCK_SCALE, CORR_BLOCK_WINDOW_DAYS,
    SBR_GUARD_ENABLED, SBR_GUARD_THRESHOLD, SBR_GUARD_STOP_MULT,
    WFO_MIN_TRAIN_SHARPE, WFO_MIN_TRAIN_PNL,
)
from risk.ticker_overlap import (
    TickerOverlapBlocker, compute_spread_correlations, filter_trade_list,
)

# Cap on shadow-trade holding length (bars). Prevents shadow positions from
# living forever when no exit signal fires.
_SHADOW_MAX_HOLD_BARS = 26 * 10  # ≈ 10 trading days at 15-min cadence

# Grid definition (same as grid.py)
ENTRY_Z_GRID = [1.65, 1.7, 1.8, 2.0, 2.2]
STOP_Z_GRID  = [3.0, 3.2, 3.5]
EXIT_Z_GRID  = [-0.1, 0.0, 0.1]
COMBOS       = [(e, x, s) for e, x, s in
                itertools.product(ENTRY_Z_GRID, EXIT_Z_GRID, STOP_Z_GRID)
                if s > e and x < e]
N_COMBOS = len(COMBOS)

# Half-life gate: skip pairs whose OU mean-reversion is too slow.
# A 12m OOS window gives ~6 months to trade; a pair with HL > 90 days
# will barely complete one full cycle — it's a 'lazy' cointegration.
HL_MAX_DAYS = 90   # calendar days (will be converted to bars inside)
WFO_SIGNAL_WARMUP_BARS = 200  # max rolling z/VWAP lookback used by build_signals()


# ── Grid kernel (same logic as grid.py, accelerated with numba) ───────────────

@njit(cache=True)
def _grid_kernel(zscore, spread, spread_mean, spread_std, vr_arr, t1_price, t2_price,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_maker, cost_taker, borrow_rate, bars_per_day):
    n_bars, n_c = len(zscore), len(entry_arr)
    pos   = np.zeros(n_c, dtype=np.int64)
    e_sp  = np.zeros(n_c); e_t1 = np.zeros(n_c)
    e_t2  = np.zeros(n_c); e_bar = np.zeros(n_c, dtype=np.int64)
    e_sma = np.zeros(n_c); e_std = np.zeros(n_c)
    res   = np.zeros((n_c, 6)); cum = np.zeros(n_c); pk = np.zeros(n_c)

    for i in range(n_bars):
        z = zscore[i]; s = spread[i]; sm = spread_mean[i]; sd = spread_std[i]
        vr = vr_arr[i]; p1 = t1_price[i]; p2 = t2_price[i]
        for c in range(n_c):
            ez, xz, sz, pc = entry_arr[c], exit_arr[c], stop_arr[c], pos[c]
            
            z_active = z
            if pc != 0 and e_std[c] > 0:
                z_active = (s - e_sma[c]) / e_std[c]
                
            if pc != 0:
                ex = (pc == 1 and z_active >= xz) or (pc == -1 and z_active <= -xz)
                st = (pc == 1 and z_active <= -sz) or (pc == -1 and z_active >= sz)
                if ex or st:
                    gross = pc * (s - e_sp[c])
                    notl  = e_t1[c] + beta * e_t2[c]
                    tx    = notl * cost_maker + notl * (cost_maker if ex else cost_taker)
                    hd    = (i - e_bar[c]) / bars_per_day
                    brw   = ((beta * e_t2[c]) if pc == 1 else e_t1[c]) * borrow_rate * hd / 252
                    net   = gross - tx - brw
                    res[c, 0] += net; res[c, 1] += 1
                    if net > 0: res[c, 2] += 1
                    res[c, 3] += net * net
                    if st: res[c, 5] += 1
                    cum[c] += net
                    if cum[c] > pk[c]: pk[c] = cum[c]
                    dd = cum[c] - pk[c]
                    if dd < res[c, 4]: res[c, 4] = dd
                    pos[c] = 0
            if pos[c] == 0 and vr >= 0.5:
                if z < -ez:
                    pos[c] = 1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd
                elif z > ez:
                    pos[c] = -1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd
    return res


def pair_train_quality_ok(best: dict | None,
                          min_sharpe: float = 0.0,
                          min_pnl: float = 0.0) -> bool:
    """
    Decide whether a pair's TRAIN window grid result is good enough to bother
    running OOS on.

    A "best" dict produced by :func:`run_grid` carries at minimum ``sharpe``
    and ``total_pnl`` fields.  The pair passes the gate iff BOTH metrics
    strictly exceed the configured floors; if either falls at or below its
    floor the function returns False and the caller is expected to skip the
    OOS pass for this pair in this window.

    A missing or non-finite metric also fails the gate — the caller must not
    silently treat NaN as "pass" because that would re-introduce the very
    losing-pair leakage this gate exists to prevent.

    Parameters
    ----------
    best
        The dict returned by ``run_grid``; ``None`` always fails the gate.
    min_sharpe, min_pnl
        Floors.  Set either to ``-math.inf`` to disable that check.

    Notes
    -----
    The gate is per-window: a pair that fails the train window of, say,
    2018-Q1 is *still* re-evaluated on 2018-Q2 train data.  This matches the
    spec preference for "soft" exclusion over persistent ban.
    """
    if not isinstance(best, dict):
        return False
    sharpe = best.get("sharpe")
    pnl = best.get("total_pnl")
    try:
        sharpe_f = float(sharpe)
        pnl_f = float(pnl)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(sharpe_f) and np.isfinite(pnl_f)):
        return False
    if sharpe_f <= float(min_sharpe):
        return False
    if pnl_f <= float(min_pnl):
        return False
    return True


def run_grid(df, t1, t2, beta, combos, days, min_trades=WFO_MIN_TRADES):
    """Run grid search. Returns best row (entry_z, exit_z, stop_z, sharpe) or None."""
    valid = [(e, x, s) for e, x, s in combos if s > e and x < e]
    if not valid or len(df) < 50:
        return None

    ea = np.array([c[0] for c in valid], dtype=np.float64)
    xa = np.array([c[1] for c in valid], dtype=np.float64)
    sa = np.array([c[2] for c in valid], dtype=np.float64)

    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_mean"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_std"].to_numpy(np.float64)),
        np.ascontiguousarray(df["vr"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),
        ea, xa, sa, float(beta),
        float(COST_MAKER), float(COST_TAKER), float(BORROW_RATE_ANNUAL), float(BARS_PER_DAY)
    )

    years = max(days / 365.25, 1e-9)
    best_sh, best_row = -np.inf, None
    for i, (ez, xz, sz) in enumerate(valid):
        n = int(res[i, 1])
        if n < min_trades:
            continue
        mean = res[i, 0] / n
        var  = max(res[i, 3] / n - mean**2, 0.0)
        std  = np.sqrt(var)
        tpy  = n / years
        sh   = mean / std * np.sqrt(tpy) if std > 0 else 0.0
        if sh > best_sh:
            best_sh  = sh
            best_row = {"entry_z": ez, "exit_z": xz, "stop_z": sz,
                        "sharpe": round(sh, 3), "trades": n,
                        "win_rate": round(res[i, 2] / n * 100, 1),
                        "total_pnl": round(res[i, 0], 4)}
    return best_row


def build_signals(closes, volumes, t1, t2, beta, half_life):
    c1 = closes[t1]
    c2 = closes[t2]
    spread = c1 - beta * c2
    window = max(20, min(int(half_life), 200))
    
    if volumes is not None and t1 in volumes.columns and t2 in volumes.columns:
        v1 = volumes[t1]
        v2 = volumes[t2]
        
        # 1. Volume-Weighted Z-Score (Tier-1)
        # V_spread = min(Dollar Volume A, Dollar Volume B)
        v_spread = np.minimum(v1 * c1, v2 * c2)
        rolling_v_sum = v_spread.rolling(window=window).sum()
        
        # VW-Mean = Sum(Spread * V_spread) / Sum(V_spread)
        vw_mean = (spread * v_spread).rolling(window=window).sum() / rolling_v_sum
        
        # VW-Variance = Sum(V_spread * (Spread - VW-Mean)^2) / Sum(V_spread)
        vw_var = (v_spread * (spread - vw_mean)**2).rolling(window=window).sum() / rolling_v_sum
        spread_std = np.sqrt(vw_var.clip(lower=1e-9))
        
        # Liquidity Filter (VR)
        v_sma = v_spread.rolling(window=window).mean()
        vr = v_spread / v_sma.replace(0, np.nan)
    else:
        # Fallback if volume is missing
        vw_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        vr = pd.Series(1.0, index=spread.index)
        
    zscore = (spread - vw_mean) / spread_std
    
    return pd.DataFrame({
        f"{t1}_close": c1,
        f"{t2}_close": c2,
        "spread": spread, "zscore": zscore,
        "spread_mean": vw_mean, "spread_std": spread_std,
        "vr": vr
    }).dropna().between_time(SIGNAL_START, RTH_END)


def trim_oos_signal_warmup(sig: pd.DataFrame,
                           oos_start: pd.Timestamp,
                           oos_end: pd.Timestamp) -> pd.DataFrame:
    """Drop warmup bars after rolling indicators are computed."""
    return sig[(sig.index >= oos_start) & (sig.index < oos_end)]

# ── Dynamic Cointegration ────────────────────────────────────────────────────

_JOH_CRIT_IDX = {0.90: 0, 0.95: 1, 0.99: 2}

def check_coint_johansen(df_daily, t1, t2, crit_level=0.95):
    """Run Johansen on daily slice; returns (is_coint, beta)."""
    pc = df_daily[[t1, t2]].dropna()
    if len(pc) < 100:
        return False, None
    try:
        res = coint_johansen(pc, det_order=0, k_ar_diff=1)
        trace = float(res.lr1[0])
        crit = float(res.cvt[0, _JOH_CRIT_IDX[crit_level]])
        if trace <= crit:
            return False, None
        evec = res.evec[:, 0]
        beta = -evec[1] / evec[0]
        return True, float(beta)
    except Exception:
        return False, None

def compute_half_life(spread_daily: pd.Series) -> float:
    aligned = pd.concat([spread_daily.diff(), spread_daily.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    try:
        theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
        return -np.log(2) / theta if theta < 0 else 200.0
    except Exception:
        return 200.0


def _precompute_hurst_series(spread_daily: pd.Series,
                              window: int = HURST_ENTRY_WINDOW,
                              hurst_lag: int = 10) -> pd.Series:
    """Rolling Hurst on daily spread → daily Series of H values."""
    if spread_daily is None or len(spread_daily) < window + hurst_lag:
        return pd.Series(dtype=float)
    hf = HurstFilter(window=window, hurst_lag=hurst_lag)
    return hf.compute_hurst(spread_daily).dropna()


def _precompute_adaptive_series(df: pd.DataFrame,
                                 w_base: int,
                                 entry_z_base: float,
                                 stop_z_base: float,
                                 alpha: float = ADAPT_ALPHA,
                                 gamma: float = ADAPT_GAMMA,
                                 eta:   float = ADAPT_ETA,
                                 delta: float = ADAPT_DELTA,
                                 sigma_fast_bars: int = ADAPT_SIGMA_FAST_BARS,
                                 sigma_slow_bars: int = ADAPT_SIGMA_SLOW_BARS,
                                 cusum_k: float = ADAPT_CUSUM_K,
                                 cusum_h: float = ADAPT_CUSUM_H,
                                 b_max:   float = ADAPT_B_MAX,
                                 w_min:   int = ADAPT_W_MIN,
                                 w_max:   int = ADAPT_W_MAX,
                                 entry_z_cap:  float = ADAPT_ENTRY_Z_CAP,
                                 stop_z_floor: float = ADAPT_STOP_Z_FLOOR,
                                 epsilon: float = ADAPT_EPSILON):
    """
    Precompute the six per-bar adaptive arrays (W_t, entry_z_t, stop_z_t,
    μ_t, σ_t, z_t) for a signal frame produced by `build_signals`. Inputs
    Bt and Ct are derived from the BASELINE z-score (df['zscore']):

      Ct = rolling_cusum(z; k, h) — same kernel used by MetaGate
      Bt = 0.5·log1p(z²/var_z) + 0.5·(Ct/h_cusum)  — fixed-β proxy

    Returns a tuple of numpy arrays aligned to df.index.
    """
    from metagate import rolling_cusum, simple_break_score
    z_arr = df["zscore"].to_numpy(dtype=np.float64)
    spread_arr = df["spread"].to_numpy(dtype=np.float64)
    c_arr = rolling_cusum(z_arr, k=cusum_k, h=cusum_h)
    b_arr = simple_break_score(z_arr, c_arr, h_cusum=cusum_h)

    # σ_fast / σ_slow on spread DIFFS (matches OnlineParameterAdapter).
    diffs = pd.Series(spread_arr).diff().fillna(0.0)
    sigma_fast = diffs.rolling(sigma_fast_bars,
                               min_periods=sigma_fast_bars).std().bfill().to_numpy(dtype=np.float64)
    sigma_slow = diffs.rolling(sigma_slow_bars,
                               min_periods=sigma_fast_bars).std().bfill().to_numpy(dtype=np.float64)

    W_arr, ez_arr, sz_arr, mu_arr, sig_arr, z_t_arr = compute_adaptive_arrays(
        spread_arr, b_arr, c_arr, sigma_fast, sigma_slow,
        w_base=w_base, entry_z_base=entry_z_base, stop_z_base=stop_z_base,
        alpha=alpha, gamma=gamma, eta=eta, delta=delta,
        w_min=w_min, w_max=w_max, b_max=b_max, cusum_h=cusum_h,
        entry_z_cap=entry_z_cap, stop_z_floor=stop_z_floor, epsilon=epsilon,
    )
    return {
        "W_t":       W_arr,
        "entry_z_t": ez_arr,
        "stop_z_t":  sz_arr,
        "mu_t":      mu_arr,
        "sigma_t":   sig_arr,
        "z_t":       z_t_arr,
        "B_t":       b_arr,
        "C_t":       c_arr,
    }


def _precompute_adf_pvalues(spread_daily: pd.Series,
                            lookback: int = 90,
                            step: int = 5) -> pd.Series:
    """Rolling ADF p-value on the daily spread, sampled every `step` days."""
    if spread_daily is None or len(spread_daily) < lookback:
        return pd.Series(dtype=float)
    from statsmodels.tsa.stattools import adfuller
    s = spread_daily.dropna()
    out: dict = {}
    for i in range(lookback, len(s) + 1, step):
        chunk = s.iloc[i - lookback:i]
        try:
            _, pval, _, _, _, _ = adfuller(chunk.values, maxlag=1,
                                           regression="c", autolag=None)
        except Exception:
            pval = 0.5
        out[s.index[i - 1]] = float(pval)
    return pd.Series(out).sort_index()


def backtest_oos(df, t1, t2, beta, entry_z, exit_z, stop_z,
                 hmm_regime=None, hurst_filter=None, spread_daily=None,
                 limit_rebate=0.05, limit_ttl=3,
                 score_frame: pd.DataFrame | None = None,
                 metagate_model: MetaGateModel | None = None,
                 training_mode: bool = False,
                 diag_sink: list | None = None,
                 adaptive: bool = False,
                 adaptive_arrays: dict | None = None,
                 adaptive_diag_sink: list | None = None,
                 post_trade: bool = False,
                 shadow_buffer: ShadowBuffer | None = None,
                 perf_tracker:  PairPerformanceTracker | None = None,
                 feedback_tracker: PerformanceFeedbackTracker | None = None,
                 pair_name: str | None = None,
                 max_hold_bars: int | None = None):
    """
    Tier-1 Backtest with MetaGate hooks.

    Modes:
      - training_mode=True: disable all risk gates (HMM/Hurst/legacy).
        Every |z| ≥ entry_z trigger becomes a virtual trade. Used to
        build the unfiltered training pool for MetaGate (no selection
        bias).

      - metagate_model usable + training_mode=False: gate entries by
        P(Win) ≥ θ and scale position size by (P - θ)/(1 - θ). Legacy
        AND-gate is bypassed because the meta-model embeds those signals.

      - metagate_model fallback / None + training_mode=False: legacy
        AND-gate behaviour (HMM block, Hurst block, soft Hurst sizing).
    """
    # Shift daily HMM regime by one day so the trading loop sees only T-1
    # information (no look-ahead).  Idempotent shift not used — caller passes
    # the raw daily series.
    hmm_regime = shift_daily_to_t1(hmm_regime)

    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = entry_sma = entry_std = 0.0
    ebar = 0
    trades: list = []
    hmm_blocked = 0
    hurst_blocked = 0
    metagate_blocked = 0
    adaptive_blocked = 0
    pair_blocked = False

    pending_pos = 0
    pending_ttl = 0
    limit_z = 0.0
    pending_h_mult = 1.0
    pending_meta_mult = 1.0
    pending_p_win = 0.5
    pending_features = None
    pending_max_hold = max_hold_bars
    current_h_mult = 1.0
    current_meta_mult = 1.0
    current_p_win = 0.5
    current_s_perf = 1.0
    current_max_hold = max_hold_bars
    pending_s_perf = 1.0
    e_features = None

    meta_active = (metagate_model is not None and metagate_model.usable
                   and not training_mode)
    bypass_legacy_gates = training_mode or meta_active

    # ── Post-Trade Learning state ───────────────────────────────────────
    # When `post_trade=True`, the ShadowBuffer drives entry decisions in
    # place of the frozen `metagate_model` (it is itself a continuously-
    # retrained logistic regression). The PairPerformanceTracker scales
    # executed position sizes by the rolling-Sharpe S_perf factor.
    pt_active = bool(post_trade) and not training_mode
    if pt_active:
        if shadow_buffer is None:
            shadow_buffer = ShadowBuffer()
        if perf_tracker is None:
            perf_tracker = PairPerformanceTracker()
    # Parallel "what-if" positions for shadow simulation. Each entry:
    #   {bar, direction, spread, sma, std, features}
    shadow_positions: list[dict] = []

    has_scores = score_frame is not None and len(score_frame) == len(df)

    # ── Adaptive arrays (Online Parameter Adaptation) ───────────────────
    # Compute on-the-fly if requested but not provided, so the caller can
    # opt in with a single bool. When `adaptive=False` AND
    # `adaptive_arrays is None` the bar loop uses the legacy scalar
    # entry_z / stop_z and the baseline df['zscore'] — guaranteeing
    # bit-for-bit identical output to the pre-adaptation code path.
    adaptive_on = bool(adaptive) or (adaptive_arrays is not None)
    if adaptive_on and adaptive_arrays is None:
        # The half-life-derived rolling window used by build_signals().
        _w_base_default = max(20, min(int(max(20, beta)), 200)) if False else 60
        adaptive_arrays = _precompute_adaptive_series(
            df,
            w_base=int(df.attrs.get("w_base", _w_base_default)),
            entry_z_base=float(entry_z),
            stop_z_base=float(stop_z),
        )
    if adaptive_on:
        adapt_entry_z = adaptive_arrays["entry_z_t"]
        adapt_stop_z  = adaptive_arrays["stop_z_t"]
        adapt_z       = adaptive_arrays["z_t"]
        adapt_W       = adaptive_arrays["W_t"]
        adapt_B       = adaptive_arrays["B_t"]
        adapt_C       = adaptive_arrays["C_t"]
    else:
        adapt_entry_z = adapt_stop_z = adapt_z = None
        adapt_W = adapt_B = adapt_C = None

    for i in range(len(df)):
        if pair_blocked:
            continue
        # Baseline z-score remains the execution signal.  The feedback
        # controller adapts the knobs around it; it does not replace the signal
        # with a different EWMA z-score.
        z = df["zscore"].iloc[i]
        s = df["spread"].iloc[i]
        try:
            p1 = df[t1c].iloc[i]
            p2 = df[t2c].iloc[i]
        except KeyError:
            print(f"KeyError in backtest_oos! cols={df.columns.tolist()}")
            raise
        vr = df["vr"].iloc[i]
        ts = df.index[i]

        if has_scores:
            regime_state = RegimeState.from_score_row(ts, score_frame.iloc[i])
        else:
            hmm_val = 0
            if hmm_regime is not None:
                d = ts.normalize().tz_localize(None)
                try:
                    val = hmm_regime.asof(d)
                    hmm_val = 0 if pd.isna(val) else int(val)
                except Exception:
                    hmm_val = 0
            break_val = float(adapt_B[i]) if adaptive_on else 0.0
            cusum_val = float(adapt_C[i]) if adaptive_on else 0.0
            regime_state = RegimeState(
                timestamp=ts,
                hmm_regime=hmm_val,
                kmeans_regime=1,
                break_score=break_val,
                cusum_val=cusum_val,
                hurst_val=0.5,
                adf_p_value=0.0,
                rcdp_score=0.5,
                vol_ratio=1.0,
                time_since_break=0,
            )

        if adaptive_on:
            knobs = compute_effective_trade_knobs(
                regime_state,
                entry_z_base=entry_z,
                stop_z_base=stop_z,
                max_hold_base=max_hold_bars if max_hold_bars is not None else _SHADOW_MAX_HOLD_BARS,
            )
            cur_entry_z = knobs.entry_z_eff
            cur_stop_z = knobs.stop_z_eff
            cur_max_hold = knobs.max_hold_eff
        else:
            cur_entry_z = entry_z
            cur_stop_z = stop_z
            cur_max_hold = max_hold_bars

        # Section 7.4: Single-Bullet Entry Guard tightens the stop in
        # high structural-break-risk regimes.  apply_sbr_guard() honours the
        # config kill-switch and never loosens stop below the adaptive value.
        cur_stop_z = apply_sbr_guard(
            regime_state,
            entry_z_base=entry_z,
            current_stop_z=cur_stop_z,
            threshold=SBR_GUARD_THRESHOLD,
            stop_mult=SBR_GUARD_STOP_MULT,
            enabled=SBR_GUARD_ENABLED,
        )

        z_active = z
        if pos != 0 and entry_std > 0:
            z_active = (s - entry_sma) / entry_std

        # ── Shadow trade exits (run in parallel with the real strategy) ──
        # Each shadow position is checked for exit/stop/time-stop using the
        # SAME exit_z / stop_z thresholds as the real strategy at entry-time.
        if pt_active and shadow_positions:
            still_open = []
            for sp in shadow_positions:
                sp_z_active = z
                if sp["std"] > 0:
                    sp_z_active = (s - sp["sma"]) / sp["std"]
                d = sp["direction"]
                bars_held = i - sp["bar"]
                sp_exit = (d == 1 and sp_z_active >= exit_z) or (d == -1 and sp_z_active <= -exit_z)
                sp_stop = (d == 1 and sp_z_active <= -sp["stop_z"]) or (d == -1 and sp_z_active >= sp["stop_z"])
                sp_time = bars_held >= int(sp.get("max_hold", _SHADOW_MAX_HOLD_BARS))
                if sp_exit or sp_stop or sp_time:
                    sp_gross = d * (s - sp["spread"])
                    sp_notl  = sp["t1"] + beta * sp["t2"]
                    sp_tx    = sp_notl * COST_MAKER + sp_notl * (COST_MAKER if sp_exit else COST_TAKER)
                    sp_hd    = bars_held / BARS_PER_DAY
                    sp_brw   = (beta * sp["t2"] if d == 1 else sp["t1"]) * BORROW_RATE_ANNUAL * sp_hd / 252
                    sp_net   = sp_gross - sp_tx - sp_brw
                    shadow_buffer.push(sp["features"], sp_net)
                else:
                    still_open.append(sp)
            shadow_positions = still_open

        # ── Exit Logic ────────────────────────────────────────────────────
        if pos != 0:
            ex = (pos == 1 and z_active >= exit_z) or (pos == -1 and z_active <= -exit_z)
            st = (pos == 1 and z_active <= -cur_stop_z) or (pos == -1 and z_active >= cur_stop_z)
            tm = current_max_hold is not None and (i - ebar) >= int(current_max_hold)
            # Count cases where the ADAPTIVE stop fires but the BASELINE
            # stop would not have — i.e., adaptive tightening caused an
            # early exit. Informational counter only.
            if adaptive_on and st and not ex and cur_stop_z < stop_z:
                static_st = (pos == 1 and z_active <= -stop_z) or (pos == -1 and z_active >= stop_z)
                if not static_st:
                    adaptive_blocked += 1
            cb = abs(z_active) >= CIRCUIT_BREAKER_Z
            if ex or st or tm or cb:
                if cb:
                    pair_blocked = True
                    st = True
                gross = pos * (s - es)
                notl = et1 + abs(beta) * et2
                tx = notl * COST_MAKER + notl * (COST_MAKER if ex and not cb else COST_TAKER)
                hd = (i - ebar) / BARS_PER_DAY
                brw = (beta * et2 if pos == 1 else et1) * BORROW_RATE_ANNUAL * hd / 252
                # Apply the multiplier that was active at entry (either
                # soft Hurst or MetaGate probabilistic sizing).
                size_mult = current_meta_mult if (meta_active or pt_active) else current_h_mult
                final_pnl = (gross - tx - brw) * size_mult
                trades.append({
                    "entry_time": df.index[ebar],
                    "exit_time":  df.index[i],
                    "direction":  "LONG" if pos == 1 else "SHORT",
                    "net_pnl":    round(final_pnl, 4),
                    "exit_reason": "STOP" if st else ("TIME_STOP" if tm else "SIGNAL"),
                    "holding_bars": i - ebar,
                    "gross_pnl":  round(gross, 4),
                    "tx_cost":    round(tx, 4),
                    "notional":   round(notl, 4),
                    "features":   e_features,
                    "metagate_p": current_p_win,
                    "size_mult":  size_mult,
                    "s_perf":     current_s_perf,
                })
                # Update rolling performance tracker(s) with this executed trade.
                if pt_active and perf_tracker is not None:
                    perf_tracker.record_trade(final_pnl)
                if feedback_tracker is not None and pair_name and notl > 0:
                    feedback_tracker.add_trade(pair_name, final_pnl, notl)
                pos = 0

        # ── Limit Order Matching ──────────────────────────────────────────
        if pos == 0 and pending_pos != 0:
            pending_ttl -= 1
            is_filled = ((pending_pos == -1 and z <= limit_z)
                         or (pending_pos == 1 and z >= limit_z))
            if is_filled:
                pos = pending_pos
                current_h_mult = pending_h_mult
                current_meta_mult = pending_meta_mult
                current_p_win = pending_p_win
                current_s_perf = pending_s_perf
                current_max_hold = pending_max_hold
                pending_pos = 0
                es = s; et1 = p1; et2 = p2; ebar = i
                entry_sma = df["spread_mean"].iloc[i]
                entry_std = df["spread_std"].iloc[i]
                e_features = pending_features
            elif pending_ttl <= 0:
                pending_pos = 0
                pending_features = None

        # ── Entry Logic ───────────────────────────────────────────────────
        if pos == 0 and pending_pos == 0 and vr >= 0.5:
            detect_long = z < -cur_entry_z
            detect_short = z > cur_entry_z
            if not (detect_long or detect_short):
                continue
            legacy_panic_blocked = (not bypass_legacy_gates) and regime_state.is_panic()

            # Compute live Hurst (used for legacy soft sizing AND for
            # diagnostic recording even when MetaGate is active).
            h_val = 0.5
            legacy_hurst_blocked = False
            if hurst_filter is not None and spread_daily is not None:
                d_prev = (df.index[i] - pd.Timedelta(days=1)).normalize()
                h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                h_tail = pd.concat([h_tail_daily,
                                    pd.Series({df.index[i]: df["spread"].iloc[i]})])
                h_blocked, h_val = hurst_filter.should_block(h_tail, df.index[i])
                if h_blocked and not bypass_legacy_gates:
                    legacy_hurst_blocked = True
                if not has_scores:
                    regime_state = RegimeState(
                        timestamp=ts,
                        hmm_regime=regime_state.hmm_regime,
                        kmeans_regime=regime_state.kmeans_regime,
                        break_score=regime_state.break_score,
                        cusum_val=regime_state.cusum_val,
                        hurst_val=h_val,
                        adf_p_value=regime_state.adf_p_value,
                        rcdp_score=regime_state.rcdp_score,
                        vol_ratio=regime_state.vol_ratio,
                        time_since_break=regime_state.time_since_break,
                    )

            # Build the MetaGate vector through the unified RegimeState API.
            feat = regime_state.to_features(abs(z), vr)

            # ── Shadow trade: open in parallel REGARDLESS of MetaGate ──
            # The shadow position will be simulated to its natural exit and
            # pushed into the ShadowBuffer to drive online retraining.
            if pt_active:
                shadow_positions.append({
                    "bar":       i,
                    "direction": 1 if detect_long else -1,
                    "spread":    s,
                    "t1":        p1,
                    "t2":        p2,
                    "sma":       df["spread_mean"].iloc[i],
                    "std":       df["spread_std"].iloc[i],
                    "features":  feat,
                    "stop_z":    cur_stop_z,
                    "max_hold":  cur_max_hold if cur_max_hold is not None else _SHADOW_MAX_HOLD_BARS,
                })

            # Real legacy blocks happen AFTER shadow capture so the shadow
            # buffer observes every raw entry-threshold crossing and does not
            # inherit the production gate's selection bias.
            if legacy_panic_blocked:
                hmm_blocked += 1
                continue
            if legacy_hurst_blocked:
                hurst_blocked += 1
                continue

            # MetaGate / Post-Trade gating.
            p_win = 0.5
            meta_size = 1.0
            decision_kind = "metagate"

            if pt_active and shadow_buffer is not None and shadow_buffer.classifier is not None:
                # Post-trade-learned classifier supersedes the frozen WFO model
                # once the shadow buffer has warmed up past `min_buffer`.
                p_win = shadow_buffer.predict_proba(feat)
                meta_size = shadow_buffer.size_multiplier(p_win)
                decision_kind = "shadow"
                if diag_sink is not None:
                    diag_sink.append({
                        "ts":       df.index[i],
                        **{name: feat[k] for k, name in enumerate(FEATURE_NAMES)},
                        "p_win":    p_win,
                        "size_mult": meta_size,
                        "decision": "ENTER" if meta_size > 0 else "BLOCK",
                        "kind":     f"shadow_theta={shadow_buffer.theta:.3f}",
                    })
                if meta_size <= 0:
                    metagate_blocked += 1
                    continue
            elif meta_active:
                p_win = metagate_model.predict_proba(feat)
                meta_size = metagate_model.size_multiplier(p_win)
                if diag_sink is not None:
                    diag_sink.append({
                        "ts":       df.index[i],
                        **{name: feat[k] for k, name in enumerate(FEATURE_NAMES)},
                        "p_win":    p_win,
                        "size_mult": meta_size,
                        "decision": "ENTER" if meta_size > 0 else "BLOCK",
                        "kind":     metagate_model.kind,
                    })
                if meta_size <= 0:
                    metagate_blocked += 1
                    continue

            # ── Pair-level performance feedback ─────────────────────────
            # Down-scale persistently underperforming pairs (rolling SR over
            # the last N executed trades). Two independent loops:
            #   • old post-trade tracker (annualised SR, S_perf ∈ [0.1, 1.0])
            #   • new RoN-based feedback tracker (S_perf ∈ [0.2, 1.2])
            s_perf = 1.0
            if pt_active and perf_tracker is not None:
                s_perf *= perf_tracker.s_perf
            if feedback_tracker is not None and pair_name:
                s_perf *= feedback_tracker.get_multiplier(pair_name)

            pending_pos = 1 if detect_long else -1
            pending_ttl = limit_ttl
            limit_z = (-cur_entry_z + limit_rebate) if detect_long else (cur_entry_z - limit_rebate)
            pending_h_mult = get_hurst_multiplier(h_val)
            pending_meta_mult = meta_size * s_perf
            pending_s_perf = s_perf
            pending_p_win = p_win
            pending_features = feat
            pending_max_hold = cur_max_hold

            # Append a row to the adaptive diagnostics sink if requested.
            if adaptive_on and adaptive_diag_sink is not None:
                adaptive_diag_sink.append({
                    "ts":              df.index[i],
                    "raw_spread":      float(s),
                    "break_score":     float(adapt_B[i]),
                    "cusum":           float(adapt_C[i]),
                    "adaptive_window": int(adapt_W[i]),
                    "adaptive_entry_z": cur_entry_z,
                    "adaptive_stop_z":  cur_stop_z,
                    "adaptive_max_hold": cur_max_hold,
                    "realized_zscore": float(z),
                })

    return trades, hmm_blocked, hurst_blocked, metagate_blocked, adaptive_blocked


def regularize_covariance(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Apply Ledoit-Wolf analytical shrinkage to the sample covariance matrix.

    The empirical covariance matrix on N assets with T observations becomes
    increasingly ill-conditioned as N approaches T.  Ledoit-Wolf shrinks the
    sample matrix toward a structured target (scaled identity) using an
    optimal shrinkage intensity computed in closed form from the data, which
    drives the smallest eigenvalues away from zero without sacrificing the
    dominant principal components.

    Falls back to the empirical covariance if scikit-learn is unavailable,
    if the data has zero rows, or if numerical issues defeat the estimator.
    """
    if returns_df is None or returns_df.empty or returns_df.shape[1] == 0:
        return np.zeros((0, 0))
    arr = np.asarray(returns_df.values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return np.cov(arr.T) if arr.size else np.zeros((arr.shape[1], arr.shape[1]))
    try:
        from sklearn.covariance import ledoit_wolf
        cov, _ = ledoit_wolf(arr)
        return np.asarray(cov, dtype=np.float64)
    except Exception:
        return np.asarray(returns_df.cov().values, dtype=np.float64)


def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """
    Recover the leaf ordering from a hierarchical-clustering linkage matrix.

    Implements Lopez de Prado, "Building Diversified Portfolios that Outperform
    Out of Sample" (JPM 2016), Algorithm 1.  The result is a permutation of
    leaf indices such that adjacent items in the ordering are close in the
    clustering tree, which produces the "quasi-diagonalised" covariance.
    """
    link = link.astype(int)
    n_items = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_items:
        new_order: list[int] = []
        for v in order:
            if v < n_items:
                new_order.append(v)
            else:
                row = link[v - n_items]
                new_order.extend([int(row[0]), int(row[1])])
        order = new_order
    return order


def _hrp_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance portfolio variance of a sub-cluster."""
    if not items:
        return 0.0
    sub = cov[np.ix_(items, items)]
    diag = np.diag(sub)
    diag = np.where(diag > 0, diag, 1e-12)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _hrp_recursive_bisect(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """
    Lopez de Prado HRP recursive bisection with inverse-variance allocation.

    Returns a weight vector indexed in the *original asset order* (not the
    quasi-diagonal order), summing to 1.0 before any subsequent cap is
    applied.  Operates entirely on integer indices to avoid pandas-label
    fragility.
    """
    n = cov.shape[0]
    w = np.ones(n, dtype=np.float64)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            v_left = _hrp_cluster_var(cov, left)
            v_right = _hrp_cluster_var(cov, right)
            total = v_left + v_right
            if total <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - v_left / total
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(n, 1.0 / n)
    return w


def _hrp_weights(returns_df: pd.DataFrame, max_weight: float) -> dict[str, float]:
    """
    Hierarchical Risk Parity with no matrix inversion.

    Pipeline:
      1. Pearson correlation distance d_ij = sqrt(0.5 * (1 - rho_ij))
      2. Single-linkage hierarchical clustering
      3. Quasi-diagonalisation
      4. Recursive bisection with inverse-variance splits

    Returns weights summing to 1.0.  When ``max_weight`` is finite, the
    function caps each weight and redistributes the excess proportionally
    over the uncapped names so the sum stays at 1.0.

    Falls back to equal weights when scipy is unavailable or the input has
    fewer than three usable assets.
    """
    if returns_df is None or returns_df.empty or returns_df.shape[1] == 0:
        return {}
    cols = list(returns_df.columns)
    if len(cols) == 1:
        return {cols[0]: 1.0}
    if len(cols) < 3:
        w = 1.0 / len(cols)
        return {c: float(w) for c in cols}
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except Exception:
        w = 1.0 / len(cols)
        return {c: float(w) for c in cols}

    arr = np.asarray(returns_df.values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    corr = np.corrcoef(arr.T)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    try:
        cond = squareform(dist, checks=False)
        link = linkage(cond, method="single")
    except Exception:
        w = 1.0 / len(cols)
        return {c: float(w) for c in cols}

    order = _hrp_quasi_diag(link)
    cov = regularize_covariance(returns_df)
    weights = _hrp_recursive_bisect(cov, order)

    if max_weight is not None and 0.0 < max_weight < 1.0 and len(cols) > 1:
        # Iteratively cap weights and redistribute excess to uncapped names
        # so the portfolio remains fully invested at sum == 1.0.
        for _ in range(20):
            excess_mask = weights > max_weight
            if not excess_mask.any():
                break
            excess = float(weights[excess_mask].sum() - excess_mask.sum() * max_weight)
            weights[excess_mask] = max_weight
            free_mask = ~excess_mask & (weights > 0)
            if not free_mask.any():
                break
            free_sum = float(weights[free_mask].sum())
            if free_sum <= 0:
                break
            weights[free_mask] += excess * (weights[free_mask] / free_sum)

    s = float(weights.sum())
    if s > 0:
        weights = weights / s
    return {c: float(w) for c, w in zip(cols, weights)}


def optimize_portfolio_weights(returns_df: pd.DataFrame,
                               target_return: float = 0.10,
                               max_weight: float = 0.20,
                               method: str | None = None) -> dict:
    """
    Dispatch portfolio weight optimization on ``method`` (defaults to
    ``config.ALLOCATION_METHOD``).

    Supported branches:

    - ``"mvo"`` / ``"markowitz"`` — Mean-Variance Optimization with a target
      return constraint.  When ``config.COVARIANCE_SHRINKAGE`` is True the
      covariance matrix is replaced with the Ledoit-Wolf analytical-shrinkage
      estimator, which stabilises the optimization for universes where the
      number of assets approaches the number of training observations.

    - ``"hrp"`` — Hierarchical Risk Parity.  No matrix inversion, so it is
      robust on collinear, large universes where MVO can produce wildly
      concentrated solutions.  Weights sum to 1.0 and respect ``max_weight``
      via an iterative cap-and-redistribute step.

    Any other / unknown method falls back to the historical MVO behaviour
    without shrinkage, preserving backward compatibility with legacy callers.

    returns_df: rows = dates, columns = pairs, values = daily PnL or returns.
    """
    if returns_df is None or returns_df.empty or returns_df.shape[1] == 0:
        return {}
    if returns_df.shape[1] == 1:
        return {returns_df.columns[0]: 1.0}

    if method is None:
        method = str(globals().get("ALLOCATION_METHOD",
                                   "markowitz")).strip().lower()
    method = (method or "markowitz").strip().lower()

    if method == "hrp":
        return _hrp_weights(returns_df, max_weight=float(max_weight))

    use_shrinkage = bool(globals().get("COVARIANCE_SHRINKAGE", True))
    mu = returns_df.mean().values
    if use_shrinkage:
        Sigma = regularize_covariance(returns_df)
    else:
        Sigma = returns_df.cov().values
    n = len(mu)
    target_daily = target_return / 252.0

    def portfolio_variance(w, S):
        return np.dot(w.T, np.dot(S, w))

    init_w = np.full(n, 1.0 / n)
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: np.sum(mu * w) - target_daily}
    ]

    res = opt.minimize(portfolio_variance, init_w, args=(Sigma,), method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        # Fallback to Min-Variance if target unreachable
        res = opt.minimize(portfolio_variance, init_w, args=(Sigma,), method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}])

    weights = res.x
    weights[weights < 1e-4] = 0.0
    weights /= np.sum(weights)
    return {col: float(w) for col, w in zip(returns_df.columns, weights)}

def calculate_systemic_risk_scaler(returns_df: pd.DataFrame, threshold: float = 0.7) -> float:
    """
    Module 4: Tail Risk Guard (Eigenvalue Spiking).
    Scales down position size when systemic correlation spikes.
    """
    if returns_df.shape[1] < 2:
        return 1.0
    try:
        # Use correlation matrix for spectral analysis
        corr_matrix = returns_df.corr().fillna(0)
        eigenvalues = np.linalg.eigvals(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Absorption Ratio = Top Eigenvalue / Total Trace
        absorption_ratio = eigenvalues[0] / np.sum(eigenvalues)
        if absorption_ratio > threshold:
            # Linear scale-down to 0.2
            scaler = max(0.2, 1.0 - (absorption_ratio - threshold) / (1.0 - threshold))
            return float(scaler)
        return 1.0
    except Exception:
        return 1.0

def _trades_to_daily_pnl(trades: list, dates: pd.DatetimeIndex) -> pd.Series:
    pnl = pd.Series(0.0, index=dates.normalize().unique())
    for t in trades:
        d = pd.Timestamp(t["exit_time"]).normalize()
        if d in pnl.index:
            pnl[d] += t["net_pnl"]
    return pnl




def load_closes():
    path = DATA_DIR / CLOSES_FILE
    closes = pd.read_csv(path, index_col=0)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    vol_path = DATA_DIR / VOLUMES_FILE
    volumes = None
    if vol_path.exists():
        volumes = pd.read_csv(vol_path, index_col=0)
        volumes.index = pd.to_datetime(volumes.index, utc=True).tz_convert("US/Eastern")
        volumes = volumes.reindex(closes.index).fillna(0.0)

    daily_path = DATA_DIR / "closes_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0)
        daily.index = pd.to_datetime(daily.index, utc=True).tz_convert("US/Eastern")
    else:
        daily = None

    return closes, daily, volumes


def make_windows(closes, expanding=WFO_EXPANDING):
    """
    Generate (train_start, train_end, test_start, test_end) tuples.

    expanding=True  (Expanding Window):
      train_start is ANCHORED to the first date in closes.
      Each subsequent window grows: 12m, 18m, 24m, …
      The algorithm accumulates memory of all past regimes.

    expanding=False (Rolling Window, classic Gatev):
      train_start slides forward by WFO_STEP_MONTHS each iteration.
      Fixed-width train window (12m).
    """
    anchor = closes.index[0].date()
    end    = closes.index[-1].date()
    windows = []
    cur = anchor
    while True:
        train_start = anchor if expanding else cur
        train_end   = cur + relativedelta(months=WFO_TRAIN_MONTHS)
        test_start  = train_end
        test_end    = test_start + relativedelta(months=WFO_TEST_MONTHS)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur = cur + relativedelta(months=WFO_STEP_MONTHS)
    return windows


# entry point


def load_global_hmm() -> pd.Series | None:
    """Load global_hmm_regime.csv (date → 0/1). Returns None if missing."""
    path = DATA_DIR / "global_hmm_regime.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename("global_hmm")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for WFO (YYYY-MM-DD). Needs 12m before first OOS.")
    parser.add_argument("--end",   type=str, default=None,
                        help="End date for WFO (YYYY-MM-DD).")
    parser.add_argument("--hl-max", type=int, default=HL_MAX_DAYS,
                        help=f"Max half-life in calendar days (default {HL_MAX_DAYS}).")
    parser.add_argument("--rolling", action="store_true",
                        help="Force rolling window mode (override config WFO_EXPANDING).")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Disable global Macro-HMM filter.")
    parser.add_argument("--no-hurst", action="store_true",
                        help="Disable Hurst exponent entry filter.")
    parser.add_argument("--pairs", type=str, default="pairs_selected.csv",
                        help="CSV file in data/ directory to read pairs from.")
    args = parser.parse_args()

    use_expanding = (not args.rolling) and WFO_EXPANDING

    closes, daily, volumes = load_closes()
    pairs_file = DATA_DIR / args.pairs
    if not pairs_file.exists():
        raise SystemExit(f"Pairs file {pairs_file} not found.")
    pairs  = pd.read_csv(pairs_file)

    if pairs.empty:
        raise SystemExit("pairs_selected.csv is empty — run pairs.py first")
    if daily is None:
        raise SystemExit("closes_daily.csv is required for WFO cointegration testing.")

    # load data
    _z_profiles: dict[str, dict] = {}
    _zp_path = DATA_DIR / "z_profiles.csv"
    if _zp_path.exists():
        _zp_df = pd.read_csv(_zp_path)
        for _, _r in _zp_df[_zp_df["tradeable"] == True].iterrows():
            _z_profiles[_r["pair"]] = {
                "entry_z": float(_r["entry_z"]),
                "exit_z":  float(_r["exit_z"]),
                "stop_z":  float(_r["stop_z"]),
                "ev":      float(_r["ev"]),
            }
        print(f"Z-Bounce profiles loaded for {len(_z_profiles)} pairs "
              f"(EV-optimal, R:R=1.3)")
    else:
        print("No z_profiles.csv — using grid search (run z_profiler.py for faster WFO)")

    # load data
    if args.no_hmm:
        global_hmm = None
        print("Macro-HMM filter: DISABLED (--no-hmm flag)")
    else:
        global_hmm = load_global_hmm()
        if global_hmm is not None:
            n_panic = int(global_hmm.sum())
            pct     = n_panic / len(global_hmm) * 100
            print(f"Macro-HMM loaded: {len(global_hmm)} days, "
                  f"panic={n_panic} ({pct:.0f}%) — entries BLOCKED during panic")
        else:
            print("Macro-HMM filter: global_hmm_regime.csv not found — DISABLED")

    # optional date filter
    if args.start:
        start_ts = pd.Timestamp(args.start).tz_localize("US/Eastern")
        closes = closes[closes.index >= start_ts]
        daily  = daily[daily.index  >= start_ts]
        print(f"Date filter applied: start={args.start}")
    if args.end:
        end_ts = pd.Timestamp(args.end).tz_localize("US/Eastern")
        closes = closes[closes.index <= end_ts]
        daily  = daily[daily.index  <= end_ts]
        print(f"Date filter applied: end={args.end}")

    hl_max_bars = args.hl_max * BARS_PER_DAY  # convert days → intraday bars

    windows = make_windows(closes, expanding=use_expanding)
    n_wins  = len(windows)

    window_mode = "EXPANDING" if use_expanding else "ROLLING"
    print("=" * 80)
    print(f"WALK-FORWARD OPTIMIZATION  ({window_mode} Window)")
    print("=" * 80)
    print(f"Initial train:  {WFO_TRAIN_MONTHS}m  |  "
          f"OOS:  {WFO_TEST_MONTHS}m  |  Step: {WFO_STEP_MONTHS}m")
    if use_expanding:
        first_train = (windows[0][1] - windows[0][0]).days // 30 if windows else 0
        last_train  = (windows[-1][1] - windows[-1][0]).days // 30 if windows else 0
        print(f"Train window: {first_train}m → {last_train}m (expanding)")
    print(f"Total windows: {n_wins}")
    print(f"Full span: {closes.index[0].date()} → {closes.index[-1].date()}")
    print(f"Pairs: {len(pairs)}")
    print(f"Half-life gate: ≤ {args.hl_max} calendar days")
    print(f"Macro-HMM: {'ACTIVE' if global_hmm is not None else 'OFF'}")
    use_hurst = not args.no_hurst
    from config import HURST_ENTRY_MAX
    hurst_label = f"ACTIVE (H>{HURST_ENTRY_MAX})" if use_hurst else "OFF"
    print(f"Hurst gate: {hurst_label}\n")

    all_oos_trades = []   # accumulate across all windows
    pair_spreads_for_corr: dict[str, pd.Series] = {}  # for Section 7.3
    wfo_params     = []   # best params log per pair per window
    total_hmm_blocked   = 0 # count of entries blocked by Macro-HMM
    total_hurst_blocked = 0 # count of entries blocked by Hurst drift guard
    total_meta_blocked  = 0 # count of entries blocked by MetaGate (P < theta)
    total_adapt_blocked = 0 # count of trades exited early by adaptive stop_z tightening

    # Global pair-level performance feedback tracker (single instance shared
    # across all pairs and windows so its FIFOs keep accumulating). Opt-in
    # via the PFB_ENABLED flag in config.py.
    feedback_tracker_global: PerformanceFeedbackTracker | None = (
        PerformanceFeedbackTracker() if PFB_ENABLED else None
    )

    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        tr_s_ts = pd.Timestamp(tr_s).tz_localize("US/Eastern")
        tr_e_ts = pd.Timestamp(tr_e).tz_localize("US/Eastern")
        te_s_ts = pd.Timestamp(te_s).tz_localize("US/Eastern")
        te_e_ts = pd.Timestamp(te_e).tz_localize("US/Eastern")

        closes_train = closes[(closes.index >= tr_s_ts) & (closes.index < tr_e_ts)]
        closes_test  = closes[(closes.index >= te_s_ts) & (closes.index < te_e_ts)]
        warmup_pos = max(0, closes.index.searchsorted(te_s_ts) - WFO_SIGNAL_WARMUP_BARS)
        warmup_start_ts = closes.index[warmup_pos]
        closes_test_with_warmup = closes[(closes.index >= warmup_start_ts) & (closes.index < te_e_ts)]

        # Johansen uses ROLLING window (last 3 years of train) even in expanding
        # mode — long histories mask structural breaks like COST-WMT.
        _joh_lookback = relativedelta(years=3)
        joh_start = max(tr_s, (tr_e - _joh_lookback))
        joh_start_ts = pd.Timestamp(joh_start).tz_localize("US/Eastern")
        daily_train = daily[(daily.index >= joh_start_ts) & (daily.index < tr_e_ts)]

        if len(closes_train) < 500 or len(closes_test) < 100 or len(daily_train) < 100:
            continue

        days_train = (tr_e - tr_s).days
        days_test  = (te_e - te_s).days

        train_months = round(days_train / 30.44)
        print(f"  Window {w_idx+1:02d}/{n_wins}  "
              f"TRAIN {tr_s} → {tr_e} ({train_months}m)  |  "
              f"OOS {te_s} → {te_e}", end="")

        window_oos_pnl    = 0.0
        window_trades     = 0
        window_coint      = 0
        window_hmm_blocks = 0
        window_hurst_blocks = 0
        window_meta_blocks = 0
        window_adapt_blocks = 0
        window_best = {}
        optimal_weights = {}
        window_best       = {}
        optimal_weights   = {}

        for _, row in pairs.iterrows():
            pair_name = row["pair"]
            t1, t2    = pair_name.split("-")

            if t1 not in closes_train.columns or t2 not in closes_train.columns:
                continue
            if t1 not in daily_train.columns or t2 not in daily_train.columns:
                continue

            # 1. Dynamic Cointegration Test (Johansen on daily train slice)
            is_coint, dynamic_beta = check_coint_johansen(daily_train, t1, t2, crit_level=0.95)
            if not is_coint or dynamic_beta < 0 or not (0.1 <= dynamic_beta <= 15.0):
                continue

            # 2. Dynamic Half-life
            spread_daily = daily_train[t1] - dynamic_beta * daily_train[t2]
            dynamic_hl   = compute_half_life(spread_daily) * BARS_PER_DAY

            # skip if mean reversion is too slow for the window
            if dynamic_hl > hl_max_bars:
                continue  # mean-reversion too slow for the OOS window
            
            window_coint += 1

            # 3. Build signals with window-specific beta and hl
            sig_train = build_signals(closes_train, volumes, t1, t2, dynamic_beta, dynamic_hl)
            sig_test  = trim_oos_signal_warmup(
                build_signals(closes_test_with_warmup, volumes, t1, t2, dynamic_beta, dynamic_hl),
                te_s_ts,
                te_e_ts,
            )

            if len(sig_train) < 100 or len(sig_test) < 20:
                continue

            # 4. Get entry params — prefer EV-optimal profile, fall back to grid
            if pair_name in _z_profiles:
                zp = _z_profiles[pair_name]
                best = {
                    "entry_z": zp["entry_z"],
                    "exit_z":  zp["exit_z"],
                    "stop_z":  zp["stop_z"],
                    "sharpe":  zp["ev"],    # EV stands in for sharpe in the log
                    "trades":  0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                }
                source = "EV-profile"
            else:
                best = run_grid(sig_train, t1, t2, dynamic_beta, COMBOS, days_train)
                source = "grid"
            if best is None:
                continue

            # ── Train-quality gate (skip losing pairs) ────────────
            # Only applied to grid results; EV-profile entries are
            # pre-curated by the upstream profiler and may carry a
            # placeholder total_pnl == 0 that would otherwise gate them.
            if source == "grid" and not pair_train_quality_ok(
                best,
                min_sharpe=WFO_MIN_TRAIN_SHARPE,
                min_pnl=WFO_MIN_TRAIN_PNL,
            ):
                print(
                    f"  {pair_name:<10} TRAIN: SKIP — Sharpe={best.get('sharpe')} "
                    f"PnL={best.get('total_pnl')} below floors "
                    f"(min_sharpe={WFO_MIN_TRAIN_SHARPE}, "
                    f"min_pnl={WFO_MIN_TRAIN_PNL})"
                )
                continue

            # ── KDE Structural Filter (Quality Check) ─────────────
            from filters import validate_kde_density
            is_kde_valid = validate_kde_density(sig_train["zscore"], best["entry_z"], threshold_ratio=0.5)
            if not is_kde_valid:
                print(f"  {pair_name:<10} TRAIN: REJECTED by KDE (Low Density Node at Z={best['entry_z']})")
                continue

            # 5. Trade OOS with the best params found on TRAIN
            pair_hurst_f = HurstFilter() if use_hurst else None
            spread_daily_full = (daily[t1] - dynamic_beta * daily[t2]).dropna() if use_hurst else None

            # ── MetaGate: build precomputed score frames ─────────────
            hurst_daily = _precompute_hurst_series(spread_daily_full) \
                if spread_daily_full is not None else pd.Series(dtype=float)
            adf_pvals = _precompute_adf_pvalues(spread_daily_full) \
                if spread_daily_full is not None else pd.Series(dtype=float)

            scores_train = build_score_frame(
                sig_train, hurst_daily, adf_pvals, global_hmm,
            )

            # ── Unfiltered training pass (no risk gates) ─────────────
            train_trades, _, _, _, _ = backtest_oos(
                sig_train, t1, t2, dynamic_beta,
                best["entry_z"], best["exit_z"], best["stop_z"],
                hmm_regime=global_hmm,
                hurst_filter=pair_hurst_f,
                spread_daily=spread_daily_full,
                score_frame=scores_train,
                training_mode=True,
                max_hold_bars=int(max(dynamic_hl * 2, 1)),
            )

            # ── Fit per-pair MetaGate on the unfiltered pool ─────────
            feats = [t["features"] for t in train_trades if t.get("features")]
            labels = [1 if t["net_pnl"] > 0 else 0
                      for t in train_trades if t.get("features")]
            pnls = [float(t["net_pnl"]) for t in train_trades if t.get("features")]
            if feats:
                X = np.asarray(feats, dtype=np.float64)
                y = np.asarray(labels, dtype=np.int64)
                pnl_arr = np.asarray(pnls, dtype=np.float64)
                metagate_model = fit_metagate(X, y, pnl_values=pnl_arr, calibrate=True)
            else:
                metagate_model = fit_metagate(
                    np.zeros((0, len(FEATURE_NAMES))),
                    np.zeros(0, dtype=np.int64),
                    pnl_values=np.zeros(0, dtype=np.float64),
                )

            best["metagate_model"]   = metagate_model
            best["metagate_kind"]    = metagate_model.kind
            best["metagate_n_train"] = metagate_model.n_train
            best["theta_calibrated"] = metagate_model.theta_entry
            best["theta_utility"]    = metagate_model.calibration_utility
            best["theta_n_oof"]      = metagate_model.calibration_n_oof
            # Backward-compat alias so legacy callers still find a model.
            best["ml_model"]         = metagate_model.clf

            if METAGATE_REQUIRE_USABLE and (
                not metagate_model.usable or metagate_model.calibration_n_oof <= 0
            ):
                print(
                    f"  {pair_name:<10} TRAIN: REJECTED by MetaGate "
                    f"(kind={metagate_model.kind}, "
                    f"n_train={metagate_model.n_train}, "
                    f"n_oof={metagate_model.calibration_n_oof})"
                )
                continue

            # Save for MVO calculation
            best["dynamic_beta"] = dynamic_beta
            best["dynamic_hl"] = dynamic_hl
            best["source"] = source
            best["train_returns"] = _trades_to_daily_pnl(train_trades, daily_train.index)
            window_best[pair_name] = best
            
        # 6. Portfolio Optimization (MVO)
        active_pairs = list(window_best.keys())
        if active_pairs:
            train_returns_df = pd.DataFrame({p: window_best[p]["train_returns"] for p in active_pairs}).fillna(0)
            optimal_weights = optimize_portfolio_weights(train_returns_df, target_return=0.10)
            
            # --- Module 4: Tail Risk Guard ---
            sys_scaler = calculate_systemic_risk_scaler(train_returns_df, threshold=0.7)
            if sys_scaler < 1.0:
                print(f"  [TAIL RISK] Absorption Ratio spike ({sys_scaler:.2f}). Scaling risk.")
                for p in optimal_weights:
                    optimal_weights[p] *= sys_scaler
        else:
            optimal_weights = {}

        # 7. OOS Execution (Second Pass with Weights)
        for pair_name in active_pairs:
            t1, t2 = pair_name.split("-")
            best = window_best[pair_name]
            weight = optimal_weights.get(pair_name, 0.0)
            if weight <= 0: continue

            # Re-build sig_test (or use cached if memory allowed, but for simplicity we rebuild)
            sig_test = trim_oos_signal_warmup(
                build_signals(closes_test_with_warmup, volumes, t1, t2, best["dynamic_beta"], best["dynamic_hl"]),
                te_s_ts,
                te_e_ts,
            )
            
            pair_hurst_f = HurstFilter() if use_hurst else None
            spread_daily_full = (daily[t1] - best["dynamic_beta"] * daily[t2]).dropna() if use_hurst else None
            # Section 7.3: stash the OOS-window spread for the post-hoc
            # correlation throttle.  Latest write wins, so the final stored
            # series reflects the most-recent dynamic beta.
            if spread_daily_full is not None and not spread_daily_full.empty:
                pair_spreads_for_corr[pair_name] = spread_daily_full

            # Score frame for OOS bars (uses the SAME hurst/adf precomputes
            # as training so live-vs-train feature distributions match).
            hurst_daily_oos = _precompute_hurst_series(spread_daily_full) \
                if spread_daily_full is not None else pd.Series(dtype=float)
            adf_pvals_oos = _precompute_adf_pvalues(spread_daily_full) \
                if spread_daily_full is not None else pd.Series(dtype=float)
            scores_test = build_score_frame(
                sig_test, hurst_daily_oos, adf_pvals_oos, global_hmm,
            )

            mg_model = best.get("metagate_model")
            diag_sink: list = []
            adapt_diag_sink: list = []

            # Decide whether to enable online parameter adaptation for this
            # OOS pair-window. ADAPT_ENABLED is the global default; callers
            # can override per-window via best["adaptive"].
            use_adaptive = bool(best.get("adaptive", ADAPT_ENABLED))
            use_post_trade = bool(best.get("post_trade", POSTTRADE_ENABLED))

            # Reuse the shadow buffer + tracker across windows when present
            # in `best` (lets them accumulate state from prior OOS slices).
            pair_shadow = None
            pair_perf   = None
            if use_post_trade:
                pair_shadow = best.get("shadow_buffer") or ShadowBuffer()
                pair_perf   = best.get("perf_tracker")  or PairPerformanceTracker()
                best["shadow_buffer"] = pair_shadow
                best["perf_tracker"]  = pair_perf

            oos_trades, pair_hmm_blocked, pair_hurst_blocked, pair_meta_blocked, pair_adapt_blocked = backtest_oos(
                sig_test, t1, t2, best["dynamic_beta"],
                best["entry_z"], best["exit_z"], best["stop_z"],
                hmm_regime=global_hmm,
                hurst_filter=pair_hurst_f,
                spread_daily=spread_daily_full,
                score_frame=scores_test,
                metagate_model=mg_model,
                training_mode=False,
                diag_sink=diag_sink,
                adaptive=use_adaptive,
                adaptive_diag_sink=adapt_diag_sink if use_adaptive else None,
                post_trade=use_post_trade,
                shadow_buffer=pair_shadow,
                perf_tracker=pair_perf,
                feedback_tracker=feedback_tracker_global,
                pair_name=pair_name,
                max_hold_bars=int(max(best["dynamic_hl"] * 2, 1)),
            )

            # (pair_shadow / pair_perf are mutated in-place by backtest_oos
            #  since `best` already holds the same references.)
            window_hmm_blocks += pair_hmm_blocked
            window_hurst_blocks += pair_hurst_blocked
            window_meta_blocks += pair_meta_blocked
            window_adapt_blocks += pair_adapt_blocked

            # Emit diagnostics (features, P, decision) for this pair-window.
            if diag_sink:
                for r in diag_sink:
                    r["pair"] = pair_name
                    r["window"] = w_idx + 1
                append_diagnostics(DATA_DIR / METAGATE_DIAG_CSV, diag_sink)

            # Emit adaptive-parameter trace for this pair-window.
            if adapt_diag_sink:
                for r in adapt_diag_sink:
                    r["pair"] = pair_name
                    r["window"] = w_idx + 1
                append_diagnostics(DATA_DIR / ADAPT_DIAG_CSV, adapt_diag_sink)

            if len(oos_trades) < WFO_MIN_TRADES:
                continue

            # ── Position-size scaling (MVO weight only — MetaGate already
            # applied probabilistic sizing inside backtest_oos). ─────────
            for t in oos_trades:
                t["net_pnl"]   *= weight
                t["gross_pnl"] *= weight
                t["tx_cost"]   *= weight

            oos_trades = [t for t in oos_trades if t["net_pnl"] != 0 or t["tx_cost"] != 0]
            if len(oos_trades) == 0:
                continue

            oos_pnl = sum(t["net_pnl"] for t in oos_trades)

            for t in oos_trades:
                t["pair"]     = pair_name
                t["window"]   = w_idx + 1
                t["entry_z"]  = best["entry_z"]
                t["exit_z"]   = best["exit_z"]
                t["stop_z"]   = best["stop_z"]
                t["source"]   = best["source"]
            all_oos_trades.extend(oos_trades)

            oos_arr = np.array([t["net_pnl"] for t in oos_trades])
            oos_sh  = (oos_arr.mean() / oos_arr.std() *
                       np.sqrt(len(oos_arr) / max(days_test / 365.25, 0.01))
                       ) if oos_arr.std() > 0 else 0.0

            wfo_params.append({
                "window":      w_idx + 1,
                "train_start": str(tr_s), "train_end": str(tr_e),
                "oos_start":   str(te_s), "oos_end":   str(te_e),
                "pair":        pair_name,
                "beta":        round(best["dynamic_beta"], 4),
                "half_life":   round(best["dynamic_hl"], 1),
                "entry_z":     best["entry_z"],
                "exit_z":      best["exit_z"],
                "stop_z":      best["stop_z"],
                "theta_calibrated": best.get("theta_calibrated"),
                "theta_utility": best.get("theta_utility"),
                "theta_n_oof": best.get("theta_n_oof"),
                "metagate_kind": best.get("metagate_kind"),
                "metagate_n_train": best.get("metagate_n_train"),
                "train_sharpe": best["sharpe"],
                "oos_sharpe":  round(oos_sh, 3),
                "oos_trades":  len(oos_trades),
                "oos_pnl":     round(oos_pnl, 4),
            })

            window_oos_pnl += oos_pnl
            window_trades  += len(oos_trades)

        total_hmm_blocked += window_hmm_blocks
        total_hurst_blocked += window_hurst_blocks
        total_meta_blocked += window_meta_blocks
        total_adapt_blocked += window_adapt_blocks
        hmm_info   = f"  HMM blocked={window_hmm_blocks}" if global_hmm is not None else ""
        hurst_info = f"  Hurst blocked={window_hurst_blocks}" if use_hurst else ""
        meta_info  = f"  MetaGate blocked={window_meta_blocks}" if window_meta_blocks else ""
        adapt_info = f"  AdaptStop={window_adapt_blocks}" if window_adapt_blocks else ""
        print(f"  →  {window_coint:2d} pairs passed Johansen  |  "
              f"{window_trades:3d} OOS trades   P&L={window_oos_pnl:+.4f}"
              f"{hmm_info}{hurst_info}{meta_info}{adapt_info}")

        if len(optimal_weights) > 0:
            import joblib
            try:
                live_state = {
                    "window_best": window_best,
                    "optimal_weights": optimal_weights,
                    "performance_feedback": (
                        feedback_tracker_global.to_dict()
                        if feedback_tracker_global is not None else None
                    ),
                }
                OUTPUT_DIR.mkdir(exist_ok=True)
                joblib.dump(live_state, OUTPUT_DIR / "live_state.pkl")
            except Exception:
                pass



    if not all_oos_trades:
        print("\nNo OOS trades accumulated. "
              "Check that pairs_selected.csv matches closes data.")
        return

    # save
    df_trades = pd.DataFrame(all_oos_trades)
    df_params = pd.DataFrame(wfo_params)

    # ── Section 7.3: Active Ticker Correlation Throttling ────────────────────
    if str(CORR_BLOCK_MODE).strip().upper() != "DISABLED" and not df_trades.empty:
        spread_corr = compute_spread_correlations(
            pair_spreads_for_corr,
            window_days=int(CORR_BLOCK_WINDOW_DAYS),
        )
        overlap_blocker = TickerOverlapBlocker(
            spread_corr,
            threshold=float(CORR_BLOCK_THRESHOLD),
            mode=str(CORR_BLOCK_MODE),
            scale=float(CORR_BLOCK_SCALE),
        )
        df_trades, overlap_stats = filter_trade_list(df_trades, overlap_blocker)
        if overlap_stats.n_dropped or overlap_stats.n_scaled:
            print(
                f"\n[Section 7.3] TickerOverlap mode={overlap_blocker.mode} "
                f"threshold={overlap_blocker.threshold:.2f}: "
                f"kept={overlap_stats.n_kept} dropped={overlap_stats.n_dropped} "
                f"scaled={overlap_stats.n_scaled}"
            )
            # Persist the blocking events for audit/diagnostics.
            try:
                if overlap_stats.blocking_pairs:
                    pd.DataFrame(
                        overlap_stats.blocking_pairs,
                        columns=["candidate_pair", "blocking_pair", "rho"],
                    ).to_csv(DATA_DIR / "wfo_correlation_blocks.csv", index=False)
            except Exception:
                pass

    df_trades.to_csv(DATA_DIR / "wfo_results.csv", index=False)
    df_params.to_csv(DATA_DIR / "wfo_params.csv",  index=False)

    # summary
    pnl       = df_trades["net_pnl"]
    win_rate  = (pnl > 0).mean() * 100
    total_pnl = pnl.sum()
    n_trades  = len(pnl)
    wins      = pnl[pnl > 0].sum()
    losses    = abs(pnl[pnl <= 0].sum())
    pf        = wins / losses if losses > 0 else float("inf")

    # Annualised Sharpe across entire OOS chain
    span_years = (pd.Timestamp(windows[-1][3]) -
                  pd.Timestamp(windows[0][2])).days / 365.25
    tpy  = n_trades / max(span_years, 0.01)
    sh   = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
    cum  = pnl.cumsum()
    dd   = float((cum - cum.cummax()).min())

    print(f"\n{'='*70}")
    print(f"WFO PORTFOLIO SUMMARY  ({n_wins} windows,  "
          f"{len(df_params['pair'].unique())} pairs)")
    print(f"{'='*70}")
    print(f"OOS Trades:      {n_trades}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Total OOS P&L:   {total_pnl:+.4f}  (spread units)")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Max Drawdown:    {dd:.4f}")
    print(f"Sharpe (chain):  {sh:.2f}")
    if global_hmm is not None:
        print(f"HMM blocked:     {total_hmm_blocked} potential entries")
    print(f"MetaGate blocked:{total_meta_blocked} potential entries")
    if total_adapt_blocked:
        print(f"AdaptStop fired: {total_adapt_blocked} early exits (stop_z tightened by CUSUM)")
    if use_hurst:
        print(f"Hurst blocked:   {total_hurst_blocked} potential entries")
    print(f"Window mode:     {window_mode}")
    print(f"\nSaved → data/wfo_results.csv  ({n_trades} trades)")
    print(f"Saved → data/wfo_params.csv   ({len(df_params)} rows)")

    # per-pair WFO stability summary
    print(f"\n{'Pair':<12}  {'Windows':>7}  {'OOS Sh':>7}  {'Train Sh':>9}  "
          f"{'OOS Trades':>10}  {'OOS P&L':>10}")
    print("-" * 65)
    for pair in sorted(df_params["pair"].unique()):
        sub   = df_params[df_params["pair"] == pair]
        n_w   = len(sub)
        avg_oos_sh   = sub["oos_sharpe"].mean()
        avg_train_sh = sub["train_sharpe"].mean()
        tot_tr = sub["oos_trades"].sum()
        tot_pnl = sub["oos_pnl"].sum()
        print(f"{pair:<12}  {n_w:>7}  {avg_oos_sh:>7.2f}  "
              f"{avg_train_sh:>9.2f}  {tot_tr:>10}  {tot_pnl:>+10.4f}")

    # visualization
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: Continuous OOS equity curve
    ax = axes[0]
    df_trades_sorted = df_trades.sort_values("exit_time")
    equity = df_trades_sorted["net_pnl"].cumsum()

    ax.plot(range(len(equity)), equity.values,
            color="steelblue", lw=1.5, label="WFO OOS equity")
    ax.fill_between(range(len(equity)), equity.values, 0,
                    where=(equity.values >= 0),
                    color="steelblue", alpha=0.15)
    ax.fill_between(range(len(equity)), equity.values, 0,
                    where=(equity.values < 0),
                    color="salmon", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    # Mark window boundaries
    window_starts = {}
    for _, r in df_params.iterrows():
        w = r["window"]
        oos_s = r["oos_start"]
        if w not in window_starts:
            # Find trade index nearest to this OOS start
            mask = df_trades_sorted["exit_time"] >= oos_s
            if mask.any():
                idx = mask.idxmax()
                pos = df_trades_sorted.index.get_loc(idx)
                window_starts[w] = pos

    for w, pos in window_starts.items():
        ax.axvline(pos, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.text(pos + 1, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else 0,
                f"W{w}", fontsize=7, color="gray")

    ax.set_title(f"WFO Continuous OOS Equity Curve  |  "
                 f"{n_trades} trades  Sharpe={sh:.2f}  P&L={total_pnl:+.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Net P&L (spread units)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Train vs OOS Sharpe per window (scatter per pair)
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_params["pair"].unique())))
    pair_colors = {p: c for p, c in zip(sorted(df_params["pair"].unique()), colors)}

    for pair in df_params["pair"].unique():
        sub = df_params[df_params["pair"] == pair]
        ax2.scatter(sub["train_sharpe"], sub["oos_sharpe"],
                    color=pair_colors[pair], alpha=0.7, s=50,
                    label=pair, zorder=3)

    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("TRAIN Sharpe")
    ax2.set_ylabel("OOS Sharpe")
    ax2.set_title("WFO Stability: Train vs OOS Sharpe per Window per Pair",
                   fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, ncol=3, loc="upper left")
    ax2.grid(True, alpha=0.2)

    # Add correlation annotation    
    if len(df_params) > 5:
        corr = df_params[["train_sharpe", "oos_sharpe"]].corr().iloc[0, 1]
        ax2.text(0.02, 0.97, f"r(train, OOS) = {corr:.2f}",
                 transform=ax2.transAxes, fontsize=10,
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    equity_path = OUTPUT_DIR / "wfo_equity.png"
    plt.savefig(equity_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {equity_path}")


if __name__ == "__main__":
    main()
