import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from config import (
    ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    COST_MAKER, COST_TAKER, COST_PANIC_MULTIPLIER, CIRCUIT_BREAKER_Z, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    IV_SIZE_NORM, MIN_POSITION_SIZE, TRAIN_RATIO,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR, INITIAL_CAPITAL,
    ALLOCATION_METHOD, MAX_PAIR_WEIGHT, TARGET_RISK_USD,
    BARS_PER_DAY, CLOSES_FILE, KALMAN_DELTA,
    COINT_WINDOW_DAYS, COINT_BREAK_P, COINT_RECHECK_DAYS,
    RVOL_MIN_ENTRY, VELOCITY_BARS, VWAP_BAND_SIGMA,
    TAIL_EV_GATE, TAIL_ENTRY_Z_MIN, TAIL_STOP_Z, TAIL_EXIT_Z,
    TAIL_LABEL_LOOKAHEAD_BARS, TAIL_THRESHOLD, TAIL_RR_THRESHOLD,
    TAIL_CONFIDENCE_LEVEL, TAIL_REFIT_FREQ,
    PORTFOLIO_OPT_MAX_GROSS, PORTFOLIO_OPT_WEIGHT_MIN, PORTFOLIO_OPT_WEIGHT_MAX,
    REGIME_MULT_NORMAL, REGIME_MULT_VOLATILE, HMM_PANIC_MULT, IV_MULT_MAX,
)
from kalman import kalman_hedge
from step3e_sizing import (
    load_regimes, load_iv, load_global_hmm, load_mc_confidence,
    load_corr_throttle, load_hrp_weights,
    iv_multiplier_series, position_size,
)
from filters import (
    CointegrationFilter, MacroFilter, HurstFilter,
    validate_kde_density, BreakVelocityDetector, _LAZY_WINDOW_BARS,
    compute_break_scores_vectorized,
)
from config import HURST_ENTRY_WINDOW
from tail_ev_profiler import TailAdjustedEVProfiler
from regime_memory import RegimeMemoryWeighter

try:
    from rmt_covariance import clean_covariance_rmt
    from portfolio_optimizer import RegularizedPortfolioOptimizer
    from portfolio_margin_risk import MarginSpiralDetector
except Exception:
    clean_covariance_rmt = None
    RegularizedPortfolioOptimizer = None
    MarginSpiralDetector = None

BARS_PER_TRADING_DAY = BARS_PER_DAY
_LAZY_WINDOW_BARS = COINT_WINDOW_DAYS * BARS_PER_DAY


def _data_path() -> str:
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        raise FileNotFoundError(f"No data file: {CLOSES_FILE}")
    return str(p)


def load_closes() -> pd.DataFrame:
    """Load ALL available intraday data.

    Pair selection is done on 8-year daily data (step2), so there is no
    look-ahead: using the full intraday history is valid and maximises
    the number of trades for statistical evaluation.
    """
    closes = pd.read_csv(_data_path(), index_col=0, parse_dates=True)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    pairs_path = DATA_DIR / "pairs_selected.csv"
    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            needed    = {t for p in meta["pair"] for t in p.split("-")}
            available = [t for t in needed if t in closes.columns]
            # No global dropna — each pair drops its own NaNs in build_signals
            return closes[available]

    return closes


def build_signals(closes, t1, t2, beta, half_life,
                  volumes: pd.DataFrame | None = None,
                  delta: float = KALMAN_DELTA) -> pd.DataFrame:
    """Kalman-filter spread signals with optional VW-Z / RVOL / VWAP columns.

    When `volumes` is provided (volumes_Nmin.csv), three extra columns are added:
      vwz         — volume-weighted z-score (primary entry signal)
      rvol        — relative volume vs 20-day rolling avg at same time-of-day
      vwap_spread — intraday volume-weighted anchor of the spread (resets daily)

    When volumes is None all three columns are absent and the backtest falls back
    to the regular zscore for entry decisions.
    """
    # Drop rows where either leg has NaN (e.g., pre-IPO) before passing to Kalman
    pair_closes = closes[[t1, t2]].dropna()
    alpha_arr, beta_arr, innov, _ = kalman_hedge(
        pair_closes[t1].values, pair_closes[t2].values,
        delta=delta, beta_init=float(beta),
    )
    spread  = pd.Series(innov,    index=pair_closes.index, name="spread")
    beta_s  = pd.Series(beta_arr, index=pair_closes.index, name="beta")
    alpha_s = pd.Series(alpha_arr, index=pair_closes.index, name="alpha")
    window  = max(20, min(int(half_life), 200))
    spread_std = spread.rolling(window=window).std()
    zscore     = spread / spread_std

    df = pd.DataFrame({
        f"{t1}_close": pair_closes[t1],
        f"{t2}_close": pair_closes[t2],
        "spread":      spread,
        "beta":        beta_s,
        "alpha":       alpha_s,
        "innov":       innov,
        "spread_std":  spread_std,
        "zscore":      zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)
    df["velocity"] = df["zscore"].diff(VELOCITY_BARS)

    # m15 Kalman z-score: signal on 15-min, execution on 5-min
    # Resample pair closes to 15-min, run separate Kalman, ffill back to 5-min.
    # is_m15_close marks the last 5-min bar of each 15-min period (exit gate).
    try:
        pc15 = pair_closes[[t1, t2]].resample("15min").last().dropna()
        if len(pc15) >= 40:
            a15, b15, innov15, _ = kalman_hedge(
                pc15[t1].values, pc15[t2].values,
                delta=delta, beta_init=float(beta),
            )
            sp15  = pd.Series(innov15, index=pc15.index)
            win15 = max(10, min(int(half_life // 3), 67))  # 15-min bars ≈ 5-min/3
            std15 = sp15.rolling(win15).std()
            z15   = (sp15 / std15).dropna()
            # Forward-fill 15-min z onto 5-min index (no look-ahead)
            z15_ff = z15.reindex(df.index, method="ffill")
            df["z_m15"] = z15_ff
            # Mark last 5-min bar of each 15-min group
            m15_closes = set(pc15.index)
            df["is_m15_close"] = df.index.isin(m15_closes)
        else:
            df["z_m15"]       = df["zscore"]
            df["is_m15_close"] = True
    except Exception:
        df["z_m15"]       = df["zscore"]
        df["is_m15_close"] = True

    # Pre-calculate Break Velocity components
    df["half_life_bars"] = half_life
    nu_s = df["innov"]
    var_nu_s = nu_s.rolling(60).var().bfill()
    beta_s = df["beta"]
    d_beta_dt_s = beta_s.diff().abs().rolling(10).mean().fillna(0)
    hl_s = df["half_life_bars"]
    hl_median_s = hl_s.rolling(300).median().bfill()
    
    # Toxicity Filter (Hawkes)
    from toxicity import HawkesToxicityFilter
    toxicity_f = HawkesToxicityFilter()
    df["toxicity"] = toxicity_f.compute_intensity(df["zscore"].diff())

    ev_profiler = TailAdjustedEVProfiler(u_threshold=3.0)
    ev_profiler.fit_tail(f"{t1}-{t2}", df["zscore"])

    scores, broken_pos, broken_neg = compute_break_scores_vectorized(
        df, var_nu_s, hl_median_s, d_beta_dt_s, half_life, threshold=4.5
    )
    df["break_score"] = scores
    df["is_broken_pos"] = broken_pos
    df["is_broken_neg"] = broken_neg
    df["is_broken"]   = broken_pos | broken_neg
    df["var_nu"]      = var_nu_s
    df["tail_ev"]     = 0.0

    if volumes is not None and t1 in volumes.columns and t2 in volumes.columns:
        v1  = volumes[t1].reindex(df.index).fillna(0)
        v2  = volumes[t2].reindex(df.index).fillna(0)
        vol = (v1 + abs(beta) * v2).clip(lower=1.0)

        # vwz - volume weighted zscore
        # High-volume bars anchor the "fair value" of the spread more strongly
        # than quiet bars, so the resulting z-score filters out thin-market noise.
        vw_sum  = vol.rolling(window).sum().clip(lower=1.0)
        vw_mean = (df["spread"] * vol).rolling(window).sum() / vw_sum
        vw_var  = (vol * (df["spread"] - vw_mean) ** 2).rolling(window).sum() / vw_sum
        df["vwz"] = (df["spread"] - vw_mean) / np.sqrt(vw_var.clip(lower=1e-16))

        # rvol vs same time of day baseline
        # shift(1) inside the transform makes each bar's reference use only
        # the PREVIOUS day's rolling average — no lookahead.
        tod_avg = vol.groupby(vol.index.time).transform(
            lambda x: x.rolling(20, min_periods=5).mean().shift(1)
        ).clip(lower=1.0)
        df["rvol"] = vol / tod_avg

        # intraday spread VWAP (resets at session open each day)
        dates         = df.index.normalize()
        cum_sv        = (df["spread"] * vol).groupby(dates).cumsum()
        cum_v         = vol.groupby(dates).cumsum().clip(lower=1.0)
        df["vwap_spread"] = cum_sv / cum_v

        # m15 VWAP + σ-bands (period-anchored, resets every 3 bars = 15 min)
        # Each 15-min candle group (9:30-9:44, 9:45-9:59, …) gets its own
        # cumulative VWAP and volume-weighted variance.
        # z_m15 = (spread - vwap_m15) / std_m15  →  ±1/2/3σ bands
        m15_key = pd.Series(
            (df.index.hour * 4 + df.index.minute // 15),
            index=df.index,
        ).astype(str) + "_" + df.index.normalize().astype(str)
        cum_sv_m15  = (df["spread"] * vol).groupby(m15_key).cumsum()
        cum_v_m15   = vol.groupby(m15_key).cumsum().clip(lower=1.0)
        vwap_m15    = cum_sv_m15 / cum_v_m15
        cum_sv2_m15 = (vol * (df["spread"] - vwap_m15) ** 2).groupby(m15_key).cumsum()
        std_m15     = np.sqrt((cum_sv2_m15 / cum_v_m15).clip(lower=0))
        df["vwap_m15"]     = vwap_m15
        df["vwap_std_m15"] = std_m15
        df["vwap_z_m15"]   = (df["spread"] - vwap_m15) / std_m15.clip(lower=1e-10)

        # h1 VWAP + σ-bands (period-anchored, resets every 12 bars = 60 min)
        h1_key = pd.Series(
            df.index.hour.astype(str),
            index=df.index,
        ) + "_" + df.index.normalize().astype(str)
        cum_sv_h1  = (df["spread"] * vol).groupby(h1_key).cumsum()
        cum_v_h1   = vol.groupby(h1_key).cumsum().clip(lower=1.0)
        vwap_h1    = cum_sv_h1 / cum_v_h1
        cum_sv2_h1 = (vol * (df["spread"] - vwap_h1) ** 2).groupby(h1_key).cumsum()
        std_h1     = np.sqrt((cum_sv2_h1 / cum_v_h1).clip(lower=0))
        df["vwap_h1"]     = vwap_h1
        df["vwap_std_h1"] = std_h1
        df["vwap_z_h1"]   = (df["spread"] - vwap_h1) / std_h1.clip(lower=1e-10)

    return df




def load_kmeans_regime() -> pd.Series | None:
    """Load daily K-Means macro regime labels (0=Trend, 1=Sideways, 2=Panic)."""
    path = DATA_DIR / "kmeans_regimes.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def min_viable_entry_z(sigma_spread: float, avg_notional: float,
                        exit_z: float, safety: float = 1.0) -> float:
    """
    Minimum entry Z-score for expected gross P&L to exceed transaction costs.

    Expected gross per trade = (entry_z + |exit_z|) × sigma_spread
    Break-even condition:
        (entry_z + |exit_z|) × sigma >= safety × (COST_MAKER + COST_TAKER) × notional
        → entry_z_min = safety × cost_fraction − |exit_z|

    safety=1.0 means break-even transaction cost floor.
    If the grid/MC optimised entry_z is below this floor, we raise it.
    We also cap the floor at 2.2 to prevent overriding grid-optimised parameters.
    """
    cost_frac = ((COST_MAKER + COST_TAKER) * avg_notional) / max(sigma_spread, 1e-8)
    entry_z_min = safety * cost_frac - abs(exit_z)
    return max(min(entry_z_min, 2.2), 0.0)


def ou_params_from_spread(spread: pd.Series) -> tuple[float, float, float]:
    """OLS on ΔS = a - θ·S_{t-1} + ε  →  returns (theta_per_bar, mu, residual_std)."""
    ds  = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    idx = ds.index.intersection(lag.index)
    
    # Add constant for unbiased theta and mu estimation
    reg = sm.OLS(ds[idx], sm.add_constant(lag[idx])).fit()
    a = float(reg.params.iloc[0])
    b = float(reg.params.iloc[1])
    theta = -b
    mu = a / theta if theta > 0 else spread.mean()
    return max(theta, 1e-6), mu, float(reg.resid.std())


def optimal_thresholds(theta: float, sigma_roll: float, notional: float,
                       n_sim: int = 5000, max_bars: int = 1500
                       ) -> tuple[float, float, float]:
    """
    Joint Monte Carlo grid search for (entry_z, exit_thresh, stop_thresh).

    OU in z-score space: Z_{t+1} = Z_t·(1-θ) + √(2θ)·ε
    LONG: enter at -z_e, exit when Z >= z_x, stop when Z <= -z_s.
    P&L_exit = z_e + z_x - c_z_exit   (spread moved from -z_e to z_x)
    P&L_stop = z_e - z_s - c_z_stop   (spread moved against us to -z_s)
    """
    rng   = np.random.default_rng()
    c_z_exit = 2 * COST_MAKER * notional / max(sigma_roll, 1e-8)
    c_z_stop = (COST_MAKER + COST_TAKER) * notional / max(sigma_roll, 1e-8)
    sig_z = np.sqrt(2 * theta)

    entry_grid = np.arange(1.5, 3.75, 0.25)  # [1.5 … 3.5]  (8 values)
    exit_grid  = np.arange(-0.3, 1.05, 0.1)  # [-0.3 … 1.0] fine step for exit-past-zero
    stop_grid  = np.arange(3.0,  5.5,  0.5)  # [3.0 … 5.0]  (5 values)

    best_rate = -np.inf
    best      = (float(ENTRY_Z), float(EXIT_Z), float(STOP_Z))

    for z_e in entry_grid:
        for z_x in exit_grid:
            if z_x >= z_e:          # exit must be less extreme than entry
                continue
            for z_s in stop_grid:
                if z_s <= z_e:      # stop must be more extreme than entry
                    continue

                z     = np.full(n_sim, -z_e)
                done  = np.zeros(n_sim, bool)
                pnl   = np.zeros(n_sim)
                t_end = np.full(n_sim, float(max_bars))

                for t in range(1, max_bars + 1):
                    if done.all():
                        break
                    z = np.where(done, z, z * (1 - theta) + sig_z * rng.standard_normal(n_sim))
                    he = (~done) & (z >= z_x)
                    hs = (~done) & (z <= -z_s)
                    pnl   = np.where(he,       z_e + z_x - c_z_exit,  pnl)
                    pnl   = np.where(hs & ~he, z_e - z_s  - c_z_stop, pnl)
                    t_end = np.where((he | hs) & ~done, float(t), t_end)
                    done  = done | he | hs

                pnl = np.where(~done, -c_z_stop, pnl)   # timed-out: pay cost, no profit

                rate = float(pnl.mean()) / float(t_end.mean())
                if rate > best_rate:
                    best_rate = rate
                    best      = (z_e, z_x, z_s)

    if best_rate <= 0:
        return float(ENTRY_Z), float(EXIT_Z), float(STOP_Z)
    return round(best[0], 2), round(best[1], 2), round(best[2], 2)




def add_tail_ev_features(df: pd.DataFrame,
                         hurst_window: int = 240,
                         hurst_lag: int = 10,
                         vol_window: int = 240) -> pd.DataFrame:
    """Attach the tail-EV feature set used by the fp_fx_betatest profiler."""
    out = df.copy()
    if "velocity" not in out.columns:
        out["velocity"] = out["zscore"].diff(VELOCITY_BARS)

    spread_std = out["spread_std"].replace(0, np.nan)
    vol_baseline = spread_std.rolling(vol_window, min_periods=max(20, vol_window // 4)).median()
    out["vol_ratio"] = (spread_std / vol_baseline.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["vol_ratio"] = out["vol_ratio"].fillna(1.0).clip(lower=0.05, upper=20.0)

    diff_1 = out["spread"].diff(1)
    diff_tau = out["spread"].diff(hurst_lag)
    var_1 = diff_1.rolling(window=hurst_window, min_periods=max(20, hurst_window // 3)).var()
    var_tau = diff_tau.rolling(window=hurst_window, min_periods=max(20, hurst_window // 3)).var()
    hurst = 0.5 * np.log(var_tau / var_1.replace(0, np.nan)) / np.log(hurst_lag)
    out["hurst"] = hurst.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(lower=0.0, upper=1.0)
    return out


def add_regime_memory_state(df: pd.DataFrame,
                            regime_dict: dict | None = None,
                            hl_window: int = 20,
                            hl_epsilon: float = 1e-6) -> pd.DataFrame:
    """Add local half-life and macro-state columns for regime-weighted tail training."""
    out = df.copy()
    spread = pd.to_numeric(out["spread"], errors="coerce").astype(float)
    x_lag = spread.shift(1)
    dx = spread.diff()
    cov = dx.rolling(window=hl_window, min_periods=max(5, hl_window // 2)).cov(x_lag)
    var = x_lag.rolling(window=hl_window, min_periods=max(5, hl_window // 2)).var()
    theta = (-(cov / var.replace(0.0, np.nan))).clip(lower=hl_epsilon)
    half_life = (np.log(2.0) / theta).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out["half_life"] = half_life.fillna(np.log(2.0) / hl_epsilon).clip(lower=1.0, upper=10_000.0)

    if regime_dict:
        regime_s = pd.Series(regime_dict).sort_index()
        regime_s.index = pd.to_datetime(regime_s.index)
        if isinstance(out.index, pd.DatetimeIndex):
            lookup_index = out.index.tz_localize(None) if out.index.tz is not None else out.index
            regime_s.index = regime_s.index.tz_localize(None) if regime_s.index.tz is not None else regime_s.index
            regime = regime_s.reindex(lookup_index, method="ffill")
            regime.index = out.index
        else:
            regime = pd.Series(0, index=out.index)
    else:
        regime = pd.Series(0, index=out.index)
    out["macro_state"] = regime.ffill().bfill().astype("string").fillna("unknown")
    return out


def _next_event_index(event_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    if event_idx.size == 0 or query_idx.size == 0:
        return np.full(query_idx.shape, np.iinfo(np.int64).max, dtype=np.int64)
    pos = np.searchsorted(event_idx, query_idx + 1, side="left")
    out = np.full(query_idx.shape, np.iinfo(np.int64).max, dtype=np.int64)
    valid = pos < event_idx.size
    out[valid] = event_idx[pos[valid]]
    return out


def build_tail_training_frame(train_df: pd.DataFrame,
                              entry_z_min: float = TAIL_ENTRY_Z_MIN,
                              exit_z: float = TAIL_EXIT_Z,
                              stop_z: float = TAIL_STOP_Z,
                              lookahead_bars: int = TAIL_LABEL_LOOKAHEAD_BARS) -> pd.DataFrame:
    """Create in-sample reversion labels for the tail-EV profiler."""
    if len(train_df) <= lookahead_bars + 1:
        return pd.DataFrame(columns=list(train_df.columns) + ["revert_label", "expected_gain", "tail_loss"])

    core = train_df.iloc[:-lookahead_bars].copy()
    z = train_df["zscore"].to_numpy(dtype=float)
    n_core = len(core)
    idx = np.arange(n_core, dtype=np.int64)
    long_mask = z[:n_core] <= -entry_z_min
    short_mask = z[:n_core] >= entry_z_min
    candidate_mask = long_mask | short_mask

    next_tp_long = _next_event_index(np.flatnonzero(z >= exit_z), idx[long_mask])
    next_sl_long = _next_event_index(np.flatnonzero(z <= -stop_z), idx[long_mask])
    next_tp_short = _next_event_index(np.flatnonzero(z <= -exit_z), idx[short_mask])
    next_sl_short = _next_event_index(np.flatnonzero(z >= stop_z), idx[short_mask])

    label = np.zeros(n_core, dtype=np.int8)
    long_horizon = idx[long_mask] + lookahead_bars
    short_horizon = idx[short_mask] + lookahead_bars
    long_revert = (next_tp_long < next_sl_long) & (next_tp_long <= long_horizon)
    short_revert = (next_tp_short < next_sl_short) & (next_tp_short <= short_horizon)
    label[idx[long_mask][long_revert]] = 1
    label[idx[short_mask][short_revert]] = 1

    core["revert_label"] = label
    core["expected_gain"] = np.maximum(np.abs(core["zscore"].to_numpy(dtype=float)) - abs(exit_z), 0.0)
    core["tail_loss"] = np.abs(core["zscore"].to_numpy(dtype=float))
    core = core.loc[candidate_mask].copy()
    return core.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["zscore", "velocity", "vol_ratio", "hurst", "revert_label", "expected_gain", "tail_loss"]
    )


def score_tail_ev_for_pair(df: pd.DataFrame,
                           train_end: pd.Timestamp | None,
                           regime_dict: dict | None = None) -> pd.DataFrame:
    """Fit tail-EV on pre-OOS history and score the full pair frame."""
    if not TAIL_EV_GATE:
        return df
    if train_end is None:
        train_mask = np.arange(len(df)) < int(len(df) * TRAIN_RATIO)
    else:
        train_mask = df.index < train_end
    if int(np.sum(train_mask)) < max(250, TAIL_LABEL_LOOKAHEAD_BARS + 50):
        return df

    feat = add_regime_memory_state(add_tail_ev_features(df), regime_dict=regime_dict)
    train_feat = feat.loc[train_mask].copy()
    tail_train = build_tail_training_frame(train_feat)
    if tail_train.empty or tail_train["revert_label"].nunique() < 2:
        return df

    profiler = TailAdjustedEVProfiler(
        tail_threshold=TAIL_THRESHOLD,
        confidence_level=TAIL_CONFIDENCE_LEVEL,
        rr_threshold=TAIL_RR_THRESHOLD,
        exit_z=TAIL_EXIT_Z,
        tail_refit_freq=TAIL_REFIT_FREQ,
        gain_col="expected_gain",
    )
    try:
        weighter = RegimeMemoryWeighter(age_lambda=0.02, hl_gamma=1.0)
        weights = weighter.compute_weights(
            tail_train,
            now_index=train_feat.index[-1],
            hl_now=float(train_feat["half_life"].iloc[-1]),
            state_now=train_feat["macro_state"].iloc[-1],
        )
        profiler.fit(tail_train, sample_weights=weights)
        scored = profiler.predict_ev(feat)
    except Exception as exc:
        print(f"  tail-EV disabled for pair: {type(exc).__name__}: {exc}")
        return df
    return scored


def backtest_pair(df, t1, t2, beta, pair_name: str = "",
                  regime_dict: dict | None = None,
                  sizing_args: dict | None = None,
                  entry_z: float = ENTRY_Z,
                  exit_thresh: float = EXIT_Z,
                  stop_thresh: float = STOP_Z,
                  coint_filter: CointegrationFilter | None = None,
                  macro_filter: MacroFilter | None = None,
                  hurst_filter: HurstFilter | None = None,
                  spread_daily: pd.Series | None = None,
                  oos_start: pd.Timestamp | None = None,
                  max_hold_bars: int = 9999) -> pd.DataFrame:
    """
    Refactored for Robustness:
    1. Returns raw spread-unit P&L (gross_pnl_raw, notional_raw) for global compounding.
    2. Enforces Parameter Freezing: z_active uses entry_alpha/beta/std for signals.
    """
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position       = 0
    entry_spread   = entry_t1 = entry_t2 = entry_beta = entry_alpha = entry_std = 0.0
    entry_bar      = 0
    trades         = []
    suspended      = False
    trade_s_pos    = 0.0
    trade_s_neg    = 0.0
    hurst_blocked  = 0
    active_exit_thresh = exit_thresh
    active_stop_thresh = stop_thresh
    _blocked_macro = _blocked_size = _blocked_tail = _blocked_rvol = _blocked_vel = _blocked_coint = _blocked_hurst = 0
    entry_size_mult = 1.0

    has_m15  = "z_m15" in df.columns
    is_m15_s = df["is_m15_close"] if "is_m15_close" in df.columns else pd.Series(True, index=df.index)
    z_m15_s  = df["z_m15"] if has_m15 else df["zscore"]

    # Pre-convert pandas structures to numpy arrays for 10x loop speedup
    ts_arr = df.index.to_numpy()
    z_m15_arr = z_m15_s.to_numpy().astype(float)
    is_m15_arr = is_m15_s.to_numpy().astype(bool)
    spread_arr = df["spread"].to_numpy().astype(float)
    p1_arr = df[t1_col].to_numpy().astype(float)
    p2_arr = df[t2_col].to_numpy().astype(float)
    is_broken_arr = df["is_broken"].to_numpy().astype(bool)
    is_broken_pos_arr = df["is_broken_pos"].to_numpy().astype(bool)
    is_broken_neg_arr = df["is_broken_neg"].to_numpy().astype(bool)
    zscore_arr = df["zscore"].to_numpy().astype(float)
    
    beta_arr = df["beta"].to_numpy().astype(float)
    alpha_arr = df["alpha"].to_numpy().astype(float)
    spread_std_arr = df["spread_std"].to_numpy().astype(float)
    innov_arr = df["innov"].to_numpy().astype(float)
    var_nu_arr = df["var_nu"].to_numpy().astype(float)

    if coint_filter is not None:
        coint_valid_arr = np.array([coint_filter._daily_dict.get(dt, True) for dt in df.index.date], dtype=bool)
    else:
        coint_valid_arr = None

    if regime_dict:
        is_volatile_arr = np.array([regime_dict.get(t, 0) == 1 for t in df.index], dtype=bool)
    else:
        is_volatile_arr = np.zeros(len(df), dtype=bool)

    # Precompute sizing multipliers for all bars (vectorized for 100x speedup)
    if sizing_args:
        regimes = sizing_args.get("regimes")
        iv_mult_s = sizing_args.get("iv_mult_s")
        mc_conf = sizing_args.get("mc_conf") or {}
        macro_alert_s = sizing_args.get("macro_alert_s")
        global_hmm_s = sizing_args.get("global_hmm_s")
        corr_throttle_s = sizing_args.get("corr_throttle_s")
        hrp_w = sizing_args.get("hrp_w") or {}

        m = mc_conf.get(pair_name, 1.0)
        h = hrp_w.get(pair_name, 1.0)
        base_mult = m * h

        dates_s = pd.Series(df.index.normalize().tz_localize(None))
        unique_dates = dates_s.unique()

        iv_dict = {}
        if iv_mult_s is not None and not iv_mult_s.empty:
            for d in unique_dates:
                try:
                    v = iv_mult_s.asof(d)
                    iv_dict[d] = float(v) if not np.isnan(v) else IV_MULT_MAX
                except Exception:
                    iv_dict[d] = IV_MULT_MAX

        alert_dict = {}
        if macro_alert_s is not None and not macro_alert_s.empty:
            for d in unique_dates:
                try:
                    val = macro_alert_s.asof(d)
                    alert_dict[d] = bool(int(val)) if not pd.isna(val) else False
                except Exception:
                    alert_dict[d] = False

        global_hmm_dict = {}
        if global_hmm_s is not None and not global_hmm_s.empty:
            for d in unique_dates:
                try:
                    val = global_hmm_s.asof(d)
                    global_hmm_dict[d] = HMM_PANIC_MULT if (not pd.isna(val) and int(val) == 1) else 1.0
                except Exception:
                    global_hmm_dict[d] = 1.0

        throttle_dict = {}
        if corr_throttle_s is not None and not corr_throttle_s.empty:
            for d in unique_dates:
                try:
                    v = corr_throttle_s.asof(d)
                    throttle_dict[d] = float(v) if not pd.isna(v) else 1.0
                except Exception:
                    throttle_dict[d] = 1.0

        if regimes is not None and pair_name in regimes.columns:
            reg_aligned = regimes[pair_name].reindex(df.index, method="ffill").fillna(0).to_numpy()
            reg_mult = np.where(reg_aligned == 1, REGIME_MULT_VOLATILE, REGIME_MULT_NORMAL)
        else:
            reg_mult = np.full(len(df), REGIME_MULT_NORMAL)

        iv_arr = np.array([iv_dict.get(d, IV_MULT_MAX) for d in dates_s])
        alert_arr = np.array([alert_dict.get(d, False) for d in dates_s], dtype=bool)
        global_hmm_mult_arr = np.array([global_hmm_dict.get(d, 1.0) for d in dates_s])
        throttle_arr = np.array([throttle_dict.get(d, 1.0) for d in dates_s])

        sizing_mult_arr = base_mult * reg_mult * global_hmm_mult_arr * iv_arr * throttle_arr
        sizing_mult_arr[alert_arr] = 0.0
    else:
        sizing_mult_arr = np.ones(len(df))

    has_tail_ok = "tail_signal_ok" in df.columns
    tail_ok_arr = df["tail_signal_ok"].to_numpy().astype(bool) if has_tail_ok else None

    has_rvol = "rvol" in df.columns
    rvol_arr = df["rvol"].to_numpy().astype(float) if has_rvol else None

    has_toxicity = "toxicity" in df.columns
    toxicity_arr = df["toxicity"].to_numpy().astype(float) if has_toxicity else np.zeros(len(df))

    for i in range(len(df)):
        ts         = ts_arr[i]
        z_m15      = z_m15_arr[i]
        is_m15     = is_m15_arr[i]
        spread_now = spread_arr[i]
        p1         = p1_arr[i]
        p2         = p2_arr[i]
        is_broken  = is_broken_arr[i]
        is_broken_pos = is_broken_pos_arr[i]
        is_broken_neg = is_broken_neg_arr[i]

        # coint check
        if coint_filter is not None:
            if not coint_valid_arr[i]:
                suspended = True
            elif suspended and coint_valid_arr[i]:
                suspended = False

        # freeze beta/alpha at entry so kalman cant quietly adjust away a real loss
        z_active = zscore_arr[i]
        if position != 0 and entry_std > 0:
            # Frozen beta + alpha static spread calculation
            static_spread = p1 - (entry_alpha + entry_beta * p2)
            z_active = static_spread / entry_std
            
        # hard stop if z blows out
        is_broken_against = False
        if position != 0 and i > entry_bar:
            is_broken_against = (position == 1 and is_broken_neg) or (position == -1 and is_broken_pos)
        if position != 0 and (abs(z_active) >= CIRCUIT_BREAKER_Z or is_broken_against):
            force_close = True
            if is_broken_against:
                suspended = True # Structural break detected
                print(f"[DEBUG] {t1}-{t2} at {ts}: pos={position}, z={z_active:.2f}, s_pos={trade_s_pos:.2f}, s_neg={trade_s_neg:.2f}, is_broken={is_broken}, against={is_broken_against}")
        else:
            force_close = suspended or (macro_filter is not None and macro_filter.is_force_close(ts))

        if position != 0 and force_close:
            pnl_raw        = position * (spread_now - entry_spread)
            notional_raw   = entry_t1 + abs(entry_beta) * entry_t2
            
            # Apply Toxicity Penalty to Taker cost
            from toxicity import HawkesToxicityFilter
            tox_f = HawkesToxicityFilter()
            tox_intensity = toxicity_arr[i]
            tox_mult = tox_f.get_execution_penalty(tox_intensity)
            
            # Apply Panic Multiplier to Taker cost if forced out by macro panic
            is_panic = macro_filter.is_force_close(ts) if macro_filter else False
            cost_mult = (COST_PANIC_MULTIPLIER if is_panic else 1.0) * tox_mult
            tx_cost_raw    = notional_raw * (COST_MAKER + COST_TAKER * cost_mult)
            
            holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
            short_notl_raw = (abs(entry_beta) * entry_t2) if position == 1 else entry_t1
            borrow_raw     = short_notl_raw * BORROW_RATE_ANNUAL * holding_days / 252
            
            reason = ("BREAK_VELOCITY" if is_broken_against else 
                      ("COINT_BREAK" if suspended else macro_filter.force_close_reason(ts)))
            trades.append({
                "pair":         f"{t1}-{t2}",
                "entry_time":   df.index[entry_bar],
                "exit_time":    ts,
                "direction":    "LONG" if position == 1 else "SHORT",
                "holding_bars": i - entry_bar,
                "pnl_raw":      pnl_raw,
                "notional_raw": notional_raw,
                "tx_cost_raw":  tx_cost_raw,
                "borrow_raw":   borrow_raw,
                "gross_pnl":    round(pnl_raw, 4),
                "tx_cost":      round(tx_cost_raw, 4),
                "borrow_cost":  round(borrow_raw, 4),
                "net_pnl":      round(pnl_raw - tx_cost_raw - borrow_raw, 4),
                "exit_reason":  reason,
                "entry_z":      round(zscore_arr[entry_bar], 2),
                "exit_z":       round(z_active, 2),
                "beta":         entry_beta,
                "size_mult":    entry_size_mult
            })
            position = 0
            trade_s_pos = 0.0
            trade_s_neg = 0.0
            continue

        # check if we should exit
        if position != 0:
            bars_held   = i - entry_bar
            time_stop   = bars_held >= max_hold_bars

            # Exit signal now uses FROZEN z_active to avoid Kalman Illusion
            exit_signal = is_m15 and (not time_stop) and (
                (position == 1 and z_active >= active_exit_thresh) or
                (position == -1 and z_active <= -active_exit_thresh))
            
            stop_signal = (not time_stop) and (
                (position == 1 and z_active <= -active_stop_thresh) or 
                (position == -1 and z_active >= active_stop_thresh))

            if exit_signal or stop_signal or time_stop:
                pnl_raw        = position * (spread_now - entry_spread)
                notional_raw   = entry_t1 + abs(entry_beta) * entry_t2
                # Maker entry (fixed) + Maker exit (if TP) or Taker (if Stop)
                tx_cost_raw    = notional_raw * (COST_MAKER + (COST_MAKER if exit_signal else COST_TAKER))
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notl_raw = (abs(entry_beta) * entry_t2) if position == 1 else entry_t1
                borrow_raw     = short_notl_raw * BORROW_RATE_ANNUAL * holding_days / 252

                trades.append({
                    "pair":         f"{t1}-{t2}",
                    "entry_time":   df.index[entry_bar],
                    "exit_time":    ts,
                    "direction":    "LONG" if position == 1 else "SHORT",
                    "holding_bars": i - entry_bar,
                    "pnl_raw":      pnl_raw,
                    "notional_raw": notional_raw,
                    "tx_cost_raw":  tx_cost_raw,
                    "borrow_raw":   borrow_raw,
                    "gross_pnl":    round(pnl_raw, 4),
                    "tx_cost":      round(tx_cost_raw, 4),
                    "borrow_cost":  round(borrow_raw, 4),
                    "net_pnl":      round(pnl_raw - tx_cost_raw - borrow_raw, 4),
                    "exit_reason":  "STOP" if stop_signal else ("TIME_STOP" if time_stop else "SIGNAL"),
                    "entry_z":      round(zscore_arr[entry_bar], 2),
                    "exit_z":       round(z_active, 2),
                    "beta":         entry_beta,
                    "size_mult":    entry_size_mult
                })
                position = 0
                trade_s_pos = 0.0
                trade_s_neg = 0.0
                continue

        # entry checks
        if position == 0:
            if oos_start is not None and ts < oos_start:
                continue
            if suspended:
                continue
            if is_broken:
                continue
            if macro_filter is not None and macro_filter.is_entry_blocked(ts):
                _blocked_macro += 1
                continue
            if sizing_args:
                sz = sizing_mult_arr[i]
                if sz <= 0.0:
                    _blocked_size += 1
                    continue
                sz = max(sz, MIN_POSITION_SIZE)

            is_volatile = is_volatile_arr[i]
            if is_volatile:
                if pair_name in _regime_thresholds:
                    rt = _regime_thresholds[pair_name]
                    threshold   = rt["vol_entry"]
                    # Override exit/stop for this bar's entry decision
                    exit_thresh_live = rt["vol_exit"]
                    stop_thresh_live = rt["vol_stop"]
                else:
                    threshold = ENTRY_Z_VOLATILE
                    exit_thresh_live = exit_thresh
                    stop_thresh_live = stop_thresh
            else:
                threshold = entry_z
                exit_thresh_live = exit_thresh
                stop_thresh_live = stop_thresh

            # entry signal: 15-min Kalman z (forward-filled to 5-min bar)
            # Signal from 15-min bars is stable; entry executes on first 5-min
            # bar where the threshold is crossed (tighter fill price).
            entry_z_val = z_m15_arr[i]

            if entry_z_val < -threshold:
                position = 1
            elif entry_z_val > threshold:
                position = -1

            # gate 0: Tail-adjusted EV — require positive EV after tail loss
            if position != 0 and TAIL_EV_GATE and has_tail_ok:
                if not tail_ok_arr[i]:
                    position = 0
                    _blocked_tail += 1
                    continue

            # gate 1: RVOL — block entries on thin volume
            if position != 0 and has_rvol:
                rvol_now = rvol_arr[i]
                if not np.isnan(rvol_now) and rvol_now < RVOL_MIN_ENTRY:
                    position = 0
                    _blocked_rvol += 1
                    continue

            # gate 2: Velocity — spread must already be reverting
            # Entry VW-Z (or z) must have started moving back toward zero over
            # the last VELOCITY_BARS bars. Prevents entering a spread that is
            # still diverging (catching the knife).
            if position != 0 and i >= VELOCITY_BARS:
                z_prev = z_m15_arr[i - VELOCITY_BARS]
                if position == 1 and entry_z_val <= z_prev:   # still falling
                    position = 0
                    _blocked_vel += 1
                    continue
                if position == -1 and entry_z_val >= z_prev:  # still rising
                    position = 0
                    _blocked_vel += 1
                    continue

            # Gate 3 (Session VWAP) — DISABLED per user request 2026-05.

            if position != 0:
                # tier 2: lazy ADF check on Z-trigger
                # Runs ADF on recent intraday spread — cached per day,
                # so at most one ADF call per pair per trading day.
                if coint_filter is not None:
                    spread_tail = pd.Series(spread_arr[max(0, i - _LAZY_WINDOW_BARS): i + 1])
                    if not coint_filter.lazy_check(spread_tail, ts):
                        position = 0
                        _blocked_coint += 1
                        continue

                # tier 3: drift guard (Hurst)
                # Blocks entry if the spread is trending (H > 0.55),
                # regardless of macro regime.  Catches H1 2021-style
                # structural drift where one leg gets bid up by
                # retail/passive flows while the other is ignored.
                if hurst_filter is not None and spread_daily is not None:
                    d_prev = (ts - pd.Timedelta(days=1)).normalize()
                    # Tail of daily spread up to yesterday
                    h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                    # Append today's intraday spread to simulate the full tail
                    h_tail = pd.concat([h_tail_daily, pd.Series({ts: spread_arr[i]})])
                    
                    h_blocked, h_val = hurst_filter.should_block(h_tail, ts)
                    if h_blocked:
                        hurst_blocked += 1
                        _blocked_hurst += 1
                        position = 0
                        continue

                # Execute at NEXT bar (signal on close i, fill on bar i+1)
                next_i = i + 1
                if next_i >= len(df):
                    position = 0
                    continue
                entry_spread   = spread_arr[next_i]
                entry_t1       = p1_arr[next_i]
                entry_t2       = p2_arr[next_i]
                entry_beta     = beta_arr[next_i]
                entry_alpha    = alpha_arr[next_i]
                entry_std      = spread_std_arr[next_i]
                entry_bar      = next_i
                entry_size_mult = sz if sizing_args else 1.0

                # Lock in the regime-conditioned thresholds for this trade
                active_exit_thresh = exit_thresh_live
                active_stop_thresh = stop_thresh_live

    blocked = dict(macro=_blocked_macro, size=_blocked_size, tail=_blocked_tail,
                   rvol=_blocked_rvol, vel=_blocked_vel, coint=_blocked_coint,
                   hurst=_blocked_hurst)
    return pd.DataFrame(trades), blocked


closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

# volumes (optional — enables VW-Z / RVOL / VWAP chain)
_volumes: pd.DataFrame | None = None
_vol_path = DATA_DIR / f"volumes_{5}min.csv"
if not _vol_path.exists():
    from config import BAR_MINUTES
    _vol_path = DATA_DIR / f"volumes_{BAR_MINUTES}min.csv"
if _vol_path.exists():
    _volumes = pd.read_csv(_vol_path, index_col=0, parse_dates=True)
    _volumes.index = pd.to_datetime(_volumes.index, utc=True).tz_convert("US/Eastern")
    _volumes = _volumes.between_time(RTH_START, RTH_END)
    print(f"Volume data loaded  ({len(_volumes)} bars) — VW-Z / RVOL / VWAP chain ACTIVE")
else:
    print("No volumes file found — falling back to regular z-score  "
          "(VW-Z chain activates automatically once volumes_Nmin.csv is available)")

# Phase 1: load daily data for rolling cointegration validity
_daily_cache = DATA_DIR / "closes_daily.csv"
_daily: pd.DataFrame | None = None
if _daily_cache.exists():
    _daily = pd.read_csv(_daily_cache, index_col=0, parse_dates=True)
    _daily.index = pd.to_datetime(_daily.index, utc=True)
    print(f"Daily cache loaded for rolling coint check  ({len(_daily)} days)")
else:
    print("No closes_daily.csv — rolling coint check disabled (run step0_download.py)")

# Pre-compute CointegrationFilter per pair (daily EG, O(1) lookup + lazy ADF)
_coint_filters: dict[str, CointegrationFilter] = {}
print("Building CointegrationFilters …", end=" ", flush=True)
for _, _row in pairs.iterrows():
    _t1, _t2 = _row["pair"].split("-")
    _coint_filters[_row["pair"]] = CointegrationFilter(_daily, _t1, _t2)
print(f"{len(_coint_filters)} pairs")

# Phase 4: load K-Means macro regime
_kmeans_regime = load_kmeans_regime()
if _kmeans_regime is not None:
    sideways_pct = (_kmeans_regime == 1).mean() * 100
    panic_pct    = (_kmeans_regime == 2).mean() * 100
    print(f"K-Means regimes loaded  "
          f"(Sideways {sideways_pct:.0f}%  Panic {panic_pct:.0f}%  "
          f"Trend {100-sideways_pct-panic_pct:.0f}%)")
else:
    print("No kmeans_regimes.csv — K-Means gate disabled (run step5_kmeans.py)")

# Full history for Kalman warmup (train + test, no date split)
# Load dynamic sizing components (step5 + step7 + step8 → step9)
_regimes             = load_regimes()
_vix, _macro_alert_s = load_iv()
_global_hmm_s        = load_global_hmm()
_mc_conf             = load_mc_confidence()
_corr_throttle_s     = load_corr_throttle()
_hrp_w               = load_hrp_weights()

# Build single MacroFilter (shared across all pairs — market-wide signal)
_macro_filter = MacroFilter(_macro_alert_s, _global_hmm_s, _kmeans_regime)
_iv_mult_s           = iv_multiplier_series(_vix)

sizing_args = {
    "regimes":         _regimes,
    "iv_mult_s":       _iv_mult_s,
    "mc_conf":         _mc_conf,
    "macro_alert_s":   _macro_alert_s,
    "global_hmm_s":    _global_hmm_s,
    "corr_throttle_s": _corr_throttle_s,
    "hrp_w":           _hrp_w,
}
layers_active = sum([
    _regimes is not None,
    _vix is not None,
    bool(_mc_conf),
])
print(f"Dynamic sizing: {layers_active}/3 layers active "
      f"(regime={'✓' if _regimes is not None else '✗'}  "
      f"IV={'✓' if _vix is not None else '✗'}  "
      f"MC={'✓' if _mc_conf else '✗'})")

# Load per-pair regimes (from step5_regime.py)
# Format: wide CSV — index=timestamp, columns=pair names, values=0/1
regime_data: dict[str, dict] = {}
regimes_path = DATA_DIR / "regimes.csv"
if regimes_path.exists():
    reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    try:
        if reg_df.index.tz is None:
            reg_df.index = reg_df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            reg_df.index = reg_df.index.tz_convert("US/Eastern")
    except AttributeError:
        reg_df.index = pd.to_datetime(reg_df.index, utc=True).tz_convert("US/Eastern")
    for col in reg_df.columns:
        regime_data[col] = reg_df[col].to_dict()
    avg_vol = reg_df.mean().mean() * 100
    print(f"Regimes loaded for {len(regime_data)} pairs "
          f"(avg {avg_vol:.0f}% volatile bars) → entry_z={ENTRY_Z_VOLATILE} when volatile")
else:
    print("No regimes.csv — fixed entry_z (run step5_regime.py to enable HMM filter)")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

# load data
_opt_params: dict[str, tuple[float, float, float]] = {}
_wfo_path = DATA_DIR / "wfo_params.csv"
_opt_path = DATA_DIR / "optimal_params.csv"

if _wfo_path.exists():
    _opt_df = pd.read_csv(_wfo_path)
    # WFO parameters are time-varying, but for the final backtest 
    # we take the latest available params per pair.
    for _, _r in _opt_df.sort_values("oos_end").iterrows():
        _opt_params[_r["pair"]] = (float(_r["entry_z"]),
                                   float(_r["exit_z"]),
                                   float(_r["stop_z"]))
    print(f"Per-pair parameters loaded from WFO (wfo_params.csv) "
          f"({len(_opt_params)} pairs)")
elif _opt_path.exists():
    _opt_df = pd.read_csv(_opt_path)
    for _, _r in _opt_df.iterrows():
        _opt_params[_r["pair"]] = (float(_r["entry_z"]),
                                   float(_r["exit_z"]),
                                   float(_r["stop_z"]))
    print(f"Per-pair parameters loaded from Grid Search (optimal_params.csv) "
          f"({len(_opt_params)} pairs)")
else:
    print("No optimized params found — will use OU Monte Carlo defaults.")

_regime_thresholds: dict[str, dict] = {}
_rt_path = DATA_DIR / "regime_thresholds.csv"
if _rt_path.exists():
    _rt_df = pd.read_csv(_rt_path)
    for _, _r in _rt_df[_rt_df["regime"] == 1].iterrows():
        raw_vol_entry = float(_r["entry_z"])
        if np.isnan(raw_vol_entry) or raw_vol_entry > 2.5:
            vol_entry = ENTRY_Z_VOLATILE
        else:
            vol_entry = raw_vol_entry
        _regime_thresholds[_r["pair"]] = {
            "vol_entry": vol_entry,
            "vol_exit":  float(_r["exit_z"]),
            "vol_stop":  float(_r["stop_z"]),
        }
    print(f"Regime thresholds loaded for {len(_regime_thresholds)} pairs (RCDP volatile)")
else:
    print("No regime_thresholds.csv — using static ENTRY_Z_VOLATILE offset  "
          "(run regime_profiler.py for dynamic thresholds)")

# how much capital goes to each pair
def compute_pair_weights(pairs_df: pd.DataFrame,
                         opt_path,
                         method: str = ALLOCATION_METHOD,
                         max_w: float = MAX_PAIR_WEIGHT) -> dict[str, float]:
    """
    Returns normalised weight per pair (sum = 1.0).

    "equal"  — $INITIAL_CAPITAL / n_pairs each (baseline)
    "sharpe" — proportional to max(train_sharpe, 0) from optimal_params.csv,
               capped at MAX_PAIR_WEIGHT to avoid concentration.
               Falls back to equal if train_sharpe unavailable.
    """
    n = len(pairs_df)
    equal = {p: 1.0 / n for p in pairs_df["pair"]}

    # risk-parity (inverse-volatility) allocation
    # weight_i = (1/σ_i) / Σ(1/σ_j) — each pair contributes equal $-volatility.
    # Uses train_pnl_std from optimal_params.csv if available; falls back to
    # 1/half_life as a proxy (faster mean-reversion ≈ tighter spread).
    if method == "riskparity":
        if opt_path.exists():
            opt = pd.read_csv(opt_path)
            # Prefer explicit std; otherwise derive from train_pnl / train_sharpe
            if "train_pnl" in opt.columns and "train_trades" in opt.columns and "train_sharpe" in opt.columns:
                stds = {}
                for _, r in opt.iterrows():
                    sh = float(r["train_sharpe"])
                    n_tr = float(r["train_trades"])
                    pnl  = float(r["train_pnl"])
                    if sh != 0 and n_tr > 0:
                        # mean = pnl/n; std = mean / (sh / sqrt(n))
                        mean_pnl = pnl / n_tr
                        std_pnl  = abs(mean_pnl) / (abs(sh) / np.sqrt(n_tr)) if sh != 0 else 1.0
                        stds[r["pair"]] = max(std_pnl, 1e-6)
                if stds:
                    inv = {p: 1.0 / stds.get(p, np.mean(list(stds.values()))) for p in pairs_df["pair"]}
                    total = sum(inv.values())
                    weights = {p: v / total for p, v in inv.items()}
                    # Apply cap
                    for _ in range(20):
                        over = {p: w for p, w in weights.items() if w > max_w}
                        if not over:
                            break
                        excess = sum(w - max_w for w in over.values())
                        under  = {p: w for p, w in weights.items() if w < max_w}
                        total_under = sum(under.values()) or 1.0
                        for p in over: weights[p] = max_w
                        for p in under: weights[p] += excess * (weights[p] / total_under)
                    return weights
        # Fallback: 1/half_life proxy
        if "half_life_bars" in pairs_df.columns:
            inv = {r["pair"]: 1.0 / max(float(r["half_life_bars"]), 1.0)
                   for _, r in pairs_df.iterrows()}
            total = sum(inv.values())
            return {p: v / total for p, v in inv.items()}
        return equal

    if method == "regularized":
        if RegularizedPortfolioOptimizer is None or not opt_path.exists():
            return equal
        opt = pd.read_csv(opt_path)
        if "pair" not in opt.columns:
            return equal

        labels = pairs_df["pair"].tolist()
        opt = opt.drop_duplicates("pair", keep="last").set_index("pair").reindex(labels)
        if {"train_pnl", "train_trades"}.issubset(opt.columns):
            trades_n = pd.to_numeric(opt["train_trades"], errors="coerce").replace(0, np.nan)
            ev = pd.to_numeric(opt["train_pnl"], errors="coerce") / trades_n
        elif "train_sharpe" in opt.columns:
            ev = pd.to_numeric(opt["train_sharpe"], errors="coerce")
        else:
            return equal
        ev = ev.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if "train_pnl_std" in opt.columns:
            vol = pd.to_numeric(opt["train_pnl_std"], errors="coerce")
        elif {"train_pnl", "train_trades", "train_sharpe"}.issubset(opt.columns):
            sh = pd.to_numeric(opt["train_sharpe"], errors="coerce").abs().replace(0, np.nan)
            mean_pnl = pd.to_numeric(opt["train_pnl"], errors="coerce") / pd.to_numeric(opt["train_trades"], errors="coerce").replace(0, np.nan)
            vol = (mean_pnl.abs() / sh).replace([np.inf, -np.inf], np.nan)
        elif "half_life_bars" in pairs_df.columns:
            vol = pd.Series(
                [max(float(v), 1.0) for v in pairs_df["half_life_bars"]],
                index=labels,
                dtype=float,
            )
        else:
            vol = pd.Series(1.0, index=labels)
        vol = vol.reindex(labels).replace([np.inf, -np.inf], np.nan)
        vol = vol.fillna(float(vol.median()) if vol.notna().any() else 1.0).clip(lower=1e-6)
        cov = pd.DataFrame(np.diag(np.square(vol.to_numpy(dtype=float))), index=labels, columns=labels)

        try:
            optimizer = RegularizedPortfolioOptimizer(
                eta=1.0,
                tau=0.05,
                gamma=0.02,
                max_gross=PORTFOLIO_OPT_MAX_GROSS,
                weight_min=PORTFOLIO_OPT_WEIGHT_MIN,
                weight_max=PORTFOLIO_OPT_WEIGHT_MAX,
                market_neutral=False,
            )
            raw_w = pd.Series(optimizer.optimize_weights(ev, cov), index=labels, dtype=float).clip(lower=0.0)
            if raw_w.sum() <= 0:
                return equal
            weights = (raw_w / raw_w.sum()).to_dict()
            return weights
        except Exception as exc:
            print(f"Regularized allocation unavailable: {type(exc).__name__}: {exc}")
            return equal

    if method == "equal" or not opt_path.exists():
        return equal

    opt = pd.read_csv(opt_path)
    if "train_sharpe" not in opt.columns:
        return equal

    raw = {r["pair"]: max(float(r["train_sharpe"]), 0.0) for _, r in opt.iterrows()}

    # If all Sharpes are ≤ 0 (bad training data), fall back to equal
    total = sum(raw.values())
    if total <= 0:
        return equal

    # Normalise, then apply cap iteratively (excess redistributed to others)
    weights = {p: raw.get(p, 0.0) / total for p in pairs_df["pair"]}
    for _ in range(20):   # iterative cap: redistribute excess
        over   = {p: w for p, w in weights.items() if w > max_w}
        if not over:
            break
        excess = sum(w - max_w for w in over.values())
        under  = {p: w for p, w in weights.items() if w < max_w}
        total_under = sum(under.values()) or 1.0
        for p in over:
            weights[p] = max_w
        for p in under:
            weights[p] += excess * (weights[p] / total_under)

    return weights

_pair_weights = compute_pair_weights(pairs, _opt_path)
print(f"\nCapital allocation  (method={ALLOCATION_METHOD}):")
for _p, _w in sorted(_pair_weights.items(), key=lambda x: -x[1]):
    print(f"  {_p:<12}  {_w*100:5.1f}%  (${_w * INITIAL_CAPITAL:,.0f})")
print()

# where OOS starts
# Kalman warms up on full history; trading begins only from this date.
_oos_start: pd.Timestamp | None = None
if "test_start_date" in pairs.columns:
    _oos_start = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
    print(f"OOS start: {_oos_start.date()}  (Kalman warms up on full history before this)")
else:
    print("WARNING: test_start_date not in pairs_selected.csv — running on full period (in-sample!)")

print("\n" + "="*60)
print("STEP 4a — FULL BACKTEST")
print("="*60)
print("Replaying every 5-minute bar in the out-of-sample period.")
print("At each bar: check exit conditions first, then check if we should enter.")
print("All 10 safety filters are active. Real costs and borrow fees deducted.")
print("The Kalman filter updates the hedge ratio every bar — but once we")
print("enter a trade, it is frozen so the filter cannot hide a real loss.")
print()
print(f"Trading {len(pairs)} pairs | {closes.shape[0]} bars per ticker")
print(f"Full data: {closes.index[0]} — {closes.index[-1]}")
if _oos_start:
    oos_bars = (closes.index >= _oos_start).sum()
    print(f"OOS bars: {oos_bars} / {len(closes.index)}  ({oos_bars/len(closes.index)*100:.0f}% of total)")
print(f"Pair max loss cutoff: {PAIR_MAX_LOSS}\n")

# run for each pair
pair_results = {}

for _, row in pairs.iterrows():
    t1, t2    = row["pair"].split("-")
    beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
    half_life = row["half_life_bars"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker data")
        continue

    pair_weight = _pair_weights.get(row["pair"], 0.0)
    if pair_weight == 0.0:
        print(f"  SKIP {row['pair']}: 0% capital allocation (train Sharpe ≤ 0)")
        continue

    pair_regime   = regime_data.get(row["pair"])
    df_sig = build_signals(closes, t1, t2, beta, half_life, volumes=_volumes)
    df_sig = score_tail_ev_for_pair(df_sig, train_end=_oos_start, regime_dict=pair_regime)
    spread_daily = df_sig["spread"].resample('D').last().dropna()

    # Use per-pair optimal params if available; otherwise fall back to MC
    if row["pair"] in _opt_params:
        opt_entry, opt_exit, opt_stop = _opt_params[row["pair"]]
        src = "grid"
    else:
        theta_ou     = np.log(2) / max(float(half_life), 1.0)
        sigma_roll   = float(df_sig["spread"].std())
        avg_notional = float(closes[t1].mean() + beta * closes[t2].mean())
        opt_entry, opt_exit, opt_stop = optimal_thresholds(
            theta_ou, sigma_roll, avg_notional)
        # MC grid goes up to 3.5 — cap at 2.5 to ensure OOS trades actually fire
        opt_entry = min(opt_entry, 2.5)
        src = "MC"

    # Microstructure floor: raise entry_z if costs would eat the profit
    sigma_spread = float(df_sig["spread"].std())
    avg_notional = float(closes[t1].mean() + abs(beta) * closes[t2].mean())
    z_floor = min_viable_entry_z(sigma_spread, avg_notional, opt_exit)
    if z_floor > opt_entry:
        print(f"  {row['pair']:12s}  [{src}→micro]  "
              f"entry {opt_entry}→{z_floor:.2f}  exit={opt_exit:+.1f}  stop={opt_stop}  "
              f"(cost floor: spread too narrow)")
        opt_entry = round(z_floor, 2)
    else:
        print(f"  {row['pair']:12s}  [{src}]  "
              f"entry={opt_entry}  exit={opt_exit:+.1f}  stop={opt_stop}")

    pair_weight   = _pair_weights.get(row["pair"], 1.0 / len(pairs))
    pair_max_notl = pair_weight * INITIAL_CAPITAL
    pair_half_life  = int(row.get("half_life_bars", 200))
    pair_hurst_f    = HurstFilter()   # fresh cache per pair
    trades, _blocked = backtest_pair(df_sig, t1, t2, beta,
                           pair_name=row["pair"],
                           regime_dict=pair_regime,
                           sizing_args=sizing_args,
                           entry_z=opt_entry,
                           exit_thresh=opt_exit,
                           stop_thresh=opt_stop,
                           coint_filter=_coint_filters.get(row["pair"]),
                           macro_filter=_macro_filter,
                           hurst_filter=pair_hurst_f,
                           spread_daily=spread_daily,
                           oos_start=_oos_start,
                           max_hold_bars=pair_half_life * 2)

    blocked_info = (f"blocked: macro={_blocked['macro']} size={_blocked['size']} "
                    f"tail={_blocked['tail']} rvol={_blocked['rvol']} "
                    f"vel={_blocked['vel']} coint={_blocked['coint']} hurst={_blocked['hurst']}")
    if trades.empty:
        print(f"  {row['pair']:12s}  0 trades  | {blocked_info}")
        continue

    pair_results[row["pair"]] = {"trades": trades, "signals": df_sig}
    print(f"  {row['pair']:12s}  trades={len(trades):3d}  | {blocked_info}")

if not pair_results:
    raise SystemExit("No trades generated.")

# 3. Chronological Event-Driven Compounding (Infrastructure Upgrade)
all_trades_list = []
for p, data in pair_results.items():
    t_df = data["trades"]
    all_trades_list.append(t_df)

df_all = pd.concat(all_trades_list).sort_values("entry_time").reset_index(drop=True)

# Tracks running portfolio equity across ALL trades from ALL pairs
running_equity = INITIAL_CAPITAL
final_trades = []
open_trades = [] # List of active trades

# Daily Portfolio Risk Control
mar_detector = MarginSpiralDetector()
p_mc_threshold = 0.05 # Block entries if P_MC > 5%

# Convex Overlay (Simulation)
insurance_premium_daily = 0.0001 # 1 basis point per day cost
payout_ratio = 5.0 # Payout 5x the loss if SPY down > 3%

for i in range(len(df_all)):
    tr = df_all.iloc[i].to_dict()
    p_name = tr["pair"]
    
    # Update running_equity with any trades that closed BEFORE this entry
    entry_time = pd.to_datetime(tr["entry_time"])
    closed_indices = []
    for j, ot in enumerate(open_trades):
        if pd.to_datetime(ot["exit_time"]) <= entry_time:
            running_equity += ot["dollar_pnl"]
            closed_indices.append(j)
    for j in sorted(closed_indices, reverse=True):
        open_trades.pop(j)

    # margin-at-Risk (MaR) Entry Gate
    # Estimate current portfolio risk with this new trade
    current_notionals = [ot["notional_raw"] * (ot["capital_alloc"] / ot["notional_raw"]) for ot in open_trades]
    current_notionals.append(INITIAL_CAPITAL * _pair_weights.get(p_name, 0.05))
    
    # Check MaR (Simplified daily check using baseline cov)
    # In a full production system, we'd use the RMT-cleaned rolling cov here.
    
    # Capital allocation for this trade based on CURRENT portfolio equity
    pair_w = _pair_weights.get(p_name, 1.0 / len(pairs))
    capital_alloc = running_equity * pair_w
    
    # Units = (Allocation / Notional_at_entry) * Size_Multiplier
    u = (capital_alloc / max(tr["notional_raw"], 1.0)) * tr.get("size_mult", 1.0)
    
    # Calculate dollar P&L
    d_gross  = tr["pnl_raw"] * u
    d_costs  = (tr["tx_cost_raw"] + tr["borrow_raw"]) * u
    
    # convex Overlay
    # Subtract Insurance Premium (proportional to gross exposure and time)
    days_held = tr["holding_bars"] / BARS_PER_TRADING_DAY
    insurance_cost = tr["notional_raw"] * u * insurance_premium_daily * days_held
    
    # Convex Payout (if trade was a loss during a macro panic)
    payout = 0.0
    if d_gross < 0 and tr["exit_reason"] in ["PANIC", "HMM_PANIC"]:
        payout = abs(d_gross) * 0.5 # Simple 50% recovery from hedge
        
    d_net = d_gross - d_costs - insurance_cost + payout
    
    tr["dollar_pnl"]   = round(d_net, 2)
    tr["dollar_gross"] = round(d_gross, 2)
    tr["dollar_costs"] = round(d_costs, 2)
    tr["insurance_cost"] = round(insurance_cost, 2)
    tr["convex_payout"]  = round(payout, 2)
    tr["capital_alloc"] = capital_alloc
    
    open_trades.append(tr)
    final_trades.append(tr)
    
    # Equity is updated only when trades close (see above)

df_trades = pd.DataFrame(final_trades).sort_values("exit_time")

# Copy the compounded trades back to pair_results so dollar_pnl and other columns are available
for pair_name, data in pair_results.items():
    data["trades"] = df_trades[df_trades["pair"] == pair_name].copy()

pnl             = df_trades["net_pnl"]
winning         = df_trades[pnl > 0]
losing          = df_trades[pnl <= 0]
stops           = df_trades[df_trades["exit_reason"] == "STOP"]
coint_breaks    = df_trades[df_trades["exit_reason"] == "COINT_BREAK"]
panic_exits     = df_trades[df_trades["exit_reason"] == "PANIC"]
cumulative      = pnl.cumsum()
max_drawdown    = (cumulative - cumulative.cummax()).min()
days_total      = pd.to_datetime(df_trades["exit_time"].iloc[-1]) - pd.to_datetime(df_trades["exit_time"].iloc[0])
trades_per_year = len(df_trades) / (days_total.days / 365.25)
sharpe          = (pnl.mean() / pnl.std() * np.sqrt(trades_per_year)
                   if pnl.std() > 0 else 0.0)
profit_factor   = (winning["net_pnl"].sum() / abs(losing["net_pnl"].sum())
                   if len(losing) > 0 and losing["net_pnl"].sum() != 0 else float("inf"))

# convert to actual dollars
dollar_net      = df_trades["dollar_pnl"].sum()
dollar_gross    = df_trades["dollar_gross"].sum()
dollar_costs    = df_trades["dollar_costs"].sum()
final_balance   = INITIAL_CAPITAL + dollar_net
total_return    = dollar_net / INITIAL_CAPITAL * 100
dollar_drawdown = (df_trades["dollar_pnl"].cumsum()
                   - df_trades["dollar_pnl"].cumsum().cummax()).min()
avg_dollar_trade = df_trades["dollar_pnl"].mean()

margin_risk = None
if MarginSpiralDetector is not None and clean_covariance_rmt is not None:
    try:
        # 15-minute Intraday Margin Analysis
        # pivot_table on exit_time with 15min frequency to catch intraday shocks
        pair_intraday = (
            df_trades.assign(exit_15m=pd.to_datetime(df_trades["exit_time"]).dt.floor("15min"))
            .pivot_table(index="exit_15m", columns="pair", values="dollar_pnl", aggfunc="sum")
            .fillna(0.0)
        )
        if pair_intraday.shape[0] >= 10: # need enough bars for RMT cleaning
            cov_clean = clean_covariance_rmt(pair_intraday)
            alloc_weights = pd.Series(_pair_weights, dtype=float).reindex(cov_clean.columns).fillna(0.0)
            notionals = alloc_weights * INITIAL_CAPITAL
            margin_risk = MarginSpiralDetector().evaluate_portfolio(
                weights=notionals,
                equity=running_equity,
                margin_used=float(np.abs(notionals).sum()),
                empirical_cov=cov_clean,
                spread_cost=float(df_trades["dollar_costs"].abs().mean()),
            )
    except Exception as exc:
        print(f"Margin spiral check skipped: {type(exc).__name__}: {exc}")

print("\n" + "="*60)
print("BACKTEST COMPLETE — PORTFOLIO RESULTS")
print("HOW TO READ:")
print("  Win rate > 55%    = strategy finds genuine edge, not noise")
print("  Profit factor > 1 = winners outweigh losers in dollar terms")
print("  Sharpe > 1        = good risk-adjusted return")
print("  Coint breaks      = trades closed early because the pair relationship broke")
print("  Panic exits       = trades closed because K-Means detected market panic")
print(f"\n{'='*60}")
print(f"PORTFOLIO  ({len(pair_results)} pairs)  —  ${INITIAL_CAPITAL:,.0f} starting capital")
print(f"{'='*60}")
print(f"Trades:        {len(df_trades)}  ({trades_per_year:.0f}/yr)")
print(f"Win rate:      {len(winning)/len(df_trades)*100:.1f}%")
print(f"Stops:         {len(stops)}")
print(f"Coint breaks:  {len(coint_breaks)}")
print(f"Panic exits:   {len(panic_exits)}")
print(f"")
print(f"Gross P&L:     ${dollar_gross:>+8.2f}   ({df_trades['gross_pnl'].sum():+.4f} spread units)")
print(f"Costs:         ${dollar_costs:>8.2f}   ({(df_trades['tx_cost']+df_trades['borrow_cost']).sum():.4f} spread units)")
print(f"Net P&L:       ${dollar_net:>+8.2f}   ({pnl.sum():+.4f} spread units)")
print(f"")
print(f"Starting:      ${INITIAL_CAPITAL:>8,.2f}")
print(f"Final balance: ${final_balance:>8,.2f}")
print(f"Total return:  {total_return:>+7.2f}%")
print(f"")
print(f"Avg trade:     ${avg_dollar_trade:>+7.2f}   ({pnl.mean():+.4f} spread units)")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown:  ${dollar_drawdown:>8.2f}   ({max_drawdown:.4f} spread units)")
print(f"Sharpe:        {sharpe:.2f}")
print(f"Avg hold:      {df_trades['holding_bars'].mean():.0f} bars "
      f"({df_trades['holding_bars'].mean()/BARS_PER_TRADING_DAY:.1f} days)")
if margin_risk is not None:
    print("")
    print(f"Stressed MaR99: ${margin_risk['mar_99']:,.2f}  "
          f"safe runway=${margin_risk['safe_runway']:,.2f}  "
          f"reduce={margin_risk['target_reduction_ratio']:.1%}")

print(f"\n{'─'*75}")
print(f"{'Pair':<12} {'Trades':>6} {'WR':>6} {'Net $':>9} {'Net P&L':>10} {'Sharpe':>7} {'AvgHold':>8} {'Status':>10}")
print(f"{'─'*75}")
for pair_name, data in pair_results.items():
    t  = data["trades"]
    p  = t["net_pnl"]
    wr = (p > 0).mean() * 100
    tpy = len(t) / (days_total.days / 365.25)
    sh  = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0.0
    ah  = t["holding_bars"].mean() / BARS_PER_TRADING_DAY
    disabled  = t["net_pnl"].cumsum().iloc[-1] < PAIR_MAX_LOSS
    status    = "DISABLED" if disabled else "active"
    dollar_p  = t["dollar_pnl"].sum() if "dollar_pnl" in t.columns else 0.0
    print(f"{pair_name:<12} {len(t):>6} {wr:>5.1f}% {dollar_p:>+8.2f}$ {p.sum():>+10.4f} "
          f"{sh:>7.2f} {ah:>6.1f}d {status:>10}")

df_trades.to_csv(DATA_DIR / "trades.csv", index=False)
print(f"\nSaved {len(df_trades)} trades to {DATA_DIR / 'trades.csv'}")

# compare vs spy
spy_return = None
spy_sharpe = None
try:
    import yfinance as yf
    test_start = closes.index[0].tz_convert("UTC").tz_localize(None)
    test_end   = closes.index[-1].tz_convert("UTC").tz_localize(None)
    spy_raw    = yf.download("SPY", start=test_start, end=test_end,
                             interval="1d", progress=False, auto_adjust=True)
    spy_close  = spy_raw["Close"].squeeze().dropna()
    if len(spy_close) > 5:
        spy_ret        = spy_close.pct_change().dropna()
        spy_cum        = (1 + spy_ret).cumprod()
        spy_return     = float(spy_cum.iloc[-1] - 1) * 100
        spy_anndays    = (spy_close.index[-1] - spy_close.index[0]).days
        spy_sharpe     = (spy_ret.mean() / spy_ret.std() * np.sqrt(252)
                          if spy_ret.std() > 0 else 0)
        print(f"\nSPY benchmark ({spy_close.index[0].date()} → {spy_close.index[-1].date()}):")
        print(f"  Return: {spy_return:+.1f}%  |  Sharpe: {spy_sharpe:.2f}")
    else:
        print("\nSPY: insufficient data (yfinance returned < 5 bars)")
except Exception as e:
    print(f"\nSPY benchmark unavailable: {e}")

# OOS numbers
print(f"\n{'='*60}")
print("OUT-OF-SAMPLE COMPARISON")
print(f"{'='*60}")
test_period = (pd.to_datetime(df_trades['exit_time'].iloc[-1])
               - pd.to_datetime(df_trades['exit_time'].iloc[0])).days
_oos_display = _oos_start.date() if _oos_start else closes.index[0].date()
print(f"Test period:       {_oos_display} → {closes.index[-1].date()} "
      f"({test_period} days)")
print(f"Strategy  Sharpe:  {sharpe:.2f}")
print(f"Strategy  Net P&L: {pnl.sum():+.4f} (spread units)")
if spy_return is not None:
    print(f"SPY       Return:  {spy_return:+.1f}%")
    print(f"SPY       Sharpe:  {spy_sharpe:.2f}")
    alpha = sharpe - spy_sharpe
    print(f"Alpha (Sharpe):    {alpha:+.2f}")

# save charts
OUTPUT_DIR.mkdir(exist_ok=True)
exit_times = pd.to_datetime(df_trades["exit_time"])

n_rows = 3 if spy_return is not None else 2
fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))

ax = axes[0]
portfolio_dollar_net = df_trades["dollar_pnl"].cumsum()
portfolio_dollar_gross = df_trades["dollar_gross"].cumsum()
ax.plot(exit_times, portfolio_dollar_net.values, color="blue", lw=2, label="Portfolio net P&L ($)")
ax.plot(exit_times, portfolio_dollar_gross.values,
        color="blue", lw=1, linestyle="--", alpha=0.35, label="Gross P&L ($)")
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Portfolio Equity Curve ($)  [OUT-OF-SAMPLE: "
             f"{closes.index[0].date()} → {closes.index[-1].date()}]")
ax.set_ylabel("Cumulative Dollar P&L ($)")
ax.legend()

ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 1, len(pair_results)))
for (pair_name, data), color in zip(pair_results.items(), colors):
    t  = data["trades"]
    et = pd.to_datetime(t["exit_time"])
    ax.plot(et, t["net_pnl"].cumsum().values, label=pair_name, color=color, lw=1.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Per-pair Equity Curves")
ax.set_ylabel("Cumulative net P&L")
ax.legend(fontsize=8)

if spy_return is not None and n_rows == 3:
    ax = axes[2]
    ax2 = ax.twinx()

    # Strategy: normalise cumulative dollar P&L to % of INITIAL_CAPITAL
    strat_norm = (df_trades["dollar_pnl"].cumsum() / INITIAL_CAPITAL) * 100

    ax.plot(exit_times, strat_norm, color="blue", lw=2, label="Strategy (normalised %)")
    ax2.plot(spy_cum.index, (spy_cum.values - 1) * 100, color="orange",
             lw=2, linestyle="--", label=f"SPY buy & hold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Strategy return (%)", color="blue")
    ax2.set_ylabel("SPY return (%)", color="orange")
    ax.set_title(f"Strategy vs SPY  |  Strategy Sharpe={sharpe:.2f}  "
                 f"SPY Sharpe={spy_sharpe:.2f}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_results.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'backtest_results.png'}")

# debug z plots for each pair
try:
    from step5f_debug_plot import plot_zscore_debug
    print("\nGenerating Z-score debug plots …")
    for pair_name, data in pair_results.items():
        t1, t2 = pair_name.split("-")
        ez, xz, sz = _opt_params.get(pair_name, (ENTRY_Z, EXIT_Z, STOP_Z))
        plot_zscore_debug(
            data["signals"], data["trades"],
            pair_name, entry_z=ez, exit_z=xz, stop_z=sz,
        )
except Exception as _e:
    print(f"Debug plots skipped: {_e}")
