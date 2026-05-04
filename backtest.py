import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from config import (
    ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    COST_PER_SIDE, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    IV_SIZE_NORM, MIN_POSITION_SIZE, TRAIN_RATIO,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR,
    BARS_PER_DAY, CLOSES_FILE, KALMAN_DELTA,
    COINT_WINDOW_DAYS, COINT_BREAK_P, COINT_RECHECK_DAYS,
)
from kalman import kalman_hedge
from sizing import (
    load_regimes, load_iv, load_mc_confidence,
    iv_multiplier_series, position_size,
)

BARS_PER_TRADING_DAY = BARS_PER_DAY


def _data_path() -> str:
    p = DATA_DIR / CLOSES_FILE
    if not p.exists():
        fallback = DATA_DIR / "closes_15min.csv"
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(f"No data file: {CLOSES_FILE}")
    return str(p)


def load_closes() -> pd.DataFrame:
    """Load ALL available intraday data.

    Pair selection is done on 8-year daily data (step2), so there is no
    look-ahead: using the full intraday history is valid and maximises
    the number of trades for statistical evaluation.
    """
    closes = pd.read_csv(_data_path(), index_col=0, parse_dates=True)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes.index = closes.index.tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    pairs_path = DATA_DIR / "pairs_selected.csv"
    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            needed    = {t for p in meta["pair"] for t in p.split("-")}
            available = [t for t in needed if t in closes.columns]
            closes    = closes[available].dropna()
            return closes

    return closes.dropna()


def build_signals(closes, t1, t2, beta, half_life,
                  delta: float = KALMAN_DELTA) -> pd.DataFrame:
    """Kalman-filter spread signals.

    The hedge ratio β is updated bar-by-bar via a Kalman filter, so it adapts
    continuously to shifting correlations (e.g. Trump tweets, macro shocks) rather
    than being fixed to an OLS estimate from months ago.  beta_daily is used only
    as the initial value; the filter takes over from bar 1.
    Spread = Kalman innovation = p1 - (α_t + β_t·p2), already ~zero-mean.
    """
    _, beta_arr, innov, _ = kalman_hedge(
        closes[t1].values, closes[t2].values,
        delta=delta, beta_init=float(beta),
    )
    spread = pd.Series(innov,    index=closes.index, name="spread")
    beta_s = pd.Series(beta_arr, index=closes.index, name="beta")
    window = max(20, min(int(half_life), 200))
    spread_std = spread.rolling(window=window).std()
    zscore = spread / spread_std   # innovations are already ~zero-mean
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread":      spread,
        "beta":        beta_s,
        "spread_std":  spread_std,
        "zscore":      zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)


def compute_rolling_coint_validity(daily: pd.DataFrame, t1: str, t2: str,
                                   window: int = COINT_WINDOW_DAYS,
                                   p_thresh: float = COINT_BREAK_P,
                                   step: int = COINT_RECHECK_DAYS) -> pd.Series:
    """
    Pre-compute a daily boolean Series: True = pair cointegrated, False = broken.
    Sampled every `step` days (default 5) and forward-filled — avoids running
    EG on every single trading day (5x faster).
    """
    pc = daily[[t1, t2]].dropna()
    if len(pc) < window:
        return pd.Series(dtype=bool)
    sampled = {}
    for i in range(window, len(pc) + 1, step):
        chunk = pc.iloc[i - window:i]
        try:
            _, pval, _ = coint(chunk[t1], chunk[t2])
            sampled[pc.index[i - 1]] = pval < p_thresh
        except Exception:
            sampled[pc.index[i - 1]] = False
    if not sampled:
        return pd.Series(dtype=bool)
    s = pd.Series(sampled)
    # forward-fill so every calendar day has a value
    return s.reindex(pc.index).ffill().fillna(True).rename(f"{t1}-{t2}")


def load_kmeans_regime() -> pd.Series | None:
    """Load daily K-Means macro regime labels (0=Trend, 1=Sideways, 2=Panic)."""
    path = DATA_DIR / "kmeans_regimes.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def ou_params_from_spread(spread: pd.Series) -> tuple[float, float]:
    """OLS on ΔS = -θ·S_{t-1} + ε  →  returns (theta_per_bar, residual_std)."""
    ds  = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    idx = ds.index.intersection(lag.index)
    reg = sm.OLS(ds[idx], lag[idx]).fit()
    theta = float(-reg.params.iloc[0])
    return max(theta, 1e-6), float(reg.resid.std())


def optimal_thresholds(theta: float, sigma_roll: float, notional: float,
                       n_sim: int = 250, max_bars: int = 1500
                       ) -> tuple[float, float, float]:
    """
    Joint Monte Carlo grid search for (entry_z, exit_thresh, stop_thresh).

    OU in z-score space: Z_{t+1} = Z_t·(1-θ) + √(2θ)·ε
    LONG: enter at -z_e, exit when Z >= z_x, stop when Z <= -z_s.
    P&L_exit = z_e + z_x - c_z   (spread moved from -z_e to z_x)
    P&L_stop = z_e - z_s - c_z   (spread moved against us to -z_s)
    """
    rng   = np.random.default_rng(42)
    c_z   = 2 * COST_PER_SIDE * notional / max(sigma_roll, 1e-8)
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
                    pnl   = np.where(he,       z_e + z_x - c_z,  pnl)
                    pnl   = np.where(hs & ~he, z_e - z_s  - c_z, pnl)
                    t_end = np.where((he | hs) & ~done, float(t), t_end)
                    done  = done | he | hs

                pnl = np.where(~done, -c_z, pnl)   # timed-out: pay cost, no profit

                rate = float(pnl.mean()) / float(t_end.mean())
                if rate > best_rate:
                    best_rate = rate
                    best      = (z_e, z_x, z_s)

    if best_rate <= 0:
        return float(ENTRY_Z), float(EXIT_Z), float(STOP_Z)
    return round(best[0], 2), round(best[1], 2), round(best[2], 2)


def backtest_pair(df, t1, t2, beta, pair_name: str = "",
                  regime_dict: dict | None = None,
                  sizing_args: dict | None = None,
                  entry_z: float = ENTRY_Z,
                  exit_thresh: float = EXIT_Z,
                  stop_thresh: float = STOP_Z,
                  coint_valid: pd.Series | None = None,
                  kmeans_regime: pd.Series | None = None,
                  oos_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    oos_start     : first timestamp of the OOS period — entries are blocked before
                    this date, but Kalman filter has already warmed up on full history.
    coint_valid   : daily boolean Series (Phase 1) — force-close + suspend entry
                    when rolling 90-day cointegration breaks.
    kmeans_regime : daily int Series (Phase 4) — 0=Trend, 1=Sideways, 2=Panic.
                    New entries only in Sideways; force-close in Panic.
    """
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position = 0
    entry_spread = entry_t1 = entry_t2 = entry_beta = 0.0
    entry_bar = 0
    cumulative_pnl = 0.0
    trades = []
    suspended = False   # True when rolling coint has broken

    # Pre-build date-keyed dicts for O(1) lookups in the hot loop
    _coint_dict: dict = {}
    if coint_valid is not None and len(coint_valid) > 0:
        cv = coint_valid.copy()
        if cv.index.tz is not None:
            cv.index = cv.index.tz_localize(None) if False else cv.index
        _coint_dict = {ts.date(): bool(v) for ts, v in cv.items()}

    _kmeans_dict: dict = {}
    if kmeans_regime is not None and len(kmeans_regime) > 0:
        kr = kmeans_regime.copy()
        _kmeans_dict = {ts.date(): int(v) for ts, v in kr.items()}

    for i in range(len(df)):
        ts         = df.index[i]
        day        = ts
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        t1_price   = df[t1_col].iloc[i]
        t2_price   = df[t2_col].iloc[i]

        # ── Phase 1: rolling cointegration validity check ─────────────────
        if _coint_dict:
            coint_ok = _coint_dict.get(day.date(), True)
            if not coint_ok and not suspended:
                suspended = True
            elif coint_ok and suspended:
                suspended = False

        # ── Phase 4: K-Means macro regime gate ───────────────────────────
        macro_regime = _kmeans_dict.get(day.date(), 1)

        # Force-close all positions in Panic regime (2) or coint break
        force_close = (macro_regime == 2) or suspended
        if position != 0 and force_close:
            gross_pnl      = position * (spread_now - entry_spread)
            notional       = entry_t1 + abs(entry_beta) * entry_t2
            tx_cost        = 2 * notional * COST_PER_SIDE
            holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
            short_notional = (abs(entry_beta) * entry_t2) if position == 1 else entry_t1
            borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
            size           = IV_SIZE_NORM
            net_pnl        = (gross_pnl - tx_cost - borrow_cost) * size
            cumulative_pnl += net_pnl
            reason = "PANIC" if macro_regime == 2 else "COINT_BREAK"
            trades.append({
                "pair":         f"{t1}-{t2}",
                "entry_time":   df.index[entry_bar],
                "exit_time":    ts,
                "direction":    "LONG" if position == 1 else "SHORT",
                "holding_bars": i - entry_bar,
                "gross_pnl":    round(gross_pnl, 4),
                "tx_cost":      round(tx_cost, 4),
                "borrow_cost":  round(borrow_cost, 4),
                "net_pnl":      round(net_pnl, 4),
                "cum_pnl":      round(cumulative_pnl, 4),
                "exit_reason":  reason,
                "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                "exit_z":       round(z, 2),
            })
            position = 0
            if cumulative_pnl < PAIR_MAX_LOSS:
                break
            continue

        if position != 0:
            exit_signal = (position == 1 and z >= exit_thresh) or (position == -1 and z <= -exit_thresh)
            stop_signal = (position == 1 and z <= -stop_thresh) or (position == -1 and z >= stop_thresh)

            if exit_signal or stop_signal:
                gross_pnl      = position * (spread_now - entry_spread)
                notional       = entry_t1 + abs(entry_beta) * entry_t2
                tx_cost        = 2 * notional * COST_PER_SIDE
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notional = (abs(entry_beta) * entry_t2) if position == 1 else entry_t1
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                if sizing_args:
                    size = position_size(
                        pair_name, df.index[entry_bar],
                        sizing_args["regimes"],
                        sizing_args["iv_mult_s"],
                        sizing_args["mc_conf"],
                    )
                else:
                    size = IV_SIZE_NORM
                net_pnl        = (gross_pnl - tx_cost - borrow_cost) * size
                cumulative_pnl += net_pnl

                trades.append({
                    "pair":         f"{t1}-{t2}",
                    "entry_time":   df.index[entry_bar],
                    "exit_time":    ts,
                    "direction":    "LONG" if position == 1 else "SHORT",
                    "holding_bars": i - entry_bar,
                    "gross_pnl":    round(gross_pnl, 4),
                    "tx_cost":      round(tx_cost, 4),
                    "borrow_cost":  round(borrow_cost, 4),
                    "net_pnl":      round(net_pnl, 4),
                    "cum_pnl":      round(cumulative_pnl, 4),
                    "exit_reason":  "STOP" if stop_signal else "SIGNAL",
                    "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                    "exit_z":       round(z, 2),
                })
                position = 0

                if cumulative_pnl < PAIR_MAX_LOSS:
                    break

        if position == 0:
            # Block new entries before OOS start (Kalman warmup period)
            if oos_start is not None and ts < oos_start:
                continue

            # Block new entries when coint broken or macro not sideways
            if suspended or (macro_regime != 1 and kmeans_regime is not None):
                continue

            if sizing_args:
                sz = position_size(pair_name, ts,
                                   sizing_args["regimes"],
                                   sizing_args["iv_mult_s"],
                                   sizing_args["mc_conf"])
                if sz < MIN_POSITION_SIZE:
                    continue

            is_volatile = regime_dict.get(ts, 0) == 1 if regime_dict else False
            threshold   = entry_z + (ENTRY_Z_VOLATILE - ENTRY_Z) if is_volatile else entry_z
            if z < -threshold:
                position = 1
            elif z > threshold:
                position = -1
            if position != 0:
                # Execute at NEXT bar — signal fires on close of bar i,
                # fill happens on bar i+1 (realistic: ~15 min later).
                next_i = i + 1
                if next_i >= len(df):
                    position = 0   # no next bar, skip
                    continue
                entry_spread = df["spread"].iloc[next_i]
                entry_t1     = df[t1_col].iloc[next_i]
                entry_t2     = df[t2_col].iloc[next_i]
                entry_beta   = df["beta"].iloc[next_i]
                entry_bar    = next_i

    return pd.DataFrame(trades)


# ── Load ─────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

# Phase 1: load daily data for rolling cointegration validity
_daily_cache = DATA_DIR / "closes_daily.csv"
_daily: pd.DataFrame | None = None
if _daily_cache.exists():
    _daily = pd.read_csv(_daily_cache, index_col=0, parse_dates=True)
    _daily.index = pd.to_datetime(_daily.index, utc=True)
    print(f"Daily cache loaded for rolling coint check  ({len(_daily)} days)")
else:
    print("No closes_daily.csv — rolling coint check disabled (run step0_download.py)")

# Pre-compute rolling cointegration validity per pair
_coint_valid: dict[str, pd.Series] = {}
if _daily is not None:
    print("Computing rolling cointegration validity …", end=" ", flush=True)
    for _, _row in pairs.iterrows():
        _t1, _t2 = _row["pair"].split("-")
        if _t1 in _daily.columns and _t2 in _daily.columns:
            _coint_valid[_row["pair"]] = compute_rolling_coint_validity(_daily, _t1, _t2)
    print(f"{len(_coint_valid)} pairs")

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
_regimes   = load_regimes()
_vix       = load_iv()
_mc_conf   = load_mc_confidence()
_iv_mult_s = iv_multiplier_series(_vix)

sizing_args = {
    "regimes":   _regimes,
    "iv_mult_s": _iv_mult_s,
    "mc_conf":   _mc_conf,
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

# ── Load per-pair optimal params (from step4d grid search on train data) ─────
_opt_params: dict[str, tuple[float, float, float]] = {}
_opt_path = DATA_DIR / "optimal_params.csv"
if _opt_path.exists():
    _opt_df = pd.read_csv(_opt_path)
    for _, _r in _opt_df.iterrows():
        _opt_params[_r["pair"]] = (float(_r["entry_z"]),
                                   float(_r["exit_z"]),
                                   float(_r["stop_z"]))
    print(f"Per-pair optimal params loaded from optimal_params.csv "
          f"({len(_opt_params)} pairs)")
else:
    print("No optimal_params.csv — will use OU Monte Carlo per pair  "
          "(run step 6e first for better results)")

# OOS start date — read from pairs_selected.csv (set by pairs.py via TRAIN_RATIO)
# Kalman warms up on full history; trading begins only from this date.
_oos_start: pd.Timestamp | None = None
if "test_start_date" in pairs.columns:
    _oos_start = pd.Timestamp(pairs["test_start_date"].iloc[0]).tz_localize("US/Eastern")
    print(f"OOS start: {_oos_start.date()}  (Kalman warms up on full history before this)")
else:
    print("WARNING: test_start_date not in pairs_selected.csv — running on full period (in-sample!)")

print(f"Trading {len(pairs)} pairs | {closes.shape[0]} bars per ticker")
print(f"Full data: {closes.index[0]} — {closes.index[-1]}")
if _oos_start:
    oos_bars = (closes.index >= _oos_start).sum()
    print(f"OOS bars: {oos_bars} / {len(closes.index)}  ({oos_bars/len(closes.index)*100:.0f}% of total)")
print(f"Pair max loss cutoff: {PAIR_MAX_LOSS}\n")

# ── Run backtest per pair ─────────────────────────────────────────────────────
pair_results = {}

for _, row in pairs.iterrows():
    t1, t2    = row["pair"].split("-")
    beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
    half_life = row["half_life_bars"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker data")
        continue

    df_sig = build_signals(closes, t1, t2, beta, half_life)

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
        src = "MC"
    print(f"  {row['pair']:12s}  [{src}]  "
          f"entry={opt_entry}  exit={opt_exit:+.1f}  stop={opt_stop}")

    pair_regime  = regime_data.get(row["pair"])
    coint_series = _coint_valid.get(row["pair"])
    trades = backtest_pair(df_sig, t1, t2, beta,
                           pair_name=row["pair"],
                           regime_dict=pair_regime,
                           sizing_args=sizing_args,
                           entry_z=opt_entry,
                           exit_thresh=opt_exit,
                           stop_thresh=opt_stop,
                           coint_valid=coint_series,
                           kmeans_regime=_kmeans_regime,
                           oos_start=_oos_start)

    if trades.empty:
        print(f"  {row['pair']:12s}  0 trades")
        continue

    pnl      = trades["net_pnl"]
    win_rate = (pnl > 0).mean() * 100
    disabled = trades["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status   = " [DISABLED — max loss hit]" if disabled else ""

    pair_results[row["pair"]] = {"trades": trades, "signals": df_sig}
    print(f"  {row['pair']:12s}  trades={len(trades):3d}  "
          f"WR={win_rate:4.1f}%  net P&L={pnl.sum():+.4f}{status}")

if not pair_results:
    raise SystemExit("No trades generated.")

# ── Combine ───────────────────────────────────────────────────────────────────
df_trades = (pd.concat([v["trades"] for v in pair_results.values()])
               .sort_values("exit_time")
               .reset_index(drop=True))

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

print(f"\n{'='*60}")
print(f"PORTFOLIO  ({len(pair_results)} pairs)")
print(f"{'='*60}")
print(f"Trades:        {len(df_trades)}  ({trades_per_year:.0f}/yr)")
print(f"Win rate:      {len(winning)/len(df_trades)*100:.1f}%")
print(f"Stops:         {len(stops)}")
print(f"Coint breaks:  {len(coint_breaks)}")
print(f"Panic exits:   {len(panic_exits)}")
print(f"Gross P&L:     {df_trades['gross_pnl'].sum():+.4f}")
print(f"Costs:         {(df_trades['tx_cost']+df_trades['borrow_cost']).sum():.4f}")
print(f"Net P&L:       {pnl.sum():+.4f}")
print(f"Avg trade:     {pnl.mean():+.4f}")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown:  {max_drawdown:.4f}")
print(f"Sharpe:        {sharpe:.2f}")
print(f"Avg hold:      {df_trades['holding_bars'].mean():.0f} bars "
      f"({df_trades['holding_bars'].mean()/BARS_PER_TRADING_DAY:.1f} days)")

print(f"\n{'─'*65}")
print(f"{'Pair':<12} {'Trades':>6} {'WR':>6} {'Net P&L':>10} {'Sharpe':>7} {'AvgHold':>8} {'Status':>10}")
print(f"{'─'*65}")
for pair_name, data in pair_results.items():
    t  = data["trades"]
    p  = t["net_pnl"]
    wr = (p > 0).mean() * 100
    tpy = len(t) / (days_total.days / 365.25)
    sh  = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0.0
    ah  = t["holding_bars"].mean() / BARS_PER_TRADING_DAY
    disabled = t["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status = "DISABLED" if disabled else "active"
    print(f"{pair_name:<12} {len(t):>6} {wr:>5.1f}% {p.sum():>+10.4f} "
          f"{sh:>7.2f} {ah:>6.1f}d {status:>10}")

df_trades.to_csv(DATA_DIR / "trades.csv", index=False)
print(f"\nSaved {len(df_trades)} trades to {DATA_DIR / 'trades.csv'}")

# ── SPY benchmark ─────────────────────────────────────────────────────────────
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

# ── OOS summary ───────────────────────────────────────────────────────────────
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

# ── Charts ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
exit_times = pd.to_datetime(df_trades["exit_time"])

n_rows = 3 if spy_return is not None else 2
fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))

ax = axes[0]
ax.plot(exit_times, cumulative.values, color="blue", lw=2, label="Portfolio net P&L")
ax.plot(exit_times, df_trades["gross_pnl"].cumsum().values,
        color="blue", lw=1, linestyle="--", alpha=0.35, label="Gross P&L")
ax.axhline(PAIR_MAX_LOSS, color="red", linestyle=":", lw=1, alpha=0.5,
           label=f"Max loss cutoff ({PAIR_MAX_LOSS})")
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Portfolio Equity Curve  [OUT-OF-SAMPLE: "
             f"{closes.index[0].date()} → {closes.index[-1].date()}]")
ax.set_ylabel("Cumulative net P&L")
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

    # Strategy: normalise cumulative to % starting from 0
    first_trade_val = cumulative.values[0]
    strat_norm = (cumulative.values - first_trade_val) / max(abs(first_trade_val), 1) * 100

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
