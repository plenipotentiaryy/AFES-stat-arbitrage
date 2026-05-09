import re
import os

def update_step3j():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()

    # Update load_closes
    old_load_closes = """def load_closes():
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    daily_path = DATA_DIR / "closes_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0)
        daily.index = pd.to_datetime(daily.index, utc=True).tz_convert("US/Eastern")
    else:
        daily = None

    return closes, daily"""
    
    new_load_closes = """def load_closes():
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)
    
    vol_path = DATA_DIR / "volumes_15min.csv"
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

    return closes, daily, volumes"""
    content = content.replace(old_load_closes, new_load_closes)

    # Update global call
    content = content.replace('closes, daily = load_closes()', 'closes, daily, volumes = load_closes()')
    
    # Update build_signals
    old_build_signals = """def build_signals(closes, t1, t2, beta, half_life):
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_mean) / spread_std
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
        "spread_mean": spread_mean, "spread_std": spread_std
    }).dropna().between_time(SIGNAL_START, RTH_END)"""
    
    new_build_signals = """def build_signals(closes, volumes, t1, t2, beta, half_life):
    c1 = closes[t1]
    c2 = closes[t2]
    spread = c1 - beta * c2
    window = max(20, min(int(half_life), 200))
    
    if volumes is not None and t1 in volumes.columns and t2 in volumes.columns:
        v1 = volumes[t1]
        v2 = volumes[t2]
        v_spread = np.minimum(v1 * c1, v2 * c2)
        v_sum = v_spread.rolling(window).sum()
        # VW-SMA
        spread_mean = (v_spread * spread).rolling(window).sum() / v_sum
        # VW-Std (using E[X^2] - E[X]^2)
        spread_sq_mean = (v_spread * (spread ** 2)).rolling(window).sum() / v_sum
        spread_var = (spread_sq_mean - (spread_mean ** 2)).clip(lower=0)
        spread_std = np.sqrt(spread_var)
        # Liquidity Filter (VR)
        v_sma = v_spread.rolling(window).mean()
        vr = v_spread / v_sma
    else:
        # Fallback if volume is missing
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std()
        vr = pd.Series(1.0, index=spread.index)
        
    zscore = (spread - spread_mean) / spread_std
    
    return pd.DataFrame({
        f"{t1}_close": c1,
        f"{t2}_close": c2,
        "spread": spread, "zscore": zscore,
        "spread_mean": spread_mean, "spread_std": spread_std,
        "vr": vr
    }).dropna().between_time(SIGNAL_START, RTH_END)"""
    content = content.replace(old_build_signals, new_build_signals)
    
    # Update backtest_oos def and loop
    old_bt = """    for i in range(len(df)):
        if pair_blocked: continue
        z = df["zscore"].iloc[i]; s = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i];    p2 = df[t2c].iloc[i]"""
    new_bt = """    for i in range(len(df)):
        if pair_blocked: continue
        z = df["zscore"].iloc[i]; s = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i];    p2 = df[t2c].iloc[i]
        vr = df["vr"].iloc[i]"""
    content = content.replace(old_bt, new_bt)
    
    old_pos0 = """        if pos == 0:
            # ── Macro HMM gate: block new entries during panic ────────────"""
    new_pos0 = """        if pos == 0 and vr >= 0.5:
            # ── Macro HMM gate: block new entries during panic ────────────"""
    content = content.replace(old_pos0, new_pos0)

    # Update _grid_kernel def and loop
    old_kernel_def = """def _grid_kernel(zscore, spread, spread_mean, spread_std, t1_price, t2_price,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_maker, cost_taker, borrow_rate, bars_per_day):"""
    new_kernel_def = """def _grid_kernel(zscore, spread, spread_mean, spread_std, vr_arr, t1_price, t2_price,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_maker, cost_taker, borrow_rate, bars_per_day):"""
    content = content.replace(old_kernel_def, new_kernel_def)
    
    old_kernel_loop = """        z = zscore[i]; s = spread[i]; sm = spread_mean[i]; sd = spread_std[i]
        p1 = t1_price[i]; p2 = t2_price[i]"""
    new_kernel_loop = """        z = zscore[i]; s = spread[i]; sm = spread_mean[i]; sd = spread_std[i]
        vr = vr_arr[i]; p1 = t1_price[i]; p2 = t2_price[i]"""
    content = content.replace(old_kernel_loop, new_kernel_loop)
    
    old_k_pos0 = """            if pos[c] == 0:
                if z < -ez:
                    pos[c] = 1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd
                elif z > ez:
                    pos[c] = -1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd"""
    new_k_pos0 = """            if pos[c] == 0 and vr >= 0.5:
                if z < -ez:
                    pos[c] = 1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd
                elif z > ez:
                    pos[c] = -1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd"""
    content = content.replace(old_k_pos0, new_k_pos0)
    
    old_run_kernel = """    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_mean"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_std"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),"""
    new_run_kernel = """    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_mean"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_std"].to_numpy(np.float64)),
        np.ascontiguousarray(df["vr"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),"""
    content = content.replace(old_run_kernel, new_run_kernel)
    
    # Update build_signals calls
    # sig_train = build_signals(closes_train, t1, t2, dynamic_beta, half_life)
    content = content.replace('build_signals(closes_train, t1, t2,', 'build_signals(closes_train, volumes, t1, t2,')
    content = content.replace('build_signals(closes_test, t1, t2,', 'build_signals(closes_test, volumes, t1, t2,')

    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_step3j()
    print("V5 patches (VW-Z and Liquidity Void Filter) applied successfully.")
