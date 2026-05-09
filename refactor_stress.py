import re

def refactor_stress():
    with open('step5e_stress.py', 'r') as f:
        content = f.read()
        
    # 1. Add imports
    if 'from step3j_wfo import' not in content:
        content = content.replace('from filters import HurstFilter', 
                                 'from filters import HurstFilter\nfrom step3j_wfo import check_coint_johansen, compute_half_life')
                                 
    # 2. Update run_backtest signature
    old_run = """def run_backtest(df: pd.DataFrame, t1: str, t2: str, beta: float,
                 entry_z: float, exit_z: float, stop_z: float,
                 hurst_filter: HurstFilter = None) -> list[dict]:"""
    new_run = """def run_backtest(df: pd.DataFrame, t1: str, t2: str, beta: float,
                 entry_z: float, exit_z: float, stop_z: float,
                 hurst_filter: HurstFilter = None,
                 spread_daily: pd.Series = None) -> list[dict]:"""
    content = content.replace(old_run, new_run)
    
    # 3. Update Hurst check inside run_backtest
    old_hurst = """                # ── Hurst drift guard ─────────────────────────────
                if hurst_filter is not None:
                    h_tail = df["spread"].iloc[max(0, i - HURST_ENTRY_WINDOW): i + 1]
                    h_blocked, _ = hurst_filter.should_block(h_tail, df.index[i])
                    if h_blocked:
                        pos = 0
                        continue"""
    new_hurst = """                # ── Hurst drift guard ─────────────────────────────
                if hurst_filter is not None and spread_daily is not None:
                    ts = df.index[i]
                    d_prev = (ts - pd.Timedelta(days=1)).normalize()
                    h_tail_daily = spread_daily.loc[:d_prev].tail(HURST_ENTRY_WINDOW - 1)
                    h_tail = pd.concat([h_tail_daily, pd.Series({ts: df["spread"].iloc[i]})])
                    
                    h_blocked, _ = hurst_filter.should_block(h_tail, ts)
                    if h_blocked:
                        pos = 0
                        continue"""
    content = content.replace(old_hurst, new_hurst)
    
    # 4. Add load_daily
    if 'def load_daily' not in content:
        old_load = "def load_closes() -> pd.DataFrame:"
        new_load = """def load_daily() -> pd.DataFrame:
    path = DATA_DIR / "closes_daily.csv"
    if not path.exists(): return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df

def load_closes() -> pd.DataFrame:"""
        content = content.replace(old_load, new_load)
        
        # also call it
        content = content.replace("closes_all = load_closes()", "closes_all = load_closes()\ndaily_all = load_daily()")
        
    # 5. Replace the Walk-Forward loop
    # We will find the exact lines to replace
    old_loop_start = """# Build signals per pair once (full dataset)
pair_signals: dict = {}"""
    old_loop_end = """wf_df = pd.DataFrame(wf_rows)"""
    
    # Extract the chunk
    idx_start = content.find(old_loop_start)
    idx_end = content.find(old_loop_end)
    
    new_loop = """print("=" * 70)
print("WALK-FORWARD TEST  (Dynamic Beta)")
print("=" * 70)

wf_rows = []

for _, row in pairs.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    if t1 not in closes_all.columns or t2 not in closes_all.columns:
        continue
        
    start = closes_all.index[0]
    end   = closes_all.index[-1]
    w     = pd.DateOffset(months=WINDOW_MONTHS)
    step  = pd.DateOffset(months=STEP_MONTHS)
    cur   = start

    windows = []
    hf = HurstFilter()
    
    while cur + w <= end + pd.DateOffset(days=1):
        win_end = cur + w
        
        # Dynamic beta calculation over trailing 3 years
        train_start = cur - pd.DateOffset(years=3)
        daily_train = daily_all[(daily_all.index >= train_start) & (daily_all.index < cur)]
        
        dyn_beta = float(row.get("beta_daily", row["beta"]))
        dyn_hl = float(row["half_life_bars"])
        
        if len(daily_train) >= 100:
            is_coint, b = check_coint_johansen(daily_train, t1, t2, crit_level=0.90)
            if is_coint and b is not None and b > 0:
                dyn_beta = b
                spread_train = daily_train[t1] - dyn_beta * daily_train[t2]
                hl = compute_half_life(spread_train) * BARS_PER_DAY
                if 20 <= hl <= 500:
                    dyn_hl = hl

        # Build signals for the whole history (fast enough) to get correct rolling Z
        df_sig = build_signals(closes_all, t1, t2, dyn_beta, dyn_hl)
        win_df = df_sig[(df_sig.index >= cur) & (df_sig.index < win_end)]
        
        if len(win_df) > 200:
            # Daily spread using dynamic beta for Hurst
            spread_daily = (daily_all[t1] - dyn_beta * daily_all[t2]).dropna()
            windows.append((cur.date(), (win_end - pd.Timedelta(days=1)).date(), win_df, dyn_beta, spread_daily))
            
        cur += step

    if not windows:
        continue

    print(f"\\n  {pair_name}  ({len(windows)} windows)")
    print(f"  {'Window':<24} {'Trades':>7} {'WR':>6} {'Sharpe':>8} {'P&L':>10}")
    print(f"  {'─'*58}")

    for w_start, w_end, win_df, dyn_beta, spread_daily in windows:
        days_w = (pd.Timestamp(w_end) - pd.Timestamp(w_start)).days
        trades = run_backtest(win_df, t1, t2, dyn_beta, ENTRY_Z, EXIT_Z, STOP_Z, hf, spread_daily)
        m      = metrics(trades, days_w)
        flag   = "  ◄ GOOD" if m["sharpe"] > 1 else ("  ✗ BAD" if m["sharpe"] < -1 else "")
        print(f"  {str(w_start)} → {str(w_end)}  "
              f"{m['n']:>7}  {m['wr']:>5.1f}%  {m['sharpe']:>8.2f}  "
              f"{m['pnl']:>+10.4f}{flag}")
        wf_rows.append({"pair": pair_name, "window_start": str(w_start),
                        "window_end": str(w_end), **m})

"""
    content = content[:idx_start] + new_loop + content[idx_end:]
    
    # We also need to fix VIX REGIME BREAKDOWN since pair_signals is gone.
    # It loops over pair_signals in VIX REGIME BREAKDOWN:
    # `for pair_name, (df_sig, t1, t2, beta, _) in pair_signals.items():`
    # We can just build signals once with static beta for VIX regime breakdown since it's an aggregate,
    # or just use static beta for VIX breakdown to keep it simple.
    
    vix_loop_old = """    for pair_name, (df_sig, t1, t2, beta, _) in pair_signals.items():"""
    vix_loop_new = """    for _, row in pairs.iterrows():
        pair_name = row["pair"]
        t1, t2 = pair_name.split("-")
        beta = float(row.get("beta_daily", row["beta"]))
        half_life = float(row["half_life_bars"])
        if t1 not in closes_all.columns or t2 not in closes_all.columns:
            continue
        df_sig = build_signals(closes_all, t1, t2, beta, half_life)
        spread_daily = (daily_all[t1] - beta * daily_all[t2]).dropna()"""
    
    content = content.replace(vix_loop_old, vix_loop_new)
    
    # inside VIX loop, we need to pass spread_daily to run_backtest
    vix_bt_old = """            regime_trades = run_backtest(sub_df, t1, t2, beta, ENTRY_Z, EXIT_Z, STOP_Z, hf)"""
    vix_bt_new = """            regime_trades = run_backtest(sub_df, t1, t2, beta, ENTRY_Z, EXIT_Z, STOP_Z, hf, spread_daily)"""
    content = content.replace(vix_bt_old, vix_bt_new)
    
    # same for TODAY SIGNAL
    today_old = """for pair_name, (df_sig, t1, t2, beta, train_beta) in pair_signals.items():"""
    today_new = """for _, row in pairs.iterrows():
    pair_name = row["pair"]
    t1, t2 = pair_name.split("-")
    beta = float(row.get("beta_daily", row["beta"]))
    train_beta = float(row["beta"])
    half_life = float(row["half_life_bars"])
    if t1 not in closes_all.columns or t2 not in closes_all.columns:
        continue
    df_sig = build_signals(closes_all, t1, t2, beta, half_life)
    spread_daily = (daily_all[t1] - beta * daily_all[t2]).dropna()"""
    content = content.replace(today_old, today_new)
    
    with open('step5e_stress.py', 'w') as f:
        f.write(content)
        
if __name__ == "__main__":
    refactor_stress()
