import re
import os

def update_config():
    with open('config.py', 'r') as f:
        content = f.read()

    # If already updated, skip
    if 'COST_MAKER' in content:
        return

    old_cost = """COST_PER_SIDE   = COST_COMMISSION + COST_SPREAD + COST_SLIPPAGE  # 0.07% per side"""
    new_cost = """# Passive Aggressor: Limit orders inside spread for entries and TP (earn spread/rebate)
COST_MAKER      = COST_COMMISSION                           # 0.03%
# Taker: Market orders for stop-loss and macro panics
COST_TAKER      = COST_COMMISSION + COST_SPREAD + COST_SLIPPAGE  # 0.07%

# ── Idiosyncratic Circuit Breaker ────────────────────────────────────────────
CIRCUIT_BREAKER_Z = 4.5  # Max Z-score before immediate hard-stop and pair block"""
    
    if old_cost in content:
        content = content.replace(old_cost, new_cost)
        with open('config.py', 'w') as f:
            f.write(content)

def update_step4a():
    with open('step4a_backtest.py', 'r') as f:
        content = f.read()

    # 1. Imports
    if 'COST_MAKER' not in content:
        content = content.replace('COST_PER_SIDE', 'COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z')

    # 2. entry_beta is already saved as entry_beta = df["beta"].iloc[next_i]
    # We need entry_std = df["spread_std"].iloc[next_i]
    if 'entry_std = 0.0' not in content:
        content = content.replace('entry_beta = 0.0', 'entry_beta = entry_std = 0.0')

    if 'entry_std      = df["spread_std"].iloc[next_i]' not in content:
        content = content.replace('entry_beta     = df["beta"].iloc[next_i]',
                                  'entry_beta     = df["beta"].iloc[next_i]\n                entry_std      = df["spread_std"].iloc[next_i]')

    # 3. Inside position != 0: calculate z_active
    old_pos_check = """        # ── Macro force-close (K-Means Panic or global HMM panic) ────────"""
    new_pos_check = """        # ── Static Baseline Z-score (Combatting Kalman Illusion) ─────────
        z_active = z
        if position != 0 and entry_std > 0:
            # Frozen beta static spread
            static_spread = p1 - entry_beta * p2
            z_active = static_spread / entry_std
            
        # ── Idiosyncratic Circuit Breaker ────────────────────────────────
        if position != 0 and abs(z_active) >= CIRCUIT_BREAKER_Z:
            force_close = True
            suspended = True # Block pair permanently for the window
        else:
            # ── Macro force-close (K-Means Panic or global HMM panic) ────────
            force_close = (
                suspended
                or (macro_filter is not None and macro_filter.is_force_close(ts))
            )"""
    if 'z_active = z' not in content:
        # replace the force_close logic block
        content = re.sub(
            r'        # ── Macro force-close \(K-Means Panic or global HMM panic\) ────────\n        force_close = \(\n            suspended\n            or \(macro_filter is not None and macro_filter\.is_force_close\(ts\)\)\n        \)',
            new_pos_check,
            content
        )

    # 4. Transaction costs
    # In force_close:
    if 'tx_cost        = 2 * notional * COST_PER_SIDE' in content:
        content = content.replace('tx_cost        = 2 * notional * COST_PER_SIDE',
                                  'tx_cost        = notional * COST_MAKER + notional * COST_TAKER')
                                  
    # In normal exit:
    old_normal_exit = """        # ── Normal exit / stop / time-stop ───────────────────────────────
        if position != 0:
            bars_held   = i - entry_bar
            time_stop   = bars_held >= max_hold_bars
            exit_signal = (not time_stop) and (
                (position == 1 and z >= active_exit_thresh) or (position == -1 and z <= -active_exit_thresh))
            stop_signal = (not time_stop) and (
                (position == 1 and z <= -active_stop_thresh) or (position == -1 and z >= active_stop_thresh))"""
    
    new_normal_exit = """        # ── Normal exit / stop / time-stop ───────────────────────────────
        if position != 0:
            bars_held   = i - entry_bar
            time_stop   = bars_held >= max_hold_bars
            exit_signal = (not time_stop) and (
                (position == 1 and z_active >= active_exit_thresh) or (position == -1 and z_active <= -active_exit_thresh))
            stop_signal = (not time_stop) and (
                (position == 1 and z_active <= -active_stop_thresh) or (position == -1 and z_active >= active_stop_thresh))"""
    if 'z_active >=' not in content:
        content = content.replace(old_normal_exit, new_normal_exit)

    # replace tx cost in normal exit
    # We need to distinguish between exit (maker) and stop/time_stop (taker)
    old_tx_calc = """            if exit_signal or stop_signal or time_stop:
                n              = entry_n_shares
                gross_pnl      = position * (spread_now - entry_spread) * n
                notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
                tx_cost        = 2 * notional * COST_PER_SIDE"""
                
    new_tx_calc = """            if exit_signal or stop_signal or time_stop:
                n              = entry_n_shares
                gross_pnl      = position * (spread_now - entry_spread) * n
                notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
                # Maker entry + Maker exit (if TP), otherwise Taker exit
                tx_cost        = notional * COST_MAKER + notional * (COST_MAKER if exit_signal else COST_TAKER)"""
    if 'tx_cost        = notional * COST_MAKER' not in content:
         content = content.replace(old_tx_calc, new_tx_calc)
         
    with open('step4a_backtest.py', 'w') as f:
        f.write(content)

def update_step3j():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()
        
    if 'CIRCUIT_BREAKER_Z' not in content:
        content = content.replace('from config import (', 'from config import (\n    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z,')
    
    # build_signals to add spread_mean and spread_std
    old_build = """    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
    })"""
    new_build = """    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_mean) / spread_std
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
        "spread_mean": spread_mean, "spread_std": spread_std
    })"""
    content = content.replace(old_build, new_build)
    
    # In backtest_oos
    old_bt = """    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = 0.0; ebar = 0; trades = []; hmm_blocked = 0; hurst_blocked = 0
    for i in range(len(df)):
        z = df["zscore"].iloc[i]; s = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i];    p2 = df[t2c].iloc[i]
        if pos != 0:
            ex = (pos == 1 and z >= exit_z)  or (pos == -1 and z <= -exit_z)
            st = (pos == 1 and z <= -stop_z) or (pos == -1 and z >= stop_z)
            if ex or st:
                gross = pos * (s - es)
                notl  = et1 + beta * et2
                tx    = 2 * notl * COST_PER_SIDE"""
                
    new_bt = """    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = entry_sma = entry_std = 0.0
    ebar = 0; trades = []; hmm_blocked = 0; hurst_blocked = 0
    pair_blocked = False
    
    for i in range(len(df)):
        if pair_blocked: continue
        z = df["zscore"].iloc[i]; s = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i];    p2 = df[t2c].iloc[i]
        
        z_active = z
        if pos != 0 and entry_std > 0:
            z_active = (s - entry_sma) / entry_std
            
        if pos != 0:
            ex = (pos == 1 and z_active >= exit_z)  or (pos == -1 and z_active <= -exit_z)
            st = (pos == 1 and z_active <= -stop_z) or (pos == -1 and z_active >= stop_z)
            cb = abs(z_active) >= CIRCUIT_BREAKER_Z
            if ex or st or cb:
                if cb:
                    pair_blocked = True
                    st = True # force taker cost
                gross = pos * (s - es)
                notl  = et1 + beta * et2
                tx    = notl * COST_MAKER + notl * (COST_MAKER if ex and not cb else COST_TAKER)"""
    
    content = content.replace(old_bt, new_bt)
    
    # Save es, entry_sma, entry_std
    old_es = """                es = s; et1 = p1; et2 = p2; ebar = i"""
    new_es = """                es = s; et1 = p1; et2 = p2; ebar = i
                entry_sma = df["spread_mean"].iloc[i]
                entry_std = df["spread_std"].iloc[i]"""
    content = content.replace(old_es, new_es)
    
    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

def update_step5e():
    with open('step5e_stress.py', 'r') as f:
        content = f.read()

    if 'CIRCUIT_BREAKER_Z' not in content:
        content = content.replace('from config import (', 'from config import (\n    COST_MAKER, COST_TAKER, CIRCUIT_BREAKER_Z,')
        
    old_build = """    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
    })"""
    new_build = """    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_mean) / spread_std
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread, "zscore": zscore,
        "spread_mean": spread_mean, "spread_std": spread_std
    })"""
    content = content.replace(old_build, new_build)

    old_bt = """    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = entry_spread = entry_t1 = entry_t2 = 0.0
    entry_bar = 0
    trades = []

    for i in range(len(df)):
        z  = df["zscore"].iloc[i]
        s  = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i]
        p2 = df[t2c].iloc[i]

        if pos != 0:
            ex = (pos == 1 and z >= exit_z)  or (pos == -1 and z <= -exit_z)
            st = (pos == 1 and z <= -stop_z) or (pos == -1 and z >= stop_z)
            if ex or st:
                gross      = pos * (s - entry_spread)
                notional   = entry_t1 + beta * entry_t2
                tx         = 2 * notional * COST_PER_SIDE"""
                
    new_bt = """    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = entry_spread = entry_t1 = entry_t2 = entry_sma = entry_std = 0.0
    entry_bar = 0
    trades = []
    pair_blocked = False

    for i in range(len(df)):
        if pair_blocked: continue
        z  = df["zscore"].iloc[i]
        s  = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i]
        p2 = df[t2c].iloc[i]
        
        z_active = z
        if pos != 0 and entry_std > 0:
            z_active = (s - entry_sma) / entry_std

        if pos != 0:
            ex = (pos == 1 and z_active >= exit_z)  or (pos == -1 and z_active <= -exit_z)
            st = (pos == 1 and z_active <= -stop_z) or (pos == -1 and z_active >= stop_z)
            cb = abs(z_active) >= CIRCUIT_BREAKER_Z
            if ex or st or cb:
                if cb:
                    pair_blocked = True
                    st = True
                gross      = pos * (s - entry_spread)
                notional   = entry_t1 + beta * entry_t2
                tx         = notional * COST_MAKER + notional * (COST_MAKER if ex and not cb else COST_TAKER)"""
    content = content.replace(old_bt, new_bt)
    
    old_es = """                entry_spread = s; entry_t1 = p1; entry_t2 = p2; entry_bar = i"""
    new_es = """                entry_spread = s; entry_t1 = p1; entry_t2 = p2; entry_bar = i
                entry_sma = df["spread_mean"].iloc[i]
                entry_std = df["spread_std"].iloc[i]"""
    content = content.replace(old_es, new_es)
    
    with open('step5e_stress.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_config()
    update_step4a()
    update_step3j()
    update_step5e()
    print("Updates applied successfully.")
