import re
import os

def update_step3j():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()

    # Add scipy optimize and numpy imports if not present
    if 'scipy.optimize' not in content:
        content = content.replace('import pandas as pd', 'import pandas as pd\nimport scipy.optimize as opt')

    # Update _grid_kernel to take spread_mean and spread_std, cost_maker, cost_taker
    old_kernel_def = """def _grid_kernel(zscore, spread, t1_price, t2_price,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_per_side, borrow_rate, bars_per_day):"""
    new_kernel_def = """def _grid_kernel(zscore, spread, spread_mean, spread_std, t1_price, t2_price,
                 entry_arr, exit_arr, stop_arr,
                 beta, cost_maker, cost_taker, borrow_rate, bars_per_day):"""
    content = content.replace(old_kernel_def, new_kernel_def)

    # Update arrays in _grid_kernel
    old_arrays = """    pos   = np.zeros(n_c, dtype=np.int64)
    e_sp  = np.zeros(n_c); e_t1 = np.zeros(n_c)
    e_t2  = np.zeros(n_c); e_bar = np.zeros(n_c, dtype=np.int64)
    res   = np.zeros((n_c, 6)); cum = np.zeros(n_c); pk = np.zeros(n_c)"""
    new_arrays = """    pos   = np.zeros(n_c, dtype=np.int64)
    e_sp  = np.zeros(n_c); e_t1 = np.zeros(n_c)
    e_t2  = np.zeros(n_c); e_bar = np.zeros(n_c, dtype=np.int64)
    e_sma = np.zeros(n_c); e_std = np.zeros(n_c)
    res   = np.zeros((n_c, 6)); cum = np.zeros(n_c); pk = np.zeros(n_c)"""
    content = content.replace(old_arrays, new_arrays)

    # Update logic in _grid_kernel
    old_loop = """        z = zscore[i]; s = spread[i]; p1 = t1_price[i]; p2 = t2_price[i]
        for c in range(n_c):
            ez, xz, sz, pc = entry_arr[c], exit_arr[c], stop_arr[c], pos[c]
            if pc != 0:
                ex = (pc == 1 and z >= xz) or (pc == -1 and z <= -xz)
                st = (pc == 1 and z <= -sz) or (pc == -1 and z >= sz)
                if ex or st:
                    gross = pc * (s - e_sp[c])
                    notl  = e_t1[c] + beta * e_t2[c]
                    tx    = 2.0 * notl * cost_per_side"""
    new_loop = """        z = zscore[i]; s = spread[i]; sm = spread_mean[i]; sd = spread_std[i]
        p1 = t1_price[i]; p2 = t2_price[i]
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
                    tx    = notl * cost_maker + notl * (cost_maker if ex else cost_taker)"""
    content = content.replace(old_loop, new_loop)

    old_entry = """            if pos[c] == 0:
                if z < -ez:  pos[c] = 1;  e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i
                elif z > ez: pos[c] = -1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i"""
    new_entry = """            if pos[c] == 0:
                if z < -ez:
                    pos[c] = 1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd
                elif z > ez:
                    pos[c] = -1; e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i; e_sma[c] = sm; e_std[c] = sd"""
    content = content.replace(old_entry, new_entry)

    # run_grid call to _grid_kernel
    old_run_kernel = """    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),
        ea, xa, sa, float(beta),
        float(COST_PER_SIDE), float(BORROW_RATE_ANNUAL), float(BARS_PER_DAY)
    )"""
    new_run_kernel = """    res = _grid_kernel(
        np.ascontiguousarray(df["zscore"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_mean"].to_numpy(np.float64)),
        np.ascontiguousarray(df["spread_std"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64)),
        np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64)),
        ea, xa, sa, float(beta),
        float(COST_MAKER), float(COST_TAKER), float(BORROW_RATE_ANNUAL), float(BARS_PER_DAY)
    )"""
    content = content.replace(old_run_kernel, new_run_kernel)
    
    # Markowitz Optimization
    mvo_code = """
def optimize_portfolio_weights(returns_df: pd.DataFrame, max_weight: float = 0.15) -> dict:
    \"\"\"Markowitz Mean-Variance Optimization.
    returns_df: rows=dates, columns=pairs, values=daily PnL.
    \"\"\"
    if returns_df.empty or returns_df.shape[1] == 0:
        return {}
    if returns_df.shape[1] == 1:
        return {returns_df.columns[0]: 1.0}
        
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)
    
    def neg_sharpe(w):
        w = np.array(w)
        port_ret = np.dot(w, mu)
        port_var = np.dot(w.T, np.dot(Sigma, w))
        if port_var <= 1e-8:
            return 0.0
        # return negative sharpe (annualized approx not needed since relative)
        return - (port_ret / np.sqrt(port_var))
        
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, max_weight) for _ in range(n))
    init_w = np.full(n, 1.0 / n)
    
    res = opt.minimize(neg_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        return {col: 1.0 / n for col in returns_df.columns}
        
    weights = res.x
    weights[weights < 1e-4] = 0.0
    weights /= np.sum(weights) # re-normalize
    
    return {col: float(w) for col, w in zip(returns_df.columns, weights)}

def _trades_to_daily_pnl(trades: list, dates: pd.DatetimeIndex) -> pd.Series:
    pnl = pd.Series(0.0, index=dates.normalize().unique())
    for t in trades:
        d = t["exit_time"].normalize()
        if d in pnl:
            pnl[d] += t["net_pnl"]
    return pnl
"""
    if 'optimize_portfolio_weights' not in content:
        # insert after backtest_oos
        content = re.sub(r'(def backtest_oos[\s\S]*?return trades, hmm_blocked, hurst_blocked)', r'\1\n' + mvo_code, content)
        
    # Inject MVO logic into main WFO loop
    # We find:
    #                 best["hmm_blocked"] = 0
    #                 best["hurst_blocked"] = 0
    #                 window_best[pair_name] = best
    old_wfo_store = """                best["hmm_blocked"] = 0
                best["hurst_blocked"] = 0
                window_best[pair_name] = best"""
    new_wfo_store = """                best["hmm_blocked"] = 0
                best["hurst_blocked"] = 0
                # Get training trades for MVO
                t_trades, _, _ = backtest_oos(
                    sig_train, t1, t2, dynamic_beta,
                    best["entry_z"], best["exit_z"], best["stop_z"],
                    hmm_regime=None, hurst_filter=None
                )
                best["train_trades"] = t_trades
                window_best[pair_name] = best"""
    content = content.replace(old_wfo_store, new_wfo_store)
    
    # After all pairs are processed for the window, we do MVO
    old_mvo_trigger = """        if not window_best:
            print("  No viable pairs for this window.")
            cur += step
            continue

        window_oos_pnl = 0.0
        window_oos_trades = 0
        window_gross = 0.0
        window_tx = 0.0"""
    new_mvo_trigger = """        if not window_best:
            print("  No viable pairs for this window.")
            cur += step
            continue

        # ── Markowitz Portfolio Optimization ─────────────────────────────────────
        train_dates = closes_train.index
        pair_returns = {}
        for pair_name, best_params in window_best.items():
            s_pnl = _trades_to_daily_pnl(best_params["train_trades"], train_dates)
            pair_returns[pair_name] = s_pnl
        returns_df = pd.DataFrame(pair_returns).fillna(0.0)
        
        from config import MAX_PAIR_WEIGHT
        optimal_weights = optimize_portfolio_weights(returns_df, max_weight=MAX_PAIR_WEIGHT)
        
        # print top weights
        top_weights = {k: v for k, v in optimal_weights.items() if v > 0.01}
        print(f"  MVO Weights: {', '.join([f'{k}: {v:.1%}' for k, v in sorted(top_weights.items(), key=lambda x: -x[1])[:5]])} ...")
        
        window_oos_pnl = 0.0
        window_oos_trades = 0
        window_gross = 0.0
        window_tx = 0.0"""
    content = content.replace(old_mvo_trigger, new_mvo_trigger)
    
    # Finally, apply weights to OOS PnL
    #                 p_gross = sum(t["gross_pnl"] for t in oos_trades)
    #                 p_tx    = sum(t["tx_cost"] for t in oos_trades)
    old_oos_sum = """                p_pnl   = sum(t["net_pnl"] for t in oos_trades)
                p_gross = sum(t["gross_pnl"] for t in oos_trades)
                p_tx    = sum(t["tx_cost"] for t in oos_trades)

                window_oos_pnl    += p_pnl
                window_gross      += p_gross
                window_tx         += p_tx
                window_oos_trades += len(oos_trades)"""
    new_oos_sum = """                w_i = optimal_weights.get(pair_name, 0.0)
                if w_i < 1e-4:
                    continue # Optimizer discarded this pair
                    
                # We scale the PnL by the portfolio weight * N_pairs, so total capital is fully deployed
                # meaning if we have 20 pairs, and weight is 10%, we trade 2x normal size for this pair.
                N_active = len(optimal_weights)
                size_multiplier = w_i * N_active
                
                p_pnl   = sum(t["net_pnl"] for t in oos_trades) * size_multiplier
                p_gross = sum(t["gross_pnl"] for t in oos_trades) * size_multiplier
                p_tx    = sum(t["tx_cost"] for t in oos_trades) * size_multiplier

                window_oos_pnl    += p_pnl
                window_gross      += p_gross
                window_tx         += p_tx
                window_oos_trades += len(oos_trades)"""
    content = content.replace(old_oos_sum, new_oos_sum)
    
    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_step3j()
    print("WFO patched with MVO successfully.")
