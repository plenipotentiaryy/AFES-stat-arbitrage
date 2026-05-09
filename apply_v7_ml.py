import re

def update_wfo():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()

    # Import sklearn inside step3j_wfo.py
    if 'from sklearn.ensemble import HistGradientBoostingClassifier' not in content:
        content = content.replace('import scipy.optimize as opt', 
                                  'import scipy.optimize as opt\nfrom sklearn.ensemble import HistGradientBoostingClassifier')

    # Update backtest_oos to record features
    old_pos0 = """        if pos == 0 and vr >= 0.5:
            # ── Macro HMM gate: block new entries during panic ────────────
            if hmm_regime is not None:
                bar_date = df.index[i].normalize().tz_localize(None)
                try:
                    hmm_val = hmm_regime.asof(bar_date)
                    if not pd.isna(hmm_val) and int(hmm_val) == 1:
                        hmm_blocked += 1
                        continue  # sit on the fence — don't enter during panic
                except Exception:
                    pass
            
            # ── Hurst gate: block entries if spread is trending (H > threshold) ──
            if hurst_filter is not None:
                bar_date = df.index[i].normalize().tz_localize(None)
                if spread_daily is not None:
                    # we pass the history up to 'today'
                    tail = spread_daily.loc[:bar_date]
                    if not hurst_filter.check(tail, bar_date):
                        hurst_blocked += 1
                        continue

            if z < -entry_z:
                pos = 1; es = s; et1 = p1; et2 = p2; ebar = i; entry_sma = df["spread_mean"].iloc[i]; entry_std = df["spread_std"].iloc[i]
            elif z > entry_z:
                pos = -1; es = s; et1 = p1; et2 = p2; ebar = i; entry_sma = df["spread_mean"].iloc[i]; entry_std = df["spread_std"].iloc[i]"""
                
    new_pos0 = """        if pos == 0 and vr >= 0.5:
            # ── Macro HMM gate: block new entries during panic ────────────
            if hmm_regime is not None:
                bar_date = df.index[i].normalize().tz_localize(None)
                try:
                    hmm_val = hmm_regime.asof(bar_date)
                    if not pd.isna(hmm_val) and int(hmm_val) == 1:
                        hmm_blocked += 1
                        continue  # sit on the fence — don't enter during panic
                except Exception:
                    pass
            
            # ── Hurst gate: block entries if spread is trending (H > threshold) ──
            if hurst_filter is not None:
                bar_date = df.index[i].normalize().tz_localize(None)
                if spread_daily is not None:
                    # we pass the history up to 'today'
                    tail = spread_daily.loc[:bar_date]
                    if not hurst_filter.check(tail, bar_date):
                        hurst_blocked += 1
                        continue

            if z < -entry_z:
                pos = 1; es = s; et1 = p1; et2 = p2; ebar = i; entry_sma = df["spread_mean"].iloc[i]; entry_std = df["spread_std"].iloc[i]
            elif z > entry_z:
                pos = -1; es = s; et1 = p1; et2 = p2; ebar = i; entry_sma = df["spread_mean"].iloc[i]; entry_std = df["spread_std"].iloc[i]
                
            if pos != 0:
                e_features = [
                    abs(z),
                    vr,
                    entry_std,
                    df.index[i].hour + df.index[i].minute / 60.0
                ]"""
    if 'e_features =' not in content:
        content = content.replace(old_pos0, new_pos0)

    # Append features to trades
    old_trades_append = """                trades.append({
                    "entry_time": df.index[ebar],
                    "exit_time":  df.index[i],
                    "direction":  "LONG" if pos == 1 else "SHORT",
                    "net_pnl":    round(gross - tx - brw, 4),
                    "exit_reason": "STOP" if st else "SIGNAL",
                    "holding_bars": i - ebar,
                    "gross_pnl":  round(gross, 4),
                    "tx_cost":    round(tx, 4)
                })"""
    new_trades_append = """                trades.append({
                    "entry_time": df.index[ebar],
                    "exit_time":  df.index[i],
                    "direction":  "LONG" if pos == 1 else "SHORT",
                    "net_pnl":    round(gross - tx - brw, 4),
                    "exit_reason": "STOP" if st else "SIGNAL",
                    "holding_bars": i - ebar,
                    "gross_pnl":  round(gross, 4),
                    "tx_cost":    round(tx, 4),
                    "features":   e_features
                })"""
    if '"features":   e_features' not in content:
        content = content.replace(old_trades_append, new_trades_append)

    # Update WFO loop to train ML model
    old_train_save = """                best["train_trades"] = t_trades
                window_best[pair_name] = best"""
    new_train_save = """                best["train_trades"] = t_trades
                
                # Train ML Model
                best["ml_model"] = None
                if len(t_trades) >= 20:
                    import numpy as np
                    X = np.array([t["features"] for t in t_trades])
                    y = np.array([1 if t["net_pnl"] > 0 else 0 for t in t_trades])
                    if len(np.unique(y)) > 1: # Need both wins and losses to train
                        clf = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=5, learning_rate=0.05, max_iter=50)
                        clf.fit(X, y)
                        best["ml_model"] = clf
                        
                window_best[pair_name] = best"""
    if 'best["ml_model"] = None' not in content:
        content = content.replace(old_train_save, new_train_save)

    # Update OOS loop to apply ML Sizing
    old_oos_sum = """                w_i = optimal_weights.get(pair_name, 0.0)
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
    new_oos_sum = """                w_i = optimal_weights.get(pair_name, 0.0)
                if w_i < 1e-4:
                    continue # Optimizer discarded this pair
                    
                N_active = len(optimal_weights)
                size_multiplier = w_i * N_active
                
                p_pnl, p_gross, p_tx, p_trades = 0.0, 0.0, 0.0, 0
                clf = window_best[pair_name].get("ml_model")
                
                for t in oos_trades:
                    ml_multiplier = 1.0
                    if clf is not None:
                        import numpy as np
                        x = np.array([t["features"]])
                        prob = clf.predict_proba(x)[0][1] # Probability of class 1 (Win)
                        ml_multiplier = max(0.0, 2.0 * (prob - 0.5))
                        
                    if ml_multiplier > 0.0:
                        final_multiplier = size_multiplier * ml_multiplier
                        p_pnl += t["net_pnl"] * final_multiplier
                        p_gross += t["gross_pnl"] * final_multiplier
                        p_tx += t["tx_cost"] * final_multiplier
                        p_trades += 1

                window_oos_pnl    += p_pnl
                window_gross      += p_gross
                window_tx         += p_tx
                window_oos_trades += p_trades"""
    if 'clf = window_best[pair_name].get("ml_model")' not in content:
        content = content.replace(old_oos_sum, new_oos_sum)

    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_wfo()
    print("V7 patches (ML-Driven Sizing) applied successfully.")
