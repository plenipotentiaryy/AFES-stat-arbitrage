import pandas as pd
import numpy as np

def calculate_drawdown(pnl: pd.Series) -> float:
    """Calculate maximum drawdown from PnL series."""
    cumulative = pnl.cumsum()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak)
    return drawdown.min()

def run_mini_backtest(
    prices: pd.DataFrame, 
    t1: str, 
    t2: str, 
    beta: float, 
    entry_z: float = 2.0, 
    exit_z: float = 0.0, 
    stop_z: float = 3.5,
    transaction_bps: float = 0.0,
    capital_per_leg: float = 100_000.0
) -> dict:
    """
    Run a simple z-score backtest on daily data.
    """
    if t1 not in prices.columns or t2 not in prices.columns:
        return {}
        
    s1 = prices[t1].dropna()
    s2 = prices[t2].dropna()
    
    # Align indices
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) < 252:
        return {}
        
    s1 = s1.loc[common_idx]
    s2 = s2.loc[common_idx]
    
    # Spread and Z-Score (using 60d rolling window for mean and std)
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    spread = log_s1 - beta * log_s2
    
    roll_mean = spread.rolling(60).mean()
    roll_std = spread.rolling(60).std()
    
    zscore = (spread - roll_mean) / roll_std
    
    # Backtest logic
    position = 0 # 1 means long spread (long t1, short t2), -1 means short spread
    entry_price_1 = 0.0
    entry_price_2 = 0.0
    
    trades = []
    pnl_series = []
    
    for i in range(len(common_idx)):
        if pd.isna(zscore.iloc[i]):
            pnl_series.append(0.0)
            continue
            
        z = zscore.iloc[i]
        p1 = s1.iloc[i]
        p2 = s2.iloc[i]
        
        daily_pnl = 0.0
        
        # Mark to market current position
        if position != 0:
            qty1 = capital_per_leg / entry_price_1 * position
            qty2 = (capital_per_leg / entry_price_2) * (-position) # Neutral
            
            # Daily P&L
            daily_pnl = qty1 * (p1 - s1.iloc[i-1]) + qty2 * (p2 - s2.iloc[i-1])
            
            # Check exit or stop
            is_exit = (position == 1 and z >= exit_z) or (position == -1 and z <= exit_z)
            is_stop = (position == 1 and z <= -stop_z) or (position == -1 and z >= stop_z)
            
            if is_exit or is_stop:
                # Close position
                cost = (capital_per_leg * 2) * (transaction_bps / 10000.0)
                daily_pnl -= cost
                
                # Record trade
                trade_pnl = qty1 * (p1 - entry_price_1) + qty2 * (p2 - entry_price_2) - cost
                trades.append({
                    "pnl": trade_pnl,
                    "is_win": trade_pnl > 0,
                    "hold_time": i - entry_idx,
                    "type": "exit" if is_exit else "stop"
                })
                position = 0
                
        # Check entry
        if position == 0:
            if z <= -entry_z:
                position = 1
                entry_price_1 = p1
                entry_price_2 = p2
                entry_idx = i
                # Entry cost
                cost = (capital_per_leg * 2) * (transaction_bps / 10000.0)
                daily_pnl -= cost
            elif z >= entry_z:
                position = -1
                entry_price_1 = p1
                entry_price_2 = p2
                entry_idx = i
                cost = (capital_per_leg * 2) * (transaction_bps / 10000.0)
                daily_pnl -= cost
                
        pnl_series.append(daily_pnl)
        
    pnl = pd.Series(pnl_series, index=common_idx)
    
    # Calculate metrics
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "avg_hold": 0.0}
        
    total_pnl = pnl.sum()
    cagr = (total_pnl / (capital_per_leg * 2)) / (len(common_idx) / 252)
    
    daily_returns = pnl / (capital_per_leg * 2)
    sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-9)
    
    max_dd = calculate_drawdown(pnl) / (capital_per_leg * 2)
    
    win_rate = sum(1 for t in trades if t["is_win"]) / len(trades)
    avg_hold = sum(t["hold_time"] for t in trades) / len(trades)
    
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_hold": avg_hold
    }
