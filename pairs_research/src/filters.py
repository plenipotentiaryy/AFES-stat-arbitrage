import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def compute_adv(volumes_df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Compute average daily dollar volume over the last `window` days."""
    if volumes_df.empty:
        return pd.Series()
    return volumes_df.iloc[-window:].mean()

def structural_break_check(
    prices: pd.DataFrame, t1: str, t2: str, 
    half_life: float, current_eg_pvalue: float
) -> tuple[bool, list[str]]:
    """
    Check for structural breaks:
    - Correlation dropped below 0.5 for >60 consecutive days in last 2 years
    - Half-life > 180 days
    - EG p-value > 0.15 in last 252D rolling window (proxy using current)
    """
    reasons = []
    
    if half_life > 180:
        reasons.append(f"Half-life too long ({half_life:.1f} > 180 days)")
        
    if current_eg_pvalue > 0.15:
        reasons.append(f"Current EG p-value too high ({current_eg_pvalue:.3f} > 0.15)")
        
    # Check correlation drop in last 2 years (~504 days)
    if t1 in prices.columns and t2 in prices.columns:
        s1 = prices[t1].iloc[-504:]
        s2 = prices[t2].iloc[-504:]
        valid_idx = s1.notna() & s2.notna()
        if valid_idx.sum() > 120:
            s1_v = s1[valid_idx]
            s2_v = s2[valid_idx]
            # Rolling 60d correlation
            corr_roll = s1_v.rolling(60).corr(s2_v)
            
            # Find max consecutive days where corr < 0.5
            is_low = corr_roll < 0.5
            max_consecutive = (is_low.groupby((~is_low).cumsum()).sum()).max()
            
            if max_consecutive > 60:
                reasons.append(f"Corr < 0.5 for {int(max_consecutive)} consecutive days in last 2 years")
                
    is_broken = len(reasons) > 0
    return is_broken, reasons
