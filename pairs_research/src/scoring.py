import numpy as np
import math

def calculate_composite_score(
    eg_pvalue: float,
    eg_pvalue_std: float,
    half_life: float,
    adv_1: float,
    adv_2: float,
    corr_std: float,
    hurst: float,
    priority: int
) -> float:
    """Calculate composite score (0-100)."""
    score = 0.0
    
    # 25% -> EG p-value
    if not np.isnan(eg_pvalue):
        eg_score = max(0.0, min(100.0, 100.0 - eg_pvalue * 1000.0))
        score += 0.25 * eg_score
        
    # 20% -> EG p-value stability
    if not np.isnan(eg_pvalue_std):
        eg_std_score = max(0.0, min(100.0, 100.0 - eg_pvalue_std * 500.0))
        score += 0.20 * eg_std_score
    else:
        # Penalize if missing
        pass
        
    # 15% -> Half-life attractiveness
    if not np.isnan(half_life):
        if 5 <= half_life <= 60:
            hl_score = 100.0
        elif 60 < half_life <= 180:
            hl_score = max(0.0, 100.0 - (half_life - 60) / 120.0 * 100.0)
        else:
            hl_score = 0.0
        score += 0.15 * hl_score
        
    # 15% -> Liquidity (min of both legs)
    min_adv = min(adv_1, adv_2)
    if min_adv > 0 and not np.isnan(min_adv):
        # 50M -> 0, 1B -> 100
        log_adv = math.log10(min_adv)
        liq_score = max(0.0, min(100.0, (log_adv - 7.69) / (9.0 - 7.69) * 100.0))
        score += 0.15 * liq_score
        
    # 10% -> Correlation stability
    if not np.isnan(corr_std):
        corr_std_score = max(0.0, min(100.0, 100.0 - corr_std * 200.0))
        score += 0.10 * corr_std_score
        
    # 10% -> Hurst exponent
    if not np.isnan(hurst):
        # 0.5 -> 0, 0.2 -> 100
        hurst_score = max(0.0, min(100.0, (0.5 - hurst) / 0.3 * 100.0))
        score += 0.10 * hurst_score
        
    # 5% -> Priority
    if priority == 1:
        score += 0.05 * 100.0
        
    return round(score, 2)
