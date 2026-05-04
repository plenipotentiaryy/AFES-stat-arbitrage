import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from hurst import compute_Hc

def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int) -> float:
    """Calculate the most recent rolling correlation over the given window."""
    valid_idx = s1.notna() & s2.notna()
    if valid_idx.sum() < window:
        return np.nan
    return s1[valid_idx].rolling(window).corr(s2[valid_idx]).iloc[-1]

def ols_hedge_ratio(s1: pd.Series, s2: pd.Series, window: int = 252) -> float:
    """Calculate OLS hedge ratio (beta) where s1 = beta * s2 over the last `window` days."""
    valid_idx = s1.notna() & s2.notna()
    if valid_idx.sum() < window:
        return np.nan
    y = s1[valid_idx].iloc[-window:]
    X = s2[valid_idx].iloc[-window:]
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        return model.params.iloc[1]
    except Exception:
        return np.nan

def engle_granger_test(s1: pd.Series, s2: pd.Series) -> float:
    """Run Engle-Granger cointegration test on log prices and return the p-value."""
    valid_idx = s1.notna() & s2.notna()
    if valid_idx.sum() < 252:
        return 1.0
    try:
        _, pvalue, _ = coint(s1[valid_idx], s2[valid_idx], maxlag=1)
        return pvalue
    except Exception:
        return 1.0

def adf_test(spread: pd.Series) -> tuple[float, float]:
    """Run ADF test on spread and return (stat, p-value)."""
    valid_spread = spread.dropna()
    if len(valid_spread) < 252:
        return 0.0, 1.0
    try:
        result = adfuller(valid_spread, maxlag=1)
        return result[0], result[1]
    except Exception:
        return 0.0, 1.0

def hurst_exponent(spread: pd.Series) -> float:
    """Calculate the Hurst exponent of the spread."""
    valid_spread = spread.dropna()
    if len(valid_spread) < 252:
        return 1.0
    try:
        H, c, data = compute_Hc(valid_spread.values, kind="price", simplified=True)
        return H
    except Exception:
        return 1.0

def half_life(spread: pd.Series) -> float:
    """Calculate the half-life of mean reversion using an Ornstein-Uhlenbeck (AR(1)) process fit."""
    valid_spread = spread.dropna()
    if len(valid_spread) < 252:
        return np.nan
    
    # y_t - y_{t-1} = \theta (\mu - y_{t-1}) + \epsilon
    # dy = \theta \mu - \theta y_{t-1} + \epsilon
    y_lag = valid_spread.shift(1).dropna()
    dy = valid_spread.diff().dropna()
    
    # Align indices
    common_idx = y_lag.index.intersection(dy.index)
    y_lag = y_lag.loc[common_idx]
    dy = dy.loc[common_idx]
    
    X = sm.add_constant(y_lag)
    try:
        model = sm.OLS(dy, X).fit()
        theta = -model.params.iloc[1]
        if theta <= 0 or theta >= 1:
            return 999.0  # Not mean-reverting or reverting too fast
        return np.log(2) / theta
    except Exception:
        return 999.0

def compute_all_metrics(prices: pd.DataFrame, t1: str, t2: str) -> dict:
    """Compute all required metrics for a single pair."""
    res = {
        "corr_60d": np.nan, "corr_120d": np.nan, "corr_252d": np.nan,
        "hedge_ratio": np.nan, "eg_pvalue": 1.0, "adf_stat": 0.0, "adf_pvalue": 1.0,
        "hurst": 1.0, "half_life": 999.0, "spread_vol": np.nan, "zscore": np.nan,
        "eg_pvalue_std": np.nan, "corr_120d_std": np.nan
    }
    
    if t1 not in prices.columns or t2 not in prices.columns:
        return res
        
    s1 = np.log(prices[t1])
    s2 = np.log(prices[t2])
    
    # Cointegration on full window
    res["eg_pvalue"] = engle_granger_test(s1, s2)
    
    # Correlations
    res["corr_60d"] = rolling_correlation(s1, s2, 60)
    res["corr_120d"] = rolling_correlation(s1, s2, 120)
    res["corr_252d"] = rolling_correlation(s1, s2, 252)
    
    # Beta and spread
    beta = ols_hedge_ratio(s1, s2, window=252)
    res["hedge_ratio"] = beta
    if np.isnan(beta):
        return res
        
    spread = s1 - beta * s2
    
    # ADF and Mean Reversion
    res["adf_stat"], res["adf_pvalue"] = adf_test(spread)
    res["hurst"] = hurst_exponent(spread)
    res["half_life"] = half_life(spread)
    
    # Volatility and z-score
    valid_spread = spread.dropna()
    if len(valid_spread) >= 60:
        roll_mean = valid_spread.rolling(60).mean()
        roll_std = valid_spread.rolling(60).std()
        res["spread_vol"] = roll_std.iloc[-1]
        
        current_spread = valid_spread.iloc[-1]
        if roll_std.iloc[-1] > 0:
            res["zscore"] = (current_spread - roll_mean.iloc[-1]) / roll_std.iloc[-1]
            
    # Stability metrics for scoring / structural break detection
    # Calculate rolling 252d EG p-value and rolling 120d corr over the last year
    # to measure standard deviation. This is computationally heavy if done per day,
    # so we'll do it sampling every 20 days.
    valid_idx = s1.notna() & s2.notna()
    if valid_idx.sum() > 500:
        s1_v = s1[valid_idx]
        s2_v = s2[valid_idx]
        
        rolling_pvals = []
        rolling_corrs = []
        
        # Sample last 252 days every 21 days
        start_idx = max(252, len(s1_v) - 252)
        for i in range(start_idx, len(s1_v), 21):
            window_s1 = s1_v.iloc[:i]
            window_s2 = s2_v.iloc[:i]
            if len(window_s1) >= 252:
                _, pval, _ = coint(window_s1.iloc[-252:], window_s2.iloc[-252:], maxlag=1)
                rolling_pvals.append(pval)
                corr = window_s1.iloc[-120:].corr(window_s2.iloc[-120:])
                rolling_corrs.append(corr)
                
        if rolling_pvals:
            res["eg_pvalue_std"] = np.nanstd(rolling_pvals)
        if rolling_corrs:
            res["corr_120d_std"] = np.nanstd(rolling_corrs)

    return res
