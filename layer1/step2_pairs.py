import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

# 1. Load Data
closes = pd.read_csv("data/closes_15min.csv", index_col=0, parse_dates=True)

# 2. Resample to Daily (Take the last price of the day)
# This converts your 15min data into clean Daily candles
closes_daily = closes.resample('D').last().dropna()

print(f"Resampled to Daily: {closes_daily.shape[0]} days, {closes_daily.shape[1]} tickers")

# 3. Use Log Prices (Crucial for Cointegration)
log_prices = np.log(closes_daily)

# 4. Correlation Filter (Using Log Returns)
log_returns = log_prices.diff().dropna()
corr_matrix = log_returns.corr()

CORR_THRESHOLD = 0.5
tickers = list(log_prices.columns)
pairs_corr = []

for t1, t2 in combinations(tickers, 2):
    corr = corr_matrix.loc[t1, t2]
    if corr > CORR_THRESHOLD:
        pairs_corr.append((t1, t2, corr))

# 5. Testing Cointegration on Daily Log Prices
pairs_selected = []

print(f"\nTesting {len(pairs_corr)} pairs for Cointegration...")

for t1, t2, corr in pairs_corr:
    # We use 'c' for trend to allow for a constant intercept in the relationship
    score, pvalue, _ = coint(log_prices[t1], log_prices[t2], trend='c')

    if pvalue < 0.05:
        # Calculate Beta via OLS on Log Prices
        X = sm.add_constant(log_prices[t2])
        model = sm.OLS(log_prices[t1], X).fit()
        beta = model.params.iloc[1]
        
        # Calculate Half-Life (How fast it reverts)
        spread = log_prices[t1] - beta * log_prices[t2]
        z_lag = spread.shift(1)
        z_diff = spread.diff()
        reg_df = pd.DataFrame({'y': z_diff, 'x': z_lag}).dropna()
        
        hl_model = sm.OLS(reg_df['y'], sm.add_constant(reg_df['x'])).fit()
        lambda_val = hl_model.params['x']
        
        # Half-life formula: -log(2) / lambda
        if lambda_val < 0:
            half_life = -np.log(2) / lambda_val
        else:
            half_life = np.nan # Doesn't revert

        pairs_selected.append({
            "pair": f"{t1}-{t2}",
            "corr": round(corr, 4),
            "p_val": round(pvalue, 4),
            "beta": round(beta, 4),
            "half_life_days": round(half_life, 1) if not np.isnan(half_life) else "Inf"
        })
        print(f"  ✓ Found: {t1}-{t2} (p={pvalue:.4f})")

# Output Results
if pairs_selected:
    df_results = pd.DataFrame(pairs_selected)
    print("\n" + "="*50)
    print("COINTEGRATED PAIRS (DAILY)")
    print("="*50)
    print(df_results.to_string(index=False))
    df_results.to_csv("data/pairs_selected_daily.csv", index=False)
else:
    print("\nNo pairs found. Try increasing your historical data range.")