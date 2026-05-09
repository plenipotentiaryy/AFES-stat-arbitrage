import pandas as pd
import numpy as np

def hurst_rs(series: np.ndarray, max_lag: int = 20) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    lags = range(2, min(max_lag + 1, n // 4))
    if len(list(lags)) < 3:
        return 0.5
    tau = [np.sqrt(np.mean((series[lag:] - series[:-lag])**2)) for lag in lags]
    if min(tau) <= 0:
        return 0.5
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

def run():
    closes = pd.read_csv("data/closes_15min.csv", index_col=0, parse_dates=True)
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index, utc=True)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes.index = closes.index.tz_convert("US/Eastern")
    closes = closes.between_time("09:30", "16:00")
    
    pairs = pd.read_csv("data/pairs_selected.csv", index_col=0)
    pair_row = pairs.loc["COST-WMT"]
    t1, t2 = "COST", "WMT"
    beta = float(pair_row.get("beta_daily", pair_row["beta"]))
    
    spread = closes[t1] - beta * closes[t2]
    spread = spread.dropna()
    
    sub = spread.loc["2023-01-01":"2023-12-31"]
    
    hurst_vals = []
    for i in range(len(sub)):
        # 200 bar tail
        end_idx = spread.index.get_loc(sub.index[i])
        start_idx = max(0, end_idx - 200)
        tail = spread.iloc[start_idx : end_idx + 1].values
        
        h = hurst_rs(tail, max_lag=20)
        hurst_vals.append(h)
        
    res = pd.Series(hurst_vals, index=sub.index)
    print("COST-WMT 2023 Hurst Stats:")
    print("Mean:", res.mean())
    print("Max:", res.max())
    print("Min:", res.min())
    print("Num > 0.55:", (res > 0.55).sum(), "out of", len(res))
    
    print("\nSample values:")
    print(res.tail(10))

if __name__ == "__main__":
    run()
