import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from step5e_stress import load_closes, build_signals, HurstFilter

def run():
    closes = load_closes()
    pairs = pd.read_csv("data/pairs_selected.csv", index_col=0)
    
    # Get COST-WMT
    pair_row = pairs[pairs["pair"] == "COST-WMT"].iloc[0]
    t1, t2 = "COST", "WMT"
    beta = float(pair_row.get("beta_daily", pair_row["beta"]))
    hl = float(pair_row["half_life_bars"])
    
    df_sig = build_signals(closes, t1, t2, beta, hl)
    
    # Focus on 2023
    df_sub = df_sig.loc["2023-01-01":"2023-12-31"].copy()
    
    hf = HurstFilter()
    hurst_vals = []
    
    for i in range(len(df_sub)):
        # Simulate what the backtester sees
        # We need the tail from df_sig, not df_sub, to get the full 200 bars at the start of 2023
        end_idx = df_sig.index.get_loc(df_sub.index[i])
        start_idx = max(0, end_idx - 200)
        tail = df_sig["spread"].iloc[start_idx : end_idx + 1]
        
        # We don't want to use the cache to see raw values
        s = tail.dropna().values
        if len(s) >= 30:
            h = hf.hurst_rs(s, max_lag=min(20, len(s)//4))
        else:
            h = 0.5
        hurst_vals.append(h)
        
    df_sub["hurst"] = hurst_vals
    
    print("COST-WMT 2023 Stats:")
    print("Mean Hurst:", df_sub["hurst"].mean())
    print("Max Hurst:", df_sub["hurst"].max())
    print("Min Hurst:", df_sub["hurst"].min())
    print(df_sub["hurst"].describe())

if __name__ == "__main__":
    run()
