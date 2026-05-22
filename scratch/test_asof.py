import pandas as pd
import numpy as np

# Create daily series
idx_daily = pd.date_range("2026-05-01", "2026-05-10", freq="D")
s_daily = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], index=idx_daily)

# Create intraday timestamps (hourly)
idx_intraday = pd.date_range("2026-05-02 09:00:00", "2026-05-08 17:00:00", freq="h")

# Normalize to tz-naive date
dates_naive = idx_intraday.normalize()

# Try asof
try:
    res = s_daily.asof(dates_naive)
    print("SUCCESS!")
    print(res.head())
    print("Length input:", len(idx_intraday), "Length output:", len(res))
except Exception as e:
    print("FAILED:", e)
