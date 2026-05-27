"""Resample volumes_5min.csv → volumes_daily.csv (sum within each session)."""
from pathlib import Path
import pandas as pd

SRC = Path("data/volumes_5min.csv")
DST = Path("data/volumes_daily.csv")

print(f"Reading {SRC} ...")
v = pd.read_csv(SRC, index_col=0, parse_dates=True)
if not isinstance(v.index, pd.DatetimeIndex):
    v.index = pd.to_datetime(v.index, utc=True).tz_convert("US/Eastern")
print(f"  shape:  {v.shape}")
print(f"  span:   {v.index.min()} → {v.index.max()}")
print(f"  memory: {v.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Index is intraday timestamps (tz-aware US/Eastern). Group by date.
print("Resampling to daily sums ...")
vd = v.groupby(v.index.normalize().tz_localize(None)).sum(min_count=1)
vd.index.name = "Date"
print(f"  daily shape: {vd.shape}")
print(f"  daily span:  {vd.index.min().date()} → {vd.index.max().date()}")

vd.to_csv(DST)
print(f"Saved → {DST}  ({DST.stat().st_size / 1e6:.1f} MB)")
