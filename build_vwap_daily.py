"""Build daily volume-weighted average prices from 5-min data.

daily_vwap[date, ticker] = Σ(close_5min × volume_5min) / Σ(volume_5min)

Used to simulate VWAP execution: signal on close N, fill on VWAP N+1.
"""
from pathlib import Path
import pandas as pd

DATA = Path("data")

print("Reading closes_5min.csv ...")
c = pd.read_csv(DATA / "closes_5min.csv", index_col=0, parse_dates=True)
if not isinstance(c.index, pd.DatetimeIndex):
    c.index = pd.to_datetime(c.index, utc=True).tz_convert("US/Eastern")
print(f"  closes shape: {c.shape}")

print("Reading volumes_5min.csv ...")
v = pd.read_csv(DATA / "volumes_5min.csv", index_col=0, parse_dates=True)
if not isinstance(v.index, pd.DatetimeIndex):
    v.index = pd.to_datetime(v.index, utc=True).tz_convert("US/Eastern")
print(f"  volumes shape: {v.shape}")

# Align (some tickers may exist in one but not the other)
common = sorted(set(c.columns) & set(v.columns))
c = c[common].reindex(v.index)
v = v[common]
print(f"  common tickers: {len(common)}")

dates = c.index.normalize().tz_localize(None)
pv = c * v
num = pv.groupby(dates).sum(min_count=1)
den = v.groupby(dates).sum(min_count=1)
daily_vwap = num / den.replace(0, pd.NA)
daily_vwap.index.name = "Date"
print(f"  daily VWAP shape: {daily_vwap.shape}")
print(f"  span: {daily_vwap.index.min().date()} → {daily_vwap.index.max().date()}")

out = DATA / "vwap_daily.csv"
daily_vwap.to_csv(out)
print(f"Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")
