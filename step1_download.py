import os
import itertools
import time
import pandas as pd
from polygon import StocksClient
from dotenv import load_dotenv
from config import TICKERS, START_DATE, END_DATE, DATA_DIR, REQUEST_SLEEP

load_dotenv()

keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS in .env")

clients = [StocksClient(api_key=k) for k in API_KEYS]
client_cycle = itertools.cycle(clients)
DATA_DIR.mkdir(exist_ok=True)


def parse_bar(bar) -> dict:
    if isinstance(bar, dict):
        return {
            "timestamp": pd.to_datetime(bar["t"], unit="ms"),
            "open": bar["o"], "high": bar["h"], "low": bar["l"],
            "close": bar["c"], "volume": bar["v"],
            "vwap": bar.get("vwap") or bar.get("vwp"),
        }
    return {
        "timestamp": pd.to_datetime(bar.timestamp, unit="ms"),
        "open": bar.open, "high": bar.high, "low": bar.low,
        "close": bar.close, "volume": bar.volume,
        "vwap": getattr(bar, "vwap", None),
    }


# ── Detect which tickers are already downloaded ───────────────────────────────
closes_path = DATA_DIR / "closes_15min.csv"
if closes_path.exists():
    existing_cols = set(pd.read_csv(closes_path, nrows=0).columns) - {"timestamp"}
else:
    existing_cols = set()

to_download = [t for t in TICKERS if t not in existing_cols]

if not to_download:
    print("All tickers already downloaded. Nothing to do.")
    raise SystemExit(0)

print(f"Already have: {len(existing_cols)} tickers")
print(f"Downloading:  {len(to_download)} missing tickers: {to_download}")
print(f"Sleep between requests: {REQUEST_SLEEP}s")
print(f"Estimated time: ~{len(to_download) * REQUEST_SLEEP // 60} min\n")

# ── Download missing tickers ──────────────────────────────────────────────────
all_data = {}

for i, ticker in enumerate(to_download):
    client = next(client_cycle)

    if i > 0:
        time.sleep(REQUEST_SLEEP)

    key_hint = API_KEYS[i % len(API_KEYS)][:4]
    print(f"[{i+1}/{len(to_download)}] {ticker}  (key {key_hint}...)")

    try:
        bars = client.get_aggregate_bars(
            symbol=ticker,
            from_date=START_DATE,
            to_date=END_DATE,
            multiplier=15,
            timespan="minute",
            limit=50000,
            full_range=True,
            warnings=False,
            run_parallel=False,
        )

        if not bars:
            print(f"  -> no data returned")
            continue

        df = pd.DataFrame([parse_bar(b) for b in bars]).set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        all_data[ticker] = df
        print(f"  -> {len(df)} bars")

    except Exception as e:
        print(f"  -> error: {e}")

if not all_data:
    print("No new data downloaded.")
    raise SystemExit(0)

# ── Merge with existing data ──────────────────────────────────────────────────
new_closes  = pd.DataFrame({t: all_data[t]["close"]  for t in all_data})
new_volumes = pd.DataFrame({t: all_data[t]["volume"] for t in all_data})
new_vwaps   = pd.DataFrame({t: all_data[t]["vwap"]   for t in all_data})

for filename, new_df in [
    ("closes_15min.csv",  new_closes),
    ("volumes_15min.csv", new_volumes),
    ("vwaps_15min.csv",   new_vwaps),
]:
    path = DATA_DIR / filename
    if path.exists():
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        combined = existing.join(new_df, how="outer")
    else:
        combined = new_df
    combined.to_csv(path)
    print(f"Saved {filename}: {combined.shape[1]} tickers total")

print(f"\nDone. Downloaded {len(all_data)} new tickers.")
