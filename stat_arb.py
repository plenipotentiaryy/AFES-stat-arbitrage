import os
import itertools
import pandas as pd
from polygon import StocksClient
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

# 2. Parse the API_KEYS string into a list
# We use .split(',') and then .strip() to remove any accidental whitespace
keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]

if not API_KEYS:
    raise ValueError("No API_KEYS found in .env file. Ensure they are comma-separated.")

key_cycle = itertools.cycle(API_KEYS)

# --- Rest of your existing logic ---

tickers = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", 
    "AMD","INTC", "AVGO", "CRM", "ORCL", 
    "ADBE", "QCOM"
]

start_date = "2024-01-01"
end_date = "2026-01-01"
all_data = {}

for i, ticker in enumerate(tickers):
    # Every 5 tickers, switch to the next API key from the list
    if i % 5 == 0:
        current_key = next(key_cycle)
        client = StocksClient(api_key=current_key)
        print(f"--- Using API Key: {current_key[:4]}... for the next batch ---")

    print(f"Скачиваю {ticker}...")

    try:
        bars = client.get_aggregate_bars(
            symbol=ticker,
            from_date=start_date,
            to_date=end_date,
            multiplier=15,
            timespan="minute",
            limit=50000,
            full_range=True,
            warnings=False,
            run_parallel=False
        )
        
        data = []
        for bar in bars:
            if isinstance(bar, dict):
                ts, o, h, l, c, v = bar.get("t"), bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"), bar.get("v")
                vwap = bar.get("vwap") or bar.get("vwp")
            else:
                ts, o, h, l, c, v = bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume
                vwap = getattr(bar, 'vwap', None)
            
            data.append({
                "timestamp": pd.to_datetime(ts, unit="ms"),
                "open": o, "high": h, "low": l, "close": c, "volume": v, "vwap": vwap
            })

        if not data:
            print(f"  {ticker}: нет данных")
            continue

        df = pd.DataFrame(data).set_index("timestamp")
        all_data[ticker] = df
        print(f"  {ticker}: {len(df)} баров")
        
    except Exception as e:
        print(f"Ошибка при загрузке {ticker}: {e}")

# Consolidation and saving
if all_data:
    # Собираем цены закрытия в одну таблицу
    closes = pd.DataFrame({t: all_data[t]["close"] for t in all_data})
    volumes = pd.DataFrame({t: all_data[t]["volume"] for t in all_data})
    vwaps = pd.DataFrame({t: all_data[t]["vwap"] for t in all_data})

    # Сохраняем в CSV чтобы не скачивать каждый раз
    closes.to_csv("closes_15min.csv")
    volumes.to_csv("volumes_15min.csv")
    vwaps.to_csv("vwaps_15min.csv")

    # Проверяем
    print("\n Результат ")
    print(f"Период: {closes.index[0]} — {closes.index[-1]}")
    print(f"Всего баров: {closes.shape[0]}")
    print(f"Тикеров: {closes.shape[1]}")
    print(f"\nПропуски:\n{closes.isnull().sum()}")
    print(f"\nПервые строки:\n{closes.head()}")

else:
    print("\n Данные не были загружены.")