import os
import pandas as pd
import time
from polygon import StocksClient
from dotenv import load_dotenv
import threading

# 1. Load the .env file
load_dotenv()

def get_bars(ticker, start_date, end_date, api_key):
    client = StocksClient(api_key=api_key)
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
        
        if not bars:
            print(f"  {ticker}: API вернул пустой список (возможно, нет данных или ошибка лимита)")
            return None

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

        df = pd.DataFrame(data).set_index("timestamp")

        df = df[~df.index.duplicated(keep='first')]
        print(f"  {ticker}: {len(df)} баров загружено")
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке {ticker}: {e}")
        return None

# 2. Parse the API_KEYS string into a list
# We use .split(',') and then .strip() to remove any accidental whitespace
keys_raw = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]
API_KEY_COUNT = len(API_KEYS)

if not API_KEYS:
    raise ValueError("No API_KEYS found in .env file.")

# Наши 12 tech-акций
tickers = [
    "KO", "PEP",
    "XOM", "CVX",
    "JPM", "BAC",
    "V", "MA",
    "HD", "LOW",
    "GOLD", "NEM"
]

start_date = "2024-01-01"
end_date = "2026-01-01"
all_data = {}
threads = []

def fetch_worker(ticker, start_date, end_date, api_key):
    df = get_bars(ticker, start_date, end_date, api_key)
    if df is not None:
        all_data[ticker] = df

for i, ticker in enumerate(tickers):
    current_key = API_KEYS[i % API_KEY_COUNT]
    print(f"--- Using API Key: {current_key[:4]}... for the next batch ---")

    print(f"Скачиваю {ticker}...")

    t = threading.Thread(target=fetch_worker, args=(ticker, start_date, end_date, current_key))
    t.start()
    threads.append(t)

    if (i+1)%API_KEY_COUNT==0 and i!=len(tickers)-1:
        print("60 second break for keys")
        time.sleep(60)

# Wait for all threads to finish
for t in threads:
    t.join()

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