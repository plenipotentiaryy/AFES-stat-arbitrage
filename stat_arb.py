import pandas as pd
from polygon import RESTClient
import dotenv
import os

dotenv.load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

tickers = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "AMD",
    "INTC", "AVGO", "CRM", "ORCL", "ADBE", "QCOM"
]

start_date = "2024-01-01"
end_date = "2026-04-28"

client = RESTClient(api_key=API_KEY)

all_data = {}

for ticker in tickers:
    print(f"Скачиваю {ticker}...")

    bars = []
    for bar in client.list_aggs(
        ticker=ticker,
        multiplier=15,
        timespan="minute",
        from_=start_date,
        to=end_date,
        limit=50000
    ):
        bars.append({
            "timestamp": pd.to_datetime(bar.timestamp, unit="ms"),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap
        })

    if not bars:
        print(f"  {ticker}: нет данных")
        continue

    df = pd.DataFrame(bars).set_index("timestamp")
    all_data[ticker] = df
    print(f"  {ticker}: {len(df)} баров")

# Собираем в таблицы
closes = pd.DataFrame({t: all_data[t]["close"] for t in all_data})
volumes = pd.DataFrame({t: all_data[t]["volume"] for t in all_data})
vwaps = pd.DataFrame({t: all_data[t]["vwap"] for t in all_data})

# Сохраняем
closes.to_csv("closes_15min.csv")
volumes.to_csv("volumes_15min.csv")
vwaps.to_csv("vwaps_15min.csv")

print(f"\nПериод: {closes.index[0]} — {closes.index[-1]}")
print(f"Баров: {closes.shape[0]}, тикеров: {closes.shape[1]}")
print(f"\nПропуски:\n{closes.isnull().sum()}")
print(f"\nПервые строки:\n{closes.head()}")