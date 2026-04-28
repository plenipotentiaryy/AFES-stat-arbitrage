import pandas as pd
from polygon import RESTClient
from datetime import datetime
import time
import os

# Вставь сюда свой API ключ от Polygon
API_KEY = "твой_ключ_pk_..."

# Наши 12 tech-акций
tickers = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "AMD",
    "INTC", "AVGO", "CRM", "ORCL", "ADBE", "QCOM"
]

# Период: 2 года
start_date = "2024-01-01"
end_date = "2026-01-01"

# Подключаемся к Polygon
client = RESTClient(api_key=API_KEY)

# Сюда соберём все данные
all_data = {}

for ticker in tickers:
    print(f"Скачиваю {ticker}...")

    bars = []
    for bar in client.list_aggs(
        ticker=ticker,
        multiplier=15,          # интервал 15
        timespan="minute",      # минут
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
            "vwap": bar.vwap       # Polygon сразу даёт VWAP!
        })

    df = pd.DataFrame(bars)
    df = df.set_index("timestamp")
    all_data[ticker] = df
    print(f"  {ticker}: {len(df)} баров")

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