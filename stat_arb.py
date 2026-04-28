import pandas as pd
from polygon import StocksClient
from datetime import datetime
import time
import dotenv
import os

dotenv.load_dotenv()  # Загружаем переменные из .env

# Вставь сюда свой API ключ от Polygon
API_KEY = os.getenv("API_KEY")

# Наши 12 tech-акций
tickers = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "AMD",
    "INTC", "AVGO", "CRM", "ORCL", "ADBE", "QCOM"
]

# Период: 30 дней (бесплатный план ограничен)
start_date = "2026-04-01"
end_date = "2026-04-28"

# Подключаемся к Polygon
client = StocksClient(api_key=API_KEY)

# Сюда соберём все данные
all_data = {}

for ticker in tickers:
    print(f"Скачиваю {ticker}...")

    bars = []
    bars = client.get_aggregate_bars(
        symbol=ticker,
        from_date=start_date,
        to_date=end_date,
        multiplier=15,
        timespan="minute",
        limit=50000,
        full_range=True,
        warnings=False,
        run_parallel=False  # последовательная загрузка
    )
    
    # Преобразуем в список словарей
    data = []
    for bar in bars:
        # API может возвращать словарь или объект
        if isinstance(bar, dict):
            ts = bar.get("t") or bar.get("timestamp")
            o = bar.get("o") or bar.get("open")
            h = bar.get("h") or bar.get("high")
            l = bar.get("l") or bar.get("low")
            c = bar.get("c") or bar.get("close")
            v = bar.get("v") or bar.get("volume")
            vwap = bar.get("vwap") or bar.get("vwp")
        else:
            ts = bar.timestamp
            o = bar.open
            h = bar.high
            l = bar.low
            c = bar.close
            v = bar.volume
            vwap = getattr(bar, 'vwap', None)
        
        data.append({
            "timestamp": pd.to_datetime(ts, unit="ms"),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "vwap": vwap
        })

    # Пропускаем тикер если нет данных
    if not data:
        print(f"  {ticker}: нет данных")
        continue

    df = pd.DataFrame(data)
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