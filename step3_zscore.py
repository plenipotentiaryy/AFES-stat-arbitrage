import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================
# Загружаем данные
# ============================================
closes = pd.read_csv("closes_15min.csv", index_col=0, parse_dates=True)
pairs = pd.read_csv("pairs_selected.csv")

print("Найденные пары:")
print(pairs.to_string(index=False))

# ============================================
# Берём лучшую пару
# ============================================
if pairs.empty:
    print("❌ ОШИБКА: Файл pairs_selected.csv пуст! Сначала запустите step2_pairs.py.")
    exit(1)

best = pairs.iloc[0]  # первая строка — лучшая по coint_pvalue
t1, t2 = best["pair"].split("-")
beta = best["beta"]

print(f"\nРаботаем с парой: {t1}-{t2}, beta={beta}")

# Проверка: есть ли эти тикеры в данных?
if t1 not in closes.columns or t2 not in closes.columns:
    print(f"❌ ОШИБКА: Один из тикеров ({t1} или {t2}) отсутствует в closes_15min.csv!")
    print(f"Доступные тикеры: {list(closes.columns)}")
    print("Пожалуйста, перезапустите stat_arb.py, чтобы докачать данные.")
    exit(1)

# ============================================
# Строим спред
# ============================================
spread = closes[t1] - beta * closes[t2]

print(f"Спред: {len(spread)} точек")
print(f"Среднее: {spread.mean():.4f}")
print(f"Стд: {spread.std():.4f}")

# ============================================
# Half-life (Орнштейн-Уленбек)
# ============================================
# Регрессия: изменение спреда = theta * предыдущее значение
# theta < 0 значит спред возвращается к среднему
# half-life = -ln(2) / theta
spread_lag = spread.shift(1)
spread_diff = spread.diff()
aligned = pd.DataFrame({
    "diff": spread_diff,
    "lag": spread_lag
}).dropna()

model = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit()
theta = model.params["lag"]
half_life = int(-np.log(2) / theta) if theta < 0 else 100

print(f"\nHalf-life: {half_life} баров")
print(f"  = {half_life * 15 / 60:.1f} часов")
print(f"  = {half_life * 15 / 60 / 6.5:.1f} торговых дней")
print(f"  (theta = {theta:.6f})")

# ============================================
# Z-score
# ============================================
# Окно = half-life (логично: окно ≈ период возврата)
# Но не меньше 20 и не больше 200
window = max(20, min(half_life, 200))
print(f"\nОкно для Z-score: {window} баров")

# Скользящее среднее и стд
spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()

# Z-score = (текущий спред - среднее) / стандартное отклонение
# Z > 2  → спред сильно вверх → шортим A, лонгим B
# Z < -2 → спред сильно вниз  → лонгим A, шортим B
# Z ≈ 0  → всё в норме, закрываем позиции
zscore = (spread - spread_mean) / spread_std

# ============================================
# Сохраняем всё в один файл
# ============================================
signals = pd.DataFrame({
    f"{t1}_close": closes[t1],
    f"{t2}_close": closes[t2],
    "spread": spread,
    "spread_mean": spread_mean,
    "spread_std": spread_std,
    "zscore": zscore
}).dropna()

signals.to_csv("signals.csv")

# ============================================
# Статистика Z-score
# ============================================
print(f"\n--- Статистика Z-score ---")
print(f"Точек: {len(signals)}")
print(f"Мин: {zscore.min():.2f}")
print(f"Макс: {zscore.max():.2f}")
print(f"Сколько раз |Z| > 2.0: {(zscore.abs() > 2.0).sum()}")
print(f"Сколько раз |Z| > 1.5: {(zscore.abs() > 1.5).sum()}")
print(f"Сколько раз |Z| > 3.0: {(zscore.abs() > 3.0).sum()}")

print(f"\nСохранено в signals.csv")
print(f"Пара: {t1}-{t2}")
print(f"Готово к Шагу 4 — торговая логика!")