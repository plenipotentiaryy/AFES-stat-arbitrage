import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

# ============================================
# Загружаем данные из Шага 1
# ============================================
closes = pd.read_csv("closes_15min.csv", index_col=0, parse_dates=True)

# ============================================
# Фильтр: Оставляем только основную сессию (RTH)
# ============================================
# Переводим в время Нью-Йорка (ET)
if closes.index.tz is None:
    closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    closes.index = closes.index.tz_convert("US/Eastern")

# Оставляем только время с 09:30 до 16:00
closes = closes.between_time("09:30", "16:00")

closes = closes.dropna()  # убираем строки с пропусками
print(f"Загружено (только основная сессия): {closes.shape[0]} баров, {closes.shape[1]} тикеров")
print(f"Тикеры: {list(closes.columns)}\n")

# ============================================
# Фильтр 1: корреляция лог-доходностей
# ============================================

# Лог-доходности: насколько акция выросла/упала в процентах
# log(цена_сегодня / цена_вчера)
log_returns = np.log(closes / closes.shift(1)).dropna()

# Матрица корреляций
corr_matrix = log_returns.corr()

# Собираем все пары с их корреляцией
CORR_THRESHOLD = 0.5          # порог корреляции (0.7 слишком жёстко для коротких периодов)
TOP_N_FALLBACK = 10           # если ни одна пара не прошла — берём топ-N

all_pairs = []
tickers = list(closes.columns)

for t1, t2 in combinations(tickers, 2):
    corr = corr_matrix.loc[t1, t2]
    all_pairs.append((t1, t2, round(corr, 4)))

# Сортируем по убыванию корреляции
all_pairs.sort(key=lambda x: -x[2])

# Фильтр по порогу
pairs_corr = [(t1, t2, c) for t1, t2, c in all_pairs if c > CORR_THRESHOLD]

if len(pairs_corr) == 0:
    print(f"⚠️  Ни одна пара не прошла порог корреляции {CORR_THRESHOLD}.")
    print(f"   Берём топ-{TOP_N_FALLBACK} пар по корреляции как fallback.")
    pairs_corr = all_pairs[:TOP_N_FALLBACK]

print(f"\nФильтр 1 (корреляция > {CORR_THRESHOLD}): {len(pairs_corr)} пар из {len(all_pairs)}")
for t1, t2, corr in pairs_corr:
    print(f"  {t1}-{t2}: корреляция = {corr}")

# ============================================
# Фильтр 2: тест Энгла-Грейнджера
# ============================================

pairs_coint = []

for t1, t2, corr in pairs_corr:
    # Тест на коинтеграцию
    score, pvalue, _ = coint(closes[t1], closes[t2])

    if pvalue < 0.05:
        # Находим бету через регрессию: цена_A = alpha + beta * цена_B
        X = sm.add_constant(closes[t2])  # добавляем константу
        model = sm.OLS(closes[t1], X).fit()
        beta = model.params.iloc[1]

        pairs_coint.append({
            "pair": f"{t1}-{t2}",
            "correlation": corr,
            "coint_pvalue": round(pvalue, 6),
            "beta": round(beta, 4)
        })

print(f"\nФильтр 2 (коинтеграция p < 0.05): {len(pairs_coint)} пар")

if len(pairs_coint) == 0:
    print("\n⚠️  Ни одна пара не прошла тест коинтеграции.")
    print("   Возможные причины:")
    print("   - Слишком короткий период данных (нужно хотя бы 2–3 месяца)")
    print("   - Акции из одного сектора, но не коинтегрированы на этом интервале")
    print("   Попробуйте увеличить период в stat_arb.py (start_date).")
    print("\n   Для диагностики — p-values всех протестированных пар:")
    for t1, t2, corr in pairs_corr:
        _, pv, _ = coint(closes[t1], closes[t2])
        print(f"     {t1}-{t2}: coint p-value = {pv:.4f}")
    # Сохраняем пустой CSV, чтобы pipeline не падал
    pd.DataFrame(columns=["pair","correlation","coint_pvalue","beta","adf_pvalue","adf_stat","half_life_bars"]).to_csv("pairs_selected.csv", index=False)
    print("\nСохранён пустой pairs_selected.csv")
    exit(0)

# ============================================
# Фильтр 3: ADF-тест на стационарность спреда
# ============================================

results = []

for pair in pairs_coint:
    t1, t2 = pair["pair"].split("-")
    beta = pair["beta"]

    # Строим спред
    spread = closes[t1] - beta * closes[t2]

    # ADF-тест: стационарен ли спред?
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(spread)

    # Half-life: за сколько баров спред возвращается к среднему
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    hl_model = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit()
    theta = hl_model.params["lag"]
    half_life = -np.log(2) / theta if theta < 0 else float("inf")

    results.append({
        "pair": pair["pair"],
        "correlation": pair["correlation"],
        "coint_pvalue": pair["coint_pvalue"],
        "beta": beta,
        "adf_pvalue": round(adf_pvalue, 6),
        "adf_stat": round(adf_stat, 4),
        "half_life_bars": round(half_life, 1)
    })

# ============================================
# Итоговая таблица
# ============================================

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("coint_pvalue")

print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА ПАР")
print("=" * 80)
print(df_results.to_string(index=False))

# Сохраняем
df_results.to_csv("pairs_selected.csv", index=False)
print(f"\nСохранено в pairs_selected.csv")

# Рекомендация
good_pairs = df_results[
    (df_results["adf_pvalue"] < 0.05) &
    (df_results["half_life_bars"] > 5) &
    (df_results["half_life_bars"] < 500)
]

print(f"\nРекомендованные пары (ADF p < 0.05, half-life 5-500 баров):")
for _, row in good_pairs.iterrows():
    print(f"  {row['pair']}: beta={row['beta']}, half-life={row['half_life_bars']} баров, coint p={row['coint_pvalue']}")