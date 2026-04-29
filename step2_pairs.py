import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

# ============================================
# Загружаем данные из Шага 1
# ============================================
closes = pd.read_csv("closes_15min.csv", index_col=0, parse_dates=True)
closes = closes.dropna()  # убираем строки с пропусками

print(f"Загружено: {closes.shape[0]} баров, {closes.shape[1]} тикеров")
print(f"Тикеры: {list(closes.columns)}\n")

# ============================================
# Фильтр 1: корреляция лог-доходностей
# ============================================

# Лог-доходности: насколько акция выросла/упала в процентах
# log(цена_сегодня / цена_вчера)
log_returns = np.log(closes / closes.shift(1)).dropna()

# Матрица корреляций
corr_matrix = log_returns.corr()

# Собираем все пары с корреляцией > 0.7
pairs_corr = []
tickers = list(closes.columns)

for t1, t2 in combinations(tickers, 2):
    corr = corr_matrix.loc[t1, t2]
    if corr > 0.7:
        pairs_corr.append((t1, t2, round(corr, 4)))

print(f"Фильтр 1 (корреляция > 0.7): {len(pairs_corr)} пар из {len(list(combinations(tickers, 2)))}")
for t1, t2, corr in sorted(pairs_corr, key=lambda x: -x[2]):
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