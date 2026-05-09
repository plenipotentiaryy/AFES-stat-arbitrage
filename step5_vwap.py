import pandas as pd
import numpy as np

# ============================================
# Загружаем данные
# ============================================
signals = pd.read_csv("signals.csv", index_col=0, parse_dates=True)
vwaps = pd.read_csv("vwaps_15min.csv", index_col=0, parse_dates=True)
pairs = pd.read_csv("pairs_selected.csv")

best = pairs.iloc[0]
t1, t2 = best["pair"].split("-")
beta = best["beta"]

# Приводим VWAP к тому же формату времени
if vwaps.index.tz is None:
    vwaps.index = vwaps.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    vwaps.index = vwaps.index.tz_convert("US/Eastern")
vwaps = vwaps.between_time("10:00", "16:00")

# Совмещаем сигналы и VWAP по индексу
signals["vwap_A"] = vwaps[t1].reindex(signals.index)
signals["vwap_B"] = vwaps[t2].reindex(signals.index)
signals = signals.dropna()

print(f"Пара: {t1}-{t2}, beta={beta}")
print(f"Данных с VWAP: {len(signals)} баров\n")

# ============================================
# Параметры
# ============================================
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5

# ============================================
# Функция бэктеста
# ============================================
def run_backtest(signals, use_vwap=False, label=""):
    position = 0
    entry_spread = 0
    entry_time = None
    entry_z = 0
    trades = []

    col_A = f"{t1}_close"
    col_B = f"{t2}_close"

    for i in range(len(signals)):
        row = signals.iloc[i]
        z = row["zscore"]
        spread_now = row["spread"]
        time_now = signals.index[i]
        price_A = row[col_A]
        price_B = row[col_B]
        vwap_A = row["vwap_A"]
        vwap_B = row["vwap_B"]

        # --- Выход ---
        if position != 0:
            exit_signal = (position == 1 and z > -EXIT_Z) or \
                          (position == -1 and z < EXIT_Z)
            stop_signal = (position == 1 and z < -STOP_Z) or \
                          (position == -1 and z > STOP_Z)

            if exit_signal or stop_signal:
                pnl = position * (spread_now - entry_spread)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": time_now,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": round(pnl, 4),
                    "exit_reason": "STOP" if stop_signal else "SIGNAL"
                })
                position = 0

        # --- Вход ---
        if position == 0:

            # Лонг спред: Z < -2, покупаем A, продаём B
            # VWAP: A ниже VWAP (дёшево), B выше VWAP (дорого)
            if z < -ENTRY_Z:
                if use_vwap:
                    vwap_ok = (price_A < vwap_A) and (price_B > vwap_B)
                    if not vwap_ok:
                        continue
                position = 1
                entry_spread = spread_now
                entry_time = time_now
                entry_z = z

            # Шорт спред: Z > 2, продаём A, покупаем B
            # VWAP: A выше VWAP (дорого), B ниже VWAP (дёшево)
            elif z > ENTRY_Z:
                if use_vwap:
                    vwap_ok = (price_A > vwap_A) and (price_B < vwap_B)
                    if not vwap_ok:
                        continue
                position = -1
                entry_spread = spread_now
                entry_time = time_now
                entry_z = z

    df = pd.DataFrame(trades)
    if df.empty:
        print(f"\n{label}: 0 сделок")
        return df

    total_pnl = df["pnl"].sum()
    wins = len(df[df["pnl"] > 0])
    stops = len(df[df["exit_reason"] == "STOP"])
    win_rate = wins / len(df) * 100

    # Sharpe — исправленный
    trades_per_year = len(df) / (len(signals) / (26 * 252)) 
    sharpe = df["pnl"].mean() / df["pnl"].std() * np.sqrt(trades_per_year) if df["pnl"].std() > 0 else 0

    # Max drawdown
    cumulative = df["pnl"].cumsum()
    drawdown = (cumulative - cumulative.cummax()).min()

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"Сделок:        {len(df)}")
    print(f"Прибыльных:    {wins} ({win_rate:.0f}%)")
    print(f"Стоп-лоссов:   {stops}")
    print(f"Общий P&L:     {total_pnl:.4f}")
    print(f"Средняя сделка:{df['pnl'].mean():.4f}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Max drawdown:  {drawdown:.4f}")

    return df

# ============================================
# Запускаем оба варианта
# ============================================
trades_no_vwap = run_backtest(signals, use_vwap=False, label="БЕЗ VWAP-фильтра")
trades_vwap = run_backtest(signals, use_vwap=True, label="С VWAP-фильтром")

# ============================================
# Сравнение
# ============================================
print(f"\n{'='*50}")
print("СРАВНЕНИЕ")
print(f"{'='*50}")

if not trades_vwap.empty and not trades_no_vwap.empty:
    diff_trades = len(trades_no_vwap) - len(trades_vwap)
    diff_pnl = trades_vwap["pnl"].sum() - trades_no_vwap["pnl"].sum()

    print(f"VWAP отфильтровал: {diff_trades} сделок")
    print(f"Разница P&L:       {diff_pnl:+.4f}")

    if trades_vwap["pnl"].mean() > trades_no_vwap["pnl"].mean():
        print("Вывод: VWAP улучшает качество сделок")
    else:
        print("Вывод: VWAP не улучшает результат на этих данных")

# Сохраняем лучший вариант
best_trades = trades_vwap if (not trades_vwap.empty and 
    trades_vwap["pnl"].mean() > trades_no_vwap["pnl"].mean()) else trades_no_vwap
best_trades.to_csv("trades.csv", index=False)
print(f"\nЛучший вариант сохранён в trades.csv")