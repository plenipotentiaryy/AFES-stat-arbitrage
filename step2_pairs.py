import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from config import (
    PAIRS, RECENT_BARS, RTH_START, RTH_END, DATA_DIR,
)

BETA_MIN = 0.1   # |beta| ниже — одна нога почти не весит, пара бессмысленна
BETA_MAX = 15.0  # |beta| выше — нереалистичное соотношение позиций


def load_closes() -> pd.DataFrame:
    closes = pd.read_csv(DATA_DIR / "closes_15min.csv", index_col=0, parse_dates=True)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes.index = closes.index.tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    min_bars = 1000
    good = [c for c in closes.columns if closes[c].notna().sum() > min_bars]
    print(f"Tickers with enough data: {len(good)} / {len(closes.columns)}")
    return closes[good].dropna()


def compute_half_life(spread: pd.Series) -> float:
    aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    aligned.columns = ["diff", "lag"]
    theta = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit().params["lag"]
    return -np.log(2) / theta if theta < 0 else float("inf")


closes = load_closes()
print(f"Loaded (RTH only): {closes.shape[0]} bars, {closes.shape[1]} tickers")

closes_recent = closes.tail(RECENT_BARS)
print(f"\nTesting {len(PAIRS)} predefined pairs on last {RECENT_BARS} bars:")
print(f"Period: {closes_recent.index[0]} — {closes_recent.index[-1]}\n")

results = []

for t1, t2 in PAIRS:
    # Skip if ticker wasn't downloaded / had no data
    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {t1}-{t2}: missing data")
        continue

    _, pvalue, _ = coint(closes_recent[t1], closes_recent[t2])

    if pvalue >= 0.05:
        print(f"  FAIL {t1}-{t2}: coint p={pvalue:.4f}")
        continue

    beta = sm.OLS(closes_recent[t1], sm.add_constant(closes_recent[t2])).fit().params.iloc[1]

    if not (BETA_MIN <= abs(beta) <= BETA_MAX):
        print(f"  SKIP {t1}-{t2}: beta={beta:.4f} outside [{BETA_MIN}, {BETA_MAX}]")
        continue

    if beta < 0:
        print(f"  SKIP {t1}-{t2}: beta={beta:.4f} is negative (stocks move opposite)")
        continue

    spread = closes_recent[t1] - beta * closes_recent[t2]
    adf_stat, adf_pvalue, *_ = adfuller(spread)
    half_life = compute_half_life(spread)

    # Log-return correlation (sanity: should be positive)
    log_ret = np.log(closes_recent / closes_recent.shift(1)).dropna()
    corr = round(log_ret[t1].corr(log_ret[t2]), 4)

    results.append({
        "pair":          f"{t1}-{t2}",
        "correlation":   corr,
        "coint_pvalue":  round(pvalue, 6),
        "beta":          round(beta, 4),
        "adf_pvalue":    round(adf_pvalue, 6),
        "adf_stat":      round(adf_stat, 4),
        "half_life_bars": round(half_life, 1),
    })
    print(f"  PASS {t1}-{t2}: coint p={pvalue:.4f}, beta={beta:.4f}, "
          f"half-life={half_life:.0f} bars, corr={corr}")

if not results:
    print("\nNo pairs passed all filters.")
    pd.DataFrame(columns=["pair","correlation","coint_pvalue","beta",
                           "adf_pvalue","adf_stat","half_life_bars"]
                 ).to_csv(DATA_DIR / "pairs_selected.csv", index=False)
    raise SystemExit(0)

df_results = pd.DataFrame(results).sort_values("coint_pvalue")

print("\n" + "=" * 75)
print(df_results.to_string(index=False))
print("=" * 75)

df_results.to_csv(DATA_DIR / "pairs_selected.csv", index=False)
print(f"\nSaved {len(df_results)} pairs to {DATA_DIR / 'pairs_selected.csv'}")

good = df_results[
    (df_results["adf_pvalue"] < 0.05) &
    (df_results["half_life_bars"].between(5, 500))
]
print(f"\nRecommended pairs (ADF p<0.05, half-life 5-500 bars): {len(good)}")
for _, row in good.iterrows():
    print(f"  {row['pair']}: beta={row['beta']}, "
          f"half-life={row['half_life_bars']} bars, coint p={row['coint_pvalue']}")
