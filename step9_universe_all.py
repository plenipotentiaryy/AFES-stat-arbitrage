"""
step9_universe_all.py — Full all-vs-all cointegration sweep across all 306 tickers.

Funnel (cheap → expensive):
  Layer 1: Correlation filter  >= 0.50         ~46k → ~2k pairs
  Layer 2: SSD pre-filter (Gatev)  bottom 80%  ~2k  → ~1.5k pairs
  Layer 3: Johansen cointegration (95%)         ~1.5k → ~few hundred
  Layer 4: Hurst < 0.50, half-life <= 500 bars

Uses closes_daily.csv — no new downloads required.
Output: data/universe_all_pairs.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from itertools import combinations

from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from config import DATA_DIR

# config
CORR_MIN      = 0.50
SSD_PCT       = 80
HURST_MAX_USE = 0.50
HL_MAX_BARS   = 500
BETA_MIN      = 0.05
BETA_MAX      = 20.0
MIN_OBS       = 252 * 3   # 3 years of daily data minimum
JOH_WINDOW    = 252 * 4   # test on last 4 years (cointegration is regime-dependent)
JOH_CRIT_IDX  = 0         # 0=90%, 1=95%, 2=99%
OUT_FILE      = DATA_DIR / "universe_all_pairs.csv"


def hurst(ts: np.ndarray) -> float:
    lags = range(2, min(100, len(ts) // 2))
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if len(tau) < 2 or min(tau) <= 0:
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def test_pair(s1: np.ndarray, s2: np.ndarray, t1: str, t2: str):
    try:
        mat = np.column_stack([s1, s2])
        res = coint_johansen(mat, det_order=0, k_ar_diff=1)
        trace_stat = res.lr1[0]
        crit_val   = res.cvt[0, JOH_CRIT_IDX]
        if trace_stat <= crit_val:
            return None
        joh_margin = (trace_stat - crit_val) / crit_val

        X   = sm.add_constant(s2)
        ols = sm.OLS(s1, X).fit()
        beta = float(ols.params[1])
        if not (BETA_MIN <= abs(beta) <= BETA_MAX):
            return None

        spread = s1 - beta * s2
        h = hurst(spread)
        if h >= HURST_MAX_USE:
            return None

        ds  = np.diff(spread)
        lag = spread[:-1]
        b   = np.polyfit(lag, ds, 1)[0]
        hl  = -np.log(2) / b if b < 0 else np.inf
        if hl <= 0 or hl > HL_MAX_BARS:
            return None

        return {
            "pair":       f"{t1}-{t2}",
            "beta":       round(beta, 4),
            "hurst":      round(h, 3),
            "half_life":  round(hl, 1),
            "joh_margin": round(joh_margin, 3),
            "trace_stat": round(trace_stat, 2),
            "crit_val":   round(crit_val, 2),
        }
    except Exception:
        return None


if __name__ == "__main__":
    # load data
    print("Loading closes_daily.csv …")
    daily = pd.read_csv(DATA_DIR / "closes_daily.csv", index_col=0, parse_dates=True)
    daily = daily.ffill().dropna(axis=1, thresh=MIN_OBS)
    tickers = list(daily.columns)
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    print(f"Tickers: {len(tickers)}  |  Total candidate pairs: {n_pairs:,}\n")

    # layer 1 - correlation filter
    print("Layer 1: correlation filter …")
    log_ret = np.log(daily / daily.shift(1)).dropna()
    corr    = log_ret.corr()
    candidates = [
        (t1, t2)
        for t1, t2 in combinations(tickers, 2)
        if abs(corr.loc[t1, t2]) >= CORR_MIN
    ]
    print(f"  After corr >= {CORR_MIN}: {len(candidates):,} pairs")

    # layer 2 - SSD filter (top pairs by price distance)
    print("Layer 2: SSD filter …")
    normed = daily / daily.iloc[0]
    ssds = [((normed[t1] - normed[t2]) ** 2).sum() for t1, t2 in candidates]
    threshold = np.percentile(ssds, SSD_PCT)
    candidates = [p for p, s in zip(candidates, ssds) if s <= threshold]
    print(f"  After SSD <= {SSD_PCT}th pct: {len(candidates):,} pairs")

    # layer 3 - johansen cointegration
    print(f"Layer 3: Johansen cointegration (sequential) …")
    results = []
    for i, (t1, t2) in enumerate(candidates):
        if i % 200 == 0:
            print(f"  {i:,} / {len(candidates):,}  ({len(results)} passed)")
        # Align by shared non-NaN dates, use last JOH_WINDOW bars
        pair_df = daily[[t1, t2]].dropna().tail(JOH_WINDOW)
        if len(pair_df) < MIN_OBS:
            continue
        s1 = pair_df[t1].values.astype(float)
        s2 = pair_df[t2].values.astype(float)
        r = test_pair(s1, s2, t1, t2)
        if r:
            results.append(r)

    print(f"\nDone: {len(results)} cointegrated pairs found")

    if results:
        df_out = (pd.DataFrame(results)
                    .sort_values("joh_margin", ascending=False)
                    .reset_index(drop=True))
        df_out.to_csv(OUT_FILE, index=False)
        print(f"Saved → {OUT_FILE}\n")
        print("Top 40 by Johansen margin:")
        print(df_out[["pair", "beta", "hurst", "half_life", "joh_margin"]].head(40).to_string(index=False))
    else:
        print("No pairs found.")
