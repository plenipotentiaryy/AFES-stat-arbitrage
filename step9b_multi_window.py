"""
step9b_multi_window.py — Multi-window cointegration filter.

Tests every pair across 3 time horizons:
  - 5 years (1260 daily bars)   — long-term structural link
  - 9 months (189 daily bars)   — medium-term regime
  - 60 days  (60 daily bars)    — current regime

Only pairs cointegrated (Johansen 90%) on ALL THREE windows are kept.
This filters out look-ahead bias and produces truly robust pairs.

Output: data/universe_multi_window.csv
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

WINDOWS = {"5y": 1260, "9m": 189, "60d": 60}
CORR_MIN = 0.50
SSD_PCT  = 80
HURST_MAX = 0.50
HL_MAX_DAYS = 200
BETA_MIN, BETA_MAX = 0.05, 20.0
JOH_IDX = 0   # 90% critical value


def hurst(ts):
    lags = range(2, min(50, len(ts) // 2))
    tau = [np.std(np.subtract(ts[l:], ts[:-l])) for l in lags]
    if len(tau) < 2 or min(tau) <= 0:
        return 0.5
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]


def test_pair(s1, s2):
    """Returns dict with metrics, or None if any filter fails."""
    if len(s1) < 30:
        return None
    try:
        res = coint_johansen(np.column_stack([s1, s2]), det_order=0, k_ar_diff=1)
        ts_stat = res.lr1[0]
        cv = res.cvt[0, JOH_IDX]
        if ts_stat <= cv:
            return None
        joh = (ts_stat - cv) / cv

        ols = sm.OLS(s1, sm.add_constant(s2)).fit()
        beta = float(ols.params[1])
        if not (BETA_MIN <= abs(beta) <= BETA_MAX):
            return None

        spread = s1 - beta * s2
        h = hurst(spread)
        if h >= HURST_MAX:
            return None

        ds = np.diff(spread)
        b = np.polyfit(spread[:-1], ds, 1)[0]
        hl = -np.log(2) / b if b < 0 else 9999
        if hl <= 0 or hl > HL_MAX_DAYS:
            return None

        return {"beta": round(beta, 4), "hurst": round(h, 3),
                "half_life": round(hl, 1), "joh_margin": round(joh, 3)}
    except Exception:
        return None


if __name__ == "__main__":
    print("Loading closes_daily.csv …")
    daily = pd.read_csv("data/closes_daily.csv", index_col=0, parse_dates=True).ffill()
    tickers = [t for t in daily.columns if daily[t].tail(WINDOWS["5y"]).dropna().shape[0] >= 1000]
    print(f"Tickers with ≥1000 obs in last 5y: {len(tickers)}\n")

    # Layer 1: correlation on the LONGEST window (most permissive)
    log_ret = np.log(daily[tickers].tail(WINDOWS["5y"]) /
                     daily[tickers].tail(WINDOWS["5y"]).shift(1)).dropna(how="all")
    corr = log_ret.corr()
    candidates = [(t1, t2) for t1, t2 in combinations(tickers, 2)
                  if abs(corr.loc[t1, t2]) >= CORR_MIN]
    print(f"Layer 1 (corr ≥ {CORR_MIN}): {len(candidates):,} pairs")

    # Layer 2: SSD on 5y
    sub5 = daily[tickers].tail(WINDOWS["5y"])
    normed = sub5 / sub5.iloc[0]
    ssds = [((normed[t1] - normed[t2]) ** 2).sum() for t1, t2 in candidates]
    thr = np.percentile(ssds, SSD_PCT)
    candidates = [p for p, s in zip(candidates, ssds) if s <= thr]
    print(f"Layer 2 (SSD ≤ {SSD_PCT}p): {len(candidates):,} pairs\n")

    # Layer 3: triple-window Johansen
    print("Layer 3: Johansen on 3 windows (5y, 9m, 60d) …")
    results = []
    for i, (t1, t2) in enumerate(candidates):
        if i % 200 == 0:
            print(f"  {i:,}/{len(candidates):,}  ({len(results)} survived)")
        pair_full = daily[[t1, t2]].dropna()
        if len(pair_full) < WINDOWS["5y"]:
            continue

        # Test on all 3 windows
        metrics = {}
        passed_all = True
        for name, w in WINDOWS.items():
            sub = pair_full.tail(w)
            if len(sub) < w * 0.8:
                passed_all = False
                break
            r = test_pair(sub[t1].values.astype(float),
                          sub[t2].values.astype(float))
            if r is None:
                passed_all = False
                break
            metrics[name] = r

        if not passed_all:
            continue

        # Last-90d correlation as bonus
        last90 = pair_full.tail(90)
        r1 = np.log(last90[t1] / last90[t1].shift(1)).dropna()
        r2 = np.log(last90[t2] / last90[t2].shift(1)).dropna()
        common = r1.index.intersection(r2.index)
        c90 = r1.loc[common].corr(r2.loc[common]) if len(common) > 30 else np.nan

        results.append({
            "pair":         f"{t1}-{t2}",
            "joh_5y":       metrics["5y"]["joh_margin"],
            "joh_9m":       metrics["9m"]["joh_margin"],
            "joh_60d":      metrics["60d"]["joh_margin"],
            "hurst_5y":     metrics["5y"]["hurst"],
            "hurst_9m":     metrics["9m"]["hurst"],
            "hurst_60d":    metrics["60d"]["hurst"],
            "hl_5y":        metrics["5y"]["half_life"],
            "hl_9m":        metrics["9m"]["half_life"],
            "hl_60d":       metrics["60d"]["half_life"],
            "beta_5y":      metrics["5y"]["beta"],
            "corr_90d":     round(c90, 3) if not np.isnan(c90) else np.nan,
        })

    print(f"\n=== {len(results)} pairs survived ALL 3 windows ===")
    if results:
        df = pd.DataFrame(results)
        # Composite score: average joh_margin across windows
        df["score"] = df[["joh_5y", "joh_9m", "joh_60d"]].mean(axis=1)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.to_csv("data/universe_multi_window.csv", index=False)
        print(f"Saved → data/universe_multi_window.csv\n")
        print("Top 30 by composite score:")
        cols = ["pair", "score", "joh_5y", "joh_9m", "joh_60d",
                "hurst_5y", "hl_5y", "corr_90d"]
        print(df[cols].head(30).to_string(index=False))
    else:
        print("No pairs survived. Try relaxing thresholds.")
