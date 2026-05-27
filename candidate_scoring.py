"""candidate_scoring.py — Honest 8-metric composite score from §3 of the
universe-expansion spec.

For each candidate pair (t1, t2), compute on a TRAIN slice only:

    S_corr   correlation strength            (15%)
    S_EG     Engle-Granger cointegration     (20%)
    S_J      Johansen trace margin           (15%)
    S_HL     half-life of mean-reversion     (15%)
    S_H      Hurst (mean-revert if H<0.5)    (10%)
    S_beta   beta-stability (rolling CV)     (10%)
    S_L      liquidity symmetry              (10%)
    S_C      cost safety                     (5%)

Then apply diversification constraints (max pairs per ticker) and select
the top 50-70 pairs.  Save selected + rejected with full audit trail.

This replaces the simple corr+coint+HL filter in bias_test.py with the
full composite spec.
"""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

DATA = Path("data")

# ── Default config (overrideable via CLI) ──────────────────────────────────
TRAIN_END_DEFAULT = "2014-12-31"
CORR_PREFILTER    = 0.30   # don't even score pairs below this corr
HL_MIN_DAYS       = 2
HL_MAX_DAYS       = 90
HURST_LO, HURST_HI = 0.50, 0.65
COST_BPS_RT       = 20.0   # 20 bps round-trip (matches our backtest)
COST_FRACTION_CAP = 0.25   # cost must be < 25% of residual move
MIN_DOLLAR_VOLUME = 5e6    # avg-daily-dollar-volume floor on each leg
MIN_PRICE         = 5.0
MIN_TRAIN_OBS     = 750    # ≥ 3y of daily data
TARGET_PAIRS      = 60
MIN_COMPOSITE_SCORE = 0.55
MAX_PAIRS_PER_TICKER = 3


# ── Metric helpers ─────────────────────────────────────────────────────────

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def s_corr(rho: float) -> float:
    """0 at rho=0.30, 1 at rho=0.70."""
    return _clip01((rho - 0.30) / (0.70 - 0.30))


def s_eg(pval: float) -> float:
    if not np.isfinite(pval):
        return 0.0
    return _clip01(1.0 - pval)


def s_johansen(trace: float, crit_95: float) -> float:
    if not (np.isfinite(trace) and np.isfinite(crit_95) and crit_95 > 0):
        return 0.0
    margin = (trace - crit_95) / abs(crit_95)
    return _clip01(margin)


def half_life_days(resid: pd.Series) -> float:
    """AR(1) half-life of residual mean-reversion (days)."""
    a = pd.concat([resid.diff(), resid.shift(1)], axis=1).dropna()
    a.columns = ["d", "lag"]
    if len(a) < 30:
        return np.nan
    try:
        phi = sm.OLS(a["d"], sm.add_constant(a["lag"])).fit().params["lag"]
    except Exception:
        return np.nan
    if not np.isfinite(phi) or phi >= 0:
        return np.nan
    return float(-np.log(2.0) / phi)


def s_half_life(hl: float, lo: float = HL_MIN_DAYS, hi: float = HL_MAX_DAYS) -> float:
    if not np.isfinite(hl) or hl <= lo or hl > hi:
        return 0.0
    return _clip01(1.0 - (hl - lo) / (hi - lo))


def hurst_rs(series: pd.Series, lags: int = 20) -> float:
    """Quick Hurst via variance scaling."""
    arr = series.dropna().values
    if len(arr) < lags + 30:
        return np.nan
    try:
        taus = np.arange(2, lags)
        std_lags = [np.std(np.subtract(arr[k:], arr[:-k])) for k in taus]
        if min(std_lags) <= 0:
            return np.nan
        slope = np.polyfit(np.log(taus), np.log(std_lags), 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def s_hurst(h: float) -> float:
    if not np.isfinite(h):
        return 0.5
    return _clip01((HURST_HI - h) / (HURST_HI - HURST_LO))


def beta_cv(p1: pd.Series, p2: pd.Series, win: int = 252) -> float:
    """CV of rolling β = log(P1) vs log(P2)."""
    lp1, lp2 = np.log(p1), np.log(p2)
    if len(lp1) < win + 30:
        return np.nan
    # Rolling OLS β
    betas = []
    for end in range(win, len(lp1), max(1, win // 4)):
        x = lp2.iloc[end - win:end].values
        y = lp1.iloc[end - win:end].values
        if np.var(x) <= 0:
            continue
        betas.append(float(np.cov(y, x, ddof=0)[0, 1] / np.var(x)))
    if len(betas) < 4:
        return np.nan
    arr = np.asarray(betas)
    mean = abs(arr.mean())
    if mean < 1e-9:
        return np.nan
    return float(arr.std() / (mean + 1e-9))


def s_beta_stability(cv: float) -> float:
    if not np.isfinite(cv):
        return 0.0
    return _clip01(1.0 - cv)


def s_liquidity(adv1: float, adv2: float) -> float:
    if not (np.isfinite(adv1) and np.isfinite(adv2)) or min(adv1, adv2) <= 0:
        return 0.0
    return _clip01(min(adv1, adv2) / max(adv1, adv2))


def s_cost(cost_bps: float, sigma_e_bps: float,
            cap_frac: float = COST_FRACTION_CAP) -> float:
    """Cost safety: cost must be < cap_frac × residual daily vol."""
    if sigma_e_bps <= 0 or not np.isfinite(sigma_e_bps):
        return 0.0
    ratio = cost_bps / (cap_frac * sigma_e_bps)
    return _clip01(1.0 - ratio)


# ── Per-pair scoring ───────────────────────────────────────────────────────

def estimate_beta_ols(p1: pd.Series, p2: pd.Series) -> float:
    x, y = np.log(p2.values), np.log(p1.values)
    var = np.var(x)
    if var <= 0:
        return np.nan
    return float(np.cov(y, x, ddof=0)[0, 1] / var)


def score_pair(closes_train: pd.DataFrame, volumes_train: pd.DataFrame,
                t1: str, t2: str) -> dict:
    """Compute the 8 metrics for one pair on TRAIN data. Returns dict."""
    pair = f"{t1}-{t2}"
    out = {"pair": pair, "t1": t1, "t2": t2}

    slc = closes_train[[t1, t2]].dropna()
    if len(slc) < MIN_TRAIN_OBS:
        out["reject_reason"] = f"insufficient_train_data ({len(slc)})"
        return out
    if slc.min().min() < MIN_PRICE:
        out["reject_reason"] = "price_below_floor"
        return out
    # Liquidity gate (cheap pre-filter)
    if volumes_train is not None and t1 in volumes_train and t2 in volumes_train:
        vol1 = (volumes_train[t1].reindex(slc.index) * slc[t1]).mean()
        vol2 = (volumes_train[t2].reindex(slc.index) * slc[t2]).mean()
    else:
        vol1 = vol2 = np.nan
    if np.isfinite(vol1) and np.isfinite(vol2):
        if min(vol1, vol2) < MIN_DOLLAR_VOLUME:
            out["reject_reason"] = "dollar_volume_below_floor"
            return out

    # Returns + correlation
    rets = np.log(slc).diff().dropna()
    if len(rets) < 100:
        out["reject_reason"] = "insufficient_returns"
        return out
    rho = float(rets[t1].corr(rets[t2]))
    if pd.isna(rho) or rho < CORR_PREFILTER:
        out["reject_reason"] = f"corr_below_prefilter ({rho:.3f})"
        return out

    # β + spread
    beta = estimate_beta_ols(slc[t1], slc[t2])
    if not np.isfinite(beta) or not (0.1 <= abs(beta) <= 15.0):
        out["reject_reason"] = "beta_out_of_range"
        return out
    spread = np.log(slc[t1]) - beta * np.log(slc[t2])

    # Engle-Granger
    try:
        _, eg_p, _ = coint(np.log(slc[t1]), np.log(slc[t2]),
                            trend="c", autolag=None, maxlag=1)
    except Exception:
        eg_p = 1.0

    # Johansen trace + crit95
    trace = crit95 = np.nan
    try:
        jres = coint_johansen(np.log(slc[[t1, t2]]), det_order=0, k_ar_diff=1)
        trace = float(jres.lr1[0])
        crit95 = float(jres.cvt[0, 1])
    except Exception:
        pass

    # Half-life
    hl = half_life_days(spread)

    # Hurst (compute on spread, not on prices)
    h = hurst_rs(spread, lags=20)

    # β stability
    bcv = beta_cv(slc[t1], slc[t2])

    # Liquidity
    sL = s_liquidity(vol1, vol2)

    # Cost safety: cost vs residual daily vol (in bps terms)
    sigma_e = float(spread.diff().std()) * 1e4   # bps
    sC = s_cost(COST_BPS_RT, sigma_e)

    # Composite
    scs = {
        "S_corr":  s_corr(rho),
        "S_EG":    s_eg(eg_p),
        "S_J":     s_johansen(trace, crit95),
        "S_HL":    s_half_life(hl),
        "S_H":     s_hurst(h),
        "S_beta":  s_beta_stability(bcv),
        "S_L":     sL,
        "S_C":     sC,
    }
    weights = {"S_corr": 0.15, "S_EG": 0.20, "S_J": 0.15, "S_HL": 0.15,
                "S_H": 0.10, "S_beta": 0.10, "S_L": 0.10, "S_C": 0.05}
    composite = sum(scs[k] * weights[k] for k in weights)

    # Hard rejection rules (per spec)
    rejects = []
    if not np.isfinite(eg_p):           rejects.append("eg_pvalue_nan")
    if not np.isfinite(trace):          rejects.append("johansen_failed")
    if not np.isfinite(hl):             rejects.append("half_life_invalid")
    elif hl <= 0:                       rejects.append("half_life_nonpositive")
    elif hl > HL_MAX_DAYS:              rejects.append(f"half_life_above_max ({hl:.0f}d)")

    out.update({
        "corr": rho, "beta": beta, "eg_p": eg_p,
        "johansen_trace": trace, "johansen_crit95": crit95,
        "half_life_days": hl, "hurst": h, "beta_cv": bcv,
        "adv1": vol1, "adv2": vol2, "sigma_e_bps": sigma_e,
        **scs, "score": composite,
    })
    if rejects:
        out["reject_reason"] = "; ".join(rejects)
    return out


# ── Universe generation ────────────────────────────────────────────────────

def generate_candidates(closes_train: pd.DataFrame, max_pairs: int = 5000) -> list[tuple]:
    """Generate candidate pairs by correlation pre-filter (cheap)."""
    rets = np.log(closes_train).diff()
    corr = rets.corr(min_periods=250)
    tickers = list(corr.columns)
    pairs = []
    for i, t1 in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            t2 = tickers[j]
            r = corr.iloc[i, j]
            if pd.notna(r) and r >= CORR_PREFILTER:
                pairs.append((t1, t2, float(r)))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:max_pairs]


# ── Selection with diversification constraints ─────────────────────────────

def select_pairs(scored: pd.DataFrame, target_n: int = TARGET_PAIRS,
                  min_score: float = MIN_COMPOSITE_SCORE,
                  max_per_ticker: int = MAX_PAIRS_PER_TICKER) -> pd.DataFrame:
    """Greedy selection: highest score first, respect ticker caps."""
    if scored.empty:
        return scored
    eligible = scored[(scored["reject_reason"].isna()) &
                       (scored["score"] >= min_score)].copy()
    eligible = eligible.sort_values("score", ascending=False)
    ticker_count: dict[str, int] = {}
    keep = []
    for _, row in eligible.iterrows():
        t1, t2 = row["t1"], row["t2"]
        if ticker_count.get(t1, 0) >= max_per_ticker:
            continue
        if ticker_count.get(t2, 0) >= max_per_ticker:
            continue
        keep.append(row.name)
        ticker_count[t1] = ticker_count.get(t1, 0) + 1
        ticker_count[t2] = ticker_count.get(t2, 0) + 1
        if len(keep) >= target_n:
            break
    return scored.loc[keep].reset_index(drop=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    p.add_argument("--max-candidates", type=int, default=3000)
    p.add_argument("--target",   type=int, default=TARGET_PAIRS)
    p.add_argument("--min-score", type=float, default=MIN_COMPOSITE_SCORE)
    args = p.parse_args()

    print(f"Loading closes_daily.csv …")
    closes = pd.read_csv(DATA / "closes_daily.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index()
    closes_train = closes.loc[:args.train_end]
    print(f"  train slice: {closes_train.index.min().date()} → "
          f"{closes_train.index.max().date()}  ({len(closes_train)} days, "
          f"{closes_train.shape[1]} tickers)")

    vol_path = DATA / "volumes_daily.csv"
    volumes_train = None
    if vol_path.exists():
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()
        volumes_train = volumes.loc[:args.train_end]
        print(f"  volumes loaded ({volumes_train.shape[1]} tickers)")

    print(f"Generating candidate pairs (corr ≥ {CORR_PREFILTER}) …")
    candidates = generate_candidates(closes_train, max_pairs=args.max_candidates)
    print(f"  {len(candidates)} pair candidates after corr pre-filter")

    print(f"Scoring all candidates …")
    rows = []
    for k, (t1, t2, r) in enumerate(candidates):
        rows.append(score_pair(closes_train, volumes_train, t1, t2))
        if (k + 1) % 500 == 0:
            print(f"  scored {k+1}/{len(candidates)}")
    scored = pd.DataFrame(rows)
    scored["reject_reason"] = scored.get("reject_reason", pd.Series([None] * len(scored)))

    # Categorise
    ok_mask = scored["reject_reason"].isna()
    print(f"  passed hard gates: {ok_mask.sum()}/{len(scored)}")
    if ok_mask.any():
        s = scored.loc[ok_mask, "score"]
        print(f"  score dist: min={s.min():.3f}  med={s.median():.3f}  "
              f"max={s.max():.3f}  ≥{args.min_score}: {(s >= args.min_score).sum()}")

    selected = select_pairs(scored, target_n=args.target,
                             min_score=args.min_score)
    print(f"\nSelected {len(selected)} pairs after diversification "
          f"(max {MAX_PAIRS_PER_TICKER} per ticker)")
    if not selected.empty:
        print("Top 15 by score:")
        cols = ["pair", "score", "corr", "beta", "eg_p", "half_life_days",
                "hurst", "beta_cv"]
        print(selected.head(15)[cols].round(3).to_string(index=False))

    # Save audit trail
    scored.to_csv(DATA / "expanded_universe_candidates.csv", index=False)
    selected.to_csv(DATA / "expanded_universe_selected.csv", index=False)
    rejected = scored[scored["reject_reason"].notna()]
    rejected.to_csv(DATA / "expanded_universe_rejected.csv", index=False)
    print(f"\nSaved:")
    print(f"  data/expanded_universe_candidates.csv  ({len(scored)})")
    print(f"  data/expanded_universe_selected.csv    ({len(selected)})")
    print(f"  data/expanded_universe_rejected.csv    ({len(rejected)})")

    # Convenience output for bias_test
    if not selected.empty:
        bt_csv = DATA / "pairs_scored_selected.csv"
        selected[["pair"]].to_csv(bt_csv, index=False)
        print(f"  data/pairs_scored_selected.csv (for bias_test/wfo_daily)")


if __name__ == "__main__":
    main()
