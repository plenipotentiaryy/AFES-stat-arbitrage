"""Factor attribution for AFES daily PnL.

This script regresses daily strategy PnL on liquid market/sector/factor ETF
returns to answer a basic but critical question:

    Is the backtest producing idiosyncratic spread alpha, or is it just a
    disguised exposure to market, sector, size, momentum, or rates factors?

The dependent variable is the daily PnL series from a trade ledger.  The
independent variables are daily log returns of available ETF proxies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_FACTORS = [
    "SPY",   # US market
    "QQQ",   # growth/tech
    "IWM",   # small-cap / size
    "MDY",   # mid-cap
    "XLF",   # financials
    "XLK",   # technology
    "XLE",   # energy
    "XLI",   # industrials
    "XLV",   # healthcare
    "XLP",   # staples
    "XLY",   # discretionary
    "XLU",   # utilities
    "VNQ",   # real estate
    "TLT",   # duration
    "HYG",   # credit risk
    "GLD",   # gold / crisis hedge
]


def load_daily_pnl(trades_path: str, factor_days: pd.DatetimeIndex) -> pd.Series:
    trades = pd.read_csv(trades_path, parse_dates=["entry", "exit"])
    if "pnl" not in trades.columns:
        raise ValueError(f"{trades_path} must contain 'pnl'")
    first_exit = trades["exit"].min().normalize()
    last_exit = trades["exit"].max().normalize()
    oos_days = factor_days[(factor_days >= first_exit) & (factor_days <= last_exit)]
    daily = trades.groupby(trades["exit"].dt.normalize())["pnl"].sum()
    return daily.reindex(oos_days, fill_value=0.0).rename("strategy_pnl")


def load_factor_returns(closes_path: str, factors: list[str]) -> pd.DataFrame:
    closes = pd.read_csv(closes_path, parse_dates=["Date"]).set_index("Date").sort_index()
    available = [f for f in factors if f in closes.columns]
    if not available:
        raise ValueError("None of the requested factor columns exist in closes file")
    ret = np.log(closes[available]).diff().replace([np.inf, -np.inf], np.nan)
    return ret.dropna(how="all")


def fit_ols(y: pd.Series, x: pd.DataFrame):
    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < max(60, x.shape[1] * 10):
        raise ValueError(f"Not enough aligned observations: {len(aligned)}")
    yy = aligned.iloc[:, 0]
    xx = sm.add_constant(aligned.iloc[:, 1:])
    model = sm.OLS(yy, xx).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model, aligned


def rolling_beta(y: pd.Series, factor: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([y, factor], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    yy = aligned.iloc[:, 0]
    xx = aligned.iloc[:, 1]
    cov = yy.rolling(window).cov(xx)
    var = xx.rolling(window).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="AFES factor attribution.")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--closes", required=True)
    parser.add_argument("--factors", nargs="*", default=DEFAULT_FACTORS)
    parser.add_argument("--out", default="output/factor_attribution_report.json")
    parser.add_argument("--csv-out", default="data/factor_attribution_daily.csv")
    args = parser.parse_args()

    factor_ret = load_factor_returns(args.closes, args.factors)
    y = load_daily_pnl(args.trades, factor_ret.index)
    model, aligned = fit_ols(y, factor_ret)

    coefs = []
    for name in model.params.index:
        coefs.append({
            "factor": name,
            "coef": float(model.params[name]),
            "tstat_hac": float(model.tvalues[name]),
            "pvalue_hac": float(model.pvalues[name]),
        })

    pred = model.predict(sm.add_constant(aligned.iloc[:, 1:]))
    residual = aligned.iloc[:, 0] - pred
    alpha_series = model.params.get("const", 0.0) + residual
    factor_neutral_sharpe = (
        float(alpha_series.mean() / alpha_series.std() * np.sqrt(252))
        if alpha_series.std() > 0 else float("nan")
    )
    raw = aligned.iloc[:, 0]
    raw_sharpe = (
        float(raw.mean() / raw.std() * np.sqrt(252))
        if raw.std() > 0 else float("nan")
    )
    factor_explained_pnl = float((pred - model.params.get("const", 0.0)).sum())
    alpha_pnl = float(alpha_series.sum())

    rolling = {}
    for f in ["SPY", "QQQ", "IWM", "XLF", "XLK"]:
        if f in aligned.columns:
            rb = rolling_beta(aligned.iloc[:, 0], aligned[f], 126)
            if len(rb):
                rolling[f] = {
                    "mean": float(rb.mean()),
                    "min": float(rb.min()),
                    "max": float(rb.max()),
                    "latest": float(rb.iloc[-1]),
                }

    out_daily = pd.DataFrame({
        "strategy_pnl": aligned.iloc[:, 0],
        "factor_pred": pred,
        "residual": residual,
        "factor_neutral_alpha": alpha_series,
    }, index=aligned.index)
    out_daily.to_csv(args.csv_out)

    payload = {
        "n_days": int(len(aligned)),
        "raw_sharpe": raw_sharpe,
        "factor_neutral_sharpe": factor_neutral_sharpe,
        "total_pnl": float(raw.sum()),
        "factor_explained_pnl": factor_explained_pnl,
        "alpha_plus_residual_pnl": alpha_pnl,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "alpha_daily": float(model.params.get("const", np.nan)),
        "alpha_tstat_hac": float(model.tvalues.get("const", np.nan)),
        "coefficients": coefs,
        "rolling_betas_126d": rolling,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("FACTOR ATTRIBUTION")
    print(f"  aligned days:       {payload['n_days']}")
    print(f"  total PnL:          {payload['total_pnl']:+.4f}")
    print(f"  raw Sharpe:         {raw_sharpe:+.3f}")
    print(f"  factor-neutral Sh:  {factor_neutral_sharpe:+.3f}")
    print(f"  R^2:                {payload['r_squared']:.3f}")
    print(f"  alpha daily:        {payload['alpha_daily']:+.5f}  "
          f"t={payload['alpha_tstat_hac']:+.2f}")
    print("\nLargest absolute factor loadings:")
    loadings = [c for c in coefs if c["factor"] != "const"]
    loadings = sorted(loadings, key=lambda x: abs(x["coef"]), reverse=True)
    for c in loadings[:10]:
        print(f"  {c['factor']:<6s} coef={c['coef']:+.4f} "
              f"t={c['tstat_hac']:+.2f} p={c['pvalue_hac']:.3f}")
    print(f"\nSaved -> {args.out}")
    print(f"Daily -> {args.csv_out}")


if __name__ == "__main__":
    main()
