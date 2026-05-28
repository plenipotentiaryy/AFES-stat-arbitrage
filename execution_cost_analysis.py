"""Execution-cost stress test for AFES trade ledgers.

The daily research stack often reports PnL after a flat spread/commission
assumption.  This script overlays a liquidity-aware Almgren-Chriss /
square-root impact model on an existing trade ledger to estimate how much
headline Sharpe survives more realistic implementation costs.

Model per trade:

    Cost_trade = flat_cost + sum_legs impact_leg + borrow_cost

For each leg:

    impact_bps = kappa * sigma_daily_bps * (participation ** alpha)

where:

    participation = leg_notional / ADV
    sigma_daily_bps = daily return volatility in bps

This is not a broker-grade simulator.  Its purpose is to detect inflated
results caused by flat costs, async listings, thin liquidity, or excessive
turnover.  Outputs are conservative diagnostics, not execution promises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CAPITAL = 10_000.0
DEFAULT_BASE_NOTIONAL = 1_000.0
DEFAULT_KAPPA = 0.10
DEFAULT_ALPHA = 0.50
DEFAULT_FLAT_BPS = 10.0
DEFAULT_BORROW_BPS = 80.0
DEFAULT_LOCAL_VOLUME_HAIRCUT = 0.25
DEFAULT_MISSING_ADV_PARTICIPATION = 0.05


SYNTHETIC_USD_LEGS = {
    "BHP.AX_USD": "BHP.AX",
    "RIO.AX_USD": "RIO.AX",
    "SHEL.L_USD": "SHEL.L",
    "BP.L_USD": "BP.L",
    "SAP.DE_USD": "SAP.DE",
    "SAN.PA_USD": "SAN.PA",
    "NOVN.SW_USD": "NOVN.SW",
    "SU.TO_USD": "SU.TO",
    "TD.TO_USD": "TD.TO",
    "RY.TO_USD": "RY.TO",
    "BNS.TO_USD": "BNS.TO",
    "CNQ.TO_USD": "CNQ.TO",
    "TRI.TO_USD": "TRI.TO",
}


def split_pair(pair: str) -> tuple[str, str]:
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot split pair {pair!r}")
    return parts[0], parts[1]


def read_price_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def read_volume_table(path: str | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def ticker_for_volume(ticker: str) -> str:
    return SYNTHETIC_USD_LEGS.get(ticker, ticker)


def daily_vol_bps(closes: pd.DataFrame, ticker: str, end_date: pd.Timestamp,
                  lookback: int) -> float:
    if ticker not in closes.columns:
        return 250.0
    px = closes.loc[:end_date, ticker].dropna().tail(lookback + 1)
    if len(px) < 30:
        return 250.0
    ret = np.log(px).diff().dropna()
    vol = float(ret.std() * 1e4)
    return float(np.clip(vol, 25.0, 1500.0))


def adv_usd(closes: pd.DataFrame, volumes: pd.DataFrame, ticker: str,
            end_date: pd.Timestamp, lookback: int,
            local_volume_haircut: float) -> float:
    vol_ticker = ticker_for_volume(ticker)
    price_ticker = ticker
    if vol_ticker not in volumes.columns or price_ticker not in closes.columns:
        return np.nan
    px = closes.loc[:end_date, price_ticker].tail(lookback)
    vo = volumes.loc[:end_date, vol_ticker].tail(lookback)
    aligned = pd.concat([px, vo], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    adv = float((aligned.iloc[:, 0] * aligned.iloc[:, 1]).median())
    if ticker in SYNTHETIC_USD_LEGS:
        adv *= local_volume_haircut
    return adv if np.isfinite(adv) and adv > 0 else np.nan


def impact_cost_per_leg(
    notional: float,
    adv: float,
    sigma_bps: float,
    kappa: float,
    alpha: float,
    missing_adv_participation: float,
) -> tuple[float, float, float]:
    if not np.isfinite(adv) or adv <= 0:
        participation = missing_adv_participation
    else:
        participation = max(notional / adv, 1e-8)
    impact_bps = kappa * sigma_bps * (participation ** alpha)
    cost = notional * impact_bps / 1e4
    return float(cost), float(impact_bps), float(participation)


def sharpe_from_trades(trades: pd.DataFrame, pnl_col: str,
                       oos_days: pd.DatetimeIndex) -> float:
    daily = trades.groupby("exit")[pnl_col].sum().reindex(oos_days, fill_value=0.0)
    sd = daily.std()
    return float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def max_drawdown_from_trades(trades: pd.DataFrame, pnl_col: str) -> float:
    eq = trades.sort_values("exit")[pnl_col].cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def apply_execution_model(
    trades: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    base_notional: float,
    capital: float,
    flat_bps: float,
    borrow_bps: float,
    kappa: float,
    alpha: float,
    lookback: int,
    local_volume_haircut: float,
    missing_adv_participation: float,
) -> pd.DataFrame:
    out = trades.copy()
    out["entry"] = pd.to_datetime(out["entry"])
    out["exit"] = pd.to_datetime(out["exit"])
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    if "size" not in out.columns:
        out["size"] = 1.0
    out["size"] = pd.to_numeric(out["size"], errors="coerce").fillna(1.0)

    rows = []
    for row in out.itertuples(index=False):
        pair = str(row.pair)
        t1, t2 = split_pair(pair)
        entry = pd.Timestamp(row.entry).tz_localize(None) if pd.Timestamp(row.entry).tzinfo else pd.Timestamp(row.entry)
        held = int(getattr(row, "held", 1))
        size = abs(float(row.size))
        gross_notional = base_notional * size
        leg_notional = gross_notional / 2.0

        flat_cost = gross_notional * flat_bps / 1e4
        borrow_cost = leg_notional * borrow_bps / 1e4 * max(held, 0) / 252.0

        leg_rows = []
        impact_total = 0.0
        for leg in (t1, t2):
            sigma = daily_vol_bps(closes, leg, entry, lookback)
            adv = adv_usd(closes, volumes, leg, entry, lookback, local_volume_haircut)
            cost, impact_bps, participation = impact_cost_per_leg(
                leg_notional, adv, sigma, kappa, alpha, missing_adv_participation
            )
            impact_total += cost
            leg_rows.append((leg, adv, sigma, impact_bps, participation))

        total_cost = flat_cost + borrow_cost + impact_total
        # Convert dollar execution drag back to the ledger's PnL scale.  The
        # input daily WFO PnL is a log-spread unit; multiplying by capital gives
        # a consistent portfolio-dollar proxy.  Dividing back by capital keeps
        # output comparable to the original ledger.
        extra_cost_units = total_cost / capital
        adjusted_pnl = float(row.pnl) - extra_cost_units
        rows.append({
            "pair": pair,
            "entry": row.entry,
            "exit": row.exit,
            "held": held,
            "orig_pnl": float(row.pnl),
            "adjusted_pnl": adjusted_pnl,
            "extra_cost_units": extra_cost_units,
            "flat_cost_dollars": flat_cost,
            "borrow_cost_dollars": borrow_cost,
            "impact_cost_dollars": impact_total,
            "gross_notional": gross_notional,
            "leg1": leg_rows[0][0],
            "leg1_adv": leg_rows[0][1],
            "leg1_sigma_bps": leg_rows[0][2],
            "leg1_impact_bps": leg_rows[0][3],
            "leg1_participation": leg_rows[0][4],
            "leg1_adv_missing": not np.isfinite(leg_rows[0][1]),
            "leg2": leg_rows[1][0],
            "leg2_adv": leg_rows[1][1],
            "leg2_sigma_bps": leg_rows[1][2],
            "leg2_impact_bps": leg_rows[1][3],
            "leg2_participation": leg_rows[1][4],
            "leg2_adv_missing": not np.isfinite(leg_rows[1][1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFES Almgren-Chriss execution-cost overlay.")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--closes", required=True)
    parser.add_argument("--volumes", default="data/volumes_daily.csv")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--base-notional", type=float, default=DEFAULT_BASE_NOTIONAL)
    parser.add_argument("--flat-bps", type=float, default=DEFAULT_FLAT_BPS)
    parser.add_argument("--borrow-bps", type=float, default=DEFAULT_BORROW_BPS)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--local-volume-haircut", type=float,
                        default=DEFAULT_LOCAL_VOLUME_HAIRCUT)
    parser.add_argument("--missing-adv-participation", type=float,
                        default=DEFAULT_MISSING_ADV_PARTICIPATION,
                        help="Assumed participation when ADV is unavailable.")
    parser.add_argument("--out", default="output/execution_cost_report.json")
    parser.add_argument("--ledger-out", default="data/execution_adjusted_trades.csv")
    args = parser.parse_args()

    trades = pd.read_csv(args.trades, parse_dates=["entry", "exit"])
    closes = read_price_table(args.closes)
    volumes = read_volume_table(args.volumes)
    adjusted = apply_execution_model(
        trades, closes, volumes,
        base_notional=args.base_notional,
        capital=args.capital,
        flat_bps=args.flat_bps,
        borrow_bps=args.borrow_bps,
        kappa=args.kappa,
        alpha=args.alpha,
        lookback=args.lookback,
        local_volume_haircut=args.local_volume_haircut,
        missing_adv_participation=args.missing_adv_participation,
    )

    oos_days = closes.loc[trades["exit"].min():trades["exit"].max()].index
    merged = trades.copy()
    merged["adjusted_pnl"] = adjusted["adjusted_pnl"].values
    orig_sharpe = sharpe_from_trades(merged, "pnl", oos_days)
    adj_sharpe = sharpe_from_trades(merged, "adjusted_pnl", oos_days)
    orig_dd = max_drawdown_from_trades(merged, "pnl")
    adj_dd = max_drawdown_from_trades(merged, "adjusted_pnl")

    pair_summary = (
        adjusted.groupby("pair")
        .agg(
            trades=("pair", "size"),
            orig_pnl=("orig_pnl", "sum"),
            adjusted_pnl=("adjusted_pnl", "sum"),
            extra_cost_units=("extra_cost_units", "sum"),
            impact_cost_dollars=("impact_cost_dollars", "sum"),
            max_participation=("leg1_participation", "max"),
        )
        .sort_values("adjusted_pnl", ascending=False)
    )

    payload = {
        "n_trades": int(len(adjusted)),
        "orig_pnl": float(adjusted["orig_pnl"].sum()),
        "adjusted_pnl": float(adjusted["adjusted_pnl"].sum()),
        "execution_drag": float(adjusted["extra_cost_units"].sum()),
        "orig_sharpe": orig_sharpe,
        "adjusted_sharpe": adj_sharpe,
        "orig_max_drawdown": orig_dd,
        "adjusted_max_drawdown": adj_dd,
        "capital": args.capital,
        "base_notional": args.base_notional,
        "flat_bps": args.flat_bps,
        "borrow_bps": args.borrow_bps,
        "kappa": args.kappa,
        "alpha": args.alpha,
        "local_volume_haircut": args.local_volume_haircut,
        "missing_adv_participation": args.missing_adv_participation,
        "missing_adv_leg_fraction": float(
            pd.concat([adjusted["leg1_adv_missing"], adjusted["leg2_adv_missing"]]).mean()
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    Path(args.ledger_out).parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(args.ledger_out, index=False)
    pair_summary.to_csv(Path(args.ledger_out).with_name("execution_cost_pair_summary.csv"))

    print("EXECUTION COST OVERLAY")
    print(f"  trades:          {len(adjusted)}")
    print(f"  orig PnL:        {payload['orig_pnl']:+.4f}")
    print(f"  adjusted PnL:    {payload['adjusted_pnl']:+.4f}")
    print(f"  execution drag:  {payload['execution_drag']:+.4f}")
    print(f"  Sharpe:          {orig_sharpe:+.3f} -> {adj_sharpe:+.3f}")
    print(f"  Max DD:          {orig_dd:+.4f} -> {adj_dd:+.4f}")
    print(f"  missing ADV leg fraction: {payload['missing_adv_leg_fraction']:.1%}")
    print("\nWorst cost drag by pair:")
    print(pair_summary.sort_values("extra_cost_units", ascending=False).head(10).to_string())
    print(f"\nSaved -> {args.out}")
    print(f"Ledger -> {args.ledger_out}")


if __name__ == "__main__":
    main()
