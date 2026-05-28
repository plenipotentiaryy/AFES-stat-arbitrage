"""Build a production `live_state.pkl` for AFES paper trading.

This script bridges the audited daily research stack and the existing
`step6_paper_trade.py` live simulator.  The legacy intraday WFO writes a
`live_state.pkl`, but the production shortlist is now defined separately as:

    baseline same-sector pairs + selected ADR/local structural pairs

For each production pair this script estimates:

* OLS hedge ratio on log prices.
* AR(1) half-life of the log-spread.
* Conservative default entry/exit/stop knobs.
* Equal production weights by default.

It intentionally does not include MetaGate, PCA baskets, broad Russell scans,
or research-only candidate scoring artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import DATA_DIR, OUTPUT_DIR
from risk import ExposureCapsController, TailEVRiskGovernor


DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.0
DEFAULT_STOP_Z = 3.5
DEFAULT_LOOKBACK_DAYS = 756
MIN_OBS = 252


def estimate_beta(log_y: pd.Series, log_x: pd.Series) -> float:
    """Estimate hedge ratio beta in log_y = alpha + beta * log_x + error."""
    aligned = pd.concat([log_y, log_x], axis=1).dropna()
    if len(aligned) < MIN_OBS:
        raise ValueError(f"not enough observations for beta: {len(aligned)}")
    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = sm.add_constant(aligned.iloc[:, 1].to_numpy(dtype=float))
    res = sm.OLS(y, x).fit()
    beta = float(res.params[1])
    if not np.isfinite(beta) or abs(beta) < 0.05 or abs(beta) > 20:
        raise ValueError(f"unstable beta estimate: {beta}")
    return beta


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from an AR(1) approximation."""
    s = spread.dropna()
    if len(s) < MIN_OBS:
        raise ValueError(f"not enough observations for half-life: {len(s)}")
    lag = s.shift(1)
    delta = s.diff()
    aligned = pd.concat([delta, lag], axis=1).dropna()
    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = sm.add_constant(aligned.iloc[:, 1].to_numpy(dtype=float))
    res = sm.OLS(y, x).fit()
    phi = float(res.params[1])
    if not np.isfinite(phi) or phi >= 0:
        return 90.0
    hl = float(-np.log(2.0) / phi)
    if not np.isfinite(hl) or hl <= 0:
        return 90.0
    return float(np.clip(hl, 2.0, 180.0))


def build_state(
    production: pd.DataFrame,
    closes: pd.DataFrame,
    lookback_days: int,
    capital: float,
    max_portfolio_es_pct: float,
    max_pair_tail_contribution: float,
    tail_contribution_mode: str,
) -> dict:
    """Build a pickle-safe state consumed by step6_paper_trade.py."""
    recent = closes.tail(lookback_days).copy()
    rows = []
    window_best = {}
    tail_governor = TailEVRiskGovernor(
        max_portfolio_es_pct=max_portfolio_es_pct,
        max_pair_tail_contribution=max_pair_tail_contribution,
        contribution_mode=tail_contribution_mode,
    )
    exposure_caps = ExposureCapsController()
    pair_sector_map = {}

    for row in production.itertuples(index=False):
        pair = str(row.pair)
        t1 = str(row.leg1)
        t2 = str(row.leg2)
        if t1 not in recent.columns or t2 not in recent.columns:
            rows.append({
                "pair": pair,
                "status": "skipped",
                "reason": f"missing close column {t1 if t1 not in recent.columns else t2}",
            })
            continue
        px = recent[[t1, t2]].dropna()
        if len(px) < MIN_OBS:
            rows.append({
                "pair": pair,
                "status": "skipped",
                "reason": f"insufficient daily history ({len(px)} obs)",
            })
            continue
        try:
            log1 = np.log(px[t1].astype(float))
            log2 = np.log(px[t2].astype(float))
            beta = estimate_beta(log1, log2)
            spread = log1 - beta * log2
            dollar_spread = px[t1].astype(float) - beta * px[t2].astype(float)
            hl = estimate_half_life(spread)
            spread_std = float(dollar_spread.tail(60).std())
            z = ((spread - spread.rolling(60).mean()) / spread.rolling(60).std()).dropna()
            tail_governor.fit_pair_gpd(pair, z.iloc[:-1].to_numpy(dtype=float))
        except Exception as exc:
            rows.append({"pair": pair, "status": "skipped", "reason": str(exc)})
            continue

        source_layer = getattr(row, "source_layer", "")
        sector = (
            f"ADR_{getattr(row, 'currency', 'FX')}"
            if source_layer == "adr_local_structural"
            else pair
        )
        pair_sector_map[pair] = str(sector)
        window_best[pair] = {
            "dynamic_beta": beta,
            "dynamic_hl": hl,
            "entry_z": DEFAULT_ENTRY_Z,
            "exit_z": DEFAULT_EXIT_Z,
            "stop_z": DEFAULT_STOP_Z,
            "adaptive": True,
            "post_trade": True,
            "source_layer": source_layer,
            "structure_type": "pair",
            "spread_std_usd": spread_std,
            "position_size_units": max(capital / max(len(production), 1), 1.0),
            "sector": str(sector),
        }
        rows.append({
            "pair": pair,
            "status": "active",
            "reason": "",
            "beta": beta,
            "half_life": hl,
            "spread_std_usd": spread_std,
            "source_layer": source_layer,
            "sector": str(sector),
        })

    if not window_best:
        raise RuntimeError("No production pairs could be activated.")

    equal_weight = 1.0 / len(window_best)
    weights = {pair: equal_weight for pair in window_best}
    return {
        "window_best": window_best,
        "optimal_weights": weights,
        "production_universe": production["pair"].tolist(),
        "production_live_state": True,
        "state_builder": "build_production_live_state.py",
        "current_capital": float(capital),
        "pair_sector_map": pair_sector_map,
        "exposure_caps": exposure_caps.to_dict(),
        "tail_ev_governor": tail_governor.to_dict(),
        "diagnostics": pd.DataFrame(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AFES production paper-trading state.")
    parser.add_argument("--universe", default="production_universe.csv",
                        help="CSV in data/ with production pairs.")
    parser.add_argument("--closes", default="closes_daily_extended.csv",
                        help="Daily closes file in data/. Must include synthetic ADR local USD legs.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Trailing daily bars used to estimate beta/half-life.")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--max-portfolio-es-pct", type=float, default=0.02)
    parser.add_argument("--max-pair-tail-contribution", type=float, default=0.25)
    parser.add_argument(
        "--tail-contribution-mode",
        default="BUDGET_RELATIVE",
        choices=["BUDGET_RELATIVE", "ACTIVE_COUNT_CONDITIONAL"],
    )
    parser.add_argument("--output", default=str(OUTPUT_DIR / "live_state_production.pkl"),
                        help="Output pickle path. Use AFES_LIVE_STATE to point step6 at it.")
    args = parser.parse_args()

    universe_path = DATA_DIR / args.universe
    closes_path = DATA_DIR / args.closes
    if not universe_path.exists():
        raise FileNotFoundError(f"{universe_path} not found. Run build_production_universe.py.")
    if not closes_path.exists():
        raise FileNotFoundError(f"{closes_path} not found. Run universe_expand.py.")

    production = pd.read_csv(universe_path)
    if "status" in production.columns:
        production = production[production["status"].astype(str).str.lower().eq("production")]
    required = {"pair", "leg1", "leg2"}
    missing = required.difference(production.columns)
    if missing:
        raise ValueError(f"{universe_path} missing columns: {sorted(missing)}")

    closes = pd.read_csv(closes_path, parse_dates=["Date"]).set_index("Date").sort_index()
    state = build_state(
        production,
        closes,
        args.lookback_days,
        capital=args.capital,
        max_portfolio_es_pct=args.max_portfolio_es_pct,
        max_pair_tail_contribution=args.max_pair_tail_contribution,
        tail_contribution_mode=args.tail_contribution_mode,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag = state.pop("diagnostics")
    joblib.dump(state, out_path)

    diag_path = DATA_DIR / "production_live_state_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    active = diag[diag["status"].eq("active")]
    skipped = diag[~diag["status"].eq("active")]
    print(f"Saved production live state -> {out_path}")
    print(f"Diagnostics -> {diag_path}")
    print(f"Active pairs: {len(active)} / {len(diag)}")
    if not active.empty:
        print(active["source_layer"].value_counts().to_string())
    if not skipped.empty:
        print("\nSkipped pairs:")
        print(skipped[["pair", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
