"""daily_signal.py - production daily signal staging for AFES live execution.

This module is intentionally broker-neutral.  It reads the production universe
and live-state parameters, computes one daily signal pass, applies local risk
guards, and stages MOC orders into the SQLite state database.  The IBKR
executor is the only module that can transmit orders.

Default behavior is safe: ``--dry-run`` prints signals without database writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - surfaced at runtime.
    yf = None

from config import DATA_DIR, OUTPUT_DIR
from risk import ExposureCapsController, TailEVRiskGovernor
from live import state as st


Z_WIN = 60
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5
MAX_HOLD = 30
BETA_TRAIN_DAYS = 504

DEFAULT_UNIVERSE = DATA_DIR / "production_universe.csv"
DEFAULT_LIVE_STATE = OUTPUT_DIR / "live_state_production.pkl"

SYNTHETIC_USD_LEGS = {
    "BHP.AX_USD": ("BHP.AX", "AUDUSD=X"),
    "RIO.AX_USD": ("RIO.AX", "AUDUSD=X"),
    "SHEL.L_USD": ("SHEL.L", "GBPUSD=X"),
    "BP.L_USD": ("BP.L", "GBPUSD=X"),
    "SAP.DE_USD": ("SAP.DE", "EURUSD=X"),
    "SAN.PA_USD": ("SAN.PA", "EURUSD=X"),
    "NOVN.SW_USD": ("NOVN.SW", "CHFUSD=X"),
}


@dataclass(frozen=True)
class PairSpec:
    pair: str
    leg1: str
    leg2: str
    trade_ticker1: str
    trade_ticker2: str
    source_layer: str
    sector: str
    status: str = "production"


@dataclass(frozen=True)
class Signal:
    pair: str
    action: str
    side: int
    z: float
    beta: float
    spread_log: float
    spread_std_usd: float
    reason: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def split_pair(pair: str) -> tuple[str, str]:
    parts = str(pair).split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad pair name {pair!r}; expected LEG1-LEG2")
    return parts[0], parts[1]


def load_live_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_production_live_state.py first."
        )
    return joblib.load(path)


def load_pair_specs(universe_path: Path, live_state: dict) -> list[PairSpec]:
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found. Run build_production_universe.py first."
        )
    df = pd.read_csv(universe_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("production")]
    required = {"pair", "leg1", "leg2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{universe_path} missing columns: {sorted(missing)}")

    active = set(live_state.get("optimal_weights", {}).keys())
    if active:
        df = df[df["pair"].astype(str).isin(active)]

    pair_sector_map = live_state.get("pair_sector_map", {})
    specs = []
    for row in df.itertuples(index=False):
        pair = str(row.pair)
        leg1 = str(row.leg1)
        leg2 = str(row.leg2)
        trade_ticker2 = leg2
        if leg2 in SYNTHETIC_USD_LEGS:
            trade_ticker2 = SYNTHETIC_USD_LEGS[leg2][0]
        specs.append(PairSpec(
            pair=pair,
            leg1=leg1,
            leg2=leg2,
            trade_ticker1=leg1 if leg1 not in SYNTHETIC_USD_LEGS else SYNTHETIC_USD_LEGS[leg1][0],
            trade_ticker2=trade_ticker2,
            source_layer=str(getattr(row, "source_layer", "")),
            sector=str(pair_sector_map.get(pair, getattr(row, "source_layer", "UNKNOWN"))),
            status=str(getattr(row, "status", "production")),
        ))
    return specs


def raw_yahoo_tickers(specs: list[PairSpec]) -> list[str]:
    raw = set()
    for spec in specs:
        for leg in (spec.leg1, spec.leg2):
            if leg in SYNTHETIC_USD_LEGS:
                local, fx = SYNTHETIC_USD_LEGS[leg]
                raw.add(local)
                raw.add(fx)
            else:
                raw.add(leg)
    return sorted(raw)


def add_synthetic_legs(closes: pd.DataFrame, specs: list[PairSpec]) -> pd.DataFrame:
    out = closes.copy()
    requested = {spec.leg1 for spec in specs} | {spec.leg2 for spec in specs}
    for synthetic in sorted(requested):
        if synthetic not in SYNTHETIC_USD_LEGS:
            continue
        local, fx = SYNTHETIC_USD_LEGS[synthetic]
        if local not in out.columns or fx not in out.columns:
            continue
        out[synthetic] = out[local].astype(float) * out[fx].astype(float)
    return out


def fetch_snapshot(specs: list[PairSpec], days: int = 760) -> pd.DataFrame:
    """Pull daily adjusted closes and build USD-normalized synthetic legs."""
    if yf is None:
        raise RuntimeError("yfinance not installed: pip install yfinance")
    tickers = raw_yahoo_tickers(specs)
    if not tickers:
        raise RuntimeError("No tickers to fetch")
    df = yf.download(
        tickers,
        period=f"{int(days)}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("yfinance returned an empty panel")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.sort_index()
    return add_synthetic_legs(close, specs)


def load_closes_csv(path: Path, specs: list[PairSpec]) -> pd.DataFrame:
    """Load a local daily close panel for offline smoke tests."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return add_synthetic_legs(df, specs)


def spread_for_pair(closes: pd.DataFrame, leg1: str, leg2: str, beta: float) -> pd.Series:
    sub = closes[[leg1, leg2]].dropna()
    if len(sub) < Z_WIN + 20:
        return pd.Series(dtype=float)
    return np.log(sub[leg1].astype(float)) - float(beta) * np.log(sub[leg2].astype(float))


def dollar_spread_for_pair(closes: pd.DataFrame, leg1: str, leg2: str, beta: float) -> pd.Series:
    """Dollar spread matching PnL units: P1 - beta * P2."""
    sub = closes[[leg1, leg2]].dropna()
    if len(sub) < Z_WIN + 20:
        return pd.Series(dtype=float)
    return sub[leg1].astype(float) - float(beta) * sub[leg2].astype(float)


def estimate_beta(closes: pd.DataFrame, leg1: str, leg2: str) -> float | None:
    sub = closes[[leg1, leg2]].dropna()
    if len(sub) < Z_WIN + 20:
        return None
    train = sub.iloc[-BETA_TRAIN_DAYS:] if len(sub) > BETA_TRAIN_DAYS else sub
    x = np.log(train[leg2].to_numpy(dtype=float))
    y = np.log(train[leg1].to_numpy(dtype=float))
    var = float(x.var())
    if var <= 0:
        return None
    beta = float(np.cov(y, x, ddof=0)[0, 1] / var)
    if not np.isfinite(beta) or not (0.05 <= abs(beta) <= 20.0):
        return None
    return beta


def signal_for_pair(
    closes: pd.DataFrame,
    spec: PairSpec,
    params: dict,
    open_pos: st.Position | None,
) -> Signal | None:
    beta = float(params.get("dynamic_beta") or params.get("beta") or 0.0)
    if not np.isfinite(beta) or abs(beta) < 0.05:
        est = estimate_beta(closes, spec.leg1, spec.leg2)
        if est is None:
            return None
        beta = est

    spread = spread_for_pair(closes, spec.leg1, spec.leg2, beta)
    if len(spread) < Z_WIN:
        return None
    dollar_spread = dollar_spread_for_pair(closes, spec.leg1, spec.leg2, beta)
    if len(dollar_spread) < Z_WIN:
        return None
    win = spread.iloc[-Z_WIN:]
    mu = float(win.mean())
    sd = float(win.std())
    if not np.isfinite(sd) or sd <= 0:
        return None

    z = float((spread.iloc[-1] - mu) / sd)
    spread_now = float(spread.iloc[-1])
    spread_std = float(dollar_spread.iloc[-Z_WIN:].std())
    if not np.isfinite(spread_std) or spread_std <= 0:
        return None
    entry_z = float(params.get("entry_z", ENTRY_Z))
    exit_z = float(params.get("exit_z", EXIT_Z))
    stop_z = float(params.get("stop_z", STOP_Z))
    max_hold = int(params.get("max_hold_days", params.get("max_hold", MAX_HOLD)))

    if open_pos is not None and open_pos.status == "open":
        try:
            entry_date = pd.Timestamp(open_pos.entry_ts).date()
            held_days = max(0, (pd.Timestamp(spread.index[-1]).date() - entry_date).days)
        except Exception:
            held_days = 0
        side = int(open_pos.side)
        hit_target = (side == -1 and z <= exit_z) or (side == +1 and z >= exit_z)
        hit_stop = abs(z) >= stop_z
        hit_time = held_days >= max_hold
        if hit_target or hit_stop or hit_time:
            reason = "target" if hit_target else ("stop" if hit_stop else "max_hold")
            return Signal(spec.pair, "EXIT", side, z, beta, spread_now, spread_std, reason)
        return None

    if open_pos is not None:
        return None
    if z > entry_z:
        return Signal(spec.pair, "ENTRY", -1, z, beta, spread_now, spread_std, "entry_short")
    if z < -entry_z:
        return Signal(spec.pair, "ENTRY", +1, z, beta, spread_now, spread_std, "entry_long")
    return None


def restore_risk(live_state: dict) -> tuple[ExposureCapsController, TailEVRiskGovernor, dict[str, str]]:
    exp = live_state.get("exposure_caps")
    tail = live_state.get("tail_ev_governor")
    exposure = ExposureCapsController.from_dict(exp) if isinstance(exp, dict) else ExposureCapsController()
    governor = TailEVRiskGovernor.from_dict(tail) if isinstance(tail, dict) else TailEVRiskGovernor(max_portfolio_es_pct=1.0)
    sector_map = {str(k): str(v) for k, v in live_state.get("pair_sector_map", {}).items()}
    return exposure, governor, sector_map


def open_positions_for_risk(
    positions: list[st.Position],
    spread_stds: dict[str, float],
) -> list[dict]:
    out = []
    for pos in positions:
        if pos.status != "open":
            continue
        out.append(pos.to_risk_dict(spread_std_usd=spread_stds.get(pos.pair, 1.0)))
    return out


def kill_switch_triggered(conn, current_capital: float, max_dd_pct: float, days: int) -> tuple[bool, str]:
    if current_capital <= 0:
        return True, "invalid capital"
    pnl = st.recent_net_pnl(conn, days=days)
    threshold = -abs(float(max_dd_pct)) * current_capital
    if pnl <= threshold:
        return True, f"{days}d net PnL {pnl:.2f} <= kill threshold {threshold:.2f}"
    return False, "OK"


def stage_exit_orders(conn, spec: PairSpec, pos: st.Position, sig: Signal, ts: str, dry_run: bool) -> None:
    act1 = "SELL" if pos.side == +1 else "BUY"
    act2 = "BUY" if pos.side == +1 else "SELL"
    oid1 = f"{sig.pair}-{ts}-EXIT-L1"
    oid2 = f"{sig.pair}-{ts}-EXIT-L2"
    if dry_run:
        return
    st.record_order(conn, oid1, sig.pair, pos.side, 1, spec.trade_ticker1, act1, pos.qty_t1, reason=sig.reason)
    st.record_order(conn, oid2, sig.pair, pos.side, 2, spec.trade_ticker2, act2, pos.qty_t2, reason=sig.reason)
    st.mark_closing(conn, sig.pair)


def stage_entry_orders(
    conn,
    spec: PairSpec,
    sig: Signal,
    closes: pd.DataFrame,
    notional_usd: float,
    params: dict,
    ts: str,
    dry_run: bool,
) -> None:
    px1 = float(closes[spec.leg1].dropna().iloc[-1])
    px2_usd = float(closes[spec.leg2].dropna().iloc[-1])
    qty_t1 = abs(float(notional_usd)) / px1
    qty_t2 = abs(sig.beta) * abs(float(notional_usd)) / px2_usd
    act1 = "BUY" if sig.side == +1 else "SELL"
    act2 = "SELL" if sig.side == +1 else "BUY"
    oid1 = f"{sig.pair}-{ts}-ENTRY-L1"
    oid2 = f"{sig.pair}-{ts}-ENTRY-L2"
    if dry_run:
        return
    st.record_order(conn, oid1, sig.pair, sig.side, 1, spec.trade_ticker1, act1, qty_t1, reason=sig.reason)
    st.record_order(conn, oid2, sig.pair, sig.side, 2, spec.trade_ticker2, act2, qty_t2, reason=sig.reason)
    st.open_position(conn, st.Position(
        pair=sig.pair,
        side=sig.side,
        qty_t1=qty_t1,
        qty_t2=qty_t2,
        entry_ts=ts,
        entry_spread_log=sig.spread_log,
        status="pending_fill",
        leg1=spec.leg1,
        leg2=spec.leg2,
        trade_ticker1=spec.trade_ticker1,
        trade_ticker2=spec.trade_ticker2,
        beta=sig.beta,
        entry_z=float(params.get("entry_z", ENTRY_Z)),
        exit_z=float(params.get("exit_z", EXIT_Z)),
        stop_z=float(params.get("stop_z", STOP_Z)),
        max_hold_days=int(params.get("max_hold_days", params.get("max_hold", MAX_HOLD))),
        notional_usd=abs(float(notional_usd)),
        sector=spec.sector,
        source_layer=spec.source_layer,
        last_z=sig.z,
        last_signal_ts=ts,
    ))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate AFES production daily signals and stage orders.")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    ap.add_argument("--live-state", default=str(DEFAULT_LIVE_STATE))
    ap.add_argument("--db", default=str(st.DB_PATH))
    ap.add_argument("--notional", type=float, default=1_000.0)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--history-days", type=int, default=760)
    ap.add_argument("--closes-csv", default="",
                    help="Optional local closes CSV with Date column; bypasses yfinance.")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--kill-dd-pct", type=float, default=0.05)
    ap.add_argument("--kill-window-days", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="Print signals without DB writes.")
    args = ap.parse_args()

    live_state = load_live_state(Path(args.live_state))
    specs = load_pair_specs(Path(args.universe), live_state)
    if not specs:
        raise RuntimeError("Production universe has no active pairs")
    current_capital = float(args.capital or live_state.get("current_capital", 10_000.0))
    window_best = live_state.get("window_best", {})
    weights = live_state.get("optimal_weights", {})
    exposure_caps, tail_governor, sector_map = restore_risk(live_state)

    conn = st.connect(Path(args.db))
    kill, kill_reason = kill_switch_triggered(conn, current_capital, args.kill_dd_pct, args.kill_window_days)

    if args.closes_csv:
        print(f"Loading daily panel from {args.closes_csv} ...")
        closes = load_closes_csv(Path(args.closes_csv), specs).tail(args.history_days)
        market_source = f"csv:{args.closes_csv}"
    else:
        print(f"Fetching daily panel for {len(specs)} production pairs ...")
        closes = fetch_snapshot(specs, days=args.history_days)
        market_source = "yfinance:1d"
    print(f"  panel: {closes.shape[0]} days x {closes.shape[1]} columns, last={closes.index[-1].date()}")
    if not args.dry_run:
        st.record_market_data_audit(
            conn,
            source=market_source,
            last_bar_ts=str(closes.index[-1]),
            n_rows=len(closes),
            n_columns=len(closes.columns),
            details=json.dumps({
                "history_days": args.history_days,
                "universe": str(args.universe),
                "live_state": str(args.live_state),
            }),
        )

    all_positions = st.list_open(conn)
    positions = {p.pair: p for p in all_positions}
    active_positions = {p.pair: p for p in st.list_active(conn)}
    spread_stds: dict[str, float] = {}
    signals: list[tuple[PairSpec, Signal]] = []
    ts = utc_now()

    for spec in specs:
        params = dict(window_best.get(spec.pair, {}))
        sig = signal_for_pair(closes, spec, params, positions.get(spec.pair))
        beta_for_risk = float(params.get("dynamic_beta", params.get("beta", 1.0)))
        spread = spread_for_pair(closes, spec.leg1, spec.leg2, beta_for_risk)
        dollar_spread = dollar_spread_for_pair(closes, spec.leg1, spec.leg2, beta_for_risk)
        if not dollar_spread.empty:
            spread_stds[spec.pair] = float(dollar_spread.tail(Z_WIN).std())
        if spec.pair in active_positions and not spread.empty:
            try:
                z_for_book = float((spread.iloc[-1] - spread.tail(Z_WIN).mean()) / spread.tail(Z_WIN).std())
                if not args.dry_run:
                    st.update_position_signal(conn, spec.pair, z_for_book, ts)
            except Exception:
                pass
        if sig is not None:
            signals.append((spec, sig))

    exits = [(s, g) for s, g in signals if g.action == "EXIT"]
    entries = [(s, g) for s, g in signals if g.action == "ENTRY"]

    if kill:
        if not args.dry_run:
            st.log_event(conn, "ERROR", "KILL_SWITCH", kill_reason)
        print(f"[kill-switch] {kill_reason}")
        entries = []
        existing_exits = {sig.pair for _, sig in exits}
        for spec in specs:
            pos = active_positions.get(spec.pair)
            if pos is None or spec.pair in existing_exits:
                continue
            exits.append((spec, Signal(spec.pair, "EXIT", pos.side, pos.last_z, pos.beta,
                                       pos.entry_spread_log, spread_stds.get(spec.pair, 1.0),
                                       "kill_switch")))

    occupied_pairs = {
        p.pair for p in all_positions
        if p.status in ("pending_fill", "open", "closing")
    }
    n_open_after_exits = len(occupied_pairs) - len(exits)
    capacity = max(0, int(args.max_concurrent) - n_open_after_exits)
    if len(entries) > capacity:
        entries.sort(key=lambda x: abs(x[1].z), reverse=True)
        dropped = entries[capacity:]
        entries = entries[:capacity]
        print(f"[cap] dropped {len(dropped)} entries beyond max-concurrent={args.max_concurrent}")

    risk_positions = open_positions_for_risk(list(active_positions.values()), spread_stds)
    staged_entries: list[tuple[PairSpec, Signal, float]] = []
    for spec, sig in entries:
        params = dict(window_best.get(spec.pair, {}))
        weight = abs(float(weights.get(spec.pair, 1.0 / max(len(specs), 1))))
        notional = min(abs(args.notional), current_capital * weight)
        allowed_exp, exp_reason = exposure_caps.check_entry_allowed(
            pair_name=spec.pair,
            direction="LONG" if sig.side == +1 else "SHORT",
            nominal_size_usd=notional,
            open_positions=risk_positions,
            current_capital=current_capital,
            pair_sector_map=sector_map,
        )
        if not allowed_exp:
            print(f"  BLOCK EXPOSURE {spec.pair:25s} {exp_reason}")
            if not args.dry_run:
                st.log_event(conn, "WARN", "BLOCKED_EXPOSURE", f"{spec.pair} {exp_reason}")
            continue
        candidate = {
            "pair": spec.pair,
            "pair_name": spec.pair,
            "legs": [spec.leg1, spec.leg2],
            "nominal_size_usd": notional,
            "current_z": sig.z,
            "spread_std_usd": max(sig.spread_std_usd, 1e-9),
            "position_size_units": max(notional, 1.0),
            "p_revert": 0.50,
            "sector": spec.sector,
        }
        allowed_tail, tail_reason, _ = tail_governor.check_candidate_entry(
            candidate, risk_positions, current_capital
        )
        if not allowed_tail:
            print(f"  BLOCK TAIL     {spec.pair:25s} {tail_reason}")
            if not args.dry_run:
                st.log_event(conn, "WARN", "BLOCKED_TAIL", f"{spec.pair} {tail_reason}")
            continue
        staged_entries.append((spec, sig, notional))
        risk_positions.append(candidate)

    print(f"\nSignals: {len(staged_entries)} entries + {len(exits)} exits")
    for spec, sig in exits:
        print(f"  EXIT  {sig.pair:25s} side={sig.side:+d} z={sig.z:+.2f} ({sig.reason})")
    for spec, sig, notional in staged_entries:
        print(f"  ENTRY {sig.pair:25s} side={sig.side:+d} z={sig.z:+.2f} notional=${notional:,.0f}")

    if args.dry_run:
        print("\n[dry-run] no DB writes")
        return

    spec_map = {spec.pair: spec for spec in specs}
    for spec, sig in exits:
        pos = positions.get(sig.pair)
        if pos is None:
            continue
        stage_exit_orders(conn, spec_map[sig.pair], pos, sig, ts, dry_run=False)
    for spec, sig, notional in staged_entries:
        stage_entry_orders(conn, spec, sig, closes, notional, window_best.get(spec.pair, {}), ts, dry_run=False)

    st.log_event(conn, "INFO", "DAILY_SIGNAL", json.dumps({
        "entries": len(staged_entries),
        "exits": len(exits),
        "dry_run": False,
        "kill_switch": kill,
    }))
    print(f"\nStaged {len(staged_entries) * 2 + len(exits) * 2} orders in {args.db}")


if __name__ == "__main__":
    main()
