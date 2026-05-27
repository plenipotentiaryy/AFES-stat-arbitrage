"""daily_signal.py — Generate today's order list from current closes.

For each production pair:
    1. Fetch latest closes via yfinance (intraday snapshot ~15:50 ET).
    2. Compute rolling z(60) of log(p1) - β·log(p2) where β estimated on
       trailing 504d window.
    3. If no open position and |z| > ENTRY_Z → emit ENTRY order.
    4. If open position and (z hits target OR stop OR max_hold) → emit EXIT.

Writes pending orders into state DB.  Does NOT submit to IBKR; that is
ibkr_executor.py's job.

CLI:
    python -m live.daily_signal --pairs data/pairs_bias_test.csv
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from live import state as st

Z_WIN, ENTRY_Z, EXIT_Z, STOP_Z, MAX_HOLD = 60, 2.0, 0.0, 3.5, 30
BETA_TRAIN_DAYS = 504


def fetch_snapshot(tickers: list[str], days: int = 700) -> pd.DataFrame:
    """Pull last `days` of daily closes for tickers via yfinance."""
    if yf is None:
        raise RuntimeError("yfinance not installed: pip install yfinance")
    df = yf.download(tickers, period=f"{days}d", interval="1d",
                      auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.sort_index()


def signal_for_pair(closes: pd.DataFrame, t1: str, t2: str,
                     open_side: int | None) -> dict | None:
    """Return {action, side, z, beta, spread_log} or None for no-op."""
    if t1 not in closes.columns or t2 not in closes.columns:
        return None
    sub = closes[[t1, t2]].dropna()
    if len(sub) < Z_WIN + 20:
        return None
    train = sub.iloc[-BETA_TRAIN_DAYS:] if len(sub) > BETA_TRAIN_DAYS else sub
    x = np.log(train[t2].values); y = np.log(train[t1].values)
    var = x.var()
    if var <= 0: return None
    beta = float(np.cov(y, x, ddof=0)[0, 1] / var)
    if not (0.1 <= abs(beta) <= 15.0):
        return None
    spread = np.log(sub[t1]) - beta * np.log(sub[t2])
    win = spread.iloc[-Z_WIN:]
    mu, sg = win.mean(), win.std()
    if sg <= 0: return None
    z_now = (spread.iloc[-1] - mu) / sg
    spread_now = float(spread.iloc[-1])

    # Exit logic for open position
    if open_side is not None:
        hit_target = (open_side == -1 and z_now <= EXIT_Z) or (open_side == +1 and z_now >= EXIT_Z)
        hit_stop   = abs(z_now) >= STOP_Z
        if hit_target or hit_stop:
            return dict(action="EXIT", side=open_side, z=z_now, beta=beta,
                         spread_log=spread_now,
                         reason="target" if hit_target else "stop")
        return None

    # Entry logic
    if z_now > ENTRY_Z:
        return dict(action="ENTRY", side=-1, z=z_now, beta=beta,
                     spread_log=spread_now, reason="entry_short")
    if z_now < -ENTRY_Z:
        return dict(action="ENTRY", side=+1, z=z_now, beta=beta,
                     spread_log=spread_now, reason="entry_long")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--notional", type=float, default=100_000)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true",
                     help="Print signals without writing to DB.")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)["pair"].tolist()
    tickers = sorted({t for p in pairs for t in p.split("-")})
    # USD-normalized legs (e.g. RY.TO_USD) need to be reconstructed —
    # for MVP we skip them; live cross-listings need FX pull (Phase 1.5+).
    cleaned_tickers = [t for t in tickers if "_USD" not in t]
    skipped = [t for t in tickers if "_USD" in t]
    if skipped:
        print(f"  [warn] skipping {len(skipped)} USD-normalized legs: "
              f"{skipped[:3]}...  (need separate FX pull)")

    print(f"Fetching {len(cleaned_tickers)} tickers from yfinance ...")
    closes = fetch_snapshot(cleaned_tickers)
    print(f"  panel: {closes.shape[0]} days × {closes.shape[1]} tickers, "
          f"last close = {closes.index[-1].date()}")

    conn = st.connect()
    open_positions = {p.pair: p for p in st.list_open(conn)}
    print(f"  state: {len(open_positions)} open positions in DB")

    entries, exits = [], []
    for pair in pairs:
        t1, t2 = pair.split("-")
        if "_USD" in t1 or "_USD" in t2:
            continue
        open_side = open_positions[pair].side if pair in open_positions else None
        sig = signal_for_pair(closes, t1, t2, open_side)
        if not sig: continue
        if sig["action"] == "EXIT":
            exits.append({"pair": pair, **sig})
        else:
            entries.append({"pair": pair, **sig})

    # Apply concurrent-trade cap to entries
    n_open_after_exits = len(open_positions) - len(exits)
    capacity = max(0, args.max_concurrent - n_open_after_exits)
    if len(entries) > capacity:
        entries.sort(key=lambda e: abs(e["z"]), reverse=True)
        skipped_n = len(entries) - capacity
        entries = entries[:capacity]
        print(f"  [cap] dropped {skipped_n} entry candidates beyond max-concurrent={args.max_concurrent}")

    print(f"\nSignals: {len(entries)} entries  +  {len(exits)} exits")
    for sig in exits:
        print(f"  EXIT  {sig['pair']:25s}  side={sig['side']:+d}  z={sig['z']:+.2f}  ({sig['reason']})")
    for sig in entries:
        print(f"  ENTRY {sig['pair']:25s}  side={sig['side']:+d}  z={sig['z']:+.2f}  β={sig['beta']:+.3f}")

    if args.dry_run:
        print("\n[dry-run] no DB writes")
        return

    # Stage orders into DB as pending (executor will submit them)
    ts = datetime.utcnow().isoformat()
    for sig in entries:
        t1, t2 = sig["pair"].split("-")
        notional = args.notional
        qty_t1 = notional / float(closes[t1].iloc[-1])
        qty_t2 = sig["beta"] * notional / float(closes[t2].iloc[-1])
        # Long spread (side=+1): BUY t1, SELL t2.  Short spread: opposite.
        act1 = "BUY" if sig["side"] == +1 else "SELL"
        act2 = "SELL" if sig["side"] == +1 else "BUY"
        oid_t1 = f"{sig['pair']}-{ts}-L1"
        oid_t2 = f"{sig['pair']}-{ts}-L2"
        st.record_order(conn, oid_t1, sig["pair"], sig["side"], 1, t1, act1, qty_t1)
        st.record_order(conn, oid_t2, sig["pair"], sig["side"], 2, t2, act2, qty_t2)
        # Pre-record position as "pending" — executor will confirm on fill
        st.open_position(conn, st.Position(
            pair=sig["pair"], side=sig["side"], qty_t1=qty_t1, qty_t2=qty_t2,
            entry_ts=ts, entry_spread_log=sig["spread_log"], status="pending_fill"))
    for sig in exits:
        pos = open_positions[sig["pair"]]
        t1, t2 = sig["pair"].split("-")
        # Close: opposite of entry
        act1 = "SELL" if pos.side == +1 else "BUY"
        act2 = "BUY"  if pos.side == +1 else "SELL"
        oid1 = f"{sig['pair']}-{ts}-EXIT-L1"
        oid2 = f"{sig['pair']}-{ts}-EXIT-L2"
        st.record_order(conn, oid1, sig["pair"], pos.side, 1, t1, act1, pos.qty_t1)
        st.record_order(conn, oid2, sig["pair"], pos.side, 2, t2, act2, pos.qty_t2)
        st.mark_closing(conn, sig["pair"])

    print(f"\nStaged {len(entries)*2 + len(exits)*2} orders in DB. "
          f"Run ibkr_executor.py to submit MOC.")


if __name__ == "__main__":
    main()
