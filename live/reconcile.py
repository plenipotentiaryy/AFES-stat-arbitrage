"""reconcile.py - compare AFES SQLite state with IBKR positions.

Run this before signal generation.  In environments without ``ib_insync`` it
prints the database book only, which is useful for local dry-runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import IB
    HAVE_IB = True
except ImportError:  # pragma: no cover
    HAVE_IB = False

from live import state as st
from live.alerts import send_alert


def db_ticker_book(positions: list[st.Position]) -> dict[str, float]:
    book: dict[str, float] = {}
    for pos in positions:
        t1 = pos.trade_ticker1 or pos.leg1 or pos.pair.split("-")[0]
        t2 = pos.trade_ticker2 or pos.leg2 or pos.pair.split("-")[1]
        q1 = +pos.qty_t1 if pos.side == +1 else -pos.qty_t1
        q2 = -pos.qty_t2 if pos.side == +1 else +pos.qty_t2
        book[t1] = book.get(t1, 0.0) + q1
        book[t2] = book.get(t2, 0.0) + q2
    return book


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile AFES DB positions against IBKR.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=8, dest="client_id")
    ap.add_argument("--db", default=str(st.DB_PATH))
    ap.add_argument("--halt-on-mismatch", action="store_true")
    ap.add_argument("--tolerance-shares", type=float, default=1.0)
    args = ap.parse_args()

    conn = st.connect(Path(args.db))
    positions = st.list_active(conn)
    db_book = db_ticker_book(positions)

    if not HAVE_IB:
        print("[mock] ib_insync not installed - DB state only")
        print(f"DB has {len(positions)} open positions:")
        for pos in positions:
            print(
                f"  {pos.pair:25s} side={pos.side:+d} "
                f"{pos.trade_ticker1 or pos.leg1}:{pos.qty_t1:.2f} "
                f"{pos.trade_ticker2 or pos.leg2}:{pos.qty_t2:.2f}"
            )
        return

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    ib_positions = {}
    for p in ib.positions():
        symbol = p.contract.localSymbol or p.contract.symbol
        ib_positions[symbol] = ib_positions.get(symbol, 0.0) + float(p.position)
        # Also populate plain symbol for SMART US contracts.
        ib_positions[p.contract.symbol] = ib_positions.get(p.contract.symbol, 0.0) + float(p.position)
    ib.disconnect()

    print(f"DB tickers: {len(db_book)}    IBKR tickers: {len(ib_positions)}")
    mismatches = []
    for ticker in sorted(set(db_book) | set(ib_positions)):
        db_q = float(db_book.get(ticker, 0.0))
        ib_q = float(ib_positions.get(ticker, 0.0))
        if abs(db_q - ib_q) > abs(float(args.tolerance_shares)):
            mismatches.append((ticker, db_q, ib_q))
            print(f"  MISMATCH {ticker:12s} DB={db_q:+.2f} IBKR={ib_q:+.2f}")

    if mismatches:
        send_alert("Reconciliation mismatch", level="ERROR", extra={"count": len(mismatches)})
        if args.halt_on_mismatch:
            raise SystemExit(2)
    else:
        print("Reconciliation clean.")


if __name__ == "__main__":
    main()
