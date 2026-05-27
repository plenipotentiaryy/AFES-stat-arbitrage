"""reconcile.py — SOD sanity check: IBKR portfolio vs state DB.

Run at start of day before signal generation.  Halts execution if
positions in DB don't match IBKR account (e.g. manual intervention,
overnight margin call, partial fill not detected).

CLI:
    python -m live.reconcile --host 127.0.0.1 --port 7497 --client-id 7
"""
from __future__ import annotations
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import IB
    HAVE_IB = True
except ImportError:
    HAVE_IB = False

from live import state as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=8, dest="client_id")
    ap.add_argument("--halt-on-mismatch", action="store_true")
    args = ap.parse_args()

    conn = st.connect()
    db_open = {p.pair: p for p in st.list_open(conn)}
    db_tickers = {}
    for pos in db_open.values():
        t1, t2 = pos.pair.split("-")
        db_tickers[t1] = db_tickers.get(t1, 0) + (+pos.qty_t1 if pos.side == +1 else -pos.qty_t1)
        db_tickers[t2] = db_tickers.get(t2, 0) + (-pos.qty_t2 if pos.side == +1 else +pos.qty_t2)

    if not HAVE_IB:
        print("[mock] ib_insync not installed — printing DB state only")
        print(f"DB has {len(db_open)} open positions:")
        for pair, pos in db_open.items():
            print(f"  {pair:25s}  side={pos.side:+d}  qty_t1={pos.qty_t1:.2f}  qty_t2={pos.qty_t2:.2f}")
        return

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    ib_positions = {p.contract.symbol: p.position for p in ib.positions()}

    print(f"DB tickers: {len(db_tickers)}    IBKR tickers: {len(ib_positions)}")
    mismatches = []
    all_tickers = set(db_tickers) | set(ib_positions)
    for tk in sorted(all_tickers):
        db_q = db_tickers.get(tk, 0)
        ib_q = ib_positions.get(tk, 0)
        if abs(db_q - ib_q) > 1.0:   # >1 share tolerance
            mismatches.append((tk, db_q, ib_q))
            print(f"  ⚠ MISMATCH {tk:8s}  DB={db_q:+.2f}  IBKR={ib_q:+.2f}")

    ib.disconnect()
    if mismatches:
        print(f"\n{len(mismatches)} mismatches detected!")
        if args.halt_on_mismatch:
            sys.exit(2)   # cron interprets exit ≥2 as fatal
    else:
        print("\n✓ Reconciliation clean.")


if __name__ == "__main__":
    main()
