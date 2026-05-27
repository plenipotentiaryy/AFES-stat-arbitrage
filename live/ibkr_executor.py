"""ibkr_executor.py — Submit staged MOC orders to IBKR via ib_insync.

Reads pending orders from state DB (written by daily_signal.py), submits
each as a MarketOnClose order, then monitors fills.  Updates DB with fill
prices and marks positions as 'open' or 'closed'.

CLI:
    python -m live.ibkr_executor --host 127.0.0.1 --port 7497 --client-id 7
                                  [--paper] [--dry-run]

Pre-reqs:
    pip install ib_insync
    TWS or IB Gateway running, API enabled, port 7497 (paper) or 7496 (live).
"""
from __future__ import annotations
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

try:
    from ib_insync import IB, Stock, MarketOrder, util
    HAVE_IB = True
except ImportError:
    HAVE_IB = False

from live import state as st

# Routing overrides for non-NYSE/NASDAQ tickers
EXCHANGE_OVERRIDES = {
    "RY.TO":   ("TSE",   "CAD"),  # TD CAD listing
    "TD.TO":   ("TSE",   "CAD"),
    "BNS.TO":  ("TSE",   "CAD"),
    "ENB.TO":  ("TSE",   "CAD"),
    "SU.TO":   ("TSE",   "CAD"),
    "SAP.DE":  ("IBIS",  "EUR"),
    "SAN.PA":  ("SBF",   "EUR"),
    "NOVN.SW": ("EBS",   "CHF"),
    "BHP.AX":  ("ASX",   "AUD"),
    "RIO.AX":  ("ASX",   "AUD"),
}


def make_contract(ticker: str):
    if ticker in EXCHANGE_OVERRIDES:
        exch, ccy = EXCHANGE_OVERRIDES[ticker]
        return Stock(ticker.split(".")[0], exch, ccy)
    return Stock(ticker, "SMART", "USD")


def submit_one(ib: IB, ticker: str, action: str, qty: float, dry_run: bool):
    contract = make_contract(ticker)
    # MOC = MarketOnClose; ib_insync exposes via MarketOrder + tif='MOC'
    order = MarketOrder(action, abs(qty))
    order.tif = "MOC"
    if dry_run:
        print(f"  [dry] {action:4s} {qty:>8.2f} {ticker:8s} MOC  "
              f"({contract.exchange}/{contract.currency})")
        return None
    trade = ib.placeOrder(contract, order)
    return trade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",      default="127.0.0.1")
    ap.add_argument("--port",      type=int, default=7497)   # 7497 paper / 7496 live
    ap.add_argument("--client-id", type=int, default=7,      dest="client_id")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--timeout",   type=int, default=120,
                    help="Seconds to wait for fills after submit.")
    args = ap.parse_args()

    conn = st.connect()
    pending = conn.execute(
        "SELECT order_id, ticker, action, qty, pair FROM orders WHERE status='pending'"
    ).fetchall()
    if not pending:
        print("No pending orders. Exiting.")
        return

    print(f"Submitting {len(pending)} pending orders ...")
    if not args.dry_run:
        if not HAVE_IB:
            raise SystemExit("ib_insync not installed: pip install ib_insync")
        ib = IB()
        ib.connect(args.host, args.port, clientId=args.client_id)
        print(f"  connected: {args.host}:{args.port} clientId={args.client_id}")

    trades = []
    for order_id, ticker, action, qty, pair in pending:
        if args.dry_run:
            submit_one(None, ticker, action, qty, dry_run=True)
            continue
        trade = submit_one(ib, ticker, action, qty, dry_run=False)
        trades.append((order_id, trade))

    if args.dry_run:
        print("\n[dry-run] no orders sent")
        return

    # Wait for fills
    print(f"\nWaiting up to {args.timeout}s for closing-auction fills ...")
    ib.sleep(2)
    deadline = util.run(ib.sleep(args.timeout))
    for order_id, trade in trades:
        ib.sleep(0.1)
        if trade.orderStatus.status in ("Filled",):
            fill_price = trade.orderStatus.avgFillPrice
            st.update_fill(conn, order_id, float(fill_price))
            print(f"  ✓ FILLED {order_id} @ {fill_price:.4f}")
        else:
            print(f"  ⚠ {order_id} status={trade.orderStatus.status}")

    # Promote pending_fill positions to open if both legs filled
    pairs_in_batch = {p for _, _, _, _, p in pending}
    for pair in pairs_in_batch:
        rows = conn.execute(
            "SELECT leg, status FROM orders WHERE pair=? AND ts >= datetime('now','-1 day')",
            (pair,)
        ).fetchall()
        legs_filled = sum(1 for _, st_ in rows if st_ == "filled")
        if legs_filled == 2:
            conn.execute("UPDATE positions SET status='open' WHERE pair=? AND status='pending_fill'",
                          (pair,))
            # If this was an exit, mark closed
            conn.execute("UPDATE positions SET status='closed' WHERE pair=? AND status='closing'",
                          (pair,))
    conn.commit()
    ib.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
