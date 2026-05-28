"""ibkr_executor.py - submit staged AFES MOC orders to IBKR.

The executor is deliberately conservative:

* It does not transmit orders unless ``--transmit`` is supplied.
* ``--dry-run`` prints the exact orders and performs no broker connection.
* Margin and borrow checks are best-effort and block when IBKR explicitly
  reports insufficient funds or no short availability.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import IB, Stock, Order
    HAVE_IB = True
except ImportError:  # pragma: no cover
    HAVE_IB = False
    IB = None
    Stock = None
    Order = None

from live import state as st
from live.alerts import send_alert


EXCHANGE_OVERRIDES = {
    "RY.TO": ("TSE", "CAD"),
    "TD.TO": ("TSE", "CAD"),
    "BNS.TO": ("TSE", "CAD"),
    "ENB.TO": ("TSE", "CAD"),
    "SU.TO": ("TSE", "CAD"),
    "SAP.DE": ("IBIS", "EUR"),
    "SAN.PA": ("SBF", "EUR"),
    "NOVN.SW": ("EBS", "CHF"),
    "BHP.AX": ("ASX", "AUD"),
    "RIO.AX": ("ASX", "AUD"),
}


@dataclass(frozen=True)
class DryContract:
    symbol: str
    exchange: str
    currency: str


@dataclass
class DryOrder:
    action: str = ""
    totalQuantity: float = 0.0
    orderType: str = "MOC"
    tif: str = "DAY"
    transmit: bool = False


def make_contract(ticker: str):
    """Build an IBKR stock contract, or a dry contract when ib_insync is absent."""
    if Stock is None:
        if ticker in EXCHANGE_OVERRIDES:
            exch, ccy = EXCHANGE_OVERRIDES[ticker]
            return DryContract(ticker.split(".")[0], exch, ccy)
        return DryContract(ticker, "SMART", "USD")
    if ticker in EXCHANGE_OVERRIDES:
        exch, ccy = EXCHANGE_OVERRIDES[ticker]
        return Stock(ticker.split(".")[0], exch, ccy)
    return Stock(ticker, "SMART", "USD")


def make_moc_order(action: str, qty: float, transmit: bool):
    """Build a Market-on-Close order object."""
    if Order is None:
        return DryOrder(action.upper(), abs(float(qty)), "MOC", "DAY", bool(transmit))
    order = Order()
    order.action = action.upper()
    order.totalQuantity = abs(float(qty))
    order.orderType = "MOC"
    order.tif = "DAY"
    order.transmit = bool(transmit)
    return order


def account_available_funds(ib: IB) -> float | None:
    """Return AvailableFunds if IBKR exposes it; otherwise None."""
    try:
        rows = ib.accountSummary()
    except Exception:
        return None
    for row in rows:
        if row.tag == "AvailableFunds" and row.currency in ("BASE", "USD"):
            try:
                return float(row.value)
            except Exception:
                return None
    return None


def estimate_pending_notional(pending_rows) -> float:
    """Cheap proxy from staged quantities only; margin check remains conservative."""
    return float(sum(abs(float(r["qty"])) for r in pending_rows))


def short_available(ib: IB, contract) -> bool:
    """Best-effort short availability check.

    IBKR generic tick 236 availability varies by account/data permissions.
    If no data arrives, return True rather than blocking all paper runs.
    Explicit zero/negative shortable values block.
    """
    try:
        ticker = ib.reqMktData(contract, "236", False, False)
        ib.sleep(1.0)
        value = getattr(ticker, "shortableShares", None)
        ib.cancelMktData(contract)
        if value is None:
            return True
        return float(value) > 0
    except Exception:
        return True


def submit_order(ib: IB, ticker: str, action: str, qty: float, dry_run: bool, transmit: bool):
    contract = make_contract(ticker)
    order = make_moc_order(action, qty, transmit=transmit)
    if dry_run:
        print(
            f"  [dry] {action:4s} {abs(float(qty)):>10.2f} {ticker:10s} "
            f"MOC ({contract.exchange}/{contract.currency}) transmit={transmit}"
        )
        return None, contract
    ib.qualifyContracts(contract)
    trade = ib.placeOrder(contract, order)
    return trade, contract


def promote_positions_after_fills(conn, order_ids_by_pair: dict[str, list[str]]) -> None:
    """Promote pending_fill/closing only from the current submitted batch."""
    for pair, order_ids in order_ids_by_pair.items():
        if not order_ids:
            continue
        placeholders = ",".join("?" for _ in order_ids)
        rows = conn.execute(
            f"SELECT leg, status FROM orders WHERE order_id IN ({placeholders})",
            tuple(order_ids),
        ).fetchall()
        by_leg = defaultdict(set)
        for row in rows:
            by_leg[int(row["leg"])].add(row["status"])
        leg1_filled = "filled" in by_leg.get(1, set())
        leg2_filled = "filled" in by_leg.get(2, set())
        if leg1_filled and leg2_filled:
            conn.execute(
                "UPDATE positions SET status='open' WHERE pair=? AND status='pending_fill'",
                (pair,),
            )
            conn.execute(
                "UPDATE positions SET status='closed' WHERE pair=? AND status='closing'",
                (pair,),
            )
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit staged AFES orders to IBKR.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=7, dest="client_id")
    ap.add_argument("--db", default=str(st.DB_PATH))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--transmit", action="store_true",
                    help="Actually transmit orders. Without this, broker orders are non-transmitting.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--margin-buffer", type=float, default=0.30,
                    help="Require AvailableFunds >= staged proxy * buffer when available.")
    ap.add_argument("--skip-borrow-check", action="store_true")
    args = ap.parse_args()

    conn = st.connect(Path(args.db))
    pending = st.pending_orders(conn)
    if not pending:
        print("No pending orders. Exiting.")
        return

    print(f"Pending orders: {len(pending)}")
    if args.dry_run:
        for row in pending:
            submit_order(None, row["ticker"], row["action"], row["qty"], dry_run=True, transmit=False)
        print("\n[dry-run] no broker connection, no orders sent")
        return

    if not HAVE_IB:
        raise SystemExit("ib_insync not installed: pip install ib_insync")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    send_alert(
        "IBKR executor connected",
        extra={"host": args.host, "port": args.port, "orders": len(pending), "transmit": args.transmit},
    )

    available = account_available_funds(ib)
    if available is not None:
        proxy = estimate_pending_notional(pending)
        required = proxy * abs(float(args.margin_buffer))
        if available < required:
            send_alert(
                "Margin pre-check blocked order batch",
                level="ERROR",
                extra={"available": f"{available:.2f}", "required": f"{required:.2f}"},
            )
            ib.disconnect()
            raise SystemExit(3)

    trades = []
    order_ids_by_pair: dict[str, list[str]] = defaultdict(list)
    try:
        for row in pending:
            contract = make_contract(row["ticker"])
            if row["action"].upper() == "SELL" and not args.skip_borrow_check:
                ib.qualifyContracts(contract)
                if not short_available(ib, contract):
                    st.update_order_status(conn, row["order_id"], "rejected")
                    send_alert("Borrow check rejected short leg", level="WARN",
                               extra={"order": row["order_id"], "ticker": row["ticker"]})
                    continue
            trade, _ = submit_order(
                ib,
                row["ticker"],
                row["action"],
                row["qty"],
                dry_run=False,
                transmit=args.transmit,
            )
            st.update_order_status(conn, row["order_id"], "submitted")
            order_ids_by_pair[row["pair"]].append(row["order_id"])
            trades.append((row["order_id"], trade))
            print(f"  submitted {row['order_id']} {row['action']} {row['qty']:.2f} {row['ticker']}")

        print(f"\nWaiting up to {args.timeout}s for order updates ...")
        ib.sleep(float(args.timeout))
        for order_id, trade in trades:
            status = trade.orderStatus.status
            if status == "Filled":
                fill_price = float(trade.orderStatus.avgFillPrice)
                st.update_fill(conn, order_id, fill_price)
                send_alert("Order filled", extra={"order": order_id, "price": f"{fill_price:.4f}"})
            elif status in ("Cancelled", "ApiCancelled"):
                st.update_order_status(conn, order_id, "cancelled")
                send_alert("Order cancelled", level="WARN", extra={"order": order_id, "status": status})
            elif status in ("Inactive",):
                st.update_order_status(conn, order_id, "rejected")
                send_alert("Order inactive/rejected", level="ERROR", extra={"order": order_id})
            else:
                st.update_order_status(conn, order_id, status.lower())
                send_alert("Order not filled yet", level="WARN", extra={"order": order_id, "status": status})

        promote_positions_after_fills(conn, dict(order_ids_by_pair))
    finally:
        ib.disconnect()

    print("Done.")


if __name__ == "__main__":
    main()
