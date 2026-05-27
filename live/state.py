"""state.py — SQLite-backed state machine for live AFES execution.

Tracks open positions, pending orders, and daily P&L history.  Single
source of truth for the live system; reconciliation cross-checks against
IBKR portfolio at SOD.

Schema:
    positions   (pair, side, qty_t1, qty_t2, entry_ts, entry_spread_log, status)
    orders      (order_id, pair, side, leg, ticker, action, qty, status, ts)
    pnl_daily   (date, gross_pnl, net_pnl, n_open, n_closed)
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "afes_live.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    pair               TEXT PRIMARY KEY,
    side               INTEGER NOT NULL,             -- +1 long-spread, -1 short-spread
    qty_t1             REAL NOT NULL,
    qty_t2             REAL NOT NULL,
    entry_ts           TEXT NOT NULL,
    entry_spread_log   REAL NOT NULL,
    status             TEXT NOT NULL DEFAULT 'open'  -- open | closing | closed
);
CREATE TABLE IF NOT EXISTS orders (
    order_id   TEXT PRIMARY KEY,
    pair       TEXT NOT NULL,
    side       INTEGER NOT NULL,
    leg        INTEGER NOT NULL,                     -- 1 or 2
    ticker     TEXT NOT NULL,
    action     TEXT NOT NULL,                        -- BUY | SELL
    qty        REAL NOT NULL,
    status     TEXT NOT NULL,                        -- pending | filled | cancelled | rejected
    ts         TEXT NOT NULL,
    fill_price REAL,
    fill_ts    TEXT
);
CREATE TABLE IF NOT EXISTS pnl_daily (
    date       TEXT PRIMARY KEY,
    gross_pnl  REAL,
    net_pnl    REAL,
    n_open     INTEGER,
    n_closed   INTEGER
);
"""


@dataclass
class Position:
    pair: str
    side: int
    qty_t1: float
    qty_t2: float
    entry_ts: str
    entry_spread_log: float
    status: str = "open"


def connect(db: Path = DB_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    return conn


def open_position(conn, p: Position) -> None:
    conn.execute("""INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)""",
                  (p.pair, p.side, p.qty_t1, p.qty_t2, p.entry_ts,
                   p.entry_spread_log, p.status))
    conn.commit()


def mark_closing(conn, pair: str) -> None:
    conn.execute("UPDATE positions SET status='closing' WHERE pair=?", (pair,))
    conn.commit()


def close_position(conn, pair: str) -> None:
    conn.execute("UPDATE positions SET status='closed' WHERE pair=?", (pair,))
    conn.commit()


def list_open(conn) -> list[Position]:
    rows = conn.execute("SELECT pair,side,qty_t1,qty_t2,entry_ts,entry_spread_log,status "
                          "FROM positions WHERE status IN ('open','closing')").fetchall()
    return [Position(*r) for r in rows]


def record_order(conn, order_id: str, pair: str, side: int, leg: int,
                  ticker: str, action: str, qty: float, status: str = "pending"):
    conn.execute("""INSERT OR REPLACE INTO orders
                     (order_id, pair, side, leg, ticker, action, qty, status, ts)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (order_id, pair, side, leg, ticker, action, qty, status,
                   datetime.utcnow().isoformat()))
    conn.commit()


def update_fill(conn, order_id: str, fill_price: float):
    conn.execute("""UPDATE orders SET status='filled', fill_price=?, fill_ts=?
                     WHERE order_id=?""",
                  (fill_price, datetime.utcnow().isoformat(), order_id))
    conn.commit()


def record_pnl_day(conn, date: str, gross: float, net: float,
                    n_open: int, n_closed: int):
    conn.execute("""INSERT OR REPLACE INTO pnl_daily VALUES (?,?,?,?,?)""",
                  (date, gross, net, n_open, n_closed))
    conn.commit()
