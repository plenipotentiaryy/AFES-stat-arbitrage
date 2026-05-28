"""state.py - SQLite-backed state machine for live AFES execution.

Tracks open positions, pending orders, and daily P&L history.  Single
source of truth for the live system; reconciliation cross-checks against
IBKR portfolio at SOD.

Schema:
    positions   current pair book and entry parameters
    orders      staged/submitted/fill status per leg
    pnl_daily   daily realized account/PnL telemetry
    events      audit log for signals, blocks, alerts, and kill switches
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
    status             TEXT NOT NULL DEFAULT 'open', -- pending_fill | open | closing | closed
    leg1               TEXT,
    leg2               TEXT,
    trade_ticker1      TEXT,
    trade_ticker2      TEXT,
    beta               REAL,
    entry_z            REAL,
    exit_z             REAL,
    stop_z             REAL,
    max_hold_days      INTEGER,
    notional_usd       REAL,
    sector             TEXT,
    source_layer       TEXT,
    last_z             REAL,
    last_signal_ts     TEXT
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
    fill_ts    TEXT,
    reason     TEXT,
    transmit   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pnl_daily (
    date       TEXT PRIMARY KEY,
    gross_pnl  REAL,
    net_pnl    REAL,
    n_open     INTEGER,
    n_closed   INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    level    TEXT NOT NULL,
    event    TEXT NOT NULL,
    details  TEXT
);
CREATE TABLE IF NOT EXISTS market_data_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts      TEXT NOT NULL,
    source      TEXT NOT NULL,
    last_bar_ts TEXT,
    n_rows      INTEGER,
    n_columns   INTEGER,
    details     TEXT
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
    leg1: str = ""
    leg2: str = ""
    trade_ticker1: str = ""
    trade_ticker2: str = ""
    beta: float = 1.0
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.5
    max_hold_days: int = 30
    notional_usd: float = 0.0
    sector: str = "UNKNOWN"
    source_layer: str = ""
    last_z: float = 0.0
    last_signal_ts: str = ""

    def to_risk_dict(self, spread_std_usd: float = 1.0) -> dict:
        """Return the dict shape consumed by exposure/tail-risk controllers."""
        return {
            "pair": self.pair,
            "pair_name": self.pair,
            "legs": [self.leg1 or self.trade_ticker1, self.leg2 or self.trade_ticker2],
            "nominal_size_usd": abs(float(self.notional_usd)),
            "sector": self.sector,
            "current_z": float(self.last_z),
            "spread_std_usd": float(max(spread_std_usd, 1e-9)),
            "position_size_units": abs(float(self.notional_usd)),
            "p_revert": 0.50,
        }


POSITION_COLUMNS = {
    "leg1": "TEXT",
    "leg2": "TEXT",
    "trade_ticker1": "TEXT",
    "trade_ticker2": "TEXT",
    "beta": "REAL",
    "entry_z": "REAL",
    "exit_z": "REAL",
    "stop_z": "REAL",
    "max_hold_days": "INTEGER",
    "notional_usd": "REAL",
    "sector": "TEXT",
    "source_layer": "TEXT",
    "last_z": "REAL",
    "last_signal_ts": "TEXT",
}

ORDER_COLUMNS = {
    "reason": "TEXT",
    "transmit": "INTEGER DEFAULT 0",
}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def connect(db: Path = DB_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _ensure_columns(conn, "positions", POSITION_COLUMNS)
    _ensure_columns(conn, "orders", ORDER_COLUMNS)
    conn.commit()
    return conn


def open_position(conn, p: Position) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO positions
           (pair, side, qty_t1, qty_t2, entry_ts, entry_spread_log, status,
            leg1, leg2, trade_ticker1, trade_ticker2, beta, entry_z, exit_z,
            stop_z, max_hold_days, notional_usd, sector, source_layer,
            last_z, last_signal_ts)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            p.pair, p.side, p.qty_t1, p.qty_t2, p.entry_ts,
            p.entry_spread_log, p.status, p.leg1, p.leg2,
            p.trade_ticker1, p.trade_ticker2, p.beta, p.entry_z,
            p.exit_z, p.stop_z, p.max_hold_days, p.notional_usd,
            p.sector, p.source_layer, p.last_z, p.last_signal_ts,
        ),
    )
    conn.commit()


def mark_closing(conn, pair: str) -> None:
    conn.execute("UPDATE positions SET status='closing' WHERE pair=?", (pair,))
    conn.commit()


def close_position(conn, pair: str) -> None:
    conn.execute("UPDATE positions SET status='closed' WHERE pair=?", (pair,))
    conn.commit()


def update_position_signal(conn, pair: str, z: float, ts: str) -> None:
    conn.execute(
        "UPDATE positions SET last_z=?, last_signal_ts=? WHERE pair=?",
        (float(z), ts, pair),
    )
    conn.commit()


def _row_to_position(row) -> Position:
    return Position(
        pair=row["pair"],
        side=int(row["side"]),
        qty_t1=float(row["qty_t1"]),
        qty_t2=float(row["qty_t2"]),
        entry_ts=row["entry_ts"],
        entry_spread_log=float(row["entry_spread_log"]),
        status=row["status"],
        leg1=row["leg1"] or "",
        leg2=row["leg2"] or "",
        trade_ticker1=row["trade_ticker1"] or "",
        trade_ticker2=row["trade_ticker2"] or "",
        beta=float(row["beta"] if row["beta"] is not None else 1.0),
        entry_z=float(row["entry_z"] if row["entry_z"] is not None else 2.0),
        exit_z=float(row["exit_z"] if row["exit_z"] is not None else 0.0),
        stop_z=float(row["stop_z"] if row["stop_z"] is not None else 3.5),
        max_hold_days=int(row["max_hold_days"] if row["max_hold_days"] is not None else 30),
        notional_usd=float(row["notional_usd"] if row["notional_usd"] is not None else 0.0),
        sector=row["sector"] or "UNKNOWN",
        source_layer=row["source_layer"] or "",
        last_z=float(row["last_z"] if row["last_z"] is not None else 0.0),
        last_signal_ts=row["last_signal_ts"] or "",
    )


def list_open(conn) -> list[Position]:
    rows = conn.execute(
        "SELECT * FROM positions WHERE status IN ('pending_fill','open','closing')"
    ).fetchall()
    return [_row_to_position(r) for r in rows]


def list_active(conn) -> list[Position]:
    rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
    return [_row_to_position(r) for r in rows]


def record_order(conn, order_id: str, pair: str, side: int, leg: int,
                  ticker: str, action: str, qty: float, status: str = "pending",
                  reason: str = "", transmit: bool = False):
    conn.execute("""INSERT OR REPLACE INTO orders
                     (order_id, pair, side, leg, ticker, action, qty, status, ts, reason, transmit)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                  (order_id, pair, side, leg, ticker, action, qty, status,
                   datetime.utcnow().isoformat(), reason, int(bool(transmit))))
    conn.commit()


def update_fill(conn, order_id: str, fill_price: float):
    conn.execute("""UPDATE orders SET status='filled', fill_price=?, fill_ts=?
                     WHERE order_id=?""",
                  (fill_price, datetime.utcnow().isoformat(), order_id))
    conn.commit()


def update_order_status(conn, order_id: str, status: str) -> None:
    conn.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order_id))
    conn.commit()


def pending_orders(conn):
    return conn.execute(
        "SELECT order_id, ticker, action, qty, pair, side, leg, reason FROM orders "
        "WHERE status='pending' ORDER BY ts, pair, leg"
    ).fetchall()


def record_pnl_day(conn, date: str, gross: float, net: float,
                    n_open: int, n_closed: int):
    conn.execute("""INSERT OR REPLACE INTO pnl_daily VALUES (?,?,?,?,?)""",
                  (date, gross, net, n_open, n_closed))
    conn.commit()


def log_event(conn, level: str, event: str, details: str = "") -> None:
    conn.execute(
        "INSERT INTO events (ts, level, event, details) VALUES (?,?,?,?)",
        (datetime.utcnow().isoformat(), level, event, details),
    )
    conn.commit()


def record_market_data_audit(
    conn,
    source: str,
    last_bar_ts: str,
    n_rows: int,
    n_columns: int,
    details: str = "",
) -> None:
    conn.execute(
        """INSERT INTO market_data_audit
           (run_ts, source, last_bar_ts, n_rows, n_columns, details)
           VALUES (?,?,?,?,?,?)""",
        (
            datetime.utcnow().isoformat(),
            str(source),
            str(last_bar_ts),
            int(n_rows),
            int(n_columns),
            str(details),
        ),
    )
    conn.commit()


def recent_net_pnl(conn, days: int = 5) -> float:
    rows = conn.execute(
        "SELECT net_pnl FROM pnl_daily ORDER BY date DESC LIMIT ?",
        (int(days),),
    ).fetchall()
    return float(sum(float(r["net_pnl"] or 0.0) for r in rows))


def mark_all_closing(conn) -> None:
    conn.execute("UPDATE positions SET status='closing' WHERE status IN ('pending_fill','open')")
    conn.commit()
