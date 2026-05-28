# AFES Live Execution

Production-oriented IBKR paper/live automation for the AFES production stack.

Default behavior is conservative:

- `daily_signal.py --dry-run` performs no DB writes.
- `ibkr_executor.py --dry-run` performs no broker connection.
- `ibkr_executor.py` does not transmit live orders unless `--transmit` is passed.
- `run_daily.sh` only transmits when `AFES_TRANSMIT=1`.

## Files

| File | Role |
|---|---|
| `state.py` | SQLite source of truth: positions, orders, daily PnL, events |
| `daily_signal.py` | Production universe signal generation, exit logic, kill switch, risk gates |
| `ibkr_executor.py` | IBKR MOC routing, margin/borrow best-effort checks, fill updates |
| `reconcile.py` | DB-vs-IBKR position reconciliation |
| `alerts.py` | Optional Telegram/Slack alerts |
| `run_daily.sh` | Cron/launchd orchestrator |

## Setup

```bash
pip install -r requirements.txt
python3 build_production_universe.py
python3 build_production_live_state.py
```

Start TWS or IB Gateway:

```text
Paper port: 7497
Live port:  7496
API setting: Enable ActiveX and Socket Clients
```

## Local Dry Run

```bash
python -m live.daily_signal \
  --universe data/production_universe.csv \
  --live-state output/live_state_production.pkl \
  --dry-run
```

To stage orders in SQLite but not send them:

```bash
python -m live.daily_signal \
  --universe data/production_universe.csv \
  --live-state output/live_state_production.pkl \
  --notional 1000

python -m live.ibkr_executor --dry-run
```

## IBKR Paper Run

Non-transmitting IBKR orders:

```bash
python -m live.ibkr_executor --host 127.0.0.1 --port 7497 --client-id 7
```

Transmitting paper orders:

```bash
python -m live.ibkr_executor --host 127.0.0.1 --port 7497 --client-id 7 --transmit
```

## Daily Orchestration

```bash
AFES_DRY_RUN=1 live/run_daily.sh
```

Paper order transmission:

```bash
AFES_TRANSMIT=1 AFES_IBKR_PORT=7497 live/run_daily.sh
```

Suggested launchd/cron timing is around 15:50 ET for signal generation and MOC
submission. IBKR MOC cutoffs vary by venue; verify with paper trading before
using live capital.

## Environment Variables

| Variable | Default | Meaning |
|---|---:|---|
| `AFES_PRODUCTION_UNIVERSE` | `data/production_universe.csv` | Production whitelist |
| `AFES_LIVE_STATE` | `output/live_state_production.pkl` | WFO/live params |
| `AFES_LIVE_DB` | `live/afes_live.db` | SQLite state file |
| `AFES_NOTIONAL` | `1000` | Per-pair max notional before weight/risk caps |
| `AFES_MAX_CONCURRENT` | `10` | Max concurrent open pairs |
| `AFES_IBKR_HOST` | `127.0.0.1` | TWS/Gateway host |
| `AFES_IBKR_PORT` | `7497` | Paper by default |
| `AFES_TRANSMIT` | `0` | Must be `1` to transmit |
| `AFES_DRY_RUN` | `0` | If `1`, signal/executor dry-run |
| `AFES_KILL_DD_PCT` | `0.05` | Kill switch drawdown threshold |
| `AFES_KILL_WINDOW_DAYS` | `5` | Kill switch lookback |
| `AFES_SLACK_WEBHOOK_URL` | empty | Optional Slack alerts |
| `AFES_TELEGRAM_BOT_TOKEN` | empty | Optional Telegram alerts |
| `AFES_TELEGRAM_CHAT_ID` | empty | Optional Telegram chat |

## Current Controls

- Production whitelist: baseline US pairs + Europe/US ADR-local pairs.
- `_USD` synthetic legs are reconstructed from local listing close times and FX.
- Entry, exit, stop, and max-hold are staged daily.
- SQLite persists open positions, pending orders, fills, and event logs.
- Exposure caps and Tail EV governor block candidate entries.
- Kill switch: recent net PnL drawdown triggers close-all staging and blocks entries.
- IBKR executor includes best-effort margin and borrow checks.
- Reconciliation halts daily run on DB-vs-IBKR mismatch when requested.

## Remaining Pre-Live Checks

- Run at least 30 market days on paper with `AFES_TRANSMIT=1` and `AFES_IBKR_PORT=7497`.
- Verify every MOC venue accepts the submitted order type before close.
- Confirm borrow check behavior on the actual IBKR account permissions.
- Confirm tax and FX bookkeeping for foreign listings.
- Define manual flatten procedure if TWS/Gateway or the machine fails near close.
