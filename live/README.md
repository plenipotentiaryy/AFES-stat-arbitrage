# AFES Live Execution (MVP)

Automated MOC-order execution against IBKR via `ib_insync`.

## Files

| File | Role |
|---|---|
| `state.py` | SQLite schema: positions, orders, pnl_daily |
| `daily_signal.py` | Pulls yfinance closes → computes z → stages orders in DB |
| `ibkr_executor.py` | Reads pending orders → submits MOC to IBKR → records fills |
| `reconcile.py` | SOD sanity check: DB positions vs IBKR account |
| `run_daily.sh` | Cron orchestrator (reconcile → signal → execute) |

## Setup

```bash
pip install ib_insync yfinance
# Start IB Gateway or TWS, enable API in settings:
#   File → Global Configuration → API → Settings → "Enable ActiveX and Socket Clients" ON
#   Socket port: 7497 (paper) or 7496 (live)
#   Master API client ID: leave blank
```

## First-time sanity check (no orders sent)

```bash
# Inspect signals on today's data
python -m live.daily_signal --pairs data/pairs_bias_test.csv --dry-run

# Submit-stage with dry-run executor (prints what would go to IB)
python -m live.daily_signal --pairs data/pairs_bias_test.csv
python -m live.ibkr_executor --dry-run
```

## Schedule

```cron
# 15:50 ET weekdays — adjust for your timezone
50 15 * * 1-5  cd /path/to/AFES_1 && live/run_daily.sh >> live/log.txt 2>&1
```

## Known MVP limitations

1. **No FX leg for cross-listings.** USD-normalized synthetic legs (e.g. `RY.TO_USD`)
   are skipped — `daily_signal.py` works only on US-listed names for now. To trade
   cross-listings you need a parallel FX fetcher and contract routing per the
   `EXCHANGE_OVERRIDES` table in `ibkr_executor.py` (which is wired but unused).

2. **No borrow-availability check.** Real production should call
   `ib.reqMktData(contract, '236')` for shortable shares + fee rate before
   submitting any short leg. If unborrowable, skip the pair for the day.

3. **No margin pre-check.** Add `ib.accountSummary()` lookup of `AvailableFunds`
   before submitting; require ≥30% buffer.

4. **No kill switch.** Add daily-DD circuit breaker (e.g., if cumulative net PnL
   over last 5 days < −5% of NAV, close all and halt).

5. **No alerting.** Add Telegram/Slack webhook on fills, rejects, mismatches.

6. **Single-broker, single-account.** No failover, no multi-account splitting.

## Pre-production checklist

- [ ] 30+ days paper trading on port 7497, all metrics match backtest expectation
- [ ] Reconcile clean every morning for 2+ weeks straight
- [ ] Borrow check + skip-on-unborrowable implemented
- [ ] Margin pre-check + 30% buffer enforced
- [ ] Kill switch: −5% 5-day DD → halt + alert
- [ ] Telegram alert on every fill, error, mismatch
- [ ] Disaster-recovery: how do you flatten if the laptop dies during market hours?
- [ ] Tax planning: cross-border FX events bookkeeping (CPA consultation)
