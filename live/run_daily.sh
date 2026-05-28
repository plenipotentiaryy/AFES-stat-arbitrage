#!/usr/bin/env bash
# Cron-callable orchestrator for AFES IBKR paper/live execution.
#
# Default mode is safe: executor does not transmit unless AFES_TRANSMIT=1.
set -euo pipefail

cd "$(dirname "$0")/.."

UNIVERSE="${AFES_PRODUCTION_UNIVERSE:-data/production_universe.csv}"
LIVE_STATE="${AFES_LIVE_STATE:-output/live_state_production.pkl}"
DB="${AFES_LIVE_DB:-live/afes_live.db}"
NOTIONAL="${AFES_NOTIONAL:-1000}"
MAX_CONCURRENT="${AFES_MAX_CONCURRENT:-10}"
IBKR_HOST="${AFES_IBKR_HOST:-127.0.0.1}"
IBKR_PORT="${AFES_IBKR_PORT:-7497}"
IBKR_CLIENT_ID="${AFES_IBKR_CLIENT_ID:-7}"
KILL_DD_PCT="${AFES_KILL_DD_PCT:-0.05}"
KILL_WINDOW_DAYS="${AFES_KILL_WINDOW_DAYS:-5}"
CLOSES_CSV="${AFES_CLOSES_CSV:-}"
NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

EXECUTOR_FLAGS=()
if [[ "${AFES_TRANSMIT:-0}" == "1" ]]; then
  EXECUTOR_FLAGS+=("--transmit")
else
  echo "AFES_TRANSMIT not set to 1; executor will submit non-transmitting IBKR orders."
fi

if [[ "${AFES_DRY_RUN:-0}" == "1" ]]; then
  SIGNAL_FLAGS=("--dry-run")
  EXECUTOR_FLAGS+=("--dry-run")
else
  SIGNAL_FLAGS=()
fi

echo "================================================================"
echo "AFES daily run @ $NOW"
echo "================================================================"

echo
echo "[1/3] Reconcile state vs IBKR ..."
python -m live.reconcile \
  --host "$IBKR_HOST" \
  --port "$IBKR_PORT" \
  --client-id "$IBKR_CLIENT_ID" \
  --db "$DB" \
  --halt-on-mismatch

echo
echo "[2/3] Generate production signals ..."
SIGNAL_EXTRA=()
if [[ -n "$CLOSES_CSV" ]]; then
  SIGNAL_EXTRA+=("--closes-csv" "$CLOSES_CSV")
fi
python -m live.daily_signal \
  --universe "$UNIVERSE" \
  --live-state "$LIVE_STATE" \
  --db "$DB" \
  --notional "$NOTIONAL" \
  --max-concurrent "$MAX_CONCURRENT" \
  --kill-dd-pct "$KILL_DD_PCT" \
  --kill-window-days "$KILL_WINDOW_DAYS" \
  "${SIGNAL_EXTRA[@]}" \
  "${SIGNAL_FLAGS[@]}"

echo
echo "[3/3] Submit staged MOC orders ..."
python -m live.ibkr_executor \
  --host "$IBKR_HOST" \
  --port "$IBKR_PORT" \
  --client-id "$IBKR_CLIENT_ID" \
  --db "$DB" \
  "${EXECUTOR_FLAGS[@]}"

echo
echo "AFES daily run complete @ $(date -u +%H:%M:%SZ)"
