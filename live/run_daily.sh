#!/usr/bin/env bash
# run_daily.sh — Cron-callable orchestrator for AFES live execution.
#
# Suggested crontab (15:50 ET, weekdays — adjust TZ accordingly):
#   50 15 * * 1-5  cd /path/to/AFES_1 && live/run_daily.sh >> live/log.txt 2>&1
#
# Steps:
#   1. Reconcile DB vs IBKR portfolio. Halt on mismatch.
#   2. Generate signals from latest snapshot.
#   3. Submit MOC orders via IBKR.
#   4. Verify fills, update state.
set -euo pipefail

cd "$(dirname "$0")/.."

PAIRS="data/pairs_bias_test.csv"
NOTIONAL=100000
MAX_CONCURRENT=10
IBKR_HOST="127.0.0.1"
IBKR_PORT=7497       # 7497 = paper, 7496 = live — CHANGE FOR PROD
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "================================================================"
echo "AFES daily run @ $NOW"
echo "================================================================"

echo
echo "[1/3] Reconcile state vs IBKR ..."
python -m live.reconcile --host "$IBKR_HOST" --port "$IBKR_PORT" --halt-on-mismatch \
    || { echo "RECONCILE FAILED — halting"; exit 2; }

echo
echo "[2/3] Generate signals ..."
python -m live.daily_signal --pairs "$PAIRS" \
    --notional "$NOTIONAL" --max-concurrent "$MAX_CONCURRENT"

echo
echo "[3/3] Submit MOC orders to IBKR ..."
python -m live.ibkr_executor --host "$IBKR_HOST" --port "$IBKR_PORT" \
    || { echo "EXECUTOR FAILED — see log"; exit 3; }

echo
echo "✓ AFES daily run complete @ $(date -u +%H:%M:%SZ)"
