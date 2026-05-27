#!/bin/bash
set -e
cd /Users/leonidpofa/VSCodeHruchevoPY/quant/AFES_1
mkdir -p output/split
for RANGE in "2006-04-03:2014-12-31:H1" "2015-01-01:2026-04-03:H2"; do
  IFS=":" read S E TAG <<< "$RANGE"
  echo "=== $TAG : $S → $E ==="
  python step3j_wfo.py --pairs pairs_baseline14.csv --start "$S" --end "$E" \
    > "output/split/log_${TAG}.txt" 2>&1
  cp data/wfo_results.csv "output/split/results_${TAG}.csv"
  tail -25 "output/split/log_${TAG}.txt" | grep -E "OOS Trades|Total OOS|Win Rate|Profit Factor|Max Drawdown|Sharpe \(chain"
  echo
done
echo DONE
