#!/bin/bash
set -e
cd /Users/leonidpofa/VSCodeHruchevoPY/quant/AFES_1
mkdir -p output/grid_sharpe
for S in 0.1 0.2 0.3 0.4; do
  echo "=== WFO_MIN_TRAIN_SHARPE = $S ==="
  python -c "
import re,pathlib
p=pathlib.Path('config.py'); t=p.read_text()
t=re.sub(r'WFO_MIN_TRAIN_SHARPE\s*=\s*[-\d.]+', f'WFO_MIN_TRAIN_SHARPE = $S', t)
p.write_text(t)
"
  python step3j_wfo.py --pairs pairs_baseline14.csv > "output/grid_sharpe/log_s${S}.txt" 2>&1
  cp data/wfo_results.csv "output/grid_sharpe/results_s${S}.csv"
  tail -25 "output/grid_sharpe/log_s${S}.txt" | grep -E "OOS Trades|Total OOS|Win Rate|Profit Factor|Max Drawdown|Sharpe \(chain"
  echo
done
echo DONE
