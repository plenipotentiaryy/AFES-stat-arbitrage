#!/bin/bash
set -e
cd /Users/leonidpofa/VSCodeHruchevoPY/quant/AFES_1
mkdir -p output/ablations

echo "=== A: baseline ==="
python step3j_wfo_daily.py --pairs pairs_selected.csv --tag base \
  > output/ablations/A_base.log 2>&1
grep -A1 "DAILY WFO SUMMARY" output/ablations/A_base.log | tail -12

echo ""
echo "=== B: +CUSUM h=15 ==="
python step3j_wfo_daily.py --pairs pairs_selected.csv --cusum 30.0 --tag cusum \
  > output/ablations/B_cusum.log 2>&1
grep -A1 "DAILY WFO SUMMARY" output/ablations/B_cusum.log | tail -12

echo ""
echo "=== C: +Kalman ==="
python step3j_wfo_daily.py --pairs pairs_selected.csv --kalman --tag kalman \
  > output/ablations/C_kalman.log 2>&1
grep -A1 "DAILY WFO SUMMARY" output/ablations/C_kalman.log | tail -12

echo ""
echo "=== D: +Kalman +CUSUM h=15 ==="
python step3j_wfo_daily.py --pairs pairs_selected.csv --kalman --cusum 30.0 --tag kalman_cusum \
  > output/ablations/D_kalman_cusum.log 2>&1
grep -A1 "DAILY WFO SUMMARY" output/ablations/D_kalman_cusum.log | tail -12

echo ""
echo "=== E: BIAS TEST (selection on first half) ==="
python bias_test.py --split 2014-12-31 \
  > output/ablations/E_bias.log 2>&1
grep -A1 "DAILY WFO SUMMARY" output/ablations/E_bias.log | tail -12

echo ""
echo DONE
