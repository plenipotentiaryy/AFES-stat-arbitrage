#!/bin/bash
set -e
cd /Users/leonidpofa/VSCodeHruchevoPY/quant/AFES_1
mkdir -p output/final

ALL_DAYS_PY='
import pandas as pd, numpy as np, sys
ALL = pd.read_csv("data/closes_daily.csv", parse_dates=["Date"], usecols=["Date"])
OOS_IDX = pd.DatetimeIndex(ALL[ALL["Date"] >= sys.argv[2]]["Date"]).normalize()
tr = pd.read_csv(sys.argv[1], parse_dates=["entry","exit"])
pnl = tr["pnl"]; eq = pnl.cumsum(); dd = (eq - eq.cummax()).min()
daily = tr.groupby(tr["exit"].dt.normalize())["pnl"].sum().reindex(OOS_IDX, fill_value=0.0)
pf = pnl[pnl>0].sum() / -pnl[pnl<0].sum() if (pnl<0).any() else float("inf")
fsh = daily.mean()/daily.std()*np.sqrt(252)
print(f"  trades={len(tr):>5}  win%={(pnl>0).mean()*100:>4.1f}  PnL={pnl.sum():>+7.3f}  DD={dd:>+6.3f}  PF={pf:.2f}  full_Sh={fsh:>+5.3f}  DD/PnL={abs(dd)/pnl.sum()*100:>5.1f}%")
'

run() {
  local name="$1"
  local split="$2"
  shift 2
  echo "=== $name ==="
  python bias_test.py --split "$split" "$@" > "output/final/$name.log" 2>&1
  cp data/wfo_daily_results_bias.csv "output/final/$name.csv"
  python -c "$ALL_DAYS_PY" "output/final/$name.csv" "$split"
}

COMBO="--maxconc 3 --pairmom --ivol"
SPLIT="2014-12-31"

run "01_combo_plain"       "$SPLIT" $COMBO
echo ""
echo "## 1. Ticker overlap"
run "02_combo_ticker_overlap" "$SPLIT" $COMBO --ticker-overlap
echo ""
echo "## 2. Stress"
run "03_stress_slip15_borrow200" "$SPLIT" $COMBO --slip 15 --borrow 200
echo ""
echo "## 2b. Stress: alternative split dates"
run "04_split_2010" "2010-01-01" $COMBO
run "05_split_2018" "2018-01-01" $COMBO
echo ""
echo "## 3. Per-pair quarter cap"
run "06_quarter_cap_2" "$SPLIT" $COMBO --per-pair-quarter-cap 2
run "07_quarter_cap_4" "$SPLIT" $COMBO --per-pair-quarter-cap 4
echo ""
echo "## 4. Vol-tight stop"
run "08_vol_tight" "$SPLIT" $COMBO --vol-tight-stop
echo ""
echo "## 5. All combined"
run "09_all_combined" "$SPLIT" $COMBO --ticker-overlap --per-pair-quarter-cap 3 --vol-tight-stop

echo ""
echo DONE
