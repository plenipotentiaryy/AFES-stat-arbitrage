"""Re-run each HMM mode and compute its two LEGACY exit-date ratios.

exit_ratio: trade PnL grouped by exit date, exit days only
full_ratio: the same with zeros on every other OOS trading day
Both are "LEGACY exit-date ratio (exit days|full days) — not a Sharpe"
(audit T10, R45): the one Sharpe is the daily mark-to-market Sharpe that
step3j_wfo_daily.py prints (and its PSR / DSR) in each run's log.  R47: each
mode's calendar is its trades' own span (from 2014-12-31 / their first OOS
start to their last exit, never into the final holdout), not the closes
file's end.
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from mtm import trades_span_days

SPLIT = "2014-12-31"
ALL = pd.read_csv("data/closes_daily.csv", parse_dates=["Date"], usecols=["Date"])
OOS = ALL[ALL["Date"] >= SPLIT]["Date"]
OOS_INDEX = pd.DatetimeIndex(OOS.values).normalize()

MODES = ["plain", "block-panic", "block-calm", "size-panic", "extreme-3state"]

print(f"OOS trading days from {SPLIT}: {len(OOS_INDEX)} (each mode: its trades' own span)")
print("exit_ratio / full_ratio: LEGACY exit-date ratio (exit days / full days) — not a Sharpe")
print(f"{'mode':<18} {'trades':>7} {'win%':>6} {'PnL':>8} {'PF':>5} "
      f"{'DD':>6} {'exit_ratio':>10} {'full_ratio':>10}  {'DD/PnL':>7}")
print("-" * 90)

for mode in MODES:
    tag = mode.replace("-", "_")
    cmd = [sys.executable, "bias_test.py", "--split", SPLIT]
    if mode != "plain":
        cmd += ["--hmm", mode]
    # Run silently; output gets overwritten in data/wfo_daily_results_bias.csv
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tr = pd.read_csv("data/wfo_daily_results_bias.csv", parse_dates=["entry", "exit"])
    if tr.empty:
        print(f"{mode:<18} EMPTY")
        continue
    pnl = tr["pnl"]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()

    daily_exit = tr.groupby(tr["exit"].dt.normalize())["pnl"].sum()
    exit_sh = daily_exit.mean() / daily_exit.std() * np.sqrt(252) if daily_exit.std() > 0 else float("nan")

    # Full-days ratio: reindex to the OOS trading days of the trades' own span, filling zeros
    daily_full = daily_exit.reindex(trades_span_days(OOS_INDEX, tr, SPLIT), fill_value=0.0)
    full_sh = daily_full.mean() / daily_full.std() * np.sqrt(252) if daily_full.std() > 0 else float("nan")

    pf = wins / losses if losses > 0 else float("inf")
    print(f"{mode:<18} {len(tr):>7d} {(pnl>0).mean()*100:>5.1f} "
          f"{pnl.sum():>+8.3f} {pf:>5.2f} {dd:>+6.2f} "
          f"{exit_sh:>+10.3f} {full_sh:>+10.3f}  {abs(dd)/pnl.sum()*100:>6.1f}%")
