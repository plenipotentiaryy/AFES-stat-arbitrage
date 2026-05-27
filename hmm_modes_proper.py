"""Re-run each HMM mode and compute both Sharpe variants properly.

exit_sharpe: groupby exit-date (current step3j_wfo_daily formula)
full_sharpe: include zeros for every OOS trading day with no exit
              (true portfolio Sharpe equivalent)
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ALL = pd.read_csv("data/closes_daily.csv", parse_dates=["Date"], usecols=["Date"])
OOS = ALL[ALL["Date"] >= "2014-12-31"]["Date"]
OOS_INDEX = pd.DatetimeIndex(OOS.values).normalize()

MODES = ["plain", "block-panic", "block-calm", "size-panic", "extreme-3state"]

print(f"OOS trading days: {len(OOS_INDEX)}")
print(f"{'mode':<18} {'trades':>7} {'win%':>6} {'PnL':>8} {'PF':>5} "
      f"{'DD':>6} {'exit_Sh':>8} {'full_Sh':>8}  {'DD/PnL':>7}")
print("-" * 90)

for mode in MODES:
    tag = mode.replace("-", "_")
    cmd = [sys.executable, "bias_test.py", "--split", "2014-12-31"]
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

    # Full Sharpe: reindex to ALL OOS trading days, filling zeros
    daily_full = daily_exit.reindex(OOS_INDEX, fill_value=0.0)
    full_sh = daily_full.mean() / daily_full.std() * np.sqrt(252) if daily_full.std() > 0 else float("nan")

    pf = wins / losses if losses > 0 else float("inf")
    print(f"{mode:<18} {len(tr):>7d} {(pnl>0).mean()*100:>5.1f} "
          f"{pnl.sum():>+8.3f} {pf:>5.2f} {dd:>+6.2f} "
          f"{exit_sh:>+8.3f} {full_sh:>+8.3f}  {abs(dd)/pnl.sum()*100:>6.1f}%")
