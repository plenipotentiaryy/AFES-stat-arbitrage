import subprocess
import sys
import time

STEPS = [
    ("step1_download.py",  "Download OHLCV data",          False),
    ("step2_pairs.py",     "Find cointegrated pairs",       True),
    ("step3_signals.py",   "Compute z-score signals",       True),
    ("step5_regime.py",    "HMM regime detection",          False),
    ("step4_backtest.py",  "Backtest all pairs",            True),
    ("step4b_grid.py",     "EXIT_Z grid search",            True),
    ("step6_montecarlo.py","Monte Carlo simulation",        True),
]

# python RUN_ALL.py      — all steps
# python RUN_ALL.py 2    — start from step 2 (skip download)
start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 1

print("\n" + "█" * 50)
print("  AFES PIPELINE")
print("█" * 50)

for step_num, (script, description, required) in enumerate(STEPS, start=1):
    if step_num < start_from:
        print(f"\n  [{step_num}] SKIP  {description}")
        continue

    print(f"\n{'█'*50}")
    print(f"  STEP {step_num} — {description}")
    print(f"{'█'*50}")

    t0     = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        if required:
            print(f"\n  FAILED — {script} exited with code {result.returncode}")
            print("  Stopping pipeline.")
            sys.exit(result.returncode)
        else:
            print(f"\n  WARNING — {script} failed but is optional, continuing...")
    else:
        print(f"\n  OK  ({elapsed:.0f}s)")

print("\n" + "█" * 50)
print("  DONE")
print("█" * 50 + "\n")
