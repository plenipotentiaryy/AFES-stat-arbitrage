import subprocess
import sys

STEPS = [
    ("step1_download.py", "Download OHLCV data from Polygon"),
    ("step2_pairs.py", "Find cointegrated pairs"),
    ("step3_signals.py", "Compute z-score signals"),
    ("step4_backtest.py", "Run backtest"),
    ("step6_montecarlo.py", "Monte Carlo simulation"),
]


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for script, description in STEPS:
        step_num = int(script[4])
        if step_num < start:
            continue

        print(f"\n{'='*60}")
        print(f"Step {step_num}: {description}")
        print(f"{'='*60}")

        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"\nFailed at {script}. Stopping.")
            sys.exit(result.returncode)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
