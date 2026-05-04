import subprocess
import sys
import re
import time

# ── Pipeline definition ───────────────────────────────────────────────────────
PIPELINE = [
    {
        "step": 1,
        "name": "Download data",
        "variants": [
            ("a", "download.py",            "Full download: Yahoo 10yr daily + Polygon 15min audit"),
            ("b", "download_incremental.py","Incremental backfill — skips already-downloaded tickers"),
            ("c", "download_redownload.py", "Re-download tickers with gaps"),
            ("d", "download_retry.py",      "Retry failed / incomplete tickers"),
        ],
        "required": False,
    },
    {
        "step": 2,
        "name": "Find cointegrated pairs",
        "variants": [
            ("a", "pairs.py", "Rolling 90d Engle-Granger + Johansen info on 161 sector pairs"),
        ],
        "required": True,
    },
    {
        "step": 3,
        "name": "Filters + Position Sizing  (run before backtest)",
        "variants": [
            ("a", "hmm.py",       "Layer 1 — HMM regime detection (per-pair, 2 states)"),
            ("b", "kmeans.py",    "Layer 2 — K-Means macro regime (Trend / Sideways / Panic)"),
            ("c", "iv.py",        "Layer 3 — Implied Volatility (Black-Scholes + VIX)"),
            ("d", "montecarlo.py","Layer 4 — OU Monte Carlo (VaR, CVaR, mc_confidence)"),
            ("e", "sizing.py",    "Layer 5 — Dynamic sizing (combine all layers)"),
            ("f", "grid.py",      "Layer 6 — Per-pair grid (train→test optimal params)"),
        ],
        "required": False,
    },
    {
        "step": 4,
        "name": "Backtest",
        "variants": [
            ("a", "backtest.py",        "Standard   — Kalman spread, rolling coint, K-Means gate"),
            ("b", "backtest_strict.py", "Strict/Sniper  entry=3.2  exit=-0.2  stop=4.4"),
        ],
        "required": True,
    },
    {
        "step": 5,
        "name": "Analysis",
        "variants": [
            ("a", "grid_exit.py",      "EXIT_Z grid search"),
            ("b", "grid_sniper.py",    "Sniper grid — 84 combos (entry × exit × stop)"),
            ("c", "bootstrap_mc.py",   "Bootstrap Monte Carlo — resampling of actual trades"),
            ("d", "dashboard.py",      "Visual dashboard — all metrics on one chart"),
            ("e", "stress.py",         "Stress test — walk-forward + VIX regimes + trade today?"),
        ],
        "required": False,
    },
]

DEFAULT_ALL = ["1a", "2a", "3a", "3b", "3c", "3d", "3e", "3f", "4a", "5d"]

# ── UI helpers ────────────────────────────────────────────────────────────────
W = 58

def box(text):
    print("╔" + "═" * W + "╗")
    print("║" + text.center(W) + "║")
    print("╚" + "═" * W + "╝")

def rule():
    print("─" * (W + 2))

def print_menu():
    print()
    box("AFES  —  Pipeline Runner")
    print()
    for entry in PIPELINE:
        req = "  (required)" if entry["required"] else ""
        print(f"  STEP {entry['step']}  {entry['name']}{req}")
        for letter, script, desc in entry["variants"]:
            tag = f"{entry['step']}{letter}"
            print(f"    {tag:<5}  {script:<26}  {desc}")
        print()

    rule()
    print("  Commands:")
    print("    all          →  full pipeline  " + "  ".join(DEFAULT_ALL))
    print("    2 5b 6d      →  steps by number (default variant = a)")
    print("    2a 4b 5b 6c  →  specific variants")
    print()

# ── Parse input ───────────────────────────────────────────────────────────────
def parse_tokens(tokens: list[str]) -> list[tuple[int, str]]:
    selected = []
    step_map = {entry["step"]: entry for entry in PIPELINE}

    for token in tokens:
        m = re.fullmatch(r"(\d+)([a-z]?)", token)
        if not m:
            print(f"  ⚠  Unknown token: '{token}' — skipped")
            continue
        step_num = int(m.group(1))
        variant  = m.group(2) or "a"

        if step_num not in step_map:
            print(f"  ⚠  Step {step_num} not found — skipped")
            continue

        valid_variants = [v[0] for v in step_map[step_num]["variants"]]
        if variant not in valid_variants:
            print(f"  ⚠  Step {step_num} has no variant '{variant}' "
                  f"(available: {valid_variants}) — using 'a'")
            variant = "a"

        selected.append((step_num, variant))

    # Preserve pipeline order, deduplicate
    seen = set()
    ordered = []
    for step_num in sorted(set(s for s, _ in selected)):
        for s, v in selected:
            if s == step_num and (s, v) not in seen:
                seen.add((s, v))
                ordered.append((s, v))
    return ordered


def resolve_script(step_num: int, variant: str) -> tuple[str, str]:
    for entry in PIPELINE:
        if entry["step"] == step_num:
            for letter, script, desc in entry["variants"]:
                if letter == variant:
                    return script, desc
    raise ValueError(f"Step {step_num}{variant} not found")


def run_step(script: str, description: str) -> bool:
    print(f"\n{'█'*W}")
    print(f"  {description}")
    print(f"  {script}")
    print(f"{'█'*W}")

    t0     = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED  ({elapsed:.0f}s)")
        return False
    print(f"\n  OK  ({elapsed:.0f}s)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if not args:
        print_menu()
        try:
            raw = input("  Enter steps to run: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
        args = raw.split()

    if not args:
        print("  Nothing to run.")
        return

    if args == ["all"]:
        tokens = DEFAULT_ALL
    else:
        tokens = args

    selected = parse_tokens(tokens)

    if not selected:
        print("  Nothing to run.")
        return

    print()
    print("  Running:")
    for step_num, variant in selected:
        script, desc = resolve_script(step_num, variant)
        print(f"    {step_num}{variant}  {script}  —  {desc}")

    print()
    for step_num, variant in selected:
        script, desc = resolve_script(step_num, variant)
        ok = run_step(script, desc)
        if not ok:
            entry = next(e for e in PIPELINE if e["step"] == step_num)
            if entry["required"]:
                print(f"\n  Required step {step_num}{variant} failed — stopping.")
                sys.exit(1)
            print(f"  Optional step {step_num}{variant} failed — continuing.")

    # Always regenerate dashboard at the end if trades exist
    dashboard_was_run = any(resolve_script(s, v)[0] == "dashboard.py" for s, v in selected)
    if not dashboard_was_run:
        from pathlib import Path
        if Path("data/trades.csv").exists():
            run_step("dashboard.py", "Visual dashboard — all metrics on one chart")

    print(f"\n{'█'*W}")
    print(f"  DONE")
    print(f"{'█'*W}\n")


if __name__ == "__main__":
    main()
