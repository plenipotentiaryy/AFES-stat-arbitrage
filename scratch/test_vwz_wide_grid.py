"""One-off: re-run L0vwz-only on honest WFO with wider entry_z grid.
Tests whether VW-Z just needs higher thresholds (right-shift hypothesis)."""
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import step4d_backtest_full as m

# Override grid
m.WFO_GRID = {
    "entry_z": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "exit_z":  [-0.2, 0.0],
    "stop_z":  [4.0, 4.5, 5.0],
    "regime":  ["off", "skip_vol", "skip_calm"],
}

sys.argv = [
    "step4d_backtest_full.py",
    "--L0", "--no-velocity", "--no-thin",
    "--L2", "--L3",
    "--tag", "L0vwz-wide-grid",
]
m.main()
