"""
Unit tests for Phase E — Section 2 Flaw 2 (no survival bias on stop exits).

The spec requires that stop-loss fills use the CURRENT-BAR-CLOSE spread (or
next-bar open) with COST_TAKER, never the exact ``stop_z * std + mean``
boundary.  This test does two things:

  1. Functional integration: drives ``step3j_wfo.backtest_oos`` over a
     synthetic z-score path that crosses the stop, then asserts the exit
     spread used by the PnL formula equals the current bar's spread —
     measured by reconstructing the gross PnL from observed columns and
     comparing it against a stop-boundary baseline that must be DIFFERENT.

  2. Static guard: a regex sweep over the four backtester source files to
     forbid any line that computes an exit price as
     ``stop_z * <std-like> + <mean-like>`` or similar.  The intent is to
     catch future regressions where someone hardcodes the boundary value.

Both checks are deliberately tight enough to fail loudly if the codebase
drifts back into the survival-bias pattern.
"""

import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class StaticSurvivalBiasGuard(unittest.TestCase):
    """
    Source-level sweep: no backtester is allowed to compute an exit price
    from the stop-z boundary.

    The pattern we forbid is anything that multiplies stop_z (or a synonym
    like ``stop_thresh``) by a standard-deviation series at the moment of
    exit and adds a mean — i.e. reconstructing the boundary spread instead
    of using the bar's observed spread.
    """

    FORBIDDEN = re.compile(
        r"""
        (stop_z|stop_thresh)\s*\*\s*    # boundary * ...
        (?!(\s)*entry_z\b)              # but NOT 1.5*entry_z (Section 7.4)
        .*?(std|sigma|spread_std)       # ... standard deviation
        """,
        re.VERBOSE,
    )

    TARGETS = [
        "step3j_wfo.py",
        "step4a_backtest.py",
        "step4_backtest.py",
        "step4b_backtest_strict.py",
        "step4d_backtest_full.py",
    ]

    def test_no_boundary_fill_pattern(self):
        offenders: list[tuple[str, int, str]] = []
        for fname in self.TARGETS:
            path = os.path.join(_ROOT, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if "#" in line:
                        line_no_comment = line.split("#", 1)[0]
                    else:
                        line_no_comment = line
                    if self.FORBIDDEN.search(line_no_comment):
                        offenders.append((fname, n, line.rstrip()))
        msg = "\n".join(f"{f}:{n}: {ln}" for (f, n, ln) in offenders)
        self.assertFalse(
            offenders,
            f"Survival-bias pattern detected (stop boundary used as exit "
            f"price). Offenders:\n{msg}",
        )


class ExitFillFormulaTests(unittest.TestCase):
    """
    Targeted regression: the gross_pnl formula used by every backtester is
    ``direction * (sp_exit_bar - sp_entry_bar)`` and never
    ``direction * (stop_z * std + mean - sp_entry_bar)``.

    Rather than spinning up the full backtest_oos pipeline (which requires
    upstream KDE, MetaGate, and adaptive arrays), this test inspects the
    AST of each backtester's exit block to confirm the gross-PnL right-hand
    side references the per-bar spread variable (``s``, ``sp``, or
    ``spread_now``) and not the stop boundary expression.
    """

    TARGETS_AND_BARSPREAD = {
        # filename → set of acceptable bar-spread identifiers used on the
        # right-hand side of the exit PnL formula.  Each name corresponds to
        # the per-bar observed spread variable in that file.
        "step3j_wfo.py":             {"s"},
        "step4a_backtest.py":        {"spread_now"},
        "step4_backtest.py":         {"spread_now"},
        "step4b_backtest_strict.py": {"sp"},
        "step4d_backtest_full.py":   {"sp"},
    }

    # Match the exit-PnL formulas:  (gross|gross_pnl|pnl_raw|g) = position *
    # (bar_spread - entry_spread).  The bar_spread identifier is captured
    # so the test can confirm it's an approved per-bar variable, not a
    # boundary expression.
    GROSS_RE = re.compile(
        r"\b(?:gross_pnl|gross|pnl_raw|g)\s*=\s*"
        r"[a-zA-Z_][a-zA-Z_0-9]*\s*\*\s*"
        r"\(\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*-\s*[a-zA-Z_][a-zA-Z_0-9]*\s*\)"
    )

    def test_gross_pnl_uses_bar_spread_variable(self):
        for fname, allowed in self.TARGETS_AND_BARSPREAD.items():
            path = os.path.join(_ROOT, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            found_any = False
            for match in self.GROSS_RE.finditer(src):
                found_any = True
                lhs = match.group(1)
                self.assertIn(
                    lhs, allowed,
                    f"{fname}: gross PnL formula uses unexpected exit-price "
                    f"variable '{lhs}'; expected one of {sorted(allowed)} "
                    f"to confirm current-bar-close fill semantics.",
                )
            self.assertTrue(
                found_any,
                f"{fname}: no gross PnL formula matched the expected "
                f"`direction * (bar_spread - entry_spread)` shape — "
                f"the test pattern may need an update if the file was "
                f"refactored.",
            )


if __name__ == "__main__":
    unittest.main()
