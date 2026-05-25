"""
Unit tests for the WFO pair-quality gate (``pair_train_quality_ok``).

Verifies the contract called out in the function docstring:
  * Sharpe AND PnL must strictly exceed the floors — neither is sufficient
    on its own.
  * Missing / NaN / non-numeric inputs always fail (never silently pass).
  * Customising the floors via ``min_sharpe`` / ``min_pnl`` works.
  * ``-math.inf`` disables a specific check.
"""

import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from step3j_wfo import pair_train_quality_ok


class PairTrainQualityGateTests(unittest.TestCase):

    def test_passes_when_both_strictly_positive(self):
        best = {"sharpe": 1.5, "total_pnl": 0.5}
        self.assertTrue(pair_train_quality_ok(best))

    def test_fails_when_sharpe_zero(self):
        # Strictly-greater semantics: 0.0 does not exceed the 0.0 floor.
        best = {"sharpe": 0.0, "total_pnl": 0.5}
        self.assertFalse(pair_train_quality_ok(best))

    def test_fails_when_pnl_zero(self):
        best = {"sharpe": 1.5, "total_pnl": 0.0}
        self.assertFalse(pair_train_quality_ok(best))

    def test_fails_when_sharpe_negative(self):
        best = {"sharpe": -0.4, "total_pnl": 1.2}
        self.assertFalse(pair_train_quality_ok(best))

    def test_fails_when_pnl_negative(self):
        best = {"sharpe": 1.2, "total_pnl": -0.05}
        self.assertFalse(pair_train_quality_ok(best))

    def test_none_fails(self):
        self.assertFalse(pair_train_quality_ok(None))

    def test_missing_key_fails(self):
        self.assertFalse(pair_train_quality_ok({"sharpe": 1.0}))
        self.assertFalse(pair_train_quality_ok({"total_pnl": 1.0}))

    def test_non_numeric_fails(self):
        self.assertFalse(pair_train_quality_ok({"sharpe": "x", "total_pnl": 1.0}))
        self.assertFalse(pair_train_quality_ok({"sharpe": 1.0, "total_pnl": None}))

    def test_nan_fails(self):
        self.assertFalse(pair_train_quality_ok({"sharpe": float("nan"), "total_pnl": 1.0}))
        self.assertFalse(pair_train_quality_ok({"sharpe": 1.0, "total_pnl": float("nan")}))

    def test_custom_floors(self):
        best = {"sharpe": 0.5, "total_pnl": 0.5}
        # Default (0.0, 0.0) passes.
        self.assertTrue(pair_train_quality_ok(best))
        # Raise sharpe floor → fails.
        self.assertFalse(pair_train_quality_ok(best, min_sharpe=1.0))
        # Raise pnl floor → fails.
        self.assertFalse(pair_train_quality_ok(best, min_pnl=1.0))

    def test_neg_inf_disables_check(self):
        # Disabling both floors with -inf admits everything finite.
        best = {"sharpe": -10.0, "total_pnl": -10.0}
        self.assertTrue(pair_train_quality_ok(
            best, min_sharpe=-math.inf, min_pnl=-math.inf,
        ))
        # But still rejects non-finite values.
        self.assertFalse(pair_train_quality_ok(
            {"sharpe": float("inf"), "total_pnl": 1.0},
            min_sharpe=-math.inf, min_pnl=-math.inf,
        ))


if __name__ == "__main__":
    unittest.main()
