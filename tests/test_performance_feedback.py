"""
Unit tests for feedback.PerformanceFeedbackTracker.

Verifies the four guarantees called out in the design doc:
  1. Return-on-Notional math is correct.
  2. Cold-start: zero trades → S_perf = 1.0 (neutral via Bayesian prior).
  3. Bayesian shrinkage converges toward the empirical Sharpe as n grows.
  4. Piecewise-linear clipping respects [S_min, S_max].

Plus a few extras: zero-variance protection, serialisation round-trip,
and pair-key isolation (one pair's history doesn't leak into another).
"""

import os
import sys
import math
import unittest

# Make sure the project root is importable when running this file directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from feedback import PerformanceFeedbackTracker
from config import (
    PFB_WINDOW, PFB_K_PRIOR, PFB_SR_PRIOR,
    PFB_S_MIN, PFB_S_MAX, PFB_SR_FLOOR, PFB_SR_TARGET,
)


class ReturnOnNotionalTests(unittest.TestCase):
    """Check R_i computation and edge-case rejection."""

    def setUp(self):
        self.t = PerformanceFeedbackTracker()

    def test_returns_normalised_by_notional(self):
        self.t.add_trade("AAA", net_pnl=50.0, notional=1000.0)
        self.t.add_trade("AAA", net_pnl=-25.0, notional=1000.0)
        diag = self.t.get_diagnostics("AAA")
        # mean of [+0.05, -0.025] = 0.0125
        self.assertAlmostEqual(diag["mean_R"], 0.0125, places=10)
        self.assertEqual(diag["n"], 2)

    def test_invalid_notional_rejected(self):
        self.t.add_trade("AAA", net_pnl=100.0, notional=0.0)
        self.t.add_trade("AAA", net_pnl=100.0, notional=-50.0)
        self.t.add_trade("AAA", net_pnl=100.0, notional=float("nan"))
        self.assertEqual(self.t.get_diagnostics("AAA")["n"], 0)

    def test_nonfinite_pnl_rejected(self):
        self.t.add_trade("AAA", net_pnl=float("inf"), notional=100.0)
        self.t.add_trade("AAA", net_pnl=float("nan"), notional=100.0)
        self.assertEqual(self.t.get_diagnostics("AAA")["n"], 0)


class ColdStartTests(unittest.TestCase):
    """With zero trades the tracker must return the neutral multiplier."""

    def test_unknown_pair_returns_neutral_multiplier(self):
        t = PerformanceFeedbackTracker()
        # sr_eff = sr_prior = 1.0 → maps to S_perf inside [S_min, S_max].
        s = t.get_multiplier("NEVER_SEEN")
        # With defaults: sr_floor=-0.1, sr_target=1.2, sr_prior=1.0 → frac ≈ 0.846
        expected = PFB_S_MIN + (PFB_S_MAX - PFB_S_MIN) * (
            (PFB_SR_PRIOR - PFB_SR_FLOOR) / (PFB_SR_TARGET - PFB_SR_FLOOR)
        )
        self.assertAlmostEqual(s, expected, places=10)
        # Sanity: well within bounds.
        self.assertGreater(s, PFB_S_MIN)
        self.assertLess(s, PFB_S_MAX)

    def test_zero_buffer_diagnostics(self):
        t = PerformanceFeedbackTracker()
        d = t.get_diagnostics("X")
        self.assertEqual(d["n"], 0)
        self.assertEqual(d["mean_R"], 0.0)
        self.assertEqual(d["std_R"], 0.0)
        self.assertEqual(d["sr_trade"], PFB_SR_PRIOR)
        self.assertEqual(d["sr_effective"], PFB_SR_PRIOR)


class BayesianShrinkageTests(unittest.TestCase):
    """As n grows, sr_effective converges to the empirical sr_trade."""

    def test_shrinkage_converges_with_more_trades(self):
        t = PerformanceFeedbackTracker(window=50, k_prior=3.0, sr_prior=1.0)
        # Pump in trades with a clearly negative Sharpe.
        # R values: alternating -0.01 / -0.02 → mean negative, std small.
        sequence = [-0.01, -0.02] * 25
        sr_eff_history = []
        for r in sequence:
            t.add_trade("AAA", net_pnl=r * 1000.0, notional=1000.0)
            sr_eff_history.append(t.get_diagnostics("AAA")["sr_effective"])
        diag = t.get_diagnostics("AAA")
        # Empirical SR_trade must be negative.
        self.assertLess(diag["sr_trade"], 0.0)
        # With many trades, sr_eff should be much closer to sr_trade
        # than to sr_prior=1.0.
        n = diag["n"]
        empirical_weight = n / (n + 3.0)
        self.assertGreater(empirical_weight, 0.9)
        self.assertLess(diag["sr_effective"], 0.0)
        # Monotone convergence: |sr_eff - sr_trade| decreases over time.
        gaps = [abs(s - diag["sr_trade"]) for s in sr_eff_history]
        # The very first gap should be the largest.
        self.assertEqual(max(gaps), gaps[0])

    def test_first_trade_pulled_toward_prior(self):
        t = PerformanceFeedbackTracker(window=10, k_prior=3.0, sr_prior=1.0)
        t.add_trade("AAA", net_pnl=-100.0, notional=1000.0)
        d = t.get_diagnostics("AAA")
        # n=1, sample std is 0 (one observation) → sr_trade = mean_R / ε
        # which is a hugely negative number. Shrinkage with K=3, n=1
        # gives sr_eff = 0.25 * sr_trade + 0.75 * 1.0.
        self.assertEqual(d["n"], 1)
        self.assertLess(d["sr_effective"], 1.0)  # pulled below the prior
        # Weight on empirical = 1/(1+3) = 0.25
        self.assertAlmostEqual(d["sr_effective"],
                               0.25 * d["sr_trade"] + 0.75 * 1.0,
                               places=10)


class PiecewiseLinearMappingTests(unittest.TestCase):
    """S_perf must stay inside [S_min, S_max] for any SR_effective."""

    def test_extreme_negative_clipped_to_smin(self):
        t = PerformanceFeedbackTracker(window=20, k_prior=0.0)
        for _ in range(20):
            t.add_trade("AAA", net_pnl=-100.0, notional=1000.0)
        self.assertAlmostEqual(t.get_multiplier("AAA"), PFB_S_MIN, places=10)

    def test_extreme_positive_clipped_to_smax(self):
        t = PerformanceFeedbackTracker(window=20, k_prior=0.0)
        for _ in range(20):
            t.add_trade("AAA", net_pnl=+100.0, notional=1000.0)
        self.assertAlmostEqual(t.get_multiplier("AAA"), PFB_S_MAX, places=10)

    def test_all_values_within_bounds(self):
        t = PerformanceFeedbackTracker()
        # Mix of positive, negative, zero trades.
        import random
        rng = random.Random(0)
        for _ in range(50):
            r = rng.uniform(-100, 100)
            t.add_trade("AAA", net_pnl=r, notional=1000.0)
            s = t.get_multiplier("AAA")
            self.assertGreaterEqual(s, PFB_S_MIN)
            self.assertLessEqual(s, PFB_S_MAX)

    def test_boundary_at_sr_floor_returns_smin(self):
        t = PerformanceFeedbackTracker()
        # Force SR_eff exactly at sr_floor by direct call to internal mapper.
        self.assertEqual(t._map_to_s_perf(PFB_SR_FLOOR), PFB_S_MIN)

    def test_boundary_at_sr_target_returns_smax(self):
        t = PerformanceFeedbackTracker()
        self.assertEqual(t._map_to_s_perf(PFB_SR_TARGET), PFB_S_MAX)


class RobustnessTests(unittest.TestCase):
    """Zero variance, NaN, serialisation, pair isolation."""

    def test_zero_variance_no_division_error(self):
        t = PerformanceFeedbackTracker(window=10, k_prior=0.0)
        # All identical returns → std_R = 0 exactly.
        for _ in range(10):
            t.add_trade("AAA", net_pnl=10.0, notional=1000.0)
        d = t.get_diagnostics("AAA")
        # numpy's sample-std on identical inputs can return tiny float
        # noise (≪1e-15) — verify it's effectively zero, not exactly.
        self.assertLess(abs(d["std_R"]), 1e-12)
        # sr_trade should be a (huge but finite) positive number, NOT NaN/Inf.
        self.assertTrue(math.isfinite(d["sr_trade"]))
        self.assertGreater(d["sr_trade"], 0.0)
        # Mapped multiplier still inside bounds.
        self.assertEqual(t.get_multiplier("AAA"), PFB_S_MAX)

    def test_pair_buffers_isolated(self):
        t = PerformanceFeedbackTracker()
        for _ in range(5):
            t.add_trade("AAA", net_pnl=+10.0, notional=1000.0)
        for _ in range(5):
            t.add_trade("BBB", net_pnl=-10.0, notional=1000.0)
        self.assertGreater(t.get_diagnostics("AAA")["mean_R"], 0)
        self.assertLess(t.get_diagnostics("BBB")["mean_R"], 0)

    def test_serialisation_round_trip(self):
        t = PerformanceFeedbackTracker(window=7, k_prior=2.5)
        for r in [10, -5, 20, -3, 7]:
            t.add_trade("AAA", net_pnl=float(r), notional=1000.0)
        for r in [-1, -2, -3]:
            t.add_trade("BBB", net_pnl=float(r), notional=1000.0)
        snap = t.to_dict()
        # Round-trip through pickle as well — that's what live_state.pkl does.
        import pickle
        snap2 = pickle.loads(pickle.dumps(snap))
        rebuilt = PerformanceFeedbackTracker.from_dict(snap2)
        # Identical multipliers and diagnostics.
        for p in ("AAA", "BBB"):
            self.assertAlmostEqual(t.get_multiplier(p), rebuilt.get_multiplier(p), places=10)
            self.assertEqual(t.get_diagnostics(p)["n"],
                             rebuilt.get_diagnostics(p)["n"])
        self.assertEqual(rebuilt.window, 7)
        self.assertEqual(rebuilt.k_prior, 2.5)

    def test_buffer_capped_at_window(self):
        t = PerformanceFeedbackTracker(window=5)
        for r in range(20):
            t.add_trade("AAA", net_pnl=float(r), notional=100.0)
        self.assertEqual(t.get_diagnostics("AAA")["n"], 5)

    def test_reset_clears_history(self):
        t = PerformanceFeedbackTracker()
        for _ in range(5):
            t.add_trade("AAA", net_pnl=10.0, notional=1000.0)
        t.reset("AAA")
        self.assertEqual(t.get_diagnostics("AAA")["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
