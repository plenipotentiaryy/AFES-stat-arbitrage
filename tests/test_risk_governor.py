"""Tests for AFES Tail EV Governor and hierarchical exposure caps."""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from risk.exposure_caps import ExposureCapsController
from risk.tail_ev_governor import GPDParameters, TailEVRiskGovernor


class ExposureCapsControllerTests(unittest.TestCase):
    def test_allows_entry_within_all_caps(self):
        ctrl = ExposureCapsController(
            max_ticker_exposure_pct=0.15,
            max_sector_exposure_pct=0.30,
            max_portfolio_leverage=2.0,
        )
        allowed, reason = ctrl.check_entry_allowed(
            pair_name="JPM-BAC",
            direction="LONG",
            nominal_size_usd=1_000.0,
            open_positions=[],
            current_capital=10_000.0,
            pair_sector_map={"JPM-BAC": "Financials"},
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")

    def test_blocks_ticker_concentration_across_pairs(self):
        ctrl = ExposureCapsController(max_ticker_exposure_pct=0.15)
        open_positions = [
            {"pair": "JPM-BAC", "nominal_size_usd": 1_000.0},
        ]
        allowed, reason = ctrl.check_entry_allowed(
            pair_name="JPM-WFC",
            direction="SHORT",
            nominal_size_usd=600.0,
            open_positions=open_positions,
            current_capital=10_000.0,
            pair_sector_map={"JPM-BAC": "Financials", "JPM-WFC": "Financials"},
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_TICKER_EXPOSURE", reason)

    def test_blocks_sector_cap(self):
        ctrl = ExposureCapsController(
            max_ticker_exposure_pct=1.0,
            max_sector_exposure_pct=0.30,
            max_portfolio_leverage=2.0,
        )
        open_positions = [
            {"pair": "JPM-BAC", "nominal_size_usd": 2_500.0},
        ]
        allowed, reason = ctrl.check_entry_allowed(
            pair_name="GS-MS",
            direction="LONG",
            nominal_size_usd=600.0,
            open_positions=open_positions,
            current_capital=10_000.0,
            pair_sector_map={"JPM-BAC": "Financials", "GS-MS": "Financials"},
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_SECTOR_EXPOSURE", reason)

    def test_blocks_global_leverage(self):
        ctrl = ExposureCapsController(
            max_ticker_exposure_pct=10.0,
            max_sector_exposure_pct=10.0,
            max_portfolio_leverage=0.20,
        )
        open_positions = [
            {"pair": "JPM-BAC", "nominal_size_usd": 1_500.0},
        ]
        allowed, reason = ctrl.check_entry_allowed(
            pair_name="HD-LOW",
            direction="LONG",
            nominal_size_usd=600.0,
            open_positions=open_positions,
            current_capital=10_000.0,
            pair_sector_map={"JPM-BAC": "Financials", "HD-LOW": "Consumer"},
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_PORTFOLIO_LEVERAGE", reason)

    def test_no_op_caps_allow_large_entry(self):
        ctrl = ExposureCapsController(
            max_ticker_exposure_pct=100.0,
            max_sector_exposure_pct=100.0,
            max_portfolio_leverage=100.0,
        )
        allowed, reason = ctrl.check_entry_allowed(
            pair_name="JPM-BAC",
            direction="LONG",
            nominal_size_usd=500_000.0,
            open_positions=[],
            current_capital=10_000.0,
            pair_sector_map={"JPM-BAC": "Financials"},
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")


class TailEVRiskGovernorTests(unittest.TestCase):
    def test_expected_shortfall_formula_exact_for_xi_lt_one(self):
        params = GPDParameters(
            shape=0.2,
            scale=1.5,
            threshold=3.0,
            n_exceedances=50,
            n_observations=500,
        )
        z = 4.0
        expected = 3.0 + (1.5 + 0.2 * (abs(z) - 3.0)) / (1.0 - 0.2)
        actual = TailEVRiskGovernor.expected_shortfall_z(z, params)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_expected_shortfall_infinite_mean_fallback(self):
        params = GPDParameters(
            shape=1.2,
            scale=1.0,
            threshold=3.0,
            n_exceedances=50,
            n_observations=500,
        )
        self.assertAlmostEqual(
            TailEVRiskGovernor.expected_shortfall_z(4.5, params),
            9.0,
            places=12,
        )

    def test_pair_es_usd_mapping(self):
        gov = TailEVRiskGovernor(u_threshold=3.0)
        gov.gpd_parameters["A-B"] = GPDParameters(
            shape=0.1,
            scale=1.0,
            threshold=3.0,
            n_exceedances=40,
            n_observations=400,
        )
        es_z = TailEVRiskGovernor.expected_shortfall_z(
            4.0, gov.gpd_parameters["A-B"]
        )
        actual = gov.calculate_pair_es_usd(
            "A-B",
            current_z=4.0,
            spread_std_usd=2.0,
            position_size_units=100.0,
        )
        self.assertAlmostEqual(actual, es_z * 2.0 * 100.0, places=12)

    def test_gpd_fit_recovers_positive_parameters(self):
        rng = np.random.default_rng(7)
        center = rng.normal(0.0, 1.0, 800)
        tail = 3.0 + rng.exponential(0.8, 120)
        z = np.concatenate([center, tail])
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            min_exceedances=20,
            min_observations=100,
        )
        gov.fit_pair_gpd("A-B", z)
        params = gov.get_pair_parameters("A-B")
        self.assertEqual(params.source, "mle")
        self.assertGreater(params.scale, 0.0)
        self.assertGreaterEqual(params.shape, 0.0)
        self.assertGreaterEqual(params.n_exceedances, 20)

    def test_prior_fallback_when_too_few_exceedances(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            prior_shape=0.1,
            prior_scale=2.0,
            min_exceedances=20,
        )
        gov.fit_pair_gpd("A-B", np.zeros(500))
        params = gov.get_pair_parameters("A-B")
        self.assertEqual(params.source, "prior")
        self.assertEqual(params.scale, 2.0)

    def test_portfolio_risk_safe_and_unsafe(self):
        gov = TailEVRiskGovernor(u_threshold=3.0, max_portfolio_es_pct=0.02)
        gov.gpd_parameters["A-B"] = GPDParameters(0.1, 1.0, 3.0, 50, 500)
        gov.gpd_parameters["C-D"] = GPDParameters(0.1, 1.0, 3.0, 50, 500)
        positions = [
            {
                "pair": "A-B",
                "current_z": 4.0,
                "spread_std_usd": 1.0,
                "position_size_units": 10.0,
                "p_revert": 0.3,
                "expected_gain_usd": 5.0,
            },
            {
                "pair": "C-D",
                "current_z": 5.0,
                "spread_std_usd": 1.0,
                "position_size_units": 50.0,
                "p_revert": 0.2,
                "expected_gain_usd": 2.0,
            },
        ]
        safe, order = gov.evaluate_portfolio_risk(positions, current_capital=10_000.0)
        self.assertFalse(safe)
        self.assertEqual(order[0], "C-D")

    def test_candidate_entry_blocked_by_tail_cap(self):
        gov = TailEVRiskGovernor(u_threshold=3.0, max_portfolio_es_pct=0.02)
        gov.gpd_parameters["A-B"] = GPDParameters(0.1, 1.0, 3.0, 50, 500)
        candidate = {
            "pair": "A-B",
            "current_z": 4.0,
            "spread_std_usd": 2.0,
            "position_size_units": 100.0,
            "p_revert": 0.5,
        }
        allowed, reason, projected = gov.check_candidate_entry(
            candidate,
            open_positions=[],
            current_capital=10_000.0,
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_TAIL_GOVERNOR", reason)
        self.assertGreater(projected, 200.0)

    def test_rolling_spread_std_is_shifted(self):
        spread = np.arange(10, dtype=float)
        import pandas as pd

        s = pd.Series(spread)
        out = TailEVRiskGovernor.rolling_spread_std_usd(s, window=3, min_periods=3)
        # At index 3, the rolling std must use values [0,1,2], not include 3.
        self.assertAlmostEqual(out.iloc[3], pd.Series([0.0, 1.0, 2.0]).std())

    def test_tail_contribution_budget_relative_blocks_entry(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            max_portfolio_es_pct=0.02,
            max_pair_tail_contribution=0.25,
            contribution_mode="BUDGET_RELATIVE",
        )
        gov.gpd_parameters["A-B"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        gov.gpd_parameters["C-D"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        open_positions = [{
            "pair": "A-B",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 5.0,  # ES = 20, contribution = 10%
        }]
        candidate = {
            "pair": "C-D",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 15.0,  # ES = 60, contribution = 30%
        }
        allowed, reason, projected = gov.check_candidate_entry(
            candidate,
            open_positions,
            current_capital=10_000.0,
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_TAIL_CONTRIBUTION C-D", reason)
        self.assertAlmostEqual(projected, 80.0, places=12)

    def test_tail_contribution_active_count_conditional_bypasses_small_portfolio(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            max_portfolio_es_pct=0.02,
            max_pair_tail_contribution=0.25,
            contribution_mode="ACTIVE_COUNT_CONDITIONAL",
        )
        gov.gpd_parameters["A-B"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        gov.gpd_parameters["C-D"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        open_positions = [{
            "pair": "A-B",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 5.0,
        }]
        candidate = {
            "pair": "C-D",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 15.0,
        }
        contributions = gov.calculate_tail_contributions(
            open_positions + [candidate],
            current_capital=10_000.0,
        )
        self.assertEqual(contributions["A-B"], 0.0)
        self.assertEqual(contributions["C-D"], 0.0)
        allowed, reason, projected = gov.check_candidate_entry(
            candidate,
            open_positions,
            current_capital=10_000.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")
        self.assertAlmostEqual(projected, 80.0, places=12)

    def test_tail_contribution_active_count_conditional_blocks_large_portfolio(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            max_portfolio_es_pct=0.02,
            max_pair_tail_contribution=0.25,
            contribution_mode="ACTIVE_COUNT_CONDITIONAL",
        )
        for pair in ["A-B", "C-D", "E-F", "G-H"]:
            gov.gpd_parameters[pair] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        open_positions = [
            {"pair": "A-B", "current_z": 4.0, "spread_std_usd": 1.0, "position_size_units": 5.0},
            {"pair": "C-D", "current_z": 4.0, "spread_std_usd": 1.0, "position_size_units": 5.0},
            {"pair": "E-F", "current_z": 4.0, "spread_std_usd": 1.0, "position_size_units": 5.0},
        ]
        candidate = {
            "pair": "G-H",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 20.0,  # ES = 80 of total 140 = 57.1%
        }
        allowed, reason, projected = gov.check_candidate_entry(
            candidate,
            open_positions,
            current_capital=10_000.0,
        )
        self.assertFalse(allowed)
        self.assertIn("BLOCKED_TAIL_CONTRIBUTION G-H", reason)
        self.assertAlmostEqual(projected, 140.0, places=12)

    def test_portfolio_risk_eval_flags_contribution_violators(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            max_portfolio_es_pct=0.02,
            max_pair_tail_contribution=0.25,
            contribution_mode="BUDGET_RELATIVE",
        )
        gov.gpd_parameters["A-B"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        gov.gpd_parameters["C-D"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        positions = [
            {
                "pair": "A-B",
                "current_z": 4.0,
                "spread_std_usd": 1.0,
                "position_size_units": 20.0,  # ES = 80, contribution = 40%
            },
            {
                "pair": "C-D",
                "current_z": 4.0,
                "spread_std_usd": 1.0,
                "position_size_units": 5.0,  # ES = 20, contribution = 10%
            },
        ]
        safe, order = gov.evaluate_portfolio_risk(positions, current_capital=10_000.0)
        self.assertFalse(safe)
        self.assertEqual(order[0], "A-B")
        self.assertNotIn("C-D", order)

    def test_tail_contribution_serialization_round_trip(self):
        gov = TailEVRiskGovernor(
            u_threshold=2.5,
            max_portfolio_es_pct=0.03,
            max_pair_tail_contribution=0.35,
            contribution_mode="ACTIVE_COUNT_CONDITIONAL",
        )
        restored = TailEVRiskGovernor.from_dict(gov.to_dict())
        self.assertEqual(restored.u_threshold, 2.5)
        self.assertEqual(restored.max_portfolio_es_pct, 0.03)
        self.assertEqual(restored.max_pair_tail_contribution, 0.35)
        self.assertEqual(restored.contribution_mode, "ACTIVE_COUNT_CONDITIONAL")

    def test_tail_contribution_no_op_when_limit_is_one(self):
        gov = TailEVRiskGovernor(
            u_threshold=3.0,
            max_portfolio_es_pct=0.02,
            max_pair_tail_contribution=1.0,
            contribution_mode="BUDGET_RELATIVE",
        )
        gov.gpd_parameters["A-B"] = GPDParameters(0.0, 1.0, 3.0, 50, 500)
        candidate = {
            "pair": "A-B",
            "current_z": 4.0,
            "spread_std_usd": 1.0,
            "position_size_units": 40.0,  # ES = 160 < total cap 200
        }
        allowed, reason, projected = gov.check_candidate_entry(
            candidate,
            open_positions=[],
            current_capital=10_000.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")
        self.assertAlmostEqual(projected, 160.0, places=12)


if __name__ == "__main__":
    unittest.main()
