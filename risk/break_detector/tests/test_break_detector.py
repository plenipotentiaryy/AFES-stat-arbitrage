from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from risk.break_detector import (
    BetaVelocityMonitor,
    BreakCompositeScorer,
    BreakDetectorConfig,
    BreakDetectorDataLayer,
    BreakRiskGate,
    CUSUMDetector,
    KalmanInnovationRatioDetector,
    LocalHalfLifeExplosionDetector,
)


def synthetic_inputs(n: int = 220, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    spread = pd.Series(np.cumsum(rng.normal(0.0, 0.6, size=n)), index=idx)
    spread.iloc[150:170] += np.linspace(0.0, 8.0, 20)
    nu = spread.diff().fillna(0.0) + rng.normal(0.0, 0.1, size=n)
    beta = pd.Series(1.0 + np.cumsum(rng.normal(0.0, 0.002, size=n)), index=idx)
    beta.iloc[160:165] += np.linspace(0.0, 0.6, 5)
    return spread, pd.Series(nu, index=idx), beta


class BreakDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = BreakDetectorConfig(
            hl_short_window=12,
            hl_median_window=12,
            innovation_window=20,
            beta_velocity_window=20,
            mu0_window=40,
            hurst_window=30,
        )
        self.spread, self.innov, self.beta = synthetic_inputs()
        self.layer = BreakDetectorDataLayer(self.config)
        self.frame = self.layer.prepare(self.spread, self.innov, self.beta)

    def test_data_layer_outputs_expected_columns(self):
        self.assertIn("x_t", self.frame.columns)
        self.assertIn("nu2_t", self.frame.columns)
        self.assertIn("nu2_rolling_mean", self.frame.columns)
        self.assertIn("beta_velocity", self.frame.columns)
        self.assertGreater(len(self.frame), 100)

    def test_cusum_detector_flags_drift_and_resets(self):
        detector = CUSUMDetector(self.config)
        out = detector.compute(self.frame)
        self.assertIn("cusum_flag", out.columns)
        self.assertGreaterEqual(int(out["cusum_flag"].sum()), 1)
        alarm_idx = out.index[out["cusum_flag"] == 1]
        if len(alarm_idx) > 0:
            first_alarm = alarm_idx[0]
            loc = out.index.get_loc(first_alarm)
            if loc + 1 < len(out):
                self.assertLessEqual(abs(out["S_plus"].iloc[loc + 1]), out["cusum_h"].iloc[loc + 1] * 2.0)

    def test_innovation_ratio_detector_flags_spike(self):
        detector = KalmanInnovationRatioDetector(self.config)
        shocked = self.frame.copy()
        shocked.loc[shocked.index[-1], "nu2_t"] = shocked["nu2_rolling_mean"].iloc[-1] * 10.0
        out = detector.compute(shocked)
        self.assertEqual(int(out["innovation_flag"].iloc[-1]), 1)

    def test_half_life_detector_flags_explosive_theta(self):
        detector = LocalHalfLifeExplosionDetector(self.config)
        explosive = pd.Series(np.concatenate([np.zeros(30), np.geomspace(0.1, 20.0, 60)]))
        out = detector.compute(explosive)
        self.assertEqual(int(out["hl_flag"].iloc[-1]), 1)
        self.assertLessEqual(float(out["theta_local"].iloc[-1]), 0.0)

    def test_beta_velocity_monitor_flags_jump(self):
        detector = BetaVelocityMonitor(self.config)
        out = detector.compute(self.frame)
        self.assertIn("beta_velocity_zscore", out.columns)
        self.assertGreaterEqual(int(out["beta_flag"].sum()), 1)

    def test_composite_scorer_and_risk_gate(self):
        merged = pd.concat(
            [
                self.frame,
                CUSUMDetector(self.config).compute(self.frame),
                KalmanInnovationRatioDetector(self.config).compute(self.frame),
                LocalHalfLifeExplosionDetector(self.config).compute(self.frame["spread"]),
                BetaVelocityMonitor(self.config).compute(self.frame),
            ],
            axis=1,
        ).dropna()
        scored = BreakCompositeScorer(self.config).compute(merged)
        self.assertIn("break_level", scored.columns)
        self.assertTrue(set(scored["break_level"].unique()).issubset({"NORMAL", "CAUTION", "ALARM", "ABORT"}))

        gate = BreakRiskGate(self.config)
        abort_action = gate.action_for_level("ABORT")
        self.assertTrue(abort_action.cancel_entries)
        self.assertEqual(abort_action.reduce_existing_size_factor, 1.0)
        self.assertTrue(abort_action.tighten_stop_or_derisk)


if __name__ == "__main__":
    unittest.main()
