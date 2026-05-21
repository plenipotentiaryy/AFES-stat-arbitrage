from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".mplconfig-tests"))
sys.path.insert(0, str(ROOT / "signal"))

from tail_ev import TailEVConfig
from tail_ev.gpd_fitter import GPDFitter, gpd_expected_shortfall
from tail_ev.revert_survival_model import RevertSurvivalModel
from tail_ev.tail_data_layer import TailDataLayer
from tail_ev.tail_ev_gate import TailEVDecisionGate
from tail_ev.tail_ev_layer import TailEVLayer


def build_synthetic_series(n: int = 900, seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    base = rng.normal(0.0, 1.0, size=n)
    jumps = rng.random(n) < 0.08
    base[jumps] += rng.standard_t(df=4, size=jumps.sum()) * 2.5
    z = pd.Series(base).rolling(5, min_periods=1).mean() * 0.4 + pd.Series(base)
    z.index = idx
    context = pd.DataFrame(
        {
            "OFI": rng.normal(0.0, 1.0, size=n),
            "macro_regime": np.where(rng.random(n) > 0.85, 2.0, 1.0),
            "hurst_exp": np.clip(0.35 + rng.normal(0.0, 0.05, size=n), 0.1, 0.8),
            "coint_score": rng.normal(-2.5, 0.4, size=n),
        },
        index=idx,
    )
    return z.astype(float), context


class TailEVTests(unittest.TestCase):
    def setUp(self):
        self.config = TailEVConfig(
            rolling_window=60,
            min_periods=30,
            min_exceedances=20,
            min_train_size=40,
            max_holding_period=20,
            ad_bootstrap_samples=40,
            walk_forward_splits=3,
        )
        self.z, self.context = build_synthetic_series()

    def test_tail_data_layer_extracts_adaptive_exceedances(self):
        layer = TailDataLayer(self.config)
        dataset = layer.fit_transform(self.z, self.context)
        self.assertGreater(len(dataset.exceedances), self.config.min_exceedances)
        self.assertIn("z_score", dataset.state_frame.columns)
        self.assertIn("vol_ratio", dataset.state_frame.columns)
        self.assertTrue(np.all(dataset.exceedances > 0))
        self.assertGreater(dataset.threshold, 0.0)

    def test_gpd_fitter_and_expected_shortfall(self):
        rng = np.random.default_rng(11)
        exceedances = stats.genpareto.rvs(c=0.22, scale=0.85, size=400, random_state=rng)
        tail = pd.Series(2.0 + exceedances, index=pd.date_range("2021-01-01", periods=400, freq="D"))
        fitter = GPDFitter(self.config)
        result = fitter.fit(exceedances, tail, threshold=2.0)
        es = gpd_expected_shortfall(3.0, result.xi, result.beta, 2.0, len(exceedances), len(tail), alpha=0.95)
        self.assertGreaterEqual(result.xi, 0.0)
        self.assertGreater(result.beta, 0.0)
        self.assertGreaterEqual(es, 3.0)
        self.assertFalse(result.qq_frame.empty)

    def test_revert_survival_model_outputs_probabilities(self):
        data_layer = TailDataLayer(self.config)
        dataset = data_layer.fit_transform(self.z, self.context)
        X = dataset.state_frame
        y = pd.Series((X["z_velocity"] < 0).astype(int).to_numpy(), index=X.index)
        model = RevertSurvivalModel(self.config)
        report = model.fit(X, y)
        preds = model.predict_proba(X.tail(10))
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(preds <= 1.0))
        self.assertGreaterEqual(report.train_size, 1)

    def test_tail_ev_layer_and_gate_preserve_empirical_path(self):
        layer = TailEVLayer(self.config)
        layer.fit(self.z, self.context)

        def empirical_ev(z: float) -> float:
            return 0.1 if abs(z) < layer.dataset_.threshold else -0.1

        gate = TailEVDecisionGate(layer, empirical_ev_fn=empirical_ev, config=self.config)
        below = gate.evaluate(0.5)
        above = gate.evaluate(layer.dataset_.threshold + 1.0, context={"macro_regime": 2.0})
        self.assertEqual(below.regime, "empirical")
        self.assertIsNotNone(below.empirical_ev)
        self.assertEqual(above.regime, "tail")
        self.assertIsNotNone(above.expected_shortfall)


if __name__ == "__main__":
    unittest.main()
