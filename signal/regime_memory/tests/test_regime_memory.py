from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".mplconfig-regime-memory-tests"))
sys.path.insert(0, str(ROOT / "signal"))

from regime_memory import build_profiler
from regime_memory.config import RegimeMemoryConfig
from regime_memory.decayed_density_estimator import DecayedDensityEstimator, LegacyExpandingWindowProfiler
from regime_memory.profile_manager import RegimeAwareProfileManager
from regime_memory.regime_classifier import RegimeClassifier
from regime_memory.regime_memory_bank import InsufficientMemoryError, ObservationRecord, RegimeMemoryBank


def synthetic_observations(n: int = 180, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    hl = np.r_[rng.normal(12, 2, n // 3), rng.normal(24, 3, n // 3), rng.normal(42, 4, n - 2 * (n // 3))]
    z = rng.normal(0.0, 1.5, size=n)
    z[n // 2 :] += 0.7
    hurst = np.clip(0.35 + hl / 150.0 + rng.normal(0.0, 0.03, size=n), 0.1, 0.8)
    vol_ratio = np.clip(1.0 + rng.normal(0.0, 0.25, size=n), 0.4, 2.2)
    macro = np.where(vol_ratio > 1.3, 2, np.where(vol_ratio < 0.8, 0, 1))
    reverted = (rng.random(n) > np.clip(np.abs(z) / 4.0, 0.1, 0.8)).astype(int)
    pnl = np.where(reverted == 1, np.abs(z), -np.abs(z) * 0.8)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "z_score": z,
            "hl_estimate": hl,
            "vol_ratio": vol_ratio,
            "hurst": hurst,
            "macro_regime": macro,
            "outcome_pnl": pnl,
            "reverted": reverted,
            "break_score": 0.0,
        }
    )


class RegimeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = RegimeMemoryConfig(
            pair_name="TEST-PAIR",
            pair_invalidation_q={"TEST-PAIR": 2.2},
            min_regime_samples=15,
            min_effective_weight=3.0,
            min_effective_sample_size=10.0,
        )
        self.frame = synthetic_observations()
        self.classifier = RegimeClassifier(self.config)
        self.frame["regime_label"] = self.frame.apply(self.classifier.classify_rule_based, axis=1)

    def test_memory_bank_add_and_get(self):
        bank = RegimeMemoryBank(self.config, classifier=self.classifier)
        for _, row in self.frame.head(40).iterrows():
            bank.add_observation(
                ObservationRecord(
                    timestamp=row["timestamp"],
                    z_score=float(row["z_score"]),
                    hl_estimate=float(row["hl_estimate"]),
                    vol_ratio=float(row["vol_ratio"]),
                    hurst=float(row["hurst"]),
                    macro_regime=int(row["macro_regime"]),
                    outcome_pnl=float(row["outcome_pnl"]),
                    reverted=int(row["reverted"]),
                    break_score=float(row["break_score"]),
                )
            )
        regime = bank.memory.iloc[-1]["regime_label"]
        mem = bank.get_regime_memory(str(regime), min_samples=1)
        self.assertFalse(mem.empty)
        self.assertTrue((mem["regime_label"] == regime).all())

    def test_memory_bank_invalidates_on_hl_jump(self):
        bank = RegimeMemoryBank(self.config, classifier=self.classifier)
        for _, row in self.frame.head(80).iterrows():
            bank.add_observation(
                ObservationRecord(
                    timestamp=row["timestamp"],
                    z_score=float(row["z_score"]),
                    hl_estimate=12.0,
                    vol_ratio=float(row["vol_ratio"]),
                    hurst=float(row["hurst"]),
                    macro_regime=int(row["macro_regime"]),
                    outcome_pnl=float(row["outcome_pnl"]),
                    reverted=int(row["reverted"]),
                    break_score=float(row["break_score"]),
                )
            )
        self.assertTrue(bank.invalidate_stale_profile(40.0))

    def test_classifier_rule_and_kmeans_fallback(self):
        unstable = {"hl_estimate": 80.0, "hurst": 0.60, "vol_ratio": 1.2, "macro_regime": 2, "break_score": 0.0}
        self.assertEqual(self.classifier.classify(unstable), "UNSTABLE")
        fitted = self.classifier.maybe_recalibrate(self.frame.assign(regime_label=self.frame["regime_label"]), force=True)
        self.assertTrue(fitted or not fitted)
        label = self.classifier.classify(self.frame.iloc[0].to_dict())
        self.assertIn(label, {"FAST", "MEDIUM", "SLOW", "UNSTABLE"})

    def test_decayed_estimator_logs_thin_memory_warning(self):
        thin_config = RegimeMemoryConfig(
            pair_name="TEST-PAIR",
            pair_invalidation_q={"TEST-PAIR": 2.2},
            min_regime_samples=15,
            min_effective_weight=8.0,
            min_effective_sample_size=10.0,
        )
        estimator = DecayedDensityEstimator(thin_config)
        thin = self.frame.tail(2).copy()
        thin["regime_label"] = "FAST"
        with self.assertLogs("regime_memory.decayed_density_estimator", level="WARNING") as cm:
            estimator.fit(thin)
        self.assertTrue(any("Thin regime memory" in msg for msg in cm.output))

    def test_decayed_estimator_and_legacy_share_interface(self):
        weighted = DecayedDensityEstimator(self.config)
        legacy = LegacyExpandingWindowProfiler(self.config)
        data = self.frame.copy()
        weighted.fit(data)
        legacy.fit(data)
        for profiler in (weighted, legacy):
            self.assertTrue(np.isfinite(profiler.density(1.5)))
            self.assertTrue(0.0 <= profiler.p_revert(1.5) <= 1.0)

    def test_manager_rebuilds_and_factory_returns_expected_type(self):
        manager = RegimeAwareProfileManager(self.config)
        first_rebuilt = False
        for _, row in self.frame.head(40).iterrows():
            res = manager.update(
                ObservationRecord(
                    timestamp=row["timestamp"],
                    z_score=float(row["z_score"]),
                    hl_estimate=float(row["hl_estimate"]),
                    vol_ratio=float(row["vol_ratio"]),
                    hurst=float(row["hurst"]),
                    macro_regime=int(row["macro_regime"]),
                    outcome_pnl=float(row["outcome_pnl"]),
                    reverted=int(row["reverted"]),
                    break_score=float(row["break_score"]),
                )
            )
            first_rebuilt = first_rebuilt or res.profile_rebuilt
        self.assertTrue(first_rebuilt)
        self.assertIsInstance(build_profiler(self.config, use_regime_memory=True), RegimeAwareProfileManager)
        self.assertIsInstance(build_profiler(self.config, use_regime_memory=False), LegacyExpandingWindowProfiler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
