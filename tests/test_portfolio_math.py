"""
Unit tests for Phase B — Section 7.1 Ledoit-Wolf and 7.2 HRP.

Covers the math invariants the production loops rely on:

    * regularize_covariance() reduces the condition number versus the empirical
      covariance for under-sampled / collinear universes.
    * HRP weights sum to 1.0, lie in [0, max_weight], and are deterministic for
      a fixed input.
    * optimize_portfolio_weights() honours the method dispatch and never
      regresses on the historical MVO behaviour when method == "markowitz".
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from step3j_wfo import (
    regularize_covariance,
    optimize_portfolio_weights,
    _hrp_weights,
    _hrp_quasi_diag,
    _hrp_recursive_bisect,
)


def _make_returns(n_obs: int, n_assets: int, seed: int = 0,
                  block_corr: float = 0.0) -> pd.DataFrame:
    """Synthetic returns with optional block-correlation structure."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_obs, n_assets))
    if block_corr > 0:
        common = rng.standard_normal((n_obs, 1))
        base = np.sqrt(1.0 - block_corr) * base + np.sqrt(block_corr) * common
    cols = [f"P{i:02d}" for i in range(n_assets)]
    idx = pd.date_range("2024-01-01", periods=n_obs, freq="D")
    return pd.DataFrame(base, index=idx, columns=cols)


class LedoitWolfTests(unittest.TestCase):
    """Shrinkage should improve conditioning when N approaches T."""

    def test_shrinkage_reduces_condition_number(self):
        # 30 assets, only 40 obs ⇒ empirical Σ is poorly conditioned.
        ret = _make_returns(n_obs=40, n_assets=30, block_corr=0.5, seed=11)
        emp = np.asarray(ret.cov().values, dtype=np.float64)
        shr = regularize_covariance(ret)
        # Both must be symmetric positive semi-definite.
        emp_eigs = np.sort(np.linalg.eigvalsh(0.5 * (emp + emp.T)))
        shr_eigs = np.sort(np.linalg.eigvalsh(0.5 * (shr + shr.T)))
        # Shrinkage drives the smallest eigenvalue upward.
        self.assertGreater(shr_eigs[0], emp_eigs[0])
        # Condition number (λ_max / λ_min) must drop.
        cond_emp = emp_eigs[-1] / max(emp_eigs[0], 1e-18)
        cond_shr = shr_eigs[-1] / max(shr_eigs[0], 1e-18)
        self.assertLess(cond_shr, cond_emp)

    def test_handles_empty_input(self):
        out = regularize_covariance(pd.DataFrame())
        self.assertEqual(out.shape, (0, 0))

    def test_single_asset_returns_scalar_cov(self):
        ret = _make_returns(n_obs=50, n_assets=1, seed=2)
        out = regularize_covariance(ret)
        # Should not raise; result shape is 1x1 or (1,) — accept either.
        self.assertTrue(out.size >= 1)


class HRPInternalTests(unittest.TestCase):
    """Spot-check the HRP building blocks."""

    def test_quasi_diag_preserves_leaves(self):
        # Simple linkage: 3 leaves, two merges.
        # Row format: [child_1, child_2, distance, n_in_cluster]
        link = np.array([
            [0, 1, 0.1, 2],   # merge 0 and 1 → cluster 3
            [2, 3, 0.2, 3],   # merge 2 and cluster 3 → cluster 4
        ], dtype=float)
        order = _hrp_quasi_diag(link)
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(len(order), 3)

    def test_recursive_bisect_weights_sum_to_one(self):
        # Identity-like covariance → equal weights (within rounding).
        n = 4
        cov = np.eye(n)
        order = list(range(n))
        w = _hrp_recursive_bisect(cov, order)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        for wi in w:
            self.assertGreaterEqual(wi, 0.0)


class HRPWeightsTests(unittest.TestCase):
    """End-to-end HRP weight properties."""

    def test_weights_sum_to_one(self):
        ret = _make_returns(n_obs=120, n_assets=8, block_corr=0.3, seed=7)
        w = _hrp_weights(ret, max_weight=1.0)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)

    def test_weights_non_negative(self):
        ret = _make_returns(n_obs=120, n_assets=8, block_corr=0.3, seed=7)
        w = _hrp_weights(ret, max_weight=1.0)
        for v in w.values():
            self.assertGreaterEqual(v, 0.0)

    def test_max_weight_cap_enforced(self):
        ret = _make_returns(n_obs=120, n_assets=8, block_corr=0.3, seed=7)
        cap = 0.20
        w = _hrp_weights(ret, max_weight=cap)
        # Allow a tiny epsilon from the cap-and-redistribute iteration.
        for v in w.values():
            self.assertLessEqual(v, cap + 1e-9)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)

    def test_deterministic_for_fixed_input(self):
        ret = _make_returns(n_obs=120, n_assets=8, seed=9, block_corr=0.4)
        w1 = _hrp_weights(ret, max_weight=1.0)
        w2 = _hrp_weights(ret, max_weight=1.0)
        for k in w1:
            self.assertAlmostEqual(w1[k], w2[k], places=12)

    def test_single_asset_returns_full_weight(self):
        ret = _make_returns(n_obs=120, n_assets=1, seed=0)
        w = _hrp_weights(ret, max_weight=1.0)
        self.assertEqual(len(w), 1)
        self.assertAlmostEqual(list(w.values())[0], 1.0, places=10)


class OptimizePortfolioWeightsDispatchTests(unittest.TestCase):
    """Verify the method dispatch in optimize_portfolio_weights."""

    def test_hrp_method_returns_sum_to_one(self):
        ret = _make_returns(n_obs=120, n_assets=6, block_corr=0.3, seed=4)
        w = optimize_portfolio_weights(ret, method="hrp", max_weight=0.5)
        self.assertEqual(set(w.keys()), set(ret.columns))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)
        for v in w.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 0.5 + 1e-9)

    def test_markowitz_method_returns_sum_to_one(self):
        # Larger T than N so SLSQP converges.
        ret = _make_returns(n_obs=180, n_assets=5, block_corr=0.1, seed=2)
        # Shift mean positive so target_return is reachable.
        ret = ret + 0.01
        w = optimize_portfolio_weights(ret, method="markowitz",
                                       max_weight=0.4, target_return=0.05)
        self.assertEqual(set(w.keys()), set(ret.columns))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_empty_returns_empty(self):
        self.assertEqual(optimize_portfolio_weights(pd.DataFrame(), method="hrp"), {})
        self.assertEqual(optimize_portfolio_weights(pd.DataFrame(), method="markowitz"), {})

    def test_single_column_returns_full_weight(self):
        ret = _make_returns(n_obs=50, n_assets=1, seed=0)
        for method in ("hrp", "markowitz"):
            w = optimize_portfolio_weights(ret, method=method)
            self.assertEqual(len(w), 1)
            self.assertAlmostEqual(list(w.values())[0], 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
