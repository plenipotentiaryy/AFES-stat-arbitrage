"""
Unit tests for Phase D — Section 7.4 Single-Bullet Entry Guard.

Covers ``regime.apply_sbr_guard`` invariants:

  * When the kill-switch is False the function is an identity pass-through.
  * Below the SBR threshold the stop is unchanged.
  * At/above the threshold the stop is the tighter of the adaptive value
    and ``stop_mult * entry_z_base``.
  * Never loosens the stop (return value <= current_stop_z when fired).
  * Handles NaN / non-finite SBR gracefully.
"""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from regime import RegimeState, apply_sbr_guard


def _state_with_sbr(target_sbr: float) -> RegimeState:
    """
    Synthesise a RegimeState whose .sbr lands at approximately ``target_sbr``.

    Recall: sbr = 1 - s_break * s_cusum.  To get sbr == X we need
    s_break * s_cusum == 1 - X.  Choosing equal s_break == s_cusum gives
    s_break = sqrt(1 - X).  Then we back out the underlying scores via
    s_break = max(0, 1 - B/B_max).
    """
    from config import METAGATE_BREAK_HI, METAGATE_CUSUM_H
    target_sbr = float(np.clip(target_sbr, 0.0, 0.9999))
    s_eq = float(np.sqrt(1.0 - target_sbr))
    # Solve s_break == 1 - B/B_max  →  B = (1 - s_eq) * B_max
    b_t = (1.0 - s_eq) * METAGATE_BREAK_HI
    c_t = (1.0 - s_eq) * METAGATE_CUSUM_H
    return RegimeState(
        timestamp="2024-01-01",
        hmm_regime=0,
        kmeans_regime=1,
        break_score=b_t,
        cusum_val=c_t,
        hurst_val=0.5,
        adf_p_value=0.0,
    )


class SBRGuardTests(unittest.TestCase):

    def test_disabled_is_identity(self):
        st = _state_with_sbr(0.9)
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=3.5,
                              enabled=False)
        self.assertEqual(out, 3.5)

    def test_below_threshold_no_change(self):
        st = _state_with_sbr(0.4)
        # threshold default is 0.60
        self.assertLessEqual(st.sbr, 0.60)
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=3.5)
        self.assertEqual(out, 3.5)

    def test_above_threshold_tightens_to_mult_x_entry(self):
        st = _state_with_sbr(0.80)
        self.assertGreater(st.sbr, 0.60)
        # entry_z=2.0, mult=1.5 → tightened = 3.0; current 4.0 → returns 3.0
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=4.0,
                              stop_mult=1.5)
        self.assertAlmostEqual(out, 3.0, places=10)

    def test_above_threshold_does_not_loosen(self):
        st = _state_with_sbr(0.80)
        # If the adaptive stop is already tighter than 1.5*entry_z, the guard
        # must leave it alone (never loosen).
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=2.5,
                              stop_mult=1.5)  # 1.5*2.0=3.0 looser than 2.5
        self.assertAlmostEqual(out, 2.5, places=10)

    def test_at_threshold_exact_no_fire(self):
        # The rule says "exceeds" — equality should not fire.
        st = _state_with_sbr(0.60)
        # Allow tiny numerical fuzz around 0.60.
        if st.sbr > 0.60 + 1e-12:
            self.skipTest("Synthetic state landed strictly above threshold")
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=4.0,
                              threshold=0.60)
        self.assertEqual(out, 4.0)

    def test_nonfinite_sbr_passthrough(self):
        # Synthesise a RegimeState whose computed sbr is finite (the dataclass
        # sanitises inputs), then patch by passing a stub with a NaN .sbr.
        class _Stub:
            sbr = float("nan")
        out = apply_sbr_guard(_Stub(), entry_z_base=2.0, current_stop_z=4.0)
        self.assertEqual(out, 4.0)

    def test_custom_threshold_and_mult(self):
        st = _state_with_sbr(0.75)
        # Raise threshold above sbr → no fire.
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=4.0,
                              threshold=0.90, stop_mult=1.5)
        self.assertEqual(out, 4.0)
        # Lower threshold below sbr, larger mult.
        out = apply_sbr_guard(st, entry_z_base=2.0, current_stop_z=4.0,
                              threshold=0.50, stop_mult=1.8)
        self.assertAlmostEqual(out, 3.6, places=10)


if __name__ == "__main__":
    unittest.main()
