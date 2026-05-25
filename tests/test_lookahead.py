"""
Unit tests for Phase A — look-ahead bias correction (Section 2 Flaw 1).

Validates that daily-frequency series shifted via ``regime.shift_daily_to_t1``
never expose information from date D when queried at intraday timestamps of
that same date.

Two layers of validation:
  1. The helper itself returns the previous day's value via ``.asof(d)``.
  2. The consumer-level integration: ``metagate.build_score_frame`` produces
     ``hmm_regime`` and ``kmeans_regime`` columns whose intraday values for
     date D equal day D-1's input, never day D's input.
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

from regime import shift_daily_to_t1
from metagate import build_score_frame


class ShiftDailyToT1Tests(unittest.TestCase):
    """Spot-check the primitive in isolation."""

    def test_none_passthrough(self):
        self.assertIsNone(shift_daily_to_t1(None))

    def test_empty_passthrough(self):
        s = pd.Series(dtype=float)
        out = shift_daily_to_t1(s)
        self.assertTrue(out.empty)

    def test_asof_returns_prior_day_value(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        # Day D carries the integer D (1..5).  After shift, asof(D) must
        # return D-1's value, never D's value.
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        shifted = shift_daily_to_t1(s)
        # Day 1 has no prior — should be NaN.
        self.assertTrue(pd.isna(shifted.iloc[0]))
        # Day 3 should expose Day 2's value (==2.0), not Day 3's value (==3.0).
        v = shifted.asof(pd.Timestamp("2024-01-03"))
        self.assertEqual(float(v), 2.0)
        # Day 5 → 4.0.
        v = shifted.asof(pd.Timestamp("2024-01-05"))
        self.assertEqual(float(v), 4.0)

    def test_intraday_query_returns_prior_day_value(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        s = pd.Series([10.0, 20.0, 30.0], index=idx)
        shifted = shift_daily_to_t1(s)
        # Mid-session bar of Day 3: must see Day 2's value (==20), not 30.
        v = shifted.asof(pd.Timestamp("2024-01-03 12:00:00"))
        self.assertEqual(float(v), 20.0)


class BuildScoreFrameIntegrationTests(unittest.TestCase):
    """
    End-to-end check: a daily HMM series with a Day-D change must not leak
    that change into intraday bars of Day D.
    """

    def _make_intraday_index(self, days: int, bars_per_day: int = 78):
        # 78 = a typical 5-min RTH count; the exact value doesn't matter here.
        start = pd.Timestamp("2024-01-01 09:30:00")
        deltas = pd.tseries.offsets.Minute(5)
        rows = []
        for d in range(days):
            day_open = start + pd.tseries.offsets.BDay(d)
            for b in range(bars_per_day):
                rows.append(day_open + b * deltas)
        return pd.DatetimeIndex(rows)

    def test_hmm_not_visible_intraday_on_change_day(self):
        idx = self._make_intraday_index(days=3, bars_per_day=10)
        sig = pd.DataFrame({
            "zscore": np.zeros(len(idx)),
            "vr":     np.ones(len(idx)),
        }, index=idx)

        # Daily HMM regime is 0 on Day 1 and Day 2, jumps to 1 on Day 3.
        day_dates = pd.to_datetime(sorted({t.normalize() for t in idx}))
        hmm = pd.Series([0, 0, 1], index=day_dates, name="hmm_regime")

        frame = build_score_frame(
            sig,
            hurst_series=None,
            adf_pvalue_series=None,
            hmm_series=hmm,
            kmeans_series=None,
        )

        # On Day 3 intraday bars we must see the PRIOR day's HMM value (0),
        # never Day 3's value (1).  This is the central look-ahead invariant.
        day3 = day_dates[2].normalize()
        day3_mask = pd.DatetimeIndex(frame.index).normalize() == day3
        self.assertGreater(int(day3_mask.sum()), 0)
        observed = set(frame.loc[day3_mask, "hmm_regime"].tolist())
        self.assertEqual(observed, {0.0},
                         f"Day-3 intraday hmm_regime leaked future value: {observed}")

        # On Day 2 intraday bars we must see Day 1's value (==0).
        day2 = day_dates[1].normalize()
        day2_mask = pd.DatetimeIndex(frame.index).normalize() == day2
        observed = set(frame.loc[day2_mask, "hmm_regime"].tolist())
        self.assertEqual(observed, {0.0})

    def test_kmeans_not_visible_intraday_on_change_day(self):
        idx = self._make_intraday_index(days=3, bars_per_day=8)
        sig = pd.DataFrame({
            "zscore": np.zeros(len(idx)),
            "vr":     np.ones(len(idx)),
        }, index=idx)

        day_dates = pd.to_datetime(sorted({t.normalize() for t in idx}))
        kmeans = pd.Series([1, 1, 2], index=day_dates, name="kmeans_regime")

        frame = build_score_frame(
            sig,
            hurst_series=None,
            adf_pvalue_series=None,
            hmm_series=None,
            kmeans_series=kmeans,
        )

        day3 = day_dates[2].normalize()
        day3_mask = pd.DatetimeIndex(frame.index).normalize() == day3
        # Must see Day-2 value (==1), never Day-3 value (==2).
        observed = set(frame.loc[day3_mask, "kmeans_regime"].tolist())
        self.assertEqual(observed, {1.0},
                         f"Day-3 intraday kmeans_regime leaked future value: {observed}")


if __name__ == "__main__":
    unittest.main()
