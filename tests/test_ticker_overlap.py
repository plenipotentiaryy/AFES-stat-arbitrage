"""
Unit tests for Phase C — Section 7.3 Active Ticker Correlation Throttling.

Coverage:
  * share_ticker — string parsing edge cases.
  * TickerOverlapBlocker — DISABLED / SCALE / HARD modes; threshold respect;
    missing-correlation defaults; symmetric pair lookup.
  * filter_trade_list — chronological enforcement; expired trades drop out
    of the open list; PnL scaling matches the chosen scale factor; no
    mutation of the input DataFrame.
  * compute_spread_correlations — basic structural sanity.
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

from risk.ticker_overlap import (
    TickerOverlapBlocker,
    OverlapDecision,
    FilterStats,
    share_ticker,
    filter_trade_list,
    compute_spread_correlations,
)


class ShareTickerTests(unittest.TestCase):

    def test_overlap_detected(self):
        self.assertTrue(share_ticker("JPM-BAC", "JPM-WFC"))
        self.assertTrue(share_ticker("JPM-BAC", "WFC-JPM"))

    def test_no_overlap(self):
        self.assertFalse(share_ticker("AAPL-MSFT", "GOOG-AMZN"))

    def test_invalid_inputs(self):
        self.assertFalse(share_ticker("", "JPM-BAC"))
        self.assertFalse(share_ticker("BADFORMAT", "JPM-BAC"))
        self.assertFalse(share_ticker(None, "JPM-BAC"))


class BlockerModeTests(unittest.TestCase):

    def _corr(self) -> pd.DataFrame:
        idx = ["JPM-BAC", "JPM-WFC", "AAPL-MSFT"]
        # JPM-BAC and JPM-WFC strongly correlated; AAPL pair independent.
        data = np.array([
            [1.0, 0.80, 0.05],
            [0.80, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ])
        return pd.DataFrame(data, index=idx, columns=idx)

    def test_disabled_mode_never_blocks(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="DISABLED")
        dec = b.evaluate("JPM-BAC", ["JPM-WFC"])
        self.assertFalse(dec.blocked)
        self.assertEqual(dec.scale, 1.0)

    def test_scale_mode_returns_scale_factor(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="SCALE", scale=0.25)
        dec = b.evaluate("JPM-BAC", ["JPM-WFC"])
        self.assertFalse(dec.blocked)
        self.assertEqual(dec.scale, 0.25)
        self.assertEqual(dec.blocking_pair, "JPM-WFC")

    def test_hard_mode_blocks_entry(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="HARD")
        dec = b.evaluate("JPM-BAC", ["JPM-WFC"])
        self.assertTrue(dec.blocked)
        self.assertEqual(dec.scale, 0.0)

    def test_threshold_respected(self):
        # Raise threshold above the 0.80 entry → no block.
        b = TickerOverlapBlocker(self._corr(), threshold=0.90, mode="HARD")
        dec = b.evaluate("JPM-BAC", ["JPM-WFC"])
        self.assertFalse(dec.blocked)

    def test_no_ticker_overlap_no_block(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="HARD")
        # AAPL-MSFT does not share a ticker with JPM-BAC, no block.
        dec = b.evaluate("AAPL-MSFT", ["JPM-BAC"])
        self.assertFalse(dec.blocked)

    def test_symmetric_lookup(self):
        # Even when correlation is stored under one ordering, the lookup
        # accepts the reversed pair tuple.
        b = TickerOverlapBlocker({"JPM-BAC": {"JPM-WFC": 0.92}},
                                 threshold=0.65, mode="HARD")
        dec1 = b.evaluate("JPM-WFC", ["JPM-BAC"])
        dec2 = b.evaluate("JPM-BAC", ["JPM-WFC"])
        self.assertTrue(dec1.blocked)
        self.assertTrue(dec2.blocked)

    def test_unknown_mode_falls_back_to_disabled(self):
        b = TickerOverlapBlocker(self._corr(), mode="NONSENSE")
        self.assertEqual(b.mode, "DISABLED")
        self.assertFalse(b.evaluate("JPM-BAC", ["JPM-WFC"]).blocked)


class FilterTradeListTests(unittest.TestCase):

    def _build_trades(self) -> pd.DataFrame:
        # Two overlapping trades on JPM-BAC and JPM-WFC, plus an independent
        # AAPL-MSFT trade. The JPM-WFC trade enters while JPM-BAC is open.
        return pd.DataFrame([
            {"pair": "JPM-BAC", "entry_time": "2024-01-02 10:00",
             "exit_time":  "2024-01-02 12:00", "net_pnl": 100.0, "size_mult": 1.0},
            {"pair": "JPM-WFC", "entry_time": "2024-01-02 11:00",
             "exit_time":  "2024-01-02 13:00", "net_pnl": 200.0, "size_mult": 1.0},
            {"pair": "AAPL-MSFT", "entry_time": "2024-01-02 11:30",
             "exit_time":  "2024-01-02 14:00", "net_pnl": 50.0, "size_mult": 1.0},
            # Later JPM-WFC trade after JPM-BAC has closed → must pass.
            {"pair": "JPM-WFC", "entry_time": "2024-01-02 13:30",
             "exit_time":  "2024-01-02 15:00", "net_pnl": 300.0, "size_mult": 1.0},
        ])

    def _corr(self) -> pd.DataFrame:
        idx = ["JPM-BAC", "JPM-WFC", "AAPL-MSFT"]
        return pd.DataFrame([
            [1.0, 0.90, 0.0],
            [0.90, 1.0, 0.0],
            [0.0,  0.0, 1.0],
        ], index=idx, columns=idx)

    def test_scale_mode_scales_overlapping_trade(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="SCALE", scale=0.25)
        out, stats = filter_trade_list(self._build_trades(), b)
        # Three input trades pass and one is scaled (the 11:00 JPM-WFC entry).
        self.assertEqual(stats.n_input, 4)
        self.assertEqual(stats.n_kept, 4)
        self.assertEqual(stats.n_dropped, 0)
        self.assertEqual(stats.n_scaled, 1)
        # Identify the scaled row: JPM-WFC at 11:00.
        scaled = out[(out["pair"] == "JPM-WFC") &
                     (out["entry_time"] == pd.Timestamp("2024-01-02 11:00"))]
        self.assertEqual(len(scaled), 1)
        self.assertAlmostEqual(float(scaled.iloc[0]["net_pnl"]), 200.0 * 0.25)
        self.assertAlmostEqual(float(scaled.iloc[0]["size_mult"]), 1.0 * 0.25)
        # Independent AAPL-MSFT must survive untouched.
        ams = out[out["pair"] == "AAPL-MSFT"]
        self.assertAlmostEqual(float(ams.iloc[0]["net_pnl"]), 50.0)

    def test_hard_mode_drops_overlapping_trade(self):
        b = TickerOverlapBlocker(self._corr(), threshold=0.65, mode="HARD")
        out, stats = filter_trade_list(self._build_trades(), b)
        # The 11:00 JPM-WFC entry must be dropped.
        self.assertEqual(stats.n_dropped, 1)
        self.assertEqual(stats.n_kept, 3)
        # The later JPM-WFC (after JPM-BAC closed) must survive.
        survived = out[out["pair"] == "JPM-WFC"]
        self.assertEqual(len(survived), 1)
        self.assertEqual(pd.Timestamp(survived.iloc[0]["entry_time"]),
                         pd.Timestamp("2024-01-02 13:30"))

    def test_disabled_mode_passes_through(self):
        b = TickerOverlapBlocker(self._corr(), mode="DISABLED")
        df_in = self._build_trades()
        out, stats = filter_trade_list(df_in, b)
        self.assertEqual(stats.n_input, 4)
        self.assertEqual(stats.n_kept, 4)
        self.assertEqual(stats.n_dropped, 0)
        self.assertEqual(stats.n_scaled, 0)
        # PnL unchanged.
        self.assertAlmostEqual(out["net_pnl"].sum(), df_in["net_pnl"].sum())

    def test_input_not_mutated(self):
        b = TickerOverlapBlocker(self._corr(), mode="HARD")
        df_in = self._build_trades()
        snapshot = df_in.copy(deep=True)
        _ = filter_trade_list(df_in, b)
        pd.testing.assert_frame_equal(df_in, snapshot)


class ComputeSpreadCorrelationsTests(unittest.TestCase):

    def test_window_respected(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        # Highly correlated spreads.
        common = rng.standard_normal(200).cumsum()
        spreads = {
            "JPM-BAC":   pd.Series(common + rng.standard_normal(200) * 0.05, index=idx),
            "JPM-WFC":   pd.Series(common + rng.standard_normal(200) * 0.05, index=idx),
            "AAPL-MSFT": pd.Series(rng.standard_normal(200).cumsum(), index=idx),
        }
        corr = compute_spread_correlations(spreads, window_days=90)
        self.assertEqual(set(corr.columns), {"JPM-BAC", "JPM-WFC", "AAPL-MSFT"})
        # Diagonal must be exactly 1.0.
        for c in corr.columns:
            self.assertAlmostEqual(corr.loc[c, c], 1.0, places=10)
        # The two JPM pairs must be highly correlated, the AAPL pair not.
        self.assertGreater(corr.loc["JPM-BAC", "JPM-WFC"], 0.5)
        self.assertLess(abs(corr.loc["JPM-BAC", "AAPL-MSFT"]), 0.3)

    def test_short_series_dropped(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        spreads = {"ABC-DEF": pd.Series(rng.standard_normal(30), index=idx)}
        out = compute_spread_correlations(spreads, window_days=90)
        # Series is shorter than window → dropped → empty frame.
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
