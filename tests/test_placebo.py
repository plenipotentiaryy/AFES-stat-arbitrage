"""Placebo tests of the daily WFO (audit task T0); see tests/README.md.

Null datasets have no edge: with causal fills every leg is a martingale, so
the mean gross Sharpe over 20 seeds must lie in [-0.2, +0.2] and the pooled
per-trade gross PnL must have |t| < 3.  The OU positive control (a real
edge) must give a mean Sharpe > 1.0.  Truncating the data must not change
the trades decided before the cut.
"""
import functools
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import placebo_harness as ph  # noqa: E402

RW_BUG = "pre-T1 behaviour: positions still open at a window end are dropped (T3)"
ASYNC_BUG = ("pre-T1 behaviour: same-close fills on date-aligned, forward-filled local "
             "prints converted with a mistimed daily FX close (T1/T2), and positions "
             "open at a window end dropped (T3)")
T3_PENDING = "T3 pending: positions still open at a window end are dropped"
T2_PENDING = ("T2 pending: fills on local-holiday dates at the forward-filled stale "
              "local print")

# Expected outcomes, measured at T0 and re-measured by each task (Ruling R5):
# (variant, dataset, test) → reason of a strict xfail.  Absent = must pass.
# The one table later tasks flip.
XFAIL = {
    ("legacy", "independent_rw",    "sharpe"): RW_BUG,
    ("legacy", "independent_rw",    "tstat"):  RW_BUG,
    ("legacy", "async_lag4.5_hol0", "sharpe"): ASYNC_BUG,
    ("legacy", "async_lag4.5_hol0", "tstat"):  ASYNC_BUG,
    ("legacy", "async_lag4.5_hol5", "sharpe"): ASYNC_BUG,
    ("legacy", "async_lag4.5_hol5", "tstat"):  ASYNC_BUG,
    ("legacy", "async_lag15_hol0",  "sharpe"): ASYNC_BUG,
    ("legacy", "async_lag15_hol0",  "tstat"):  ASYNC_BUG,
    ("legacy", "async_lag15_hol5",  "sharpe"): ASYNC_BUG,
    ("legacy", "async_lag15_hol5",  "tstat"):  ASYNC_BUG,
    ("fixed",  "independent_rw",    "sharpe"): T3_PENDING,
    ("fixed",  "independent_rw",    "tstat"):  T3_PENDING,
    ("fixed",  "async_lag4.5_hol5", "sharpe"): T2_PENDING,
    ("fixed",  "async_lag4.5_hol5", "tstat"):  T2_PENDING,
    ("fixed",  "async_lag15_hol5",  "sharpe"): T2_PENDING,
    ("fixed",  "async_lag15_hol5",  "tstat"):  T2_PENDING,
}

TRUNCATION_DATASETS = ["independent_rw", "async_lag15_hol5", "ou_positive_control"]


def cases(test: str, datasets: list[str]) -> list:
    out = []
    for variant in ph.VARIANTS:
        for dataset in datasets:
            reason = XFAIL.get((variant, dataset, test))
            marks = ([pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
                     if reason else [])
            out.append(pytest.param(variant, dataset, marks=marks, id=f"{variant}-{dataset}"))
    return out


@pytest.fixture(scope="session")
def placebo():
    return ph.results(ph.all_jobs())


def require_pooled_trades(s: ph.Summary) -> None:
    """The trade-count floor.  pytest.fail, not assert: a strict xfail mark
    (raises=AssertionError) must not absorb a floor failure."""
    if s.n_trades < ph.MIN_POOLED_TRADES:
        pytest.fail(f"only {s.n_trades} pooled trades (floor {ph.MIN_POOLED_TRADES})")


@pytest.mark.parametrize("variant,dataset", cases("sharpe", ph.NULL_DATASETS))
def test_null_mean_gross_sharpe_is_zero(placebo, variant, dataset):
    s = placebo[(variant, dataset)]
    require_pooled_trades(s)
    assert -0.2 <= s.mean_sharpe <= 0.2, f"mean Sharpe {s.mean_sharpe:+.3f} ± {s.se_sharpe:.3f}"


@pytest.mark.parametrize("variant,dataset", cases("tstat", ph.NULL_DATASETS))
def test_null_mean_gross_trade_pnl_is_zero(placebo, variant, dataset):
    s = placebo[(variant, dataset)]
    require_pooled_trades(s)
    assert abs(s.tstat) < 3, f"per-trade t {s.tstat:+.1f} over {s.n_trades} trades"


def test_ou_positive_control_shows_its_edge(placebo):
    s = placebo[("fixed", ph.POSITIVE_CONTROL)]
    assert s.mean_sharpe > 1.0, f"mean Sharpe {s.mean_sharpe:+.3f} ± {s.se_sharpe:.3f}"


@functools.lru_cache(maxsize=None)
def full_and_truncated(key: tuple):
    """Runs of seed 0 on the full panel and on the panel cut at its middle
    date D (inclusive), and D."""
    cfg, builder, dataset = key
    closes, pairs = ph.make_panel(dataset, 0, builder)
    cut = closes.index[len(closes) // 2]
    return ph.run_quiet(closes, pairs, cfg), ph.run_quiet(closes.loc[:cut], pairs, cfg), cut


@pytest.mark.parametrize("variant,dataset", cases("truncation", TRUNCATION_DATASETS))
def test_truncation_invariance(variant, dataset):
    # Trades of the windows the truncated run has (windows still reset, T3):
    # every trade whose exit was decided before the cut date D is identical
    # in all columns, and every entry decided before D has the same decision
    # date, fill date, side and size.  Decisions at or after D are excluded
    # (an exit decided on D fills on D+1 or, in the truncated run, at D's
    # close as end_of_data).
    full, truncated, cut = full_and_truncated(ph.run_key(variant, dataset))
    n = truncated.stats["n_windows"]
    assert 0 < n < full.stats["n_windows"] and not truncated.trades.empty
    ref = full.trades[full.trades["window"] <= n]

    def decided_before_cut(tr: pd.DataFrame, decision: str, cols=slice(None)) -> pd.DataFrame:
        return tr.loc[tr[decision] < cut, cols].reset_index(drop=True)

    exits = decided_before_cut(truncated.trades, "exit_signal")
    assert not exits.empty
    pd.testing.assert_frame_equal(exits, decided_before_cut(ref, "exit_signal"), check_exact=True)
    entry_cols = ["pair", "window", "entry_signal", "entry", "side", "size"]
    pd.testing.assert_frame_equal(decided_before_cut(truncated.trades, "entry_signal", entry_cols),
                                  decided_before_cut(ref, "entry_signal", entry_cols),
                                  check_exact=True)
