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
PENDING = "T1–T4 pending"

# Expected outcomes, measured at T0 (Ruling R5): (variant, dataset, test) →
# reason of a strict xfail.  Absent = must pass.  The one table later tasks flip.
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
    ("fixed",  "independent_rw",    "sharpe"): PENDING,
    ("fixed",  "independent_rw",    "tstat"):  PENDING,
    ("fixed",  "async_lag4.5_hol0", "sharpe"): PENDING,
    ("fixed",  "async_lag4.5_hol0", "tstat"):  PENDING,
    ("fixed",  "async_lag4.5_hol5", "sharpe"): PENDING,
    ("fixed",  "async_lag4.5_hol5", "tstat"):  PENDING,
    ("fixed",  "async_lag15_hol0",  "sharpe"): PENDING,
    ("fixed",  "async_lag15_hol0",  "tstat"):  PENDING,
    ("fixed",  "async_lag15_hol5",  "sharpe"): PENDING,
    ("fixed",  "async_lag15_hol5",  "tstat"):  PENDING,
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


@pytest.mark.parametrize("variant,dataset", cases("sharpe", ph.NULL_DATASETS))
def test_null_mean_gross_sharpe_is_zero(placebo, variant, dataset):
    s = placebo[(variant, dataset)]
    assert s.n_trades >= ph.MIN_POOLED_TRADES, f"only {s.n_trades} trades"
    assert -0.2 <= s.mean_sharpe <= 0.2, f"mean Sharpe {s.mean_sharpe:+.3f} ± {s.se_sharpe:.3f}"


@pytest.mark.parametrize("variant,dataset", cases("tstat", ph.NULL_DATASETS))
def test_null_mean_gross_trade_pnl_is_zero(placebo, variant, dataset):
    s = placebo[(variant, dataset)]
    assert s.n_trades >= ph.MIN_POOLED_TRADES, f"only {s.n_trades} trades"
    assert abs(s.tstat) < 3, f"per-trade t {s.tstat:+.1f} over {s.n_trades} trades"


def test_ou_positive_control_shows_its_edge(placebo):
    s = placebo[("fixed", ph.POSITIVE_CONTROL)]
    assert s.mean_sharpe > 1.0, f"mean Sharpe {s.mean_sharpe:+.3f} ± {s.se_sharpe:.3f}"


@functools.lru_cache(maxsize=None)
def full_and_truncated(key: tuple):
    """Runs of seed 0 on the full panel and on the panel cut at its middle date."""
    cfg, builder, dataset = key
    closes, pairs = ph.make_panel(dataset, 0, builder)
    cut = closes.index[len(closes) // 2]
    return ph.run_quiet(closes, pairs, cfg), ph.run_quiet(closes.loc[:cut], pairs, cfg)


@pytest.mark.parametrize("variant,dataset", cases("truncation", TRUNCATION_DATASETS))
def test_truncation_invariance(variant, dataset):
    # Ruling R4: at T0 compare the trades of every window the truncated run
    # has; T3 tightens this to every decision before the cut date.
    full, truncated = full_and_truncated(ph.run_key(variant, dataset))
    n = truncated.stats["n_windows"]
    assert 0 < n < full.stats["n_windows"] and not truncated.trades.empty
    before = full.trades[full.trades["window"] <= n].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated.trades, before, check_exact=True)
