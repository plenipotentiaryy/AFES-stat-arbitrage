"""Placebo tests of the daily WFO (audit task T0); see tests/README.md.

Null datasets have no edge: log prices are martingales and fills are causal
(the USD convexity of fixed-quantity legs cancels between long and short
spreads, see tests/README.md), so the mean gross Sharpe over 20 seeds must
lie in [-0.2, +0.2] and the pooled per-trade gross PnL must have |t| < 3.  The OU positive control (a real
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
from wfo_daily_core import make_windows  # noqa: E402

RW_BUG = "pre-T1 behaviour: positions still open at a window end are dropped (T3)"
ASYNC_BUG = ("pre-T1 behaviour: same-close fills on date-aligned, forward-filled local "
             "prints converted with a mistimed daily FX close (T1/T2), and positions "
             "open at a window end dropped (T3)")

# Expected outcomes, measured at T0 and re-measured by each task (Ruling R5):
# (variant, dataset, test) → reason of a strict xfail.  Absent = must pass.
# The one table later tasks flip.  Since T3 every "fixed" null passes.
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
    date D (inclusive), and C = min(D, the truncated run's last OOS end)."""
    cfg, builder, dataset = key
    closes, pairs = ph.make_panel(dataset, 0, builder)
    cut = closes.index[len(closes) // 2]
    last_oos_e = make_windows(closes.loc[:cut], cfg.train, cfg.oos, cfg.step,
                              start=cfg.start, end=cfg.end)[-1][3]
    return (ph.run_quiet(closes, pairs, cfg), ph.run_quiet(closes.loc[:cut], pairs, cfg),
            min(cut, last_oos_e))


ENTRY_COLS = ["pair", "window_entered", "entry_signal", "entry", "side", "size", "beta",
              "p1_entry", "p2_entry"]


@pytest.mark.parametrize("variant,dataset", cases("truncation", TRUNCATION_DATASETS))
def test_truncation_invariance(variant, dataset):
    # Run on the panel cut at its middle date D and on the full panel.  With
    # C = min(D, the truncated run's last OOS end): every entry decided before
    # C has the same decision, fill, side, size, β and entry leg prices, and
    # every exit decided before C is identical in all columns (exit date,
    # reason, exit leg prices, pnl, ...).  Decisions at or after C are
    # excluded: the truncated run decides nothing after its last OOS end and
    # closes what is still open then as end_of_data (T3), so its end_of_data
    # closes are excluded too.
    full, truncated, c = full_and_truncated(ph.run_key(variant, dataset))
    assert 0 < truncated.stats["n_windows"] < full.stats["n_windows"] and not truncated.trades.empty

    def entries(tr: pd.DataFrame) -> pd.DataFrame:
        sel = tr[tr["entry_signal"] < c].sort_values(["pair", "entry_signal"])
        return sel[ENTRY_COLS].reset_index(drop=True)

    def exits(tr: pd.DataFrame) -> pd.DataFrame:
        sel = tr[(tr["exit_signal"] < c) & (tr["reason"] != "end_of_data")]
        return sel.sort_values(["pair", "exit_signal"]).reset_index(drop=True)

    assert not exits(truncated.trades).empty
    pd.testing.assert_frame_equal(exits(truncated.trades), exits(full.trades), check_exact=True)
    pd.testing.assert_frame_equal(entries(truncated.trades), entries(full.trades), check_exact=True)
