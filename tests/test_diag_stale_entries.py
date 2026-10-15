"""diag_stale_entries.py (audit task T2): the share of entries where either
leg had an exactly zero return on the signal day — on hand-made prices and
on the synthetic async holiday dataset (legacy vs fixed run)."""
import contextlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic  # noqa: E402
from diag_stale_entries import main as diag_main, stale_entries  # noqa: E402
from placebo_harness import VARIANTS  # noqa: E402
from wfo_daily_core import run_wfo_daily  # noqa: E402

DAYS = pd.bdate_range("2021-01-04", periods=5)


def test_zero_return_rule_on_hand_made_prices():
    closes = pd.DataFrame({"A": [1.0, 1.1, 1.1, np.nan, 1.2],
                           "B": [2.0, 2.05, 2.1, 2.2, 2.3]}, index=DAYS)
    trades = pd.DataFrame({"pair": ["A-B"] * 4, "entry_signal": DAYS[1:], "entry": DAYS[1:] + pd.Timedelta(days=1)})
    # moved | A unchanged | A missing | A vs its previous non-NaN close (1.1): moved
    expected = [False, True, True, False]
    assert stale_entries(trades, closes).tolist() == expected
    # Old trades files have no entry_signal: the entry date is the signal day.
    old = trades.drop(columns="entry").rename(columns={"entry_signal": "entry"})
    assert stale_entries(old, closes).tolist() == expected


def test_usd_leg_is_judged_on_its_forward_filled_local_series():
    # The USD leg moves every day (FX); the local market is closed on days 2 and 4.
    panel = pd.DataFrame({"U": [10.0, 11.0, 12.0, 13.0, 14.0],
                          "L.SY_USD": [5.0, 5.1, 5.2, 5.3, 5.4]}, index=DAYS)
    local = pd.DataFrame({"L.SY": [7.0, 7.1, 7.2]}, index=DAYS[[0, 1, 3]])
    trades = pd.DataFrame({"pair": ["U-L.SY_USD"] * 4, "entry_signal": DAYS[1:]})
    assert stale_entries(trades, panel).tolist() == [False] * 4
    assert stale_entries(trades, panel, local).tolist() == [False, True, False, True]


# ── The synthetic async holiday dataset ─────────────────────────────────────

DECISIONS_AND_FILLS = ["entry_signal", "entry", "exit_signal", "exit"]


@pytest.fixture(scope="module")
def async_runs(tmp_path_factory):
    """Legacy and fixed runs on one async dataset with 10 local holidays a
    year; closes, trades and the raw local closes written as CSVs."""
    raw = synthetic.async_same_asset(n_years=6, n_pairs=6, close_lag_hours=15,
                                     holidays_per_year=10, seed=0)
    out = tmp_path_factory.mktemp("diag")
    raw.local.to_csv(out / "local.csv")
    trades = {}
    for variant in ("legacy", "fixed"):
        cfg, builder = VARIANTS[variant]
        closes, pairs = builder(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            trades[variant] = run_wfo_daily(closes, pairs, cfg).trades
        closes.to_csv(out / f"closes_{variant}.csv")
        trades[variant].to_csv(out / f"trades_{variant}.csv", index=False)
    return raw, out, trades


def diag(capsys, out, variant: str, local: bool) -> tuple[int, int]:
    """(stale, entries) printed by the script for a run."""
    args = ["--trades", str(out / f"trades_{variant}.csv"), "--closes", str(out / f"closes_{variant}.csv")]
    if local:
        args += ["--local-closes", str(out / "local.csv")]
    diag_main(args)
    m = re.search(r"stale entries: (\d+) of (\d+)", capsys.readouterr().out)
    return int(m.group(1)), int(m.group(2))


def test_legacy_run_enters_on_stale_local_prints(async_runs, capsys):
    raw, out, trades = async_runs
    on_holiday = trades["legacy"]["entry_signal"].isin(raw.holidays)
    stale, n = diag(capsys, out, "legacy", local=True)
    assert n == len(trades["legacy"]) and stale > 0
    assert stale == on_holiday.sum()          # exactly the entries signalled on a local holiday


@pytest.mark.parametrize("local", [True, False], ids=["local_closes", "panel_only"])
def test_fixed_run_has_no_stale_entry(async_runs, capsys, local):
    _, out, trades = async_runs
    assert diag(capsys, out, "fixed", local=local) == (0, len(trades["fixed"]))


def test_fixed_run_never_decides_or_fills_on_a_local_holiday(async_runs):
    raw, _, trades = async_runs
    assert trades["legacy"][DECISIONS_AND_FILLS].isin(raw.holidays).any().all()
    assert len(trades["fixed"]) > 100
    assert not trades["fixed"][DECISIONS_AND_FILLS].isin(raw.holidays).any().any()
