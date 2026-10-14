"""The extracted core reproduces the pre-refactor CLI (audit task T0).

tests/fixtures/t0_golden/ holds the trades CSVs that step3j_wfo_daily.py as
of commit 46ffdfb wrote for small synthetic panels (see make_golden.py).
`run_wfo_daily`, given the same inputs and the WfoConfig the CLI builds from
the same arguments, must reproduce every fixture byte for byte.  Later tasks
run these cases with their legacy_* flags on (LEGACY_FLAGS), so the legacy
path keeps reproducing the pre-audit code.
"""
import contextlib
import dataclasses
import importlib.util
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from step3j_wfo_daily import build_parser  # noqa: E402
from wfo_daily_core import WfoConfig, run_wfo_daily  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "t0_make_golden", os.path.join(_HERE, "fixtures", "t0_golden", "make_golden.py"))
make_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_golden)

# WfoConfig fields that select the pre-audit behaviour.  Empty at T0; each
# later task adds its legacy_* flag here.
LEGACY_FLAGS: dict = {}


def as_cli_reads(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """`df` as the CLI reads it back from its CSV (same text, same parser)."""
    if df is None:
        return None
    return pd.read_csv(io.StringIO(df.to_csv()), parse_dates=["Date"]).set_index("Date").sort_index()


def cli_config(args: list[str]) -> WfoConfig:
    """The WfoConfig step3j_wfo_daily.main() builds from these arguments."""
    ns = build_parser().parse_args(args)
    return WfoConfig(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(WfoConfig)})


def run_case(case: str, **overrides):
    inp = make_golden.case_inputs(case)
    cfg = dataclasses.replace(cli_config(inp["args"]), **LEGACY_FLAGS, **overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(as_cli_reads(inp["closes"]), inp["pairs"], cfg,
                             volumes=as_cli_reads(inp["volumes"]),
                             vwap=as_cli_reads(inp["vwap"]))


@pytest.mark.parametrize("case", list(make_golden.CASES))
def test_core_reproduces_pre_refactor_cli(case):
    res = run_case(case)
    fixture = make_golden.fixture_path(case)
    if fixture.exists():
        assert res.trades.to_csv(index=False).encode() == fixture.read_bytes()
    else:                                   # the pre-refactor CLI found no trades
        assert res.trades.empty
        assert res.stats["n_trades"] == 0 and np.isnan(res.stats["sharpe_full"])
        assert len(res.daily_pnl) > 0 and (res.daily_pnl == 0).all()


def test_result_daily_pnl_and_stats_match_trades():
    res = run_case("flat")
    tr, daily = res.trades, res.daily_pnl
    assert daily.index.is_monotonic_increasing and daily.index.is_unique
    assert daily.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-12)
    assert set(tr["exit"].dt.normalize()) <= set(daily.index)
    assert res.stats["n_trades"] == len(tr)
    assert res.stats["total_pnl"] == tr["pnl"].sum()
    assert res.stats["sharpe_full"] == daily.mean() / daily.std() * np.sqrt(252)


def test_cli_options_are_the_wfo_config_fields_with_same_defaults():
    defaults = vars(build_parser().parse_args([]))
    fields = {f.name for f in dataclasses.fields(WfoConfig)}
    assert set(defaults) - {"pairs", "closes", "tag"} == fields
    assert {k: defaults[k] for k in fields} == dataclasses.asdict(WfoConfig())


@pytest.mark.parametrize("flag", ["metagate", "hrp"])
def test_metagate_and_hrp_keep_pre_existing_unbound_local_error(flag):
    # Ruling R7: main() referenced cost_log before assignment with these flags;
    # T0 preserves the failure (see AUDIT_FIX_LOG.md, known limitations).
    with pytest.raises(UnboundLocalError):
        run_case("flat", **{flag: True})


@pytest.mark.parametrize("flag", ["vwz", "vwap_exec"])
def test_flags_needing_aux_frames_raise_without_them(flag):
    inp = make_golden.case_inputs("flat")
    with pytest.raises(ValueError):
        run_wfo_daily(inp["closes"], inp["pairs"], WfoConfig(**{flag: True}))
