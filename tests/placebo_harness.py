"""Placebo harness: run the daily WFO core on synthetic datasets over many seeds.

Used by tests/test_placebo.py and tests/placebo_report.py (see tests/README.md).
Seeds are fanned out over a process pool (worker stdout suppressed).  Results
are cached per process and keyed by (cfg, panel builder, dataset), so the
Sharpe and t-stat tests share runs and variants with identical settings are
computed once.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic  # noqa: E402
from wfo_daily_core import WfoConfig, run_wfo_daily  # noqa: E402

# Gross = before costs: flat model with zero cost, slippage and borrow (Ruling R3).
GROSS = WfoConfig(execution_model="flat", cost=0.0, slip=0.0, borrow=0.0)

# Variant table — the single place later tasks edit: name → (cfg, builder of
# the async panel from AsyncRaw).  "legacy" = GROSS with every legacy_* flag
# on and the legacy panel builder (the pre-audit behaviour); "fixed" = GROSS
# with the corrected defaults and the corrected builder (since T2).  Each
# task adds its legacy_* cfg field to "legacy".
VARIANTS: dict[str, tuple[WfoConfig, Callable]] = {
    "legacy": (replace(GROSS, legacy_same_close=True, legacy_spread_pnl=True,
                       legacy_stale_prints=True, legacy_window_reset=True,
                       legacy_exit_date_pnl=True),
               synthetic.build_async_panel),
    "fixed":  (GROSS, synthetic.build_async_panel_fixed),
}

# name → (generator, kwargs).  AsyncRaw outputs go through the variant's builder.
DATASETS: dict[str, tuple[Callable, dict]] = {
    "independent_rw":      (synthetic.independent_rw, {}),
    "async_lag4.5_hol0":   (synthetic.async_same_asset, {"close_lag_hours": 4.5, "holidays_per_year": 0}),
    "async_lag4.5_hol5":   (synthetic.async_same_asset, {"close_lag_hours": 4.5, "holidays_per_year": 5}),
    "async_lag15_hol0":    (synthetic.async_same_asset, {"close_lag_hours": 15, "holidays_per_year": 0}),
    "async_lag15_hol5":    (synthetic.async_same_asset, {"close_lag_hours": 15, "holidays_per_year": 5}),
    "ou_positive_control": (synthetic.ou_positive_control, {}),
}
POSITIVE_CONTROL = "ou_positive_control"
NULL_DATASETS = [d for d in DATASETS if d != POSITIVE_CONTROL]
NULL_SEEDS = range(20)
CONTROL_SEEDS = range(10)           # Ruling R9
MIN_POOLED_TRADES = 1000            # a null test never passes on (almost) no trades
MAX_WORKERS = 8


def make_panel(dataset: str, seed: int, builder: Callable | None) -> tuple[pd.DataFrame, list[str]]:
    """(closes, pairs) of one dataset and seed (builder: async datasets only)."""
    gen, kwargs = DATASETS[dataset]
    out = gen(seed=seed, **kwargs)
    return builder(out) if isinstance(out, synthetic.AsyncRaw) else out


def run_quiet(closes: pd.DataFrame, pairs: list[str], cfg: WfoConfig):
    """run_wfo_daily with its progress prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, pairs, cfg)


def daily_sharpe(daily_pnl: pd.Series) -> float:
    """Annualised Sharpe of a daily PnL series (0.0 if its std is 0 or NaN)."""
    sd = daily_pnl.std()
    if not sd > 0:
        return 0.0
    return float(daily_pnl.mean() / sd * math.sqrt(252))


def t_stat(x: np.ndarray) -> float:
    """mean / (std / √n) with the sample std (NaN for fewer than 2 values)."""
    if len(x) < 2:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


@dataclass
class SeedRun:
    sharpe: float           # daily_sharpe(WfoResult.daily_pnl)
    trade_pnl: np.ndarray   # WfoResult.trades["pnl"]


def run_seed(cfg: WfoConfig, builder: Callable | None, dataset: str, seed: int) -> SeedRun:
    """Worker: one dataset seed through run_wfo_daily."""
    res = run_quiet(*make_panel(dataset, seed, builder), cfg)
    pnl = res.trades["pnl"].to_numpy(dtype=float) if not res.trades.empty else np.empty(0)
    return SeedRun(sharpe=daily_sharpe(res.daily_pnl), trade_pnl=pnl)


@dataclass
class Summary:
    """One (variant, dataset) over its seeds."""
    sharpes: np.ndarray     # per seed
    trade_pnl: np.ndarray   # all trades of all seeds

    @property
    def mean_sharpe(self) -> float:
        return float(self.sharpes.mean())

    @property
    def se_sharpe(self) -> float:
        return float(self.sharpes.std(ddof=1) / math.sqrt(len(self.sharpes)))

    @property
    def tstat(self) -> float:
        return t_stat(self.trade_pnl)

    @property
    def n_trades(self) -> int:
        return len(self.trade_pnl)

    @property
    def win_rate(self) -> float:
        return float((self.trade_pnl > 0).mean()) if self.n_trades else float("nan")


_CACHE: dict[tuple, Summary] = {}


def run_key(variant: str, dataset: str) -> tuple:
    """(cfg, builder or None, dataset): what determines a run's output."""
    cfg, builder = VARIANTS[variant]
    uses_builder = DATASETS[dataset][0] is synthetic.async_same_asset
    return cfg, (builder if uses_builder else None), dataset


def results(jobs: list[tuple[str, str]]) -> dict[tuple[str, str], Summary]:
    """Summaries of (variant, dataset) jobs; each distinct run_key is computed
    once per process, its seeds fanned out over a process pool."""
    todo = {run_key(v, d) for v, d in jobs} - _CACHE.keys()
    if todo:
        tasks = [(key, seed) for key in todo
                 for seed in (CONTROL_SEEDS if key[2] == POSITIVE_CONTROL else NULL_SEEDS)]
        workers = min(MAX_WORKERS, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_seed, *key, seed) for key, seed in tasks]
            runs = [f.result() for f in futures]
        for key in todo:
            mine = [r for (k, _), r in zip(tasks, runs) if k == key]
            _CACHE[key] = Summary(sharpes=np.array([r.sharpe for r in mine]),
                                  trade_pnl=np.concatenate([r.trade_pnl for r in mine]))
    return {(v, d): _CACHE[run_key(v, d)] for v, d in jobs}


def all_jobs() -> list[tuple[str, str]]:
    """Every null dataset under every variant, plus the positive control ("fixed")."""
    return [(v, d) for v in VARIANTS for d in NULL_DATASETS] + [("fixed", POSITIVE_CONTROL)]
