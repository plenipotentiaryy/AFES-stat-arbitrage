"""
risk.ticker_overlap — Section 7.3 Active Ticker Correlation Throttling.

When AFES scales beyond a few dozen pairs the universe inevitably contains
shared tickers (e.g. JPM-BAC and JPM-WFC).  Treating those pairs as
independent under-counts the true factor exposure of the book.  This module
enforces a configurable per-pair check:

    For a candidate trade on pair A at time t:
        if any concurrently-open trade on pair B
           shares a ticker with A
           and the 90-day spread correlation rho(A, B) >= threshold:
        then APPLY the configured rule (HARD block / SCALE size / DISABLED).

The current pair-by-pair backtest pipeline processes pairs sequentially, so a
true per-bar enforcement would require an inter-pair event loop.  Because the
existing AFES sizing model is *linear in size* (final_pnl = (gross - tx -
borrow) * size_mult), applying the rule as a deterministic post-hoc filter on
the already-generated trade list is mathematically identical to applying it
at entry, modulo the per-pair feedback loops that depend on R_i values --
which are unaffected because filtered trades are still recorded (with scaled
size or removed) on the same event stream.

Public surface
--------------
TickerOverlapBlocker
    Configurable evaluator.  Holds a spread correlation matrix (DataFrame
    or dict-of-dicts) and the rule parameters.

filter_trade_list
    Convenience helper: walk a chronologically-sorted trade list, apply the
    blocker at each entry, return the filtered list and per-pair stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


VALID_MODES = ("DISABLED", "SCALE", "HARD")


def _normalise_mode(mode: str) -> str:
    """Coerce mode to upper-case canonical name; default to DISABLED."""
    if not isinstance(mode, str):
        return "DISABLED"
    m = mode.strip().upper()
    return m if m in VALID_MODES else "DISABLED"


def _tickers_of(pair: str) -> tuple[str, str] | None:
    """Split ``"AAA-BBB"`` style pair names into a ticker tuple."""
    if not isinstance(pair, str):
        return None
    parts = pair.split("-")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def share_ticker(pair_a: str, pair_b: str) -> bool:
    """Return True iff the two pair labels share at least one ticker."""
    ta = _tickers_of(pair_a)
    tb = _tickers_of(pair_b)
    if ta is None or tb is None:
        return False
    return bool(set(ta) & set(tb))


@dataclass
class OverlapDecision:
    """Outcome of evaluating a candidate trade against open positions."""
    blocked: bool = False
    scale: float = 1.0
    blocking_pair: str | None = None
    rho: float | None = None
    mode: str = "DISABLED"


class TickerOverlapBlocker:
    """
    Deterministic, configurable correlation-overlap evaluator.

    Construction
    ------------
    corr_matrix
        Either a square ``pandas.DataFrame`` whose index and columns are pair
        labels and whose values are the rolling 90-day spread correlations,
        or a dict-of-dicts with the same structure.  Missing entries are
        treated as zero (no blocking).

    threshold
        Correlation level above which a shared-ticker overlap counts as
        hidden concentration risk.  Spec default 0.65.

    mode
        ``"DISABLED"`` skips all blocking; ``"SCALE"`` multiplies the trade
        size by ``scale``; ``"HARD"`` drops the trade entirely.  The mode is
        validated and any unknown value falls back to ``DISABLED``.

    scale
        Multiplier applied when ``mode == "SCALE"``.  Spec default 0.25.
    """

    def __init__(self,
                 corr_matrix: pd.DataFrame | Mapping[str, Mapping[str, float]] | None,
                 threshold: float = 0.65,
                 mode: str = "SCALE",
                 scale: float = 0.25):
        self.threshold = float(threshold)
        self.mode = _normalise_mode(mode)
        self.scale = float(scale)
        self._corr_lookup = self._build_lookup(corr_matrix)
        self.blocked_count = 0
        self.scaled_count = 0

    @staticmethod
    def _build_lookup(corr: pd.DataFrame | Mapping | None) -> dict[tuple[str, str], float]:
        """Flatten the correlation source into a symmetric pair-tuple dict."""
        out: dict[tuple[str, str], float] = {}
        if corr is None:
            return out
        if isinstance(corr, pd.DataFrame):
            cols = list(corr.columns)
            for i, ci in enumerate(cols):
                for j, cj in enumerate(cols):
                    if i == j:
                        continue
                    val = corr.iat[i, j]
                    if pd.isna(val) or not np.isfinite(val):
                        continue
                    out[(str(ci), str(cj))] = float(val)
        elif isinstance(corr, Mapping):
            for a, sub in corr.items():
                if not isinstance(sub, Mapping):
                    continue
                for b, val in sub.items():
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(fv):
                        continue
                    out[(str(a), str(b))] = fv
        return out

    def _rho(self, pair_a: str, pair_b: str) -> float:
        """Return rho(A,B) treating missing entries as zero (no block)."""
        if pair_a == pair_b:
            return 1.0
        key1 = (str(pair_a), str(pair_b))
        if key1 in self._corr_lookup:
            return self._corr_lookup[key1]
        key2 = (str(pair_b), str(pair_a))
        if key2 in self._corr_lookup:
            return self._corr_lookup[key2]
        return 0.0

    def evaluate(self, candidate_pair: str,
                 open_pairs: Iterable[str]) -> OverlapDecision:
        """
        Decide whether a new trade on ``candidate_pair`` should be blocked or
        scaled given the currently-open ``open_pairs`` (any iterable of pair
        labels).

        Returns a single OverlapDecision describing the most-blocking open
        pair, if any.  The evaluator picks the pair with the highest spread
        correlation among ticker-sharing overlaps so that the worst observed
        overlap drives the decision.
        """
        decision = OverlapDecision(mode=self.mode)
        if self.mode == "DISABLED":
            return decision
        worst_rho = -np.inf
        worst_pair: str | None = None
        for other in open_pairs:
            if other == candidate_pair:
                continue
            if not share_ticker(candidate_pair, other):
                continue
            rho = self._rho(candidate_pair, other)
            if rho >= self.threshold and rho > worst_rho:
                worst_rho = rho
                worst_pair = other
        if worst_pair is None:
            return decision
        decision.rho = float(worst_rho)
        decision.blocking_pair = worst_pair
        if self.mode == "HARD":
            decision.blocked = True
            decision.scale = 0.0
            self.blocked_count += 1
        else:
            decision.blocked = False
            decision.scale = float(self.scale)
            self.scaled_count += 1
        return decision


@dataclass
class FilterStats:
    """Counters returned from ``filter_trade_list`` for telemetry."""
    n_input: int = 0
    n_kept: int = 0
    n_dropped: int = 0
    n_scaled: int = 0
    blocking_pairs: list[tuple[str, str, float]] = field(default_factory=list)


def filter_trade_list(trades: pd.DataFrame,
                      blocker: TickerOverlapBlocker,
                      entry_col: str = "entry_time",
                      exit_col: str = "exit_time",
                      pair_col: str = "pair",
                      pnl_col: str = "net_pnl",
                      size_col: str | None = "size_mult") -> tuple[pd.DataFrame, FilterStats]:
    """
    Apply a TickerOverlapBlocker chronologically to a trade list.

    Algorithm
    ---------
    Sort by entry time.  Maintain a list of "open" trades (entry_time <= t,
    exit_time > t).  For each new entry at time t:
        1. Drop expired trades from the open list (exit_time <= t).
        2. Evaluate the blocker against the open pair labels.
        3. If HARD-blocked, the trade is removed from the output.
        4. If SCALE-blocked, the trade is kept but ``pnl_col`` and the
           optional ``size_col`` are multiplied by the scale factor.
        5. Append the (possibly-modified) trade to the open list.

    Returns the filtered DataFrame plus a FilterStats record.  When the
    blocker mode is DISABLED the input is returned unchanged.

    The function does not mutate the input DataFrame.
    """
    stats = FilterStats(n_input=len(trades))
    if trades is None or trades.empty:
        return trades.copy() if trades is not None else trades, stats
    if blocker.mode == "DISABLED":
        stats.n_kept = len(trades)
        return trades.copy(), stats

    required = {entry_col, exit_col, pair_col, pnl_col}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"filter_trade_list missing columns: {sorted(missing)}")

    work = trades.copy()
    work[entry_col] = pd.to_datetime(work[entry_col])
    work[exit_col] = pd.to_datetime(work[exit_col])
    work = work.sort_values(entry_col, kind="mergesort").reset_index(drop=True)

    kept_rows: list[pd.Series] = []
    open_trades: list[tuple[pd.Timestamp, str]] = []  # (exit_time, pair)

    for _, row in work.iterrows():
        t_entry = row[entry_col]
        # Expire trades whose exit_time <= t_entry.
        open_trades = [(et, p) for (et, p) in open_trades if et > t_entry]
        decision = blocker.evaluate(str(row[pair_col]),
                                    (p for (_, p) in open_trades))
        if decision.blocked:
            stats.n_dropped += 1
            if decision.blocking_pair is not None:
                stats.blocking_pairs.append(
                    (str(row[pair_col]), decision.blocking_pair,
                     float(decision.rho if decision.rho is not None else np.nan))
                )
            continue
        if decision.scale != 1.0:
            stats.n_scaled += 1
            if decision.blocking_pair is not None:
                stats.blocking_pairs.append(
                    (str(row[pair_col]), decision.blocking_pair,
                     float(decision.rho if decision.rho is not None else np.nan))
                )
            row = row.copy()
            row[pnl_col] = float(row[pnl_col]) * decision.scale
            if size_col is not None and size_col in row.index:
                try:
                    row[size_col] = float(row[size_col]) * decision.scale
                except (TypeError, ValueError):
                    pass
        kept_rows.append(row)
        open_trades.append((row[exit_col], str(row[pair_col])))

    stats.n_kept = len(kept_rows)
    if kept_rows:
        out = pd.DataFrame(kept_rows).reset_index(drop=True)
    else:
        out = work.iloc[0:0].copy()
    return out, stats


def compute_spread_correlations(spreads_by_pair: Mapping[str, pd.Series],
                                window_days: int = 90) -> pd.DataFrame:
    """
    Compute the pairwise correlation matrix of the most-recent ``window_days``
    of each pair's daily spread *changes* (diff).

    Differences are used instead of raw spread levels because spread levels
    are typically non-stationary; correlating levels yields spuriously high
    coefficients dominated by common trends.  Spread changes are first-order
    stationary by construction for cointegrated pairs.

    Pairs with fewer than ``window_days`` observations after diffing are
    dropped from the matrix.  The output is symmetric with unit diagonal.
    """
    if not spreads_by_pair:
        return pd.DataFrame()
    cols: dict[str, pd.Series] = {}
    for pair, series in spreads_by_pair.items():
        if series is None:
            continue
        diff = series.diff().dropna()
        if len(diff) < window_days:
            continue
        cols[str(pair)] = diff.iloc[-window_days:]
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).dropna(how="all")
    if df.shape[1] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns,
                            data=np.eye(df.shape[1]))
    return df.corr().fillna(0.0)
