"""wfo_daily_core.py — engine of the daily walk-forward backtest.

Extracted from `step3j_wfo_daily.py` (audit task T0) so the backtest can run
on in-memory data: `run_wfo_daily(closes, pairs, cfg)` takes a close panel,
a pair list and a `WfoConfig`, and returns a `WfoResult`.  The CLI in
`step3j_wfo_daily.py` keeps all file IO, builds the `WfoConfig` from its
arguments and calls `run_wfo_daily`.  The defaults carry the audit fixes
(AUDIT_FIX_LOG.md); with every `legacy_*` flag on, behaviour is identical
to the pre-refactor CLI (commit 46ffdfb).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import BAR_TIMEFRAME
from cross_listing import mask_stale_prints, repeats_previous_print
from kalman import kalman_hedge
from metagate_daily import (
    MetaGateDaily, FEATURE_NAMES as MG_FEATURES,
    entry_features as mg_entry_features,
    rolling_hurst as mg_rolling_hurst,
    harvest_train_trades as mg_harvest,
)
from hrp_weights import hrp_weights
from almgren_chriss_repricing import (
    classify_ticker as ac_classify,
    leg_cost_log as ac_leg_cost,
    borrow_log_per_day as ac_borrow_per_day,
    TIER_TABLE as AC_TIER_TABLE,
)

# Local import (HMM is optional dep but already used elsewhere in the repo).
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None

# ── Default WFO geometry (mirrors step3j_wfo.py for comparability) ──────────
TRAIN_INIT_MO = 6     # initial expanding-window train length (months)
OOS_MO        = 3     # out-of-sample window length per step
STEP_MO       = 3     # walk-forward step

# ── Default strategy params ────────────────────────────────────────────────
Z_WIN     = 60        # rolling z-score window (days)
ENTRY_Z   = 2.0
EXIT_Z    = 0.0
STOP_Z    = 3.5
MAX_HOLD  = 30        # days (cap on holding period)


# ── Engine ──────────────────────────────────────────────────────────────────

_AUSSIE_JP_SUFFIXES = (".AX_USD", "_AX_USD", ".T_USD", "_T_USD")


def is_aussie_jp_pair(pair: str) -> bool:
    return any(s in pair for s in _AUSSIE_JP_SUFFIXES)


def aussie_blackout_series(spread: pd.Series) -> pd.Series:
    """Return boolean Series aligned to `spread`: True = skip entry.

    Gates (cumulative — any True triggers blackout):
      1. Vol-spike: 5d rolling std of spread > 95th-pct of trailing 252d.
      2. Half-life: rolling 60d AR(1) half-life > 5 trading days, NaN, or non-mean-reverting.
    """
    s = spread.astype(float)
    diffs = s.diff()
    vol5  = diffs.rolling(5).std()
    vol_p95 = vol5.rolling(252, min_periods=63).quantile(0.95)
    spike = (vol5 > vol_p95)

    # Rolling AR(1) on first-differences vs lagged level (Engle-Granger style).
    def _hl(window):
        if len(window) < 30:
            return np.nan
        d = np.diff(window)
        lag = window[:-1]
        # OLS: d = a + phi * lag
        x = lag - lag.mean()
        y = d  - d.mean()
        denom = (x * x).sum()
        if denom <= 0:
            return np.nan
        phi = (x * y).sum() / denom
        if phi >= 0:
            return np.nan  # non-reverting
        return -np.log(2) / np.log1p(phi)

    hl = s.rolling(60, min_periods=30).apply(_hl, raw=True)
    hl_bad = (hl.isna()) | (hl > 5.0)

    blackout = spike.fillna(True) | hl_bad.fillna(True)
    return blackout


def estimate_beta(p1: pd.Series, p2: pd.Series) -> float:
    x, y = np.log(p2.values), np.log(p1.values)
    return float(np.cov(y, x, ddof=0)[0, 1] / np.var(x))


def spread_series(df: pd.DataFrame, t1: str, t2: str, beta: float) -> pd.Series:
    return (np.log(df[t1]) - beta * np.log(df[t2])).dropna()


def backtest_pair(spread_full: pd.Series, oos_start, oos_end,
                  cost_per_trade: float,
                  entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                  z_win=Z_WIN, max_hold=MAX_HOLD,
                  cusum_h: float | None = None,
                  z_series: pd.Series | None = None,
                  vr_series: pd.Series | None = None,
                  vr_min: float = 0.5,
                  panic_series: pd.Series | None = None,
                  panic_size_mult: float = 0.0,
                  borrow_per_day: float = 0.0,
                  metagate: MetaGateDaily | None = None,
                  hurst_series: pd.Series | None = None,
                  vol_tight_stop: bool = False,
                  vol_tight_threshold: float = 1.5,
                  vol_tight_stop_z: float = 2.5,
                  exec_spread: pd.Series | None = None,
                  entry_blackout: pd.Series | None = None,
                  legacy_same_close: bool = False,
                  decision_end=None,
                  leg_prices: pd.DataFrame | None = None,
                  exec_leg_prices: pd.DataFrame | None = None,
                  beta: float | pd.Series | None = None,
                  ) -> pd.DataFrame:
    """Generate mean-reversion trades on a daily spread.

    Entries open only inside the OOS window; z is computed on the warmup-
    extended series so rolling statistics are stable from bar 0 of OOS.

    cusum_h: if set, run two-sided CUSUM on the standardized spread and
             abort all further entries for this pair-window once
             max(S+, S-) > cusum_h (k=0.5 slack, classic defaults).
    z_series: pre-computed z (overrides rolling); used by Kalman path
              which produces innovations / sqrt(innov_var).

    Fills (audit T1): an entry or exit decided on bar i (a row of
    spread_full; the decision uses data up to bar i) fills on bar i+1, at
    its close or, with exec_spread, at exec_spread[i] (the VWAP of bar i+1;
    an exit whose VWAP is NaN fills at the close of bar i+1).  The holding
    period and borrow start at the entry fill; entry/exit are the fill
    dates, entry_signal/exit_signal the decision dates.  No bar i+1: an
    entry is not filled; an exit fills at the close of bar i with reason
    "end_of_data".
    decision_end: no entry, exit or CUSUM decision on bars after this date
                  (fill-only bars).
    legacy_same_close: pre-audit fills on the deciding bar itself.

    PnL (audit T2).  leg_prices: the pair's two USD closes [P1, P2] on the
    rows of spread_full; beta: the β the spread was built with (a scalar, or
    one per bar on the Kalman path — the trade uses β of its entry decision
    bar).  With leg_prices the PnL comes from the leg fills, the spread drives
    the signals only: at the entry fill q1 = 1/P1 and q2 = β/P2 (per $1 of
    leg-1 notional) are fixed, and gross = side·(q1·(P1_exit − P1_entry) −
    q2·(P2_exit − P2_entry)) at the exit-fill prices.  exec_leg_prices (with
    exec_spread): each leg's VWAP of bar i+1 on row i; a leg fills at its
    VWAP, an exit leg without one at its close (an entry without a VWAP
    spread is not filled, as above).  Without leg_prices (legacy): gross =
    side·(exit_spread − entry_spread).  Either way pnl = size·(gross −
    cost_per_trade − held·borrow_per_day).
    """
    if z_series is None:
        mu  = spread_full.rolling(z_win).mean()
        sig = spread_full.rolling(z_win).std()
        z   = (spread_full - mu) / sig
    else:
        z = z_series.reindex(spread_full.index)

    # CUSUM state (only used if cusum_h is set)
    Sp = Sn = 0.0
    cusum_k = 0.5
    broken = False

    trades = []
    in_pos, side = False, 0
    entry_idx, entry_spread = None, 0.0
    entry_sig = None  # bar on which the open position's entry was decided
    entry_size = 1.0  # applied to pnl/cost at exit (1.0 normal, 0.5 panic-sized)
    cur_stop_z = stop_z  # per-trade stop (mutated at entry if vol_tight_stop on)

    idx = spread_full.index
    n = len(idx)
    oos_mask = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    # Decisions on bars up to decision_end only; later bars are fill-only.
    n_decide = (n if decision_end is None
                else int(idx.searchsorted(pd.Timestamp(decision_end), side="right")))

    # Per-bar inputs as numpy arrays, taken once before the loop.  Reading
    # arr[i] gives the same values as the former `.iloc[i]` (speed only).
    z_vals  = z.to_numpy()
    z_isna  = pd.isna(z_vals)
    sp_vals = spread_full.to_numpy()
    vr_vals = vr_series.to_numpy() if vr_series is not None else None
    panic_vals = panic_series.to_numpy() if panic_series is not None else None
    blackout_vals = entry_blackout.to_numpy() if entry_blackout is not None else None
    exec_vals = exec_spread.to_numpy() if exec_spread is not None else None
    # Audit T2: leg prices and β; the open position's entry legs and quantities.
    assert leg_prices is None or (beta is not None
                                  and (exec_spread is None) == (exec_leg_prices is None))
    leg_vals = leg_prices.reindex(idx).to_numpy(dtype=float) if leg_prices is not None else None
    xleg_vals = (exec_leg_prices.reindex(idx).to_numpy(dtype=float)
                 if exec_leg_prices is not None else None)
    beta_vals = beta.reindex(idx).to_numpy(dtype=float) if isinstance(beta, pd.Series) else None
    nan = float("nan")
    beta_e = p1e = p2e = q1 = q2 = nan

    def legs_at(bar: int, vwap_row: int | None = None) -> tuple[float, float]:
        """(P1, P2) of a fill on `bar`: with vwap_row, each leg's VWAP from that
        row of exec_leg_prices (the VWAP of `bar`), its close where it has none."""
        if leg_vals is None:
            return nan, nan
        p1, p2 = leg_vals[bar]
        if vwap_row is not None and xleg_vals is not None:
            v1, v2 = xleg_vals[vwap_row]
            p1, p2 = (v1 if pd.notna(v1) else p1), (v2 if pd.notna(v2) else p2)
        return p1, p2

    def gross_of(exit_spread: float, p1x: float, p2x: float) -> float:
        """Gross PnL per unit size of the open position closed at these prices."""
        if leg_vals is None:                       # legacy: the spread change
            return side * (exit_spread - entry_spread)
        return side * (q1 * (p1x - p1e) - q2 * (p2x - p2e))

    def leg_cols(p1x: float, p2x: float, gross: float) -> dict:
        return {"beta": beta_e, "p1_entry": p1e, "p2_entry": p2e, "p1_exit": p1x,
                "p2_exit": p2x, "q1": q1, "q2": q2, "gross": gross}

    def exit_fill(i: int, reason: str) -> tuple[int, float, float, float, str]:
        """(fill bar, fill spread, P1, P2, reason) of an exit decided on bar i."""
        if i + 1 >= n:                     # no bar i+1: the data ends here
            return (i, sp_vals[i], *legs_at(i), "end_of_data")
        vwap_row = i if exec_vals is not None else None
        if exec_vals is not None and pd.notna(exec_vals[i]):
            return (i + 1, exec_vals[i], *legs_at(i + 1, vwap_row), reason)   # VWAP of bar i+1
        return (i + 1, sp_vals[i + 1], *legs_at(i + 1, vwap_row), reason)

    def causal(decision: int, fill: int, reason: str) -> bool:
        """Hard rule 2: a fill comes after its decision; only an exit at the
        end of the data fills on its decision bar, the last bar."""
        return fill > decision or (reason == "end_of_data" and fill == decision == n - 1)

    for i in range(n_decide):
        if z_isna[i]:
            continue
        zi = z_vals[i]
        # Update CUSUM on every bar (train+oos) so the test is informed.
        if cusum_h is not None and not broken:
            Sp = max(0.0, Sp + zi - cusum_k)
            Sn = max(0.0, Sn - zi - cusum_k)
            if max(Sp, Sn) > cusum_h:
                broken = True
                # Force-close any open position on break.
                if in_pos:
                    if legacy_same_close:
                        exit_idx, exit_spread, reason = i, sp_vals[i], "cusum_break"
                        p1x, p2x = legs_at(i)
                    else:
                        exit_idx, exit_spread, p1x, p2x, reason = exit_fill(i, "cusum_break")
                    held_bars  = exit_idx - entry_idx
                    gross = gross_of(exit_spread, p1x, p2x)
                    pnl = entry_size * (gross - cost_per_trade - held_bars * borrow_per_day)
                    assert legacy_same_close or (causal(entry_sig, entry_idx, "entry")
                                                 and causal(i, exit_idx, reason))
                    trades.append({
                        "entry":  idx[entry_idx], "exit": idx[exit_idx],
                        "held":   held_bars, "side": side, "pnl": pnl,
                        "size":   entry_size,
                        "reason": reason,
                        "entry_signal": idx[entry_sig], "exit_signal": idx[i],
                        **leg_cols(p1x, p2x, gross),
                    })
                    in_pos = False

        if broken:
            continue

        if not in_pos:
            if not oos_mask[i]:
                continue
            # Liquidity gate: VR < 0.5 → skip entry (legacy convention).
            if vr_series is not None:
                vri = vr_vals[i]
                if pd.notna(vri) and vri < vr_min:
                    continue
            # Macro-HMM panic gate. When panic active:
            #   panic_size_mult == 0  → skip entry entirely (block-mode)
            #   panic_size_mult > 0   → enter with reduced size (sizing-mode)
            this_size = 1.0
            if panic_series is not None:
                pi = panic_vals[i]
                if pd.notna(pi) and bool(pi):
                    if panic_size_mult <= 0.0:
                        continue
                    this_size = panic_size_mult
            # External blackout (e.g., Aussie filter: vol-spike / HL gate).
            if entry_blackout is not None:
                bo = blackout_vals[i]
                if pd.notna(bo) and bool(bo):
                    continue
            # Pre-compute provisional side for MetaGate scoring; we still
            # need a hard z-threshold cross to consider entering.
            prov_side = -1 if zi > entry_z else (+1 if zi < -entry_z else 0)
            if prov_side == 0:
                continue
            mg_size = 1.0
            if metagate is not None and metagate.usable and hurst_series is not None:
                feat = mg_entry_features(spread_full, z, i, prov_side, hurst_series)
                p_win = metagate.predict_proba(feat)
                mg_size = metagate.size_multiplier(p_win)
                if mg_size <= 0.0:
                    continue
            # If exec_spread given, fill uses next-day VWAP price.
            # exec_spread is assumed pre-shifted: exec_spread.iloc[i] holds
            # the spread that would be realised by an order entered at the
            # end of bar i (i.e. day i+1 VWAP). NaN → skip (no fill yet).
            # Otherwise the fill is the close of bar i+1 (audit T1); no bar
            # i+1 → not filled.  legacy_same_close: the close of bar i.
            if legacy_same_close:
                fill_idx = i
                fill_spread_at_entry = (
                    exec_vals[i] if exec_spread is not None
                    else sp_vals[i]
                )
            elif i + 1 < n:
                fill_idx = i + 1
                fill_spread_at_entry = (
                    exec_vals[i] if exec_spread is not None
                    else sp_vals[i + 1]
                )
            else:
                continue
            if exec_spread is not None and pd.isna(fill_spread_at_entry):
                continue
            side = prov_side
            in_pos = True
            entry_idx, entry_spread = fill_idx, fill_spread_at_entry
            entry_sig = i
            # Audit T2: quantities fixed at the entry fill, with β of this decision bar.
            p1e, p2e = legs_at(fill_idx, i if exec_spread is not None else None)
            beta_e = (beta_vals[i] if beta_vals is not None
                      else (nan if beta is None else float(beta)))
            q1, q2 = 1.0 / p1e, beta_e / p2e
            entry_size = this_size * mg_size
            cur_stop_z = stop_z
            if vol_tight_stop and i >= 60:
                v20 = spread_full.iloc[i - 19: i + 1].std()
                v60 = spread_full.iloc[i - 59: i + 1].std()
                if v60 > 0 and (v20 / v60) > vol_tight_threshold:
                    cur_stop_z = vol_tight_stop_z
        else:
            held = i - entry_idx
            hit_target = (side == -1 and zi <= exit_z) or (side == +1 and zi >= exit_z)
            hit_stop   = abs(zi) >= cur_stop_z
            timeout    = held >= max_hold
            if hit_target or hit_stop or timeout:
                reason = "target" if hit_target else ("stop" if hit_stop else "time")
                if legacy_same_close:
                    exit_idx = i
                    fill_spread_at_exit = (
                        exec_vals[i] if exec_spread is not None
                        else sp_vals[i]
                    )
                    p1x, p2x = legs_at(i, i if exec_spread is not None else None)
                    if exec_spread is not None and pd.isna(fill_spread_at_exit):
                        # Roll forward one bar — if last bar of OOS, force-close
                        # at last available exec price (no future data leakage
                        # because exec_spread is pre-shifted).
                        last_known = exec_spread.iloc[:i + 1].dropna()
                        if last_known.empty:
                            continue
                        fill_spread_at_exit = last_known.iloc[-1]
                        p1x, p2x = legs_at(i, int(np.flatnonzero(pd.notna(exec_vals[:i + 1]))[-1]))
                else:
                    exit_idx, fill_spread_at_exit, p1x, p2x, reason = exit_fill(i, reason)
                exit_spread = fill_spread_at_exit
                held_bars  = exit_idx - entry_idx
                borrow_cost = held_bars * borrow_per_day
                gross = gross_of(exit_spread, p1x, p2x)
                pnl = entry_size * gross - entry_size * cost_per_trade - entry_size * borrow_cost
                assert legacy_same_close or (causal(entry_sig, entry_idx, "entry")
                                             and causal(i, exit_idx, reason))
                trades.append({
                    "entry":  idx[entry_idx], "exit": idx[exit_idx],
                    "held":   held_bars, "side": side, "pnl": pnl,
                    "size":   entry_size,
                    "reason": reason,
                    "entry_signal": idx[entry_sig], "exit_signal": idx[i],
                    **leg_cols(p1x, p2x, gross),
                })
                in_pos = False
    return pd.DataFrame(trades)


def fit_macro_hmm(spy_returns_train: np.ndarray,
                  n_components: int = 2,
                  random_state: int = 0):
    """Fit Gaussian HMM (2 or 3 states) on SPY/macro daily returns.

    Returns (model, panic_state_idx).  Panic state is the one with the
    largest emission variance.  For 3-state mode this is the 'extreme'
    state (highest vol).  Returns (None, None) on fit failure.
    """
    if GaussianHMM is None or len(spy_returns_train) < 250:
        return None, None
    try:
        x = spy_returns_train.reshape(-1, 1)
        m = GaussianHMM(n_components=n_components, covariance_type="full",
                        n_iter=200, random_state=random_state)
        m.fit(x)
        panic = int(np.argmax(m.covars_.flatten()))
        return m, panic
    except Exception:
        return None, None


def vwz_spread_z(df: pd.DataFrame, vol: pd.DataFrame, t1: str, t2: str,
                 beta: float, z_win: int = Z_WIN
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Volume-weighted z-score on log-spread (mirrors legacy build_signals).

    v_spread  = min(P1·V1, P2·V2)  — dollar-volume of weaker leg
    vw_mean   = Σ(spread·v_spread) / Σ(v_spread)        rolling z_win
    vw_var    = Σ(v_spread·(spread-vw_mean)²) / Σ(v_spread)
    vr        = v_spread / SMA(v_spread)                liquidity ratio

    Falls back to plain rolling z if volume is missing.
    """
    spread = np.log(df[t1]) - beta * np.log(df[t2])
    if vol is None or t1 not in vol.columns or t2 not in vol.columns:
        mu  = spread.rolling(z_win).mean()
        sig = spread.rolling(z_win).std()
        vr  = pd.Series(1.0, index=spread.index)
        return spread, (spread - mu) / sig, vr

    v1 = vol[t1].reindex(spread.index).fillna(0.0)
    v2 = vol[t2].reindex(spread.index).fillna(0.0)
    v_sp = np.minimum(v1 * df[t1], v2 * df[t2]).reindex(spread.index)

    roll_v = v_sp.rolling(z_win).sum()
    vw_mean = (spread * v_sp).rolling(z_win).sum() / roll_v
    vw_var  = (v_sp * (spread - vw_mean) ** 2).rolling(z_win).sum() / roll_v
    spread_std = np.sqrt(vw_var.clip(lower=1e-12))
    z   = (spread - vw_mean) / spread_std
    v_sma = v_sp.rolling(z_win).mean()
    vr   = v_sp / v_sma.replace(0, np.nan)
    return spread, z, vr


def kalman_spread_z(df: pd.DataFrame, t1: str, t2: str,
                    delta: float = 1e-6,
                    z_win: int = Z_WIN) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Dynamic-β spread via Kalman, then rolling z on the spread.

    Kalman provides the time-varying hedge ratio β_t (random-walk prior).
    The tradable spread is spread_t = log P1_t − β_t · log P2_t — a path
    you can hold and exit later. We then standardise it with the same
    rolling-z used by the static-β baseline so PnL semantics match.

    Returns (spread_series, z_series, beta_series) aligned to df.index;
    β_t sizes a trade entered on bar t (audit T2).
    """
    p1 = np.log(df[t1].values)
    p2 = np.log(df[t2].values)
    _, beta, _, _ = kalman_hedge(p1, p2, delta=delta, beta_init=1.0)
    # Use only the dynamic β; ignore the Kalman α (intercept).  α tracks
    # the spread mean in real-time, which collapses the residual to noise
    # and produces a 80%+ win-rate artefact (looks like edge, isn't).
    spread = pd.Series(p1 - beta * p2, index=df.index, name="kalman_spread")
    mu  = spread.rolling(z_win).mean()
    sig = spread.rolling(z_win).std()
    z   = (spread - mu) / sig
    return spread, z, pd.Series(beta, index=df.index, name="kalman_beta")


def make_windows(closes: pd.DataFrame,
                 train_init_mo: int, oos_mo: int, step_mo: int,
                 start=None, end=None) -> list[tuple]:
    """Return list of (train_start, train_end, oos_start, oos_end) tuples."""
    first = pd.Timestamp(start) if start else closes.index.min()
    last  = pd.Timestamp(end)   if end   else closes.index.max()
    windows = []
    tr_s = first
    tr_e = tr_s + relativedelta(months=train_init_mo)
    while tr_e + relativedelta(months=oos_mo) <= last:
        oos_s = tr_e
        oos_e = oos_s + relativedelta(months=oos_mo)
        windows.append((tr_s, tr_e, oos_s, oos_e))
        tr_e = tr_e + relativedelta(months=step_mo)
    return windows


# ── Config / result ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WfoConfig:
    """Settings of one daily WFO run.

    Field names are exactly the argparse `dest` names of the CLI options in
    `step3j_wfo_daily.py` (minus the file-IO options --pairs/--closes/--tag),
    with identical defaults.  Scalars only, so the config stays hashable.
    """
    start: str | None = None
    end: str | None = None
    cost: float = 5.0
    slip: float = 5.0
    borrow: float = 80.0
    train: int = TRAIN_INIT_MO
    oos: int = OOS_MO
    step: int = STEP_MO
    kalman: bool = False
    vwz: bool = False
    metagate: bool = False
    hrp: bool = False
    ivol: bool = False
    volvol: bool = False
    pairmom: bool = False
    maxconc: int = 0
    ticker_overlap: bool = False
    per_pair_quarter_cap: int = 0
    vol_tight_stop: bool = False
    exec_mode: str = "next_close"
    legacy_same_close: bool = False
    legacy_spread_pnl: bool = False
    legacy_stale_prints: bool = False
    rotation: bool = False
    voltarget: float = 0.0
    hmm: str = "off"
    cusum: float | None = None
    aussie_filter: bool = False
    execution_model: str = "almgren"
    notional: float = 100_000


@dataclass
class WfoResult:
    """Output of `run_wfo_daily`.

    trades:    the trade list the CLI writes to wfo_daily_results*.csv.
    daily_pnl: portfolio PnL per day of the OOS span (zeros on empty days).
    stats:     summary metrics, see `trade_stats`.
    """
    trades: pd.DataFrame
    daily_pnl: pd.Series
    stats: dict


# ── Reporting ───────────────────────────────────────────────────────────────

def exit_date_daily_pnl(tr: pd.DataFrame, oos_index) -> pd.Series:
    """Trade PnL booked on exit dates, summed per day, reindexed to
    `oos_index` with zeros on days without an exit."""
    oos = pd.DatetimeIndex(oos_index).normalize()
    if tr.empty:
        return pd.Series(0.0, index=oos, name="pnl")
    daily_exit = tr.groupby(pd.to_datetime(tr["exit"]).dt.normalize())["pnl"].sum()
    return daily_exit.reindex(oos, fill_value=0.0)


def trade_stats(tr: pd.DataFrame, n_windows: int, n_pairs: int,
                oos_index: pd.DatetimeIndex | None = None) -> dict:
    """Portfolio metrics of a trade list (the numbers `summarise` prints).

    sharpe_full uses every OOS trading day (zeros on no-exit days);
    sharpe_exit uses exit days only (legacy, inflated).  No trades →
    n_trades = 0 and NaN metrics.
    """
    nan = float("nan")
    stats = {
        "n_windows": n_windows, "n_pairs": n_pairs, "n_pairs_active": 0,
        "n_trades": 0, "win_rate": nan, "total_pnl": nan,
        "profit_factor": nan, "max_dd": nan, "sharpe_full": nan,
        "sharpe_exit": nan, "exit_reasons": {},
    }
    if tr.empty:
        return stats
    pnl = tr["pnl"]
    eq  = pnl.cumsum()
    dd  = (eq - eq.cummax()).min()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()

    daily_exit = tr.groupby(pd.to_datetime(tr["exit"]).dt.normalize())["pnl"].sum()
    exit_sh = (daily_exit.mean() / daily_exit.std() * np.sqrt(252)
                if daily_exit.std() > 0 else float("nan"))

    if oos_index is not None and len(oos_index) > 0:
        daily_full = exit_date_daily_pnl(tr, oos_index)
        full_sh = (daily_full.mean() / daily_full.std() * np.sqrt(252)
                    if daily_full.std() > 0 else float("nan"))
    else:
        full_sh = float("nan")

    stats.update({
        "n_pairs_active": tr["pair"].nunique(),
        "n_trades":       len(tr),
        "win_rate":       (pnl > 0).mean(),
        "total_pnl":      pnl.sum(),
        "profit_factor":  wins/losses if losses>0 else float('inf'),
        "max_dd":         dd,
        "sharpe_full":    full_sh,
        "sharpe_exit":    exit_sh,
        "exit_reasons":   tr["reason"].value_counts().to_dict(),
    })
    return stats


def summarise(tr: pd.DataFrame, label: str, n_windows: int, n_pairs: int,
              oos_index: pd.DatetimeIndex | None = None):
    """Report portfolio metrics.

    full_sharpe uses every OOS trading day (zeros on no-exit days) — this
    is the correct annualised portfolio Sharpe.  The legacy exit-day-only
    Sharpe is also printed for backwards-compatibility; it inflates by
    a factor of ~√(trading_days / exit_days) for sparse strategies.
    """
    if tr.empty:
        print(f"\n[{label}] NO TRADES across {n_windows} windows × {n_pairs} pairs")
        return
    st = trade_stats(tr, n_windows, n_pairs, oos_index)

    print(f"\n{'='*70}")
    print(f"DAILY WFO SUMMARY — {label}")
    print(f"{'='*70}")
    print(f"Windows:        {st['n_windows']}")
    print(f"Pairs (active): {st['n_pairs_active']} / {st['n_pairs']}")
    print(f"OOS Trades:     {st['n_trades']}")
    print(f"Win rate:       {st['win_rate']*100:.1f}%")
    print(f"Total PnL:      {st['total_pnl']:+.4f}  (log-spread units)")
    print(f"Profit factor:  {st['profit_factor']:.2f}")
    print(f"Max DD:         {st['max_dd']:+.4f}")
    print(f"DD / PnL:       {abs(st['max_dd'])/st['total_pnl']*100:.1f}%")
    if not np.isnan(st["sharpe_full"]):
        print(f"Sharpe (full):  {st['sharpe_full']:+.3f}   ← correct portfolio Sharpe")
        print(f"Sharpe (exit):  {st['sharpe_exit']:+.3f}   (inflated; legacy metric)")
    else:
        print(f"Sharpe (exit):  {st['sharpe_exit']:+.3f}")
    print(f"Exit reasons:   {st['exit_reasons']}")

    by_pair = tr.groupby("pair")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
    print(f"\nPer-pair PnL (top {min(15, len(by_pair))} of {len(by_pair)}):")
    print(by_pair.head(15).round(4).to_string())
    if len(by_pair) > 15:
        print(f"... + {len(by_pair) - 15} more pairs")


def apply_quarterly_rotation(trades: pd.DataFrame,
                              sr_prior: float = 0.5,
                              k_shrink: float = 5.0,
                              sr_target: float = 1.2,
                              sr_floor:  float = 0.0,
                              drop_sr6m_below: float = 0.0) -> pd.DataFrame:
    """Per-pair quarterly rescaling and drop policy (§5 of upgrade spec).

    For each trade, look at the pair's CLOSED trades that ended strictly
    before this trade's entry. Compute realised Sharpe over the last 6
    months of those closed trades, apply Bayesian shrinkage, derive an
    S_perf size multiplier, and drop the trade entirely if 6m-Sharpe of
    the pair stays below `drop_sr6m_below`.
    """
    if trades.empty:
        return trades
    t = trades.sort_values("entry").reset_index(drop=False).copy()
    t["entry"] = pd.to_datetime(t["entry"])
    t["exit"]  = pd.to_datetime(t["exit"])

    drop_idx, mults = [], []
    for pair, grp in t.groupby("pair"):
        grp = grp.sort_values("entry")
        for _, row in grp.iterrows():
            # Same-pair trades with exit STRICTLY before this entry.
            past = grp[(grp["exit"] < row["entry"])]
            mult = 1.0
            keep = True
            if len(past) >= 3:
                # 6-month window for SR, 3-month window for drop check
                cutoff_6m = row["entry"] - pd.DateOffset(months=6)
                cutoff_3m = row["entry"] - pd.DateOffset(months=3)
                past_6m = past[past["exit"] >= cutoff_6m]
                past_3m = past[past["exit"] >= cutoff_3m]
                if len(past_6m) >= 3:
                    pnl_6m = past_6m["pnl"]
                    if pnl_6m.std() > 0:
                        sr_raw = pnl_6m.mean() / pnl_6m.std() * np.sqrt(252)
                        n = len(pnl_6m)
                        sr = (n * sr_raw + k_shrink * sr_prior) / (n + k_shrink)
                        mult = float(np.clip(
                            0.2 + (sr - sr_floor) / (sr_target - sr_floor),
                            0.2, 1.2))
                        # Drop rule: SR_6m below threshold AND SR_3m also weak
                        if sr_raw < drop_sr6m_below and len(past_3m) >= 3:
                            pnl_3m = past_3m["pnl"]
                            if pnl_3m.std() > 0:
                                sr_3m = pnl_3m.mean() / pnl_3m.std() * np.sqrt(252)
                                if sr_3m < drop_sr6m_below:
                                    keep = False
            if not keep:
                drop_idx.append(row["index"])
            mults.append((row["index"], mult))
    mult_map = dict(mults)
    surviving_mask = ~t["index"].isin(drop_idx)
    t = t[surviving_mask]
    multipliers = t["index"].map(mult_map).fillna(1.0).values
    trades_out = trades.loc[t["index"]].copy().reset_index(drop=True)
    trades_out["pnl"] = trades_out["pnl"].values * multipliers
    if "size" in trades_out.columns:
        trades_out["size"] = trades_out["size"].values * multipliers
    return trades_out


def apply_vol_target(trades: pd.DataFrame, target_vol: float,
                      oos_index: pd.DatetimeIndex,
                      lookback: int = 20,
                      max_lev: float = 2.0, min_lev: float = 0.5,
                      smooth_alpha: float = 0.20) -> pd.DataFrame:
    """Causal portfolio-level vol targeting.

    For each day in the OOS window, compute realised vol from the previous
    `lookback` days of portfolio PnL. Leverage_t = target / realised_t,
    clipped and smoothed.  Scale each trade's PnL by the leverage that was
    available at its entry date (causal — no future info).
    """
    if trades.empty or target_vol <= 0:
        return trades
    t = trades.copy()
    t["entry"] = pd.to_datetime(t["entry"])
    t["exit"]  = pd.to_datetime(t["exit"])
    # Daily portfolio PnL, indexed by exit date.
    daily = t.groupby(t["exit"].dt.normalize())["pnl"].sum()
    daily = daily.reindex(pd.DatetimeIndex(oos_index).normalize(), fill_value=0.0)
    rolling_std = daily.shift(1).rolling(lookback, min_periods=lookback).std()
    realised_vol = rolling_std * np.sqrt(252)
    raw_lev = (target_vol / realised_vol).clip(lower=min_lev, upper=max_lev)
    raw_lev = raw_lev.fillna(1.0)
    # EWMA smoothing
    smooth = raw_lev.ewm(alpha=smooth_alpha, adjust=False).mean()
    smooth = smooth.clip(lower=min_lev, upper=max_lev)
    # Look up leverage at each trade's entry date.
    entry_dates = t["entry"].dt.normalize()
    lev = entry_dates.map(smooth).fillna(1.0).values
    t["pnl"] = t["pnl"].values * lev
    if "size" in t.columns:
        t["size"] = t["size"].values * lev
    return t


# ── Main loop ───────────────────────────────────────────────────────────────

def run_wfo_daily(closes: pd.DataFrame, pairs: list[str], cfg: WfoConfig,
                  volumes: pd.DataFrame | None = None,
                  vwap: pd.DataFrame | None = None) -> WfoResult:
    """Run the expanding-window daily WFO over `pairs` on the `closes` panel.

    closes:  daily closes (DatetimeIndex × tickers), as read from data/;
             NaN = no print (no price is forward-filled here).
    pairs:   pair names "T1-T2" (both tickers must be columns of closes).
    volumes: daily share volumes (volumes_daily.csv); required by cfg.vwz,
             used for the stale-print mask and the Almgren-Chriss ADV if present.
    vwap:    daily VWAPs (vwap_daily.csv); required by cfg.exec_mode "vwap".
    """
    if cfg.exec_mode not in ("next_close", "vwap"):
        raise ValueError(f"cfg.exec_mode must be 'next_close' or 'vwap', not {cfg.exec_mode!r}")
    if cfg.vwz and volumes is None:
        raise ValueError("cfg.vwz needs the volumes frame (volumes_daily.csv)")
    if cfg.exec_mode == "vwap" and vwap is None:
        raise ValueError("cfg.exec_mode 'vwap' needs the vwap frame (vwap_daily.csv)")

    if not cfg.legacy_stale_prints:
        # Audit T2: an exactly unchanged close with zero or missing volume is
        # no print (NaN).  Each pair's dropna() below then keeps only the dates
        # on which both legs printed: no signal, no fill and no z-window day on
        # a date when either market is closed.
        closes = mask_stale_prints(closes, volumes)

    # Macro returns for HMM.  Prefer SPY; otherwise use an equal-weighted
    # cross-sectional mean of log-returns as a market-wide panic proxy.
    spy_returns = None
    hmm_active = (cfg.hmm != "off")
    if hmm_active:
        if "SPY" in closes.columns:
            spy_returns = np.log(closes["SPY"]).diff()
            print(f"  hmm mode: {cfg.hmm}   macro proxy: SPY")
        else:
            spy_returns = np.log(closes).diff().mean(axis=1)
            print(f"  hmm mode: {cfg.hmm}   macro proxy: EW mean of {closes.shape[1]} names")

    vwap_daily = None
    if cfg.exec_mode == "vwap":
        vwap_daily = vwap
        print(f"  vwap-exec: ON ({vwap_daily.shape[0]} days × {vwap_daily.shape[1]} tickers)")

    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step,
                            start=cfg.start, end=cfg.end)
    # Cost breakdown:
    #   flat:    commission+slippage round-trip = 2 × (cost + slip) bps
    #   almgren: per-pair tiered cost, computed inside per-pair loop
    cost_log_flat       = 2 * (cfg.cost + cfg.slip) / 1e4
    borrow_per_day_flat = cfg.borrow / (252 * 1e4)

    # Pre-load $ ADV (price × share-volume) for Almgren-Chriss impact term.
    ac_dollar_vol = None
    if cfg.execution_model == "almgren":
        if volumes is not None:
            common = [c for c in volumes.columns if c in closes.columns]
            ac_dollar_vol = (closes[common] * volumes[common]).rolling(20, min_periods=10).mean()
        else:
            print(f"  almgren: volumes_daily.csv missing, falling back to tier-default ADV")

    AC_FALLBACK_ADV = {
        "us_large_cap":     2.0e9, "us_mid_cap":     3.0e8, "us_small_cap": 2.5e7,
        "canadian_us_list": 1.5e8, "canadian_local": 1.0e8,
        "european_adr_us":  2.0e8, "european_local": 6.0e7,
        "aussie_adr_us":    2.5e8, "aussie_local":   3.0e7,
    }

    def ac_pair_cost(t1: str, t2: str, train_df: pd.DataFrame) -> tuple[float, float]:
        """Return (round-trip cost_log, per-day borrow_log) for an Almgren-Chriss pair."""
        rets = np.log(train_df).diff()
        sig1 = float(rets[t1].std()) if t1 in rets.columns else 0.015
        sig2 = float(rets[t2].std()) if t2 in rets.columns else 0.015
        if not np.isfinite(sig1) or sig1 <= 0: sig1 = 0.015
        if not np.isfinite(sig2) or sig2 <= 0: sig2 = 0.015

        def _adv(tk: str) -> float:
            if ac_dollar_vol is not None and tk in ac_dollar_vol.columns:
                v = ac_dollar_vol.loc[train_df.index.max() if train_df.index.max() in ac_dollar_vol.index
                                       else ac_dollar_vol.index[-1], tk]
                if pd.notna(v) and v > 0:
                    return float(v)
            return AC_FALLBACK_ADV[ac_classify(tk)]

        c1 = ac_leg_cost(t1, sig1, cfg.notional, _adv(t1))
        c2 = ac_leg_cost(t2, sig2, cfg.notional, _adv(t2))
        round_trip = 2 * (c1 + c2)   # 2 legs × 2 sides (enter+exit)
        # Borrow on the short leg (leg2 by convention).
        borrow_per_d = ac_borrow_per_day(t2)
        return round_trip, borrow_per_d

    print(f"DAILY WFO  (BAR_TIMEFRAME={BAR_TIMEFRAME})")
    print(f"  closes:     {closes.shape[0]} days × {closes.shape[1]} tickers")
    print(f"  span:       {closes.index.min().date()} → {closes.index.max().date()}")
    print(f"  pairs:      {len(pairs)}")
    print(f"  windows:    {len(windows)}  (init={cfg.train}m / oos={cfg.oos}m / step={cfg.step}m)")
    print(f"  z-window:   {Z_WIN}d  | entry/exit/stop = {ENTRY_Z}/{EXIT_Z}/{STOP_Z}  | hold≤{MAX_HOLD}d")
    if cfg.execution_model == "flat":
        print(f"  costs:      FLAT — commission+slip = 2×({cfg.cost}+{cfg.slip}) bps = "
              f"{cost_log_flat*1e4:.1f} bps round-trip")
        print(f"              borrow on short = {cfg.borrow} bps/year "
              f"= {borrow_per_day_flat*1e4:.3f} bps/day")
    else:
        print(f"  costs:      ALMGREN-CHRISS — tiered per-pair, notional=${cfg.notional:,.0f}/leg")
        print(f"              half_spread + κ·σ·√(notional/ADV) per leg, "
              f"borrow per-tier (50-300 bps/yr)")
    print(f"  kalman:     {'ON' if cfg.kalman else 'off'}   "
          f"vwz:       {'ON' if cfg.vwz else 'off'}   "
          f"hmm:       {cfg.hmm}   "
          f"metagate:  {'ON' if cfg.metagate else 'off'}   "
          f"cusum:     {('h='+str(cfg.cusum)) if cfg.cusum else 'off'}")
    exec_label = "next_close" if cfg.exec_mode == "next_close" else "next-day VWAP"
    if cfg.legacy_same_close:
        exec_label = ("SAME-CLOSE (legacy)" if cfg.exec_mode == "next_close"
                      else exec_label + " (legacy same-bar bookkeeping)")
    print(f"  exec:       {exec_label}")
    print("  prints:     " + ("LEGACY — no stale-print mask, no zero-return entry guard"
                               if cfg.legacy_stale_prints else
                               "unchanged close with zero/missing volume = no print; "
                               "no entry on a zero-return signal day"))
    print("  pnl:        " + ("LEGACY — spread change between the fills" if cfg.legacy_spread_pnl
                               else "leg fills, q1 = 1/P1 and q2 = β/P2 fixed at the entry fill"))
    print()

    # Last date of the run's data: fills never use a row after cfg.end.
    data_end = pd.Timestamp(cfg.end) if cfg.end else closes.index.max()
    has_px = None if cfg.legacy_same_close else closes.notna()   # for the fill-only bar
    # Audit T2 entry guard: no entry decision on a date where either leg's close
    # equals its previous print exactly (zero return), whatever the volume.
    zero_ret = None if cfg.legacy_stale_prints else repeats_previous_print(closes)

    all_trades = []
    for w_idx, (tr_s, tr_e, oos_s, oos_e) in enumerate(windows, start=1):
        win_trades_n = 0
        win_pnl      = 0.0

        # Window-scoped MetaGate + HRP: both need the train-trade harvest.
        mg_model: MetaGateDaily | None = None
        hrp_w_pair: dict[str, float] = {}

        # Inverse-vol sizing: weight ∝ 1 / std(train spread) per pair,
        # mean-normalised to 1.  No train-trade panel needed; uses raw spread.
        if cfg.ivol:
            vols = {}
            for pair in pairs:
                t1, t2 = pair.split("-")
                if t1 not in closes.columns or t2 not in closes.columns:
                    continue
                tr_slice = closes.loc[tr_s:tr_e, [t1, t2]].dropna()
                if len(tr_slice) < 60:
                    continue
                beta = estimate_beta(tr_slice[t1], tr_slice[t2])
                sp = np.log(tr_slice[t1]) - beta * np.log(tr_slice[t2])
                v = sp.diff().std()
                if v > 0:
                    vols[pair] = v
            if vols:
                inv = pd.Series({p: 1.0 / v for p, v in vols.items()})
                inv = (inv / inv.mean()).clip(lower=0.3, upper=3.0)
                hrp_w_pair.update(inv.to_dict())
                top = inv.sort_values(ascending=False).head(3)
                bot = inv.sort_values().head(3)
                print(f"  ▶ window {w_idx}: ivol weights — "
                      f"top: {dict((k, round(v,2)) for k,v in top.items())}  "
                      f"bot: {dict((k, round(v,2)) for k,v in bot.items())}")

        if cfg.metagate or cfg.hrp:
            X, y, pnl, train_panel = mg_harvest(
                closes.loc[tr_s:tr_e], pairs, tr_s, tr_e,
                z_win=Z_WIN, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                max_hold=MAX_HOLD, cost_per_trade=cost_log,
                borrow_per_day=borrow_per_day,
                estimate_beta_fn=estimate_beta,
                spread_series_fn=spread_series,
            )
            if cfg.metagate:
                mg_model = MetaGateDaily().fit(X, y, pnl)
                if mg_model.usable:
                    print(f"  ▶ window {w_idx}: MetaGate fit on {mg_model.n_train} "
                          f"train trades ({mg_model.kind}, θ={mg_model.theta:.3f})")
                else:
                    print(f"  ▶ window {w_idx}: MetaGate fallback ({mg_model.n_train} train trades)")
            if cfg.hrp and not train_panel.empty and train_panel.shape[0] >= 5:
                w_sum1 = hrp_weights(train_panel)
                # Renormalise so MEAN weight == 1.  Avoid huge multipliers
                # in degenerate windows by clipping to [0.1, 3.0].
                w_mean1 = (w_sum1 * len(w_sum1)).clip(lower=0.1, upper=3.0)
                hrp_w_pair = w_mean1.to_dict()
                top = w_mean1.sort_values(ascending=False).head(3)
                bot = w_mean1.sort_values().head(3)
                print(f"  ▶ window {w_idx}: HRP weights — "
                      f"top: {dict((k, round(v,2)) for k,v in top.items())}  "
                      f"bot: {dict((k, round(v,2)) for k,v in bot.items())}")

        # Window-scoped Macro-HMM: fit on macro-returns(train), predict on full slice.
        panic_full = None
        panic_size_mult = 0.0   # 0 = block, 0.5 = half size, etc.
        if hmm_active and spy_returns is not None:
            spy_train = spy_returns.loc[tr_s:tr_e].dropna().values
            spy_full  = spy_returns.loc[tr_s:oos_e].dropna()
            n_comp = 3 if cfg.hmm == "extreme-3state" else 2
            m, panic_st = fit_macro_hmm(spy_train, n_components=n_comp)
            if m is not None and len(spy_full) > 0:
                states = m.predict(spy_full.values.reshape(-1, 1))
                if cfg.hmm == "block-calm":
                    # Invert: panic-flag = "in calm state" (block in calm)
                    panic_mask = (states != panic_st)
                else:
                    # Block-panic / size-panic / extreme-3state — same mask
                    # (extreme-3state's panic_st is the highest-vol of 3)
                    panic_mask = (states == panic_st)
                panic_full = pd.Series(panic_mask, index=spy_full.index)
                if cfg.hmm == "size-panic":
                    panic_size_mult = 0.5

        # Row range after the OOS end, up to data_end: the fill-only bar's candidates.
        after_lo = closes.index.searchsorted(oos_e, side="right")
        after_hi = closes.index.searchsorted(data_end, side="right")

        for pair in pairs:
            t1, t2 = pair.split("-")
            if t1 not in closes.columns or t2 not in closes.columns:
                continue
            train = closes.loc[tr_s:tr_e, [t1, t2]].dropna()
            if len(train) < 60:
                continue
            full = closes.loc[tr_s:oos_e, [t1, t2]].dropna()
            if len(full) < Z_WIN + 10:
                continue
            if has_px is not None:
                # Audit T1: append ONE fill-only bar, the first later row with
                # both legs, so that a decision on the window's last bar can
                # fill.  No decision is taken on it (decision_end=oos_e);
                # positions still open after the last decision bar are
                # dropped as before (T3).
                both = (has_px[t1].to_numpy()[after_lo:after_hi]
                        & has_px[t2].to_numpy()[after_lo:after_hi])
                if both.any():
                    full = pd.concat([full, closes[[t1, t2]].iloc[[after_lo + int(both.argmax())]]])

            # Per-pair cost: flat scalars OR Almgren-Chriss tiered.
            if cfg.execution_model == "almgren":
                cost_log, borrow_per_day = ac_pair_cost(t1, t2, train)
            else:
                cost_log, borrow_per_day = cost_log_flat, borrow_per_day_flat

            # Audit T2: PnL from the legs' fill prices (legacy: from the spread);
            # the zero-return entry guard on the pair's rows (legacy: none).
            legs = None if cfg.legacy_spread_pnl else full[[t1, t2]]
            no_entry = None if zero_ret is None else (zero_ret[t1] | zero_ret[t2])

            if cfg.kalman:
                sp, z_kal, beta_kal = kalman_spread_z(full, t1, t2)
                tr = backtest_pair(sp, oos_s, oos_e, cost_log,
                                   cusum_h=cfg.cusum, z_series=z_kal,
                                   borrow_per_day=borrow_per_day,
                                   entry_blackout=(None if no_entry is None
                                                   else no_entry.reindex(sp.index)),
                                   legacy_same_close=cfg.legacy_same_close,
                                   decision_end=oos_e,
                                   leg_prices=legs, beta=beta_kal)
            elif cfg.vwz:
                beta = estimate_beta(train[t1], train[t2])
                if not (0.1 <= abs(beta) <= 15.0):
                    continue
                sp, z_vw, vr = vwz_spread_z(full, volumes, t1, t2, beta)
                tr = backtest_pair(
                    sp, oos_s, oos_e, cost_log,
                    z_series=z_vw, vr_series=vr,
                    cusum_h=cfg.cusum,
                    panic_series=(panic_full.reindex(sp.index)
                                   if panic_full is not None else None),
                    panic_size_mult=panic_size_mult,
                    borrow_per_day=borrow_per_day,
                    entry_blackout=None if no_entry is None else no_entry.reindex(sp.index),
                    legacy_same_close=cfg.legacy_same_close,
                    decision_end=oos_e,
                    leg_prices=legs, beta=beta,
                )
            else:
                beta = estimate_beta(train[t1], train[t2])
                if not (0.1 <= abs(beta) <= 15.0):
                    continue
                sp = spread_series(full, t1, t2, beta)
                hurst_sp = (mg_rolling_hurst(sp, window=Z_WIN)
                             if mg_model is not None and mg_model.usable else None)

                # Build exec_spread (next-day VWAP) if requested.
                exec_sp = exec_legs = None
                if vwap_daily is not None:
                    if t1 in vwap_daily.columns and t2 in vwap_daily.columns:
                        vw = vwap_daily.loc[full.index.min():full.index.max(),
                                              [t1, t2]].reindex(full.index)
                        vw_spread = np.log(vw[t1]) - beta * np.log(vw[t2])
                        # Order signalled at end of bar i fills at vw of bar i+1.
                        exec_sp = vw_spread.shift(-1)
                        exec_legs = vw.shift(-1)        # each leg's VWAP of bar i+1 (T2)

                # Aussie/JP conditional gate.
                pair_entry_z = ENTRY_Z
                pair_blackout = None
                if cfg.aussie_filter and is_aussie_jp_pair(pair):
                    pair_entry_z = max(ENTRY_Z, 2.5)
                    pair_blackout = aussie_blackout_series(sp)
                if no_entry is not None:
                    guard = no_entry.reindex(sp.index)
                    pair_blackout = guard if pair_blackout is None else (pair_blackout | guard)

                tr = backtest_pair(
                    sp, oos_s, oos_e, cost_log,
                    entry_z=pair_entry_z,
                    cusum_h=cfg.cusum,
                    panic_series=(panic_full.reindex(sp.index)
                                   if panic_full is not None else None),
                    panic_size_mult=panic_size_mult,
                    borrow_per_day=borrow_per_day,
                    metagate=mg_model if (mg_model is not None and mg_model.usable) else None,
                    hurst_series=hurst_sp,
                    vol_tight_stop=cfg.vol_tight_stop,
                    exec_spread=exec_sp,
                    entry_blackout=pair_blackout,
                    legacy_same_close=cfg.legacy_same_close,
                    decision_end=oos_e,
                    leg_prices=legs, exec_leg_prices=exec_legs, beta=beta,
                )
            if not tr.empty:
                # HRP / inverse-vol: scale the entire trade (pnl is linear in
                # size, so are commission and borrow baked in by backtest_pair).
                if (cfg.hrp or cfg.ivol) and pair in hrp_w_pair:
                    w = hrp_w_pair[pair]
                    tr["pnl"]  = tr["pnl"] * w
                    if "size" in tr.columns:
                        tr["size"] = tr["size"] * w
                tr["pair"]    = pair
                tr["window"]  = w_idx
                tr["oos_start"] = oos_s
                all_trades.append(tr)
                win_trades_n += len(tr)
                win_pnl      += tr["pnl"].sum()
        print(f"  Window {w_idx:02d}/{len(windows)}  "
              f"TRAIN {tr_s.date()} → {tr_e.date()}  |  "
              f"OOS {oos_s.date()} → {oos_e.date()}  "
              f"→  {win_trades_n} trades   P&L = {win_pnl:+.4f}")

    if not all_trades:
        no_trades = pd.DataFrame()
        oos_idx = (closes.loc[windows[0][2]:windows[-1][3]].index if windows
                   else pd.DatetimeIndex([]))
        return WfoResult(trades=no_trades,
                         daily_pnl=exit_date_daily_pnl(no_trades, oos_idx),
                         stats=trade_stats(no_trades, len(windows), len(pairs)))

    trades = pd.concat(all_trades, ignore_index=True)
    trades["entry"] = pd.to_datetime(trades["entry"])
    trades["exit"]  = pd.to_datetime(trades["exit"])
    trades["entry_signal"] = pd.to_datetime(trades["entry_signal"])
    trades["exit_signal"]  = pd.to_datetime(trades["exit_signal"])

    # ── volvol: per-entry size ∝ 1/recent_20d_spread_vol (window-normalised)
    if cfg.volvol:
        size_multipliers = []
        for _, row in trades.sort_values("entry").iterrows():
            t1, t2 = row["pair"].split("-")
            entry_dt = row["entry_signal"]      # closes up to the entry decision
            window_start = entry_dt - pd.Timedelta(days=30)
            sl = closes.loc[window_start:entry_dt, [t1, t2]].dropna()
            if len(sl) < 20:
                size_multipliers.append((row.name, 1.0))
                continue
            beta = estimate_beta(sl[t1], sl[t2])
            sp = np.log(sl[t1]) - beta * np.log(sl[t2])
            recent_vol = sp.diff().tail(20).std()
            size_multipliers.append((row.name, 1.0 / recent_vol if recent_vol > 0 else 1.0))
        # Normalise so mean = 1, clip to [0.3, 3.0]
        idx, mults = zip(*size_multipliers)
        mults = np.array(mults)
        mults = (mults / mults.mean()).clip(0.3, 3.0)
        mult_series = pd.Series(mults, index=idx)
        trades["pnl"]  = trades["pnl"]  * mult_series
        trades["size"] = trades.get("size", 1.0) * mult_series

    # ── pairmom: scale trade by rolling Sharpe of prior same-pair trades
    if cfg.pairmom:
        N_LOOKBACK = 10
        adjusted = []
        for pair, grp in trades.sort_values("entry").groupby("pair"):
            past = []
            for _, row in grp.iterrows():
                if len(past) < 3:
                    mult = 1.0
                else:
                    arr = np.array(past[-N_LOOKBACK:])
                    if arr.std() > 0:
                        sh = arr.mean() / arr.std()
                        # Map Sharpe ∈ [-2, +2] → multiplier ∈ [0.5, 1.5]
                        mult = float(np.clip(1.0 + 0.25 * sh, 0.5, 1.5))
                    else:
                        mult = 1.0
                adjusted.append((row.name, mult))
                past.append(row["pnl"])
        idx, mults = zip(*adjusted)
        mult_series = pd.Series(mults, index=idx)
        trades["pnl"]  = trades["pnl"]  * mult_series
        trades["size"] = trades.get("size", 1.0) * mult_series

    # ── ticker-overlap: block entry if either ticker is already exposed
    if cfg.ticker_overlap:
        ts = trades.sort_values("entry").reset_index(drop=False)
        in_use: list[tuple[pd.Timestamp, set]] = []  # (exit_time, {t1,t2})
        keep = []
        for _, row in ts.iterrows():
            in_use = [(et, tks) for et, tks in in_use if et > row["entry"]]
            t1, t2 = row["pair"].split("-")
            this = {t1, t2}
            if any(this & tks for _, tks in in_use):
                continue
            in_use.append((row["exit"], this))
            keep.append(row["index"])
        n_before = len(trades)
        trades = trades.loc[keep].sort_values("entry").reset_index(drop=True)
        print(f"  → ticker-overlap: kept {len(trades)}/{n_before} trades")

    # ── per-pair quarter cap
    if cfg.per_pair_quarter_cap > 0:
        ts = trades.sort_values("entry").reset_index(drop=False)
        counts: dict[tuple[str, pd.Period], int] = {}
        keep = []
        for _, row in ts.iterrows():
            key = (row["pair"], row["entry"].to_period("Q"))
            if counts.get(key, 0) >= cfg.per_pair_quarter_cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            keep.append(row["index"])
        n_before = len(trades)
        trades = trades.loc[keep].sort_values("entry").reset_index(drop=True)
        print(f"  → per-pair-quarter-cap={cfg.per_pair_quarter_cap}: kept {len(trades)}/{n_before}")

    # ── maxconc: drop trades that would exceed the global concurrency cap
    if cfg.maxconc > 0:
        trades_sorted = trades.sort_values("entry").reset_index(drop=False)
        open_exits = []   # list of exit timestamps currently open
        keep_idx = []
        for _, row in trades_sorted.iterrows():
            # Expire trades that have closed by this entry time.
            open_exits = [e for e in open_exits if e > row["entry"]]
            if len(open_exits) >= cfg.maxconc:
                continue
            open_exits.append(row["exit"])
            keep_idx.append(row["index"])
        trades = trades.loc[keep_idx].sort_values("entry").reset_index(drop=True)
        print(f"  → maxconc={cfg.maxconc}: kept {len(trades)}/{len(trades_sorted)} trades")

    # Full OOS index = every trading day from the first OOS window start
    # to the last OOS window end. Used for the correct portfolio Sharpe.
    first_oos = windows[0][2]
    last_oos  = windows[-1][3]
    # Audit T1: an exit decided on the last OOS bar fills on the next row, so
    # the index runs to the last exit fill; no booked PnL falls outside it.
    last_oos  = max(last_oos, trades["exit"].max())
    oos_idx = closes.loc[first_oos:last_oos].index

    # ── Quarterly per-pair rotation (Option B §5) ─────────────────────────
    if cfg.rotation:
        n_before = len(trades)
        trades = apply_quarterly_rotation(trades)
        print(f"  → rotation: kept {len(trades)}/{n_before} trades after "
              "Sharpe-based pair drop + per-trade S_perf rescaling")

    # ── Portfolio vol targeting (Option B §7) ─────────────────────────────
    if cfg.voltarget > 0:
        trades = apply_vol_target(trades, cfg.voltarget, oos_idx)
        print(f"  → voltarget={cfg.voltarget*100:.1f}%: causal leverage applied")

    if not cfg.legacy_same_close:
        # Hard rule 2 on the final table: every fill after its decision; an
        # exit fills on its decision date only at the end of the data.
        eod = trades["reason"] == "end_of_data"
        assert (trades["entry"] > trades["entry_signal"]).all()
        assert (trades["exit"] > trades["exit_signal"])[~eod].all()
        assert (trades["exit"] == trades["exit_signal"])[eod].all()
    if not cfg.legacy_stale_prints:
        # Audit T2: both legs have a genuine print on every entry's signal date
        # and fill date.
        for pair, g in trades.groupby("pair"):
            px = closes[pair.split("-")]
            assert px.loc[g["entry_signal"]].notna().all().all()
            assert px.loc[g["entry"]].notna().all().all()

    # exit-date booking over panel trading days; replaced by daily MTM over all business days in T4
    daily_pnl = exit_date_daily_pnl(trades, oos_idx)
    return WfoResult(trades=trades, daily_pnl=daily_pnl,
                     stats=trade_stats(trades, len(windows), len(pairs), oos_idx))
