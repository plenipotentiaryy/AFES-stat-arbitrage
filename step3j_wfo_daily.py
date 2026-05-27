"""step3j_wfo_daily.py — Walk-Forward Optimization on the daily timeframe.

Built on top of the sanity-tested engine in `daily_sanity.py`.  The intraday
`step3j_wfo.py` carries a stack of filters (Johansen on rolling 3y, MetaGate
trained on intraday features, limit-order matching, VW-zscore, grid search
on TRAIN) which all degrade on daily bars — the result on baseline 14 pairs
was Sharpe ≈ 0 with 57 trades vs. sanity's Sharpe ≈ +1.59 with 4564 trades.

This file is the daily replacement: expanding-window WFO with the simple,
robust pieces only.  Filters and gates (HMM / Hurst / MetaGate / sizing /
ticker-overlap) will be layered back in subsequent phases on top of this
backbone, each justified by an A/B comparison vs. this raw baseline.

CLI:
    python step3j_wfo_daily.py --pairs pairs_baseline14.csv
    python step3j_wfo_daily.py --pairs pairs_selected.csv --cost 10
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import DATA_DIR, OUTPUT_DIR, BAR_TIMEFRAME
from kalman import kalman_hedge
from metagate_daily import (
    MetaGateDaily, FEATURE_NAMES as MG_FEATURES,
    entry_features as mg_entry_features,
    rolling_hurst as mg_rolling_hurst,
    harvest_train_trades as mg_harvest,
)
from hrp_weights import hrp_weights

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
                  ) -> pd.DataFrame:
    """Generate mean-reversion trades on a daily spread.

    Entries open only inside the OOS window; z is computed on the warmup-
    extended series so rolling statistics are stable from bar 0 of OOS.

    cusum_h: if set, run two-sided CUSUM on the standardized spread and
             abort all further entries for this pair-window once
             max(S+, S-) > cusum_h (k=0.5 slack, classic defaults).
    z_series: pre-computed z (overrides rolling); used by Kalman path
              which produces innovations / sqrt(innov_var).
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
    entry_size = 1.0  # applied to pnl/cost at exit (1.0 normal, 0.5 panic-sized)
    cur_stop_z = stop_z  # per-trade stop (mutated at entry if vol_tight_stop on)

    idx = spread_full.index
    oos_mask = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))

    for i, t in enumerate(idx):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        # Update CUSUM on every bar (train+oos) so the test is informed.
        if cusum_h is not None and not broken:
            Sp = max(0.0, Sp + zi - cusum_k)
            Sn = max(0.0, Sn - zi - cusum_k)
            if max(Sp, Sn) > cusum_h:
                broken = True
                # Force-close any open position on break.
                if in_pos:
                    exit_spread = spread_full.iloc[i]
                    held_bars  = i - entry_idx
                    gross = side * (exit_spread - entry_spread)
                    pnl = entry_size * (gross - cost_per_trade - held_bars * borrow_per_day)
                    trades.append({
                        "entry":  idx[entry_idx], "exit": t,
                        "held":   held_bars, "side": side, "pnl": pnl,
                        "size":   entry_size,
                        "reason": "cusum_break",
                    })
                    in_pos = False

        if broken:
            continue

        if not in_pos:
            if not oos_mask[i]:
                continue
            # Liquidity gate: VR < 0.5 → skip entry (legacy convention).
            if vr_series is not None:
                vri = vr_series.iloc[i]
                if pd.notna(vri) and vri < vr_min:
                    continue
            # Macro-HMM panic gate. When panic active:
            #   panic_size_mult == 0  → skip entry entirely (block-mode)
            #   panic_size_mult > 0   → enter with reduced size (sizing-mode)
            this_size = 1.0
            if panic_series is not None:
                pi = panic_series.iloc[i]
                if pd.notna(pi) and bool(pi):
                    if panic_size_mult <= 0.0:
                        continue
                    this_size = panic_size_mult
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
            fill_spread_at_entry = (
                exec_spread.iloc[i] if exec_spread is not None
                else spread_full.iloc[i]
            )
            if exec_spread is not None and pd.isna(fill_spread_at_entry):
                continue
            side = prov_side
            in_pos = True
            entry_idx, entry_spread = i, fill_spread_at_entry
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
                fill_spread_at_exit = (
                    exec_spread.iloc[i] if exec_spread is not None
                    else spread_full.iloc[i]
                )
                if exec_spread is not None and pd.isna(fill_spread_at_exit):
                    # Roll forward one bar — if last bar of OOS, force-close
                    # at last available exec price (no future data leakage
                    # because exec_spread is pre-shifted).
                    last_known = exec_spread.iloc[:i + 1].dropna()
                    if last_known.empty:
                        continue
                    fill_spread_at_exit = last_known.iloc[-1]
                exit_spread = fill_spread_at_exit
                held_bars  = i - entry_idx
                borrow_cost = held_bars * borrow_per_day
                gross = side * (exit_spread - entry_spread)
                pnl = entry_size * gross - entry_size * cost_per_trade - entry_size * borrow_cost
                trades.append({
                    "entry":  idx[entry_idx], "exit": t,
                    "held":   held, "side": side, "pnl": pnl,
                    "size":   entry_size,
                    "reason": "target" if hit_target else ("stop" if hit_stop else "time"),
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
                    z_win: int = Z_WIN) -> tuple[pd.Series, pd.Series]:
    """Dynamic-β spread via Kalman, then rolling z on the spread.

    Kalman provides the time-varying hedge ratio β_t (random-walk prior).
    The tradable spread is spread_t = log P1_t − β_t · log P2_t — a path
    you can hold and exit later. We then standardise it with the same
    rolling-z used by the static-β baseline so PnL semantics match.

    Returns (spread_series, z_series) aligned to df.index.
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
    return spread, z


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


# ── Reporting ───────────────────────────────────────────────────────────────

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
    pnl = tr["pnl"]
    eq  = pnl.cumsum()
    dd  = (eq - eq.cummax()).min()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()

    daily_exit = tr.groupby(pd.to_datetime(tr["exit"]).dt.normalize())["pnl"].sum()
    exit_sh = (daily_exit.mean() / daily_exit.std() * np.sqrt(252)
                if daily_exit.std() > 0 else float("nan"))

    if oos_index is not None and len(oos_index) > 0:
        daily_full = daily_exit.reindex(pd.DatetimeIndex(oos_index).normalize(),
                                          fill_value=0.0)
        full_sh = (daily_full.mean() / daily_full.std() * np.sqrt(252)
                    if daily_full.std() > 0 else float("nan"))
    else:
        full_sh = float("nan")

    print(f"\n{'='*70}")
    print(f"DAILY WFO SUMMARY — {label}")
    print(f"{'='*70}")
    print(f"Windows:        {n_windows}")
    print(f"Pairs (active): {tr['pair'].nunique()} / {n_pairs}")
    print(f"OOS Trades:     {len(tr)}")
    print(f"Win rate:       {(pnl > 0).mean()*100:.1f}%")
    print(f"Total PnL:      {pnl.sum():+.4f}  (log-spread units)")
    print(f"Profit factor:  {wins/losses if losses>0 else float('inf'):.2f}")
    print(f"Max DD:         {dd:+.4f}")
    print(f"DD / PnL:       {abs(dd)/pnl.sum()*100:.1f}%")
    if not np.isnan(full_sh):
        print(f"Sharpe (full):  {full_sh:+.3f}   ← correct portfolio Sharpe")
        print(f"Sharpe (exit):  {exit_sh:+.3f}   (inflated; legacy metric)")
    else:
        print(f"Sharpe (exit):  {exit_sh:+.3f}")
    print(f"Exit reasons:   {tr['reason'].value_counts().to_dict()}")

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


def save_equity_chart(tr: pd.DataFrame, path: Path):
    if tr.empty:
        return
    eq = tr.sort_values("exit")["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr.sort_values("exit")["exit"].values, eq.values)
    ax.set_title("Daily WFO — cumulative log-spread PnL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── Main loop ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Daily Walk-Forward Optimization")
    p.add_argument("--pairs",  type=str, default="pairs_selected.csv",
                   help="CSV in data/ with a 'pair' column (e.g. AMT-AWK).")
    p.add_argument("--start",  type=str, default=None)
    p.add_argument("--end",    type=str, default=None)
    p.add_argument("--cost",   type=float, default=5.0,
                   help="Commission + half-spread, bps per leg (default 5).")
    p.add_argument("--slip",   type=float, default=5.0,
                   help="Slippage, bps per leg (default 5).")
    p.add_argument("--borrow", type=float, default=80.0,
                   help="Borrow rate on short leg, bps/year (default 80).")
    p.add_argument("--train",  type=int, default=TRAIN_INIT_MO,
                   help=f"Initial train length in months (default {TRAIN_INIT_MO}).")
    p.add_argument("--oos",    type=int, default=OOS_MO,
                   help=f"OOS length in months (default {OOS_MO}).")
    p.add_argument("--step",   type=int, default=STEP_MO,
                   help=f"Walk-forward step in months (default {STEP_MO}).")
    p.add_argument("--kalman", action="store_true",
                   help="Use Kalman dynamic β (innovations as spread, e/√S as z).")
    p.add_argument("--vwz",    action="store_true",
                   help="Use volume-weighted z + VR≥0.5 liquidity gate.")
    p.add_argument("--metagate", action="store_true",
                   help="Per-window MetaGate ML entry classifier "
                        "(features: abs_z, side, vol_20/60, vol_ratio, hurst).")
    p.add_argument("--hrp", action="store_true",
                   help="Per-window HRP weights scale each pair's trade size. "
                        "Weights renormalized so MEAN weight = 1 (low-corr "
                        "pairs scaled up, high-corr pairs scaled down).")
    p.add_argument("--ivol", action="store_true",
                   help="Inverse-volatility sizing: weight ∝ 1/std(spread) on train, "
                        "mean-normalized to 1. Simpler/more robust than HRP.")
    p.add_argument("--volvol", action="store_true",
                   help="Dynamic vol-targeting: per-entry size ∝ 1/recent_20d_spread_vol, "
                        "normalised against window mean. Pro-cyclical.")
    p.add_argument("--pairmom", action="store_true",
                   help="Rolling-Sharpe pair momentum: hot pair → up-weight, cold → down. "
                        "Causal: only uses pair's executed OOS trades so far.")
    p.add_argument("--maxconc", type=int, default=0,
                   help="Hard cap on concurrent open trades across all pairs "
                        "(post-hoc filter, 0 = no cap).")
    p.add_argument("--ticker-overlap", action="store_true",
                   help="Block new entry if either of its tickers is already "
                        "exposed in another open trade.")
    p.add_argument("--per-pair-quarter-cap", type=int, default=0,
                   help="Max trades per pair per calendar quarter (0 = no cap).")
    p.add_argument("--vol-tight-stop", action="store_true",
                   help="Tighten stop_z to 2.5 when entry's 20d/60d vol ratio > 1.5.")
    p.add_argument("--vwap-exec", action="store_true",
                   help="Use next-day VWAP for entry/exit fills (signal still "
                        "computed on close). Requires data/vwap_daily.csv.")
    p.add_argument("--rotation", action="store_true",
                   help="Quarterly per-pair S_perf rescaling (§5): trades sized by "
                        "rolling pair-Sharpe with Bayesian shrinkage. Drop pairs "
                        "with SR_6m < 0 for the current quarter.")
    p.add_argument("--voltarget", type=float, default=0.0,
                   help="Portfolio vol target (annualised, e.g. 0.10 = 10%%). "
                        "Applied as causal leverage scaling. 0 = off.")
    p.add_argument("--hmm",    type=str, default="off",
                   choices=["off", "block-panic", "block-calm",
                            "size-panic", "extreme-3state"],
                   help="HMM mode: off / block-panic / block-calm (invert) / "
                        "size-panic (50%% size in panic) / extreme-3state "
                        "(3-state, block only highest-vol).")
    p.add_argument("--cusum",  type=float, default=None,
                   help="CUSUM break threshold h (e.g. 5.0). If unset, CUSUM off.")
    p.add_argument("--tag",    type=str, default="",
                   help="Suffix for output filenames (e.g. 'kalman_cusum').")
    args = p.parse_args()

    if BAR_TIMEFRAME != "daily":
        raise SystemExit(
            f"BAR_TIMEFRAME = {BAR_TIMEFRAME!r}; this script is for daily mode. "
            "Set BAR_TIMEFRAME='daily' in config.py."
        )

    closes_path = DATA_DIR / "closes_daily.csv"
    closes = pd.read_csv(closes_path, parse_dates=["Date"]).set_index("Date").sort_index()
    pairs  = pd.read_csv(DATA_DIR / args.pairs)["pair"].tolist()

    # Macro returns for HMM.  Prefer SPY; otherwise use an equal-weighted
    # cross-sectional mean of log-returns as a market-wide panic proxy.
    spy_returns = None
    hmm_active = (args.hmm != "off")
    if hmm_active:
        if "SPY" in closes.columns:
            spy_returns = np.log(closes["SPY"]).diff()
            print(f"  hmm mode: {args.hmm}   macro proxy: SPY")
        else:
            spy_returns = np.log(closes).diff().mean(axis=1)
            print(f"  hmm mode: {args.hmm}   macro proxy: EW mean of {closes.shape[1]} names")

    vwap_daily = None
    if args.vwap_exec:
        vp = DATA_DIR / "vwap_daily.csv"
        if not vp.exists():
            raise SystemExit("vwap_daily.csv missing — run build_vwap_daily.py")
        vwap_daily = pd.read_csv(vp, parse_dates=["Date"]).set_index("Date").sort_index()
        print(f"  vwap-exec: ON ({vwap_daily.shape[0]} days × {vwap_daily.shape[1]} tickers)")

    volumes = None
    if args.vwz:
        vol_path = DATA_DIR / "volumes_daily.csv"
        if not vol_path.exists():
            raise SystemExit("volumes_daily.csv missing — run build_volumes_daily.py")
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()

    windows = make_windows(closes, args.train, args.oos, args.step,
                            start=args.start, end=args.end)
    # Cost breakdown:
    #   commission+slippage on round-trip = 2 × (cost + slip) bps
    #   borrow on short leg, accrued daily, applied per holding-bar at exit
    cost_log         = 2 * (args.cost + args.slip) / 1e4
    borrow_per_day   = args.borrow / (252 * 1e4)   # bps/year → log per day on short leg

    print(f"DAILY WFO  (BAR_TIMEFRAME={BAR_TIMEFRAME})")
    print(f"  closes:     {closes.shape[0]} days × {closes.shape[1]} tickers")
    print(f"  span:       {closes.index.min().date()} → {closes.index.max().date()}")
    print(f"  pairs:      {len(pairs)} from {args.pairs}")
    print(f"  windows:    {len(windows)}  (init={args.train}m / oos={args.oos}m / step={args.step}m)")
    print(f"  z-window:   {Z_WIN}d  | entry/exit/stop = {ENTRY_Z}/{EXIT_Z}/{STOP_Z}  | hold≤{MAX_HOLD}d")
    print(f"  costs:      commission+slip = 2×({args.cost}+{args.slip}) bps = "
          f"{cost_log*1e4:.1f} bps round-trip")
    print(f"              borrow on short = {args.borrow} bps/year "
          f"= {borrow_per_day*1e4:.3f} bps/day")
    print(f"  kalman:     {'ON' if args.kalman else 'off'}   "
          f"vwz:       {'ON' if args.vwz else 'off'}   "
          f"hmm:       {args.hmm}   "
          f"metagate:  {'ON' if args.metagate else 'off'}   "
          f"cusum:     {('h='+str(args.cusum)) if args.cusum else 'off'}")
    print()

    all_trades = []
    for w_idx, (tr_s, tr_e, oos_s, oos_e) in enumerate(windows, start=1):
        win_trades_n = 0
        win_pnl      = 0.0

        # Window-scoped MetaGate + HRP: both need the train-trade harvest.
        mg_model: MetaGateDaily | None = None
        hrp_w_pair: dict[str, float] = {}

        # Inverse-vol sizing: weight ∝ 1 / std(train spread) per pair,
        # mean-normalised to 1.  No train-trade panel needed; uses raw spread.
        if args.ivol:
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

        if args.metagate or args.hrp:
            X, y, pnl, train_panel = mg_harvest(
                closes.loc[tr_s:tr_e], pairs, tr_s, tr_e,
                z_win=Z_WIN, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                max_hold=MAX_HOLD, cost_per_trade=cost_log,
                borrow_per_day=borrow_per_day,
                estimate_beta_fn=estimate_beta,
                spread_series_fn=spread_series,
            )
            if args.metagate:
                mg_model = MetaGateDaily().fit(X, y, pnl)
                if mg_model.usable:
                    print(f"  ▶ window {w_idx}: MetaGate fit on {mg_model.n_train} "
                          f"train trades ({mg_model.kind}, θ={mg_model.theta:.3f})")
                else:
                    print(f"  ▶ window {w_idx}: MetaGate fallback ({mg_model.n_train} train trades)")
            if args.hrp and not train_panel.empty and train_panel.shape[0] >= 5:
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
            n_comp = 3 if args.hmm == "extreme-3state" else 2
            m, panic_st = fit_macro_hmm(spy_train, n_components=n_comp)
            if m is not None and len(spy_full) > 0:
                states = m.predict(spy_full.values.reshape(-1, 1))
                if args.hmm == "block-calm":
                    # Invert: panic-flag = "in calm state" (block in calm)
                    panic_mask = (states != panic_st)
                else:
                    # Block-panic / size-panic / extreme-3state — same mask
                    # (extreme-3state's panic_st is the highest-vol of 3)
                    panic_mask = (states == panic_st)
                panic_full = pd.Series(panic_mask, index=spy_full.index)
                if args.hmm == "size-panic":
                    panic_size_mult = 0.5

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

            if args.kalman:
                sp, z_kal = kalman_spread_z(full, t1, t2)
                tr = backtest_pair(sp, oos_s, oos_e, cost_log,
                                   cusum_h=args.cusum, z_series=z_kal,
                                   borrow_per_day=borrow_per_day)
            elif args.vwz:
                beta = estimate_beta(train[t1], train[t2])
                if not (0.1 <= abs(beta) <= 15.0):
                    continue
                sp, z_vw, vr = vwz_spread_z(full, volumes, t1, t2, beta)
                tr = backtest_pair(
                    sp, oos_s, oos_e, cost_log,
                    z_series=z_vw, vr_series=vr,
                    cusum_h=args.cusum,
                    panic_series=(panic_full.reindex(sp.index)
                                   if panic_full is not None else None),
                    panic_size_mult=panic_size_mult,
                    borrow_per_day=borrow_per_day,
                )
            else:
                beta = estimate_beta(train[t1], train[t2])
                if not (0.1 <= abs(beta) <= 15.0):
                    continue
                sp = spread_series(full, t1, t2, beta)
                hurst_sp = (mg_rolling_hurst(sp, window=Z_WIN)
                             if mg_model is not None and mg_model.usable else None)

                # Build exec_spread (next-day VWAP) if requested.
                exec_sp = None
                if vwap_daily is not None:
                    if t1 in vwap_daily.columns and t2 in vwap_daily.columns:
                        vw = vwap_daily.loc[full.index.min():full.index.max(),
                                              [t1, t2]].reindex(full.index)
                        vw_spread = np.log(vw[t1]) - beta * np.log(vw[t2])
                        # Order signalled at end of bar i fills at vw of bar i+1.
                        exec_sp = vw_spread.shift(-1)

                tr = backtest_pair(
                    sp, oos_s, oos_e, cost_log,
                    cusum_h=args.cusum,
                    panic_series=(panic_full.reindex(sp.index)
                                   if panic_full is not None else None),
                    panic_size_mult=panic_size_mult,
                    borrow_per_day=borrow_per_day,
                    metagate=mg_model if (mg_model is not None and mg_model.usable) else None,
                    hurst_series=hurst_sp,
                    vol_tight_stop=args.vol_tight_stop,
                    exec_spread=exec_sp,
                )
            if not tr.empty:
                # HRP / inverse-vol: scale the entire trade (pnl is linear in
                # size, so are commission and borrow baked in by backtest_pair).
                if (args.hrp or args.ivol) and pair in hrp_w_pair:
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
        print("\nNo trades.")
        return

    trades = pd.concat(all_trades, ignore_index=True)
    trades["entry"] = pd.to_datetime(trades["entry"])
    trades["exit"]  = pd.to_datetime(trades["exit"])

    # ── volvol: per-entry size ∝ 1/recent_20d_spread_vol (window-normalised)
    if args.volvol:
        size_multipliers = []
        for _, row in trades.sort_values("entry").iterrows():
            t1, t2 = row["pair"].split("-")
            entry_dt = row["entry"]
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
    if args.pairmom:
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
    if args.ticker_overlap:
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
    if args.per_pair_quarter_cap > 0:
        ts = trades.sort_values("entry").reset_index(drop=False)
        counts: dict[tuple[str, pd.Period], int] = {}
        keep = []
        for _, row in ts.iterrows():
            key = (row["pair"], row["entry"].to_period("Q"))
            if counts.get(key, 0) >= args.per_pair_quarter_cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            keep.append(row["index"])
        n_before = len(trades)
        trades = trades.loc[keep].sort_values("entry").reset_index(drop=True)
        print(f"  → per-pair-quarter-cap={args.per_pair_quarter_cap}: kept {len(trades)}/{n_before}")

    # ── maxconc: drop trades that would exceed the global concurrency cap
    if args.maxconc > 0:
        trades_sorted = trades.sort_values("entry").reset_index(drop=False)
        open_exits = []   # list of exit timestamps currently open
        keep_idx = []
        for _, row in trades_sorted.iterrows():
            # Expire trades that have closed by this entry time.
            open_exits = [e for e in open_exits if e > row["entry"]]
            if len(open_exits) >= args.maxconc:
                continue
            open_exits.append(row["exit"])
            keep_idx.append(row["index"])
        trades = trades.loc[keep_idx].sort_values("entry").reset_index(drop=True)
        print(f"  → maxconc={args.maxconc}: kept {len(trades)}/{len(trades_sorted)} trades")

    # Full OOS index = every trading day from the first OOS window start
    # to the last OOS window end. Used for the correct portfolio Sharpe.
    first_oos = windows[0][2]
    last_oos  = windows[-1][3]
    oos_idx = closes.loc[first_oos:last_oos].index

    # ── Quarterly per-pair rotation (Option B §5) ─────────────────────────
    if args.rotation:
        n_before = len(trades)
        trades = apply_quarterly_rotation(trades)
        print(f"  → rotation: kept {len(trades)}/{n_before} trades after "
              "Sharpe-based pair drop + per-trade S_perf rescaling")

    # ── Portfolio vol targeting (Option B §7) ─────────────────────────────
    if args.voltarget > 0:
        trades = apply_vol_target(trades, args.voltarget, oos_idx)
        print(f"  → voltarget={args.voltarget*100:.1f}%: causal leverage applied")

    summarise(trades, label=f"NET {args.cost} bps/leg",
              n_windows=len(windows), n_pairs=len(pairs),
              oos_index=oos_idx)

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = DATA_DIR / f"wfo_daily_results{suffix}.csv"
    trades.to_csv(out_csv, index=False)
    chart = OUTPUT_DIR / f"wfo_daily_equity{suffix}.png"
    save_equity_chart(trades, chart)
    print(f"\nSaved → {out_csv} ({len(trades)} trades)")
    print(f"Chart  → {chart}")


if __name__ == "__main__":
    main()
