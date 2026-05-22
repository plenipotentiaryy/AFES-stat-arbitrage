"""
step4d_backtest_full.py
=======================
Прогрессивная сборка полного бэктеста поверх numba-кернела из step4b.

Слои (включаются флагами):
  L0 — Microstructure: VW-Z + velocity gate + thin-market gate
  L1 — HMM regime filter (skip entry в волатильном режиме)
  L2 — Walk-forward оптимизация (rolling train/test refit параметров)
  L3 — Portfolio sizing + concurrent cap
  L4 — Stress overlay

Запуск:
  python step4d_backtest_full.py             # baseline (= step4b)
  python step4d_backtest_full.py --L0        # + microstructure filters
  python step4d_backtest_full.py --L1        # + regime filter
"""
import argparse
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from config import (
    CLOSES_FILE, VOLUMES_FILE,
    COST_TAKER, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR, BARS_PER_DAY,
)

ENTRY_Z = 3.2
EXIT_Z  = -0.2
STOP_Z  = 4.4
BARS_PER_TRADING_DAY = BARS_PER_DAY


# ── L0 config (microstructure) ────────────────────────────────────────────────
@dataclass
class L0Config:
    use_vwz: bool        = True   # volume-weighted z-score instead of plain rolling z
    use_velocity: bool   = True   # block entry when |Δz| over K bars is too large
    use_thin: bool       = True   # block entry when current bar dollar-volume is thin
    vel_bars: int        = 5
    vel_thr: float       = 2.0    # directional: sign(z)·(z[i]−z[i−K]) > vel_thr → block
    thin_lookback: int   = 60     # rolling window for typical dollar-volume
    thin_frac: float     = 0.5    # block if dv_pair[i] < frac × rolling-median dv_pair

L0_OFF = L0Config(use_vwz=False, use_velocity=False, use_thin=False)


# ── Numba kernel with optional regime mask ────────────────────────────────────
@njit(cache=True)
def _backtest_kernel(zscore, spread, t1_price, t2_price, regime_skip, beta,
                     entry_z, exit_z, stop_z,
                     cost_taker, borrow_rate, bars_per_day, pair_max_loss,
                     use_regime):
    n = zscore.shape[0]
    entry_idx  = np.empty(n, dtype=np.int64)
    exit_idx   = np.empty(n, dtype=np.int64)
    direction  = np.empty(n, dtype=np.int8)
    gross_pnl  = np.empty(n, dtype=np.float64)
    tx_cost_a  = np.empty(n, dtype=np.float64)
    borrow_a   = np.empty(n, dtype=np.float64)
    net_pnl_a  = np.empty(n, dtype=np.float64)
    cum_pnl_a  = np.empty(n, dtype=np.float64)
    is_stop    = np.empty(n, dtype=np.int8)
    k = 0

    position = 0
    entry_spread = 0.0
    entry_t1 = 0.0
    entry_t2 = 0.0
    entry_bar = 0
    cumulative_pnl = 0.0

    for i in range(n):
        z = zscore[i]
        sp = spread[i]
        p1 = t1_price[i]
        p2 = t2_price[i]

        if position != 0:
            exit_signal = (position == 1 and z > -exit_z) or (position == -1 and z < exit_z)
            stop_signal = (position == 1 and z < -stop_z) or (position == -1 and z > stop_z)

            if exit_signal or stop_signal:
                g = position * (sp - entry_spread)
                notional = entry_t1 + beta * entry_t2
                tx = 2.0 * notional * cost_taker
                hold_days = (i - entry_bar) / bars_per_day
                short_notional = (beta * entry_t2) if position == 1 else entry_t1
                bc = short_notional * borrow_rate * hold_days / 252.0
                net = g - tx - bc
                cumulative_pnl += net

                entry_idx[k] = entry_bar
                exit_idx[k]  = i
                direction[k] = position
                gross_pnl[k] = g
                tx_cost_a[k] = tx
                borrow_a[k]  = bc
                net_pnl_a[k] = net
                cum_pnl_a[k] = cumulative_pnl
                is_stop[k]   = 1 if stop_signal else 0
                k += 1
                position = 0

                if cumulative_pnl < pair_max_loss:
                    break

        if position == 0:
            # L1: skip entries while regime says "volatile"
            if use_regime and regime_skip[i] == 1:
                continue
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
            if position != 0:
                entry_spread = sp
                entry_t1 = p1
                entry_t2 = p2
                entry_bar = i

    return (entry_idx[:k], exit_idx[:k], direction[:k], gross_pnl[:k],
            tx_cost_a[:k], borrow_a[:k], net_pnl_a[:k], cum_pnl_a[:k], is_stop[:k])


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_closes(bars_tail: int = RECENT_BARS) -> pd.DataFrame:
    closes = pd.read_csv(DATA_DIR / CLOSES_FILE, index_col=0)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    return closes.between_time(RTH_START, RTH_END).tail(bars_tail)


def load_volumes(bars_tail: int = RECENT_BARS) -> pd.DataFrame | None:
    p = DATA_DIR / VOLUMES_FILE
    if not p.exists():
        return None
    vols = pd.read_csv(p, index_col=0)
    vols.index = pd.to_datetime(vols.index, utc=True).tz_convert("US/Eastern")
    return vols.between_time(RTH_START, RTH_END).tail(bars_tail)


def load_regimes() -> pd.DataFrame | None:
    p = DATA_DIR / "regimes.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p, index_col=0)
    r.index = pd.to_datetime(r.index, utc=True).tz_convert("US/Eastern")
    return r


def _vw_zscore(spread: pd.Series, weights: pd.Series, window: int) -> pd.Series:
    """Volume-weighted z-score: VW-mean (high-volume bars anchor fair value) but
    plain rolling std for the denominator.

    Rationale: weighted variance systematically underestimates dispersion when
    high-volume bars cluster — that inflates z and produces phantom signals. Use
    VW only where the prior is strong (location of fair value), and let std come
    from the unweighted dispersion of the spread itself.
    """
    mp = max(5, window // 2)
    w = weights.fillna(0.0).clip(lower=0.0)
    sw  = w.rolling(window, min_periods=mp).sum()
    swx = (w * spread).rolling(window, min_periods=mp).sum()
    vw_mean = swx / sw.replace(0.0, np.nan)
    std = spread.rolling(window, min_periods=mp).std()
    z = (spread - vw_mean) / std.replace(0.0, np.nan)
    return z


def build_signals(closes, t1, t2, beta, half_life,
                  volumes: pd.DataFrame | None = None,
                  l0: L0Config = L0_OFF) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))

    # Dollar-volume per leg + binding pair liquidity (min of the two legs)
    dv_pair = None
    if volumes is not None and t1 in volumes.columns and t2 in volumes.columns:
        v1 = volumes[t1].reindex(closes.index)
        v2 = volumes[t2].reindex(closes.index)
        dv1 = closes[t1] * v1
        dv2 = closes[t2] * v2 * float(beta)
        dv_pair = pd.concat([dv1, dv2], axis=1).min(axis=1)

    # Z-score: volume-weighted if L0.use_vwz and we have volumes; else plain rolling
    if l0.use_vwz and dv_pair is not None:
        zscore = _vw_zscore(spread, dv_pair, window)
    else:
        zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    out = pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread,
        "zscore": zscore,
    })

    # L0 entry-block mask: 1 = skip entry on this bar
    block = pd.Series(0, index=out.index, dtype=np.int8)
    if l0.use_velocity:
        # Directional velocity: block only if z is moving AWAY from zero fast.
        # |z| already extreme + still diverging in same direction → structural,
        # not mean-reverting. Plain |Δz| also blocks reversion moves, which kills alpha.
        dz = zscore - zscore.shift(l0.vel_bars)
        divergence = np.sign(zscore) * dz
        vel_gate = divergence > l0.vel_thr
        block = block | vel_gate.fillna(False).astype(np.int8)
    if l0.use_thin and dv_pair is not None:
        thin_med = dv_pair.rolling(l0.thin_lookback,
                                   min_periods=max(10, l0.thin_lookback // 3)).median()
        thin_mask = dv_pair < (l0.thin_frac * thin_med)
        block = block | thin_mask.fillna(False).astype(np.int8)
    out["entry_block_l0"] = block.astype(np.int8)

    return out.dropna(subset=[f"{t1}_close", f"{t2}_close", "spread", "zscore"]) \
              .between_time(SIGNAL_START, RTH_END)


def _combined_skip(df, regime_series, use_regime, use_l0) -> tuple[np.ndarray, bool]:
    """OR per-bar entry-skip from regime filter (L1) and L0 microstructure block."""
    n = len(df)
    skip = np.zeros(n, dtype=np.int8)
    active = False
    if use_regime and regime_series is not None:
        rs = regime_series.reindex(df.index).fillna(0).astype(np.int8).to_numpy()
        skip = (skip | rs).astype(np.int8)
        active = True
    if use_l0 and "entry_block_l0" in df.columns:
        l0 = df["entry_block_l0"].fillna(0).astype(np.int8).to_numpy()
        skip = (skip | l0).astype(np.int8)
        active = True
    return skip, active


def backtest_pair(df, t1, t2, beta, regime_series=None,
                  use_regime=False, use_l0=False) -> pd.DataFrame:
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    zscore   = df["zscore"].to_numpy(dtype=np.float64)
    spread   = df["spread"].to_numpy(dtype=np.float64)
    p1       = df[t1_col].to_numpy(dtype=np.float64)
    p2       = df[t2_col].to_numpy(dtype=np.float64)

    rs, skip_active = _combined_skip(df, regime_series, use_regime, use_l0)

    (ei, xi, dir_, g, tx, bc, net, cum, stop) = _backtest_kernel(
        zscore, spread, p1, p2, rs, float(beta),
        float(ENTRY_Z), float(EXIT_Z), float(STOP_Z),
        float(COST_TAKER), float(BORROW_RATE_ANNUAL),
        float(BARS_PER_TRADING_DAY), float(PAIR_MAX_LOSS),
        bool(skip_active),
    )

    if ei.size == 0:
        return pd.DataFrame()

    idx = df.index
    return pd.DataFrame({
        "pair":         f"{t1}-{t2}",
        "entry_time":   idx[ei],
        "exit_time":    idx[xi],
        "direction":    np.where(dir_ == 1, "LONG", "SHORT"),
        "holding_bars": xi - ei,
        "gross_pnl":    np.round(g, 4),
        "tx_cost":      np.round(tx, 4),
        "borrow_cost":  np.round(bc, 4),
        "net_pnl":      np.round(net, 4),
        "cum_pnl":      np.round(cum, 4),
        "exit_reason":  np.where(stop == 1, "STOP", "SIGNAL"),
        "entry_z":      np.round(zscore[ei], 2),
        "exit_z":       np.round(zscore[xi], 2),
    })


# ── L2: Walk-forward optimization ─────────────────────────────────────────────
WFO_GRID = {
    # Wide grid: empirically 75% of folds want stop_z=5.0 and 28% want entry_z>=4.5.
    # Old narrow grid (entry≤4.0, stop≤4.5) was a bottleneck; widening alone took
    # honest OOS Sharpe 2.06 → 2.63 and MaxDD −97 → −66.
    "entry_z": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "exit_z":  [-0.2, 0.0],
    "stop_z":  [4.0, 4.5, 5.0],
    "regime":  ["off", "skip_vol", "skip_calm"],  # 0=off, 1=skip_vol, 2=skip_calm
}


def _kernel_with_params(df, beta, regime_series, entry_z, exit_z, stop_z,
                        regime_mode, use_l0=False):
    """Run numba kernel with given params on a slice. Returns trades DataFrame."""
    zscore = df["zscore"].to_numpy(dtype=np.float64)
    spread = df["spread"].to_numpy(dtype=np.float64)
    p1 = df.iloc[:, 0].to_numpy(dtype=np.float64)
    p2 = df.iloc[:, 1].to_numpy(dtype=np.float64)

    n = len(df)
    rs = np.zeros(n, dtype=np.int8)
    use_reg = regime_mode != "off"
    if use_reg and regime_series is not None:
        r = regime_series.reindex(df.index).fillna(0).astype(np.int8)
        if regime_mode == "skip_calm":
            r = (1 - r).astype(np.int8)
        rs = (rs | r.to_numpy()).astype(np.int8)
    else:
        use_reg = False

    if use_l0 and "entry_block_l0" in df.columns:
        l0 = df["entry_block_l0"].fillna(0).astype(np.int8).to_numpy()
        rs = (rs | l0).astype(np.int8)
        use_reg = True  # kernel just needs "use_skip_mask = True"

    (ei, xi, _, _, _, _, net, _, _) = _backtest_kernel(
        zscore, spread, p1, p2, rs, float(beta),
        float(entry_z), float(exit_z), float(stop_z),
        float(COST_TAKER), float(BORROW_RATE_ANNUAL),
        float(BARS_PER_TRADING_DAY), float(PAIR_MAX_LOSS),
        bool(use_reg),
    )
    return ei, xi, net


def _score_train(ei, xi, net):
    """Sharpe-ish objective on train: total PnL × sqrt(N). Robust to small N."""
    if net.size < 5:
        return -1e9
    std = net.std()
    if std == 0:
        return -1e9
    return (net.mean() / std) * np.sqrt(net.size)


def walk_forward_pair(df_full, t1, t2, beta, regime_series,
                      train_bars, test_bars, step_bars, use_l0=False):
    """
    Rolling fixed-window WFO on a single pair.
    For each fold: grid-search on train slice, apply best on test slice.
    Returns concatenated test-period trades.
    """
    n = len(df_full)
    if n < train_bars + test_bars:
        return pd.DataFrame(), []

    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    cols = [t1_col, t2_col, "spread", "zscore"]
    if "entry_block_l0" in df_full.columns:
        cols.append("entry_block_l0")
    df_ker = df_full[cols].copy()

    test_trades_all = []
    fold_log = []
    start = 0
    fold = 0
    while start + train_bars + test_bars <= n:
        train = df_ker.iloc[start : start + train_bars]
        test  = df_ker.iloc[start + train_bars : start + train_bars + test_bars]

        # Grid search on train
        best_score = -1e18
        best_params = None
        for ez in WFO_GRID["entry_z"]:
            for xz in WFO_GRID["exit_z"]:
                for sz in WFO_GRID["stop_z"]:
                    for rm in WFO_GRID["regime"]:
                        ei, xi, net = _kernel_with_params(train, beta, regime_series,
                                                          ez, xz, sz, rm, use_l0=use_l0)
                        s = _score_train(ei, xi, net)
                        if s > best_score:
                            best_score = s
                            best_params = (ez, xz, sz, rm)

        # Apply on test
        if best_params is not None:
            ez, xz, sz, rm = best_params
            ei, xi, net = _kernel_with_params(test, beta, regime_series, ez, xz, sz, rm,
                                              use_l0=use_l0)
            if ei.size > 0:
                idx = test.index
                trades = pd.DataFrame({
                    "pair":        f"{t1}-{t2}",
                    "entry_time":  idx[ei],
                    "exit_time":   idx[xi],
                    "net_pnl":     np.round(net, 4),
                    "fold":        fold,
                    "entry_z":     ez, "exit_z": xz, "stop_z": sz, "regime": rm,
                })
                test_trades_all.append(trades)
            fold_log.append({
                "fold": fold,
                "train_end": train.index[-1],
                "test_end":  test.index[-1],
                "params":    best_params,
                "train_score": round(best_score, 3),
                "test_trades": ei.size,
                "test_pnl":  round(float(net.sum()) if ei.size > 0 else 0.0, 4),
            })
        start += step_bars
        fold += 1

    if test_trades_all:
        return pd.concat(test_trades_all, ignore_index=True), fold_log
    return pd.DataFrame(), fold_log


# ── L3: portfolio sizing + concurrent cap ────────────────────────────────────
def apply_concurrent_cap(df_trades: pd.DataFrame, max_concurrent: int) -> pd.DataFrame:
    """Greedy: keep trade only if open positions at entry < cap."""
    df = df_trades.sort_values("entry_time").reset_index(drop=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"]  = pd.to_datetime(df["exit_time"])

    open_ends = []  # exit_times of currently-kept open trades
    kept = np.zeros(len(df), dtype=bool)
    for i, row in df.iterrows():
        et = row["entry_time"]
        open_ends = [e for e in open_ends if e > et]
        if len(open_ends) < max_concurrent:
            kept[i] = True
            open_ends.append(row["exit_time"])
    return df[kept].reset_index(drop=True)


def apply_inv_vol_sizing(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Per-pair inverse-vol weight. Normalized so mean weight = 1."""
    df = df_trades.copy()
    pair_vol = df.groupby("pair")["net_pnl"].std().replace(0, np.nan)
    w = 1.0 / pair_vol
    w = w / w.mean()
    w = w.fillna(1.0).to_dict()
    df["size_mult"] = df["pair"].map(w)
    df["net_pnl_sized"] = df["net_pnl"] * df["size_mult"]
    return df


# ── L4: stress analysis on equity ─────────────────────────────────────────────
CRISIS_WINDOWS = [
    ("2022 inflation/rate-hike", "2022-01-01", "2022-10-31"),
    ("2023 banking crisis",      "2023-03-01", "2023-05-31"),
    ("2024 yen-carry unwind",    "2024-07-15", "2024-09-15"),
    ("2025 tariff shock",        "2025-02-01", "2025-05-31"),
]


def _safe_sharpe(pnl: pd.Series, days: int) -> float:
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    tpy = len(pnl) / (days / 365.25) if days > 0 else 0
    return float(pnl.mean() / pnl.std() * np.sqrt(tpy))


def stress_report(df: pd.DataFrame, tag: str):
    df = df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    pnl = df["net_pnl"]
    cum = pnl.cumsum()

    print(f"\n{'='*60}\nL4 STRESS  [{tag}]\n{'='*60}")

    # 1. Crisis periods
    print("\nCrisis-period P&L:")
    for name, s, e in CRISIS_WINDOWS:
        ts = pd.Timestamp(s, tz="US/Eastern")
        te = pd.Timestamp(e, tz="US/Eastern")
        m = (df["exit_time"] >= ts) & (df["exit_time"] <= te)
        sub = df[m]
        if len(sub) == 0:
            print(f"  {name:30s}  (no trades in window)")
            continue
        p = sub["net_pnl"]
        c = p.cumsum()
        dd = (c - c.cummax()).min()
        sh = _safe_sharpe(p, (te - ts).days)
        wr = (p > 0).mean() * 100
        print(f"  {name:30s}  trades={len(sub):4d}  PnL={p.sum():+8.2f}  "
              f"WR={wr:4.1f}%  Sharpe={sh:+.2f}  DD={dd:+.2f}")

    # 2. Worst rolling drawdowns
    print("\nWorst rolling-period equity drawdowns:")
    eq_daily = df.set_index("exit_time")["net_pnl"].resample("D").sum().cumsum()
    for win_days in [5, 10, 30, 60]:
        roll = eq_daily - eq_daily.rolling(win_days, min_periods=1).max()
        worst = roll.min()
        idx = roll.idxmin()
        print(f"  {win_days:3d}d:  {worst:+.2f}  ending {idx.date() if pd.notna(idx) else 'n/a'}")

    # 3. Block-bootstrap Sharpe CI (same annualization as headline)
    days_span = (df["exit_time"].iloc[-1] - df["exit_time"].iloc[0]).days or 1
    tpy = len(pnl) / (days_span / 365.25)
    print(f"\nBlock-bootstrap Sharpe CI (1000 resamples, block=20, tpy={tpy:.0f}):")
    n = len(pnl); block = 20; reps = 1000
    arr = pnl.to_numpy()
    rng = np.random.default_rng(42)
    sharpes = np.empty(reps)
    nblocks = max(1, n // block)
    for r in range(reps):
        starts = rng.integers(0, max(1, n - block), size=nblocks)
        sample = np.concatenate([arr[s:s+block] for s in starts])
        std = sample.std()
        sharpes[r] = (sample.mean() / std * np.sqrt(tpy)) if std > 0 else 0.0
    lo, med, hi = np.percentile(sharpes, [5, 50, 95])
    p_pos = (sharpes > 0).mean() * 100
    print(f"  median={med:.2f}  90% CI=[{lo:.2f}, {hi:.2f}]  P(Sharpe>0)={p_pos:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L0", action="store_true",
                    help="enable microstructure layer: VW-Z + velocity gate + thin-market gate")
    ap.add_argument("--L1", action="store_true", help="enable HMM regime filter")
    ap.add_argument("--L2", action="store_true", help="enable walk-forward optimization")
    ap.add_argument("--L3", action="store_true", help="apply concurrent cap + inv-vol sizing on result")
    ap.add_argument("--L4", action="store_true", help="run stress analysis on final equity")
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--wfo-years", type=int, default=5, help="L2: years of history")
    ap.add_argument("--wfo-train-mo", type=int, default=12, help="L2: train months")
    ap.add_argument("--wfo-test-mo",  type=int, default=3,  help="L2: test months")
    ap.add_argument("--invert-regime", action="store_true",
                    help="L1: skip CALM bars instead of VOLATILE (test inversion)")
    ap.add_argument("--clip-to-regime", action="store_true",
                    help="restrict backtest to regime-data window (fair compare)")
    # L0 sub-knobs (theoretical defaults; rarely need tuning)
    ap.add_argument("--no-vwz",      action="store_true", help="L0: disable VW-Z (keep plain rolling z)")
    ap.add_argument("--no-velocity", action="store_true", help="L0: disable velocity gate")
    ap.add_argument("--no-thin",     action="store_true", help="L0: disable thin-market gate")
    ap.add_argument("--vel-bars",    type=int,   default=5)
    ap.add_argument("--vel-thr",     type=float, default=2.0,
                    help="L0: directional velocity threshold (sign(z)·Δz over vel-bars)")
    ap.add_argument("--thin-frac",   type=float, default=0.5,
                    help="L0: block if bar dollar-volume < frac × rolling-60 median")
    ap.add_argument("--tag", default="", help="tag for output files")
    args = ap.parse_args()

    layers = []
    if args.L0: layers.append("L0-micro")
    if args.L1: layers.append("L1-regime")
    if args.L2: layers.append("L2-wfo")
    if args.L3: layers.append("L3-portfolio")
    if args.L4: layers.append("L4-stress")
    tag = args.tag or ("+".join(layers) if layers else "baseline")

    # L0 config from CLI
    if args.L0:
        l0_cfg = L0Config(
            use_vwz      = not args.no_vwz,
            use_velocity = not args.no_velocity,
            use_thin     = not args.no_thin,
            vel_bars     = args.vel_bars,
            vel_thr      = args.vel_thr,
            thin_frac    = args.thin_frac,
        )
    else:
        l0_cfg = L0_OFF

    bars_tail = BARS_PER_TRADING_DAY * 252 * args.wfo_years if args.L2 else RECENT_BARS
    closes  = load_closes(int(bars_tail))
    volumes = load_volumes(int(bars_tail)) if args.L0 else None
    pairs   = pd.read_csv(DATA_DIR / "pairs_selected.csv")
    regimes = load_regimes() if (args.L1 or args.L2 or args.clip_to_regime) else None

    if args.L0 and volumes is None:
        raise SystemExit(f"{VOLUMES_FILE} не найден — L0 требует volume-данные")
    if args.L1 and regimes is None:
        raise SystemExit("regimes.csv не найден — запусти step3a_hmm.py")

    if args.clip_to_regime and regimes is not None:
        start = regimes.index.min()
        closes = closes.loc[closes.index >= start]
        print(f"Clipped to regime window: from {start}")

    print(f"[{tag}] {closes.shape[0]} bars  range={closes.index[0].date()}→{closes.index[-1].date()}")
    print(f"Trading {len(pairs)} pairs")

    if args.L2:
        train_bars = BARS_PER_TRADING_DAY * 21 * args.wfo_train_mo
        test_bars  = BARS_PER_TRADING_DAY * 21 * args.wfo_test_mo
        step_bars  = test_bars
        print(f"WFO: train={args.wfo_train_mo}mo ({train_bars} bars) "
              f"test={args.wfo_test_mo}mo ({test_bars} bars)  step={args.wfo_test_mo}mo")

        all_test_trades = []
        all_folds = []
        for _, row in pairs.iterrows():
            t1, t2 = row["pair"].split("-")
            beta = row["beta"]; hl = row["half_life_bars"]
            if t1 not in closes.columns or t2 not in closes.columns:
                continue
            pc = closes[[t1, t2]].dropna()
            if pc.empty:
                continue
            df_sig = build_signals(pc, t1, t2, beta, hl,
                                   volumes=volumes, l0=l0_cfg)
            reg = regimes[row["pair"]] if (regimes is not None and row["pair"] in regimes.columns) else None
            trades, folds = walk_forward_pair(df_sig, t1, t2, beta, reg,
                                              int(train_bars), int(test_bars), int(step_bars),
                                              use_l0=args.L0)
            if not trades.empty:
                all_test_trades.append(trades)
            for f in folds:
                f["pair"] = row["pair"]
                all_folds.append(f)
            n_tr = len(trades) if not trades.empty else 0
            print(f"  {row['pair']:12s}  folds={len(folds):2d}  test_trades={n_tr:3d}")

        if not all_test_trades:
            raise SystemExit("WFO: no test trades")
        df = pd.concat(all_test_trades).sort_values("exit_time").reset_index(drop=True)
        pd.DataFrame(all_folds).to_csv(DATA_DIR / f"wfo_folds_{tag}.csv", index=False)

        if args.L3:
            n_before = len(df)
            df = apply_concurrent_cap(df, args.max_concurrent)
            print(f"L3: concurrent cap ≤{args.max_concurrent} kept {len(df)}/{n_before}")
            df = apply_inv_vol_sizing(df)
            df["net_pnl"] = df["net_pnl_sized"]
            print(f"L3: inv-vol sizing applied (size_mult range "
                  f"{df['size_mult'].min():.2f}–{df['size_mult'].max():.2f})")

        pnl = df["net_pnl"]
        days = (pd.to_datetime(df["exit_time"].iloc[-1])
                - pd.to_datetime(df["exit_time"].iloc[0])).days or 1
        tpy = len(df) / (days / 365.25)
        sh  = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
        wr  = (pnl > 0).mean() * 100
        win = df[pnl > 0]["net_pnl"].sum()
        los = abs(df[pnl <= 0]["net_pnl"].sum())
        pf  = win / los if los > 0 else float("inf")
        cum = pnl.cumsum()
        mdd = (cum - cum.cummax()).min()

        print(f"\n{'='*60}\n[{tag}] OOS PORTFOLIO (WFO)\n{'='*60}")
        print(f"Test trades:  {len(df)}  ({tpy:.0f}/yr)  span {days}d")
        print(f"Win rate:     {wr:.1f}%")
        print(f"Net P&L:      {pnl.sum():+.4f}")
        print(f"Profit fact:  {pf:.2f}")
        print(f"Max DD:       {mdd:.4f}")
        print(f"OOS Sharpe:   {sh:.2f}")
        df.to_csv(DATA_DIR / f"trades_{tag}.csv", index=False)
        print(f"Saved data/trades_{tag}.csv  &  wfo_folds_{tag}.csv")

        if args.L4:
            stress_report(df, tag)
        return

    pair_results = {}
    for _, row in pairs.iterrows():
        t1, t2 = row["pair"].split("-")
        beta   = row["beta"]
        hl     = row["half_life_bars"]
        if t1 not in closes.columns or t2 not in closes.columns:
            continue
        pc = closes[[t1, t2]].dropna()
        if pc.empty:
            continue
        df_sig = build_signals(pc, t1, t2, beta, hl,
                               volumes=volumes, l0=l0_cfg)

        regime_series = None
        if args.L1 and row["pair"] in regimes.columns:
            regime_series = regimes[row["pair"]]
            if args.invert_regime:
                regime_series = 1 - regime_series.fillna(0)

        trades = backtest_pair(df_sig, t1, t2, beta,
                               regime_series=regime_series,
                               use_regime=args.L1, use_l0=args.L0)
        if trades.empty:
            print(f"  {row['pair']:12s}  0 trades")
            continue
        pair_results[row["pair"]] = trades
        wr = (trades["net_pnl"] > 0).mean() * 100
        print(f"  {row['pair']:12s}  trades={len(trades):3d}  "
              f"WR={wr:4.1f}%  net P&L={trades['net_pnl'].sum():+.4f}")

    if not pair_results:
        raise SystemExit("No trades")

    df = pd.concat(pair_results.values()).sort_values("exit_time").reset_index(drop=True)
    pnl = df["net_pnl"]
    days = (pd.to_datetime(df["exit_time"].iloc[-1])
            - pd.to_datetime(df["exit_time"].iloc[0])).days or 1
    tpy = len(df) / (days / 365.25)
    sh  = pnl.mean() / pnl.std() * np.sqrt(tpy) if pnl.std() > 0 else 0.0
    wr  = (pnl > 0).mean() * 100
    win = df[pnl > 0]["net_pnl"].sum()
    los = abs(df[pnl <= 0]["net_pnl"].sum())
    pf  = win / los if los > 0 else float("inf")
    cum = pnl.cumsum()
    mdd = (cum - cum.cummax()).min()

    print(f"\n{'='*60}\n[{tag}] PORTFOLIO\n{'='*60}")
    print(f"Trades:       {len(df)}  ({tpy:.0f}/yr)")
    print(f"Win rate:     {wr:.1f}%")
    print(f"Net P&L:      {pnl.sum():+.4f}")
    print(f"Profit fact:  {pf:.2f}")
    print(f"Max DD:       {mdd:.4f}")
    print(f"Sharpe:       {sh:.2f}")

    out = DATA_DIR / f"trades_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
