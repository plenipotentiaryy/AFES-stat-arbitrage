"""Quick daily-horizon mean-reversion sanity check on baseline 14 pairs.

No MetaGate, no HMM, no Hurst, no SBR — just rolling-z trades on daily spreads
with an expanding-window WFO. Goal: does daily mean-reversion have any edge
on this universe? If raw Sharpe ≤ 0.2, changing horizon won't save the system.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
PAIRS_CSV = DATA / "pairs_baseline14.csv"
CLOSES_CSV = DATA / "closes_daily.csv"

Z_WIN     = 60       # rolling z-score window (days)
ENTRY_Z   = 2.0
EXIT_Z    = 0.0
STOP_Z    = 3.5
MAX_HOLD  = 30       # days
TRAIN_YRS = 4
OOS_YRS   = 1
STEP_MO   = 3        # quarterly walk-forward
COST_BPS  = 10.0     # per-leg round-trip cost; set 0 for gross


def load_data():
    df = pd.read_csv(CLOSES_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
    pairs = pd.read_csv(PAIRS_CSV)["pair"].tolist()
    return df, pairs


def estimate_beta(p1: pd.Series, p2: pd.Series) -> float:
    x = np.log(p2.values)
    y = np.log(p1.values)
    return float(np.cov(y, x, ddof=0)[0, 1] / np.var(x))


def spread_series(prices: pd.DataFrame, t1: str, t2: str, beta: float) -> pd.Series:
    s = np.log(prices[t1]) - beta * np.log(prices[t2])
    return s.dropna()


def backtest_pair_oos(spread_full: pd.Series, oos_start, oos_end,
                       cost_per_trade: float):
    """Trade z>ENTRY → short spread (bet on mean revert), z<-ENTRY → long.
    Exit on |z| ≤ EXIT_Z or |z| ≥ STOP_Z or MAX_HOLD bars. Open only in OOS."""
    mu  = spread_full.rolling(Z_WIN).mean()
    sig = spread_full.rolling(Z_WIN).std()
    z   = (spread_full - mu) / sig

    trades = []
    in_pos = False
    side = 0           # +1 long-spread, -1 short-spread
    entry_idx = None
    entry_spread = 0.0

    idx = spread_full.index
    oos_mask = (idx >= oos_start) & (idx < oos_end)

    for i, t in enumerate(idx):
        if pd.isna(z.iloc[i]):
            continue
        zi = z.iloc[i]
        if not in_pos:
            if not oos_mask[i]:
                continue
            if zi > ENTRY_Z:
                side, in_pos = -1, True
                entry_idx, entry_spread = i, spread_full.iloc[i]
            elif zi < -ENTRY_Z:
                side, in_pos = +1, True
                entry_idx, entry_spread = i, spread_full.iloc[i]
        else:
            held = i - entry_idx
            hit_target = (side == -1 and zi <= EXIT_Z) or (side == +1 and zi >= EXIT_Z)
            hit_stop   = abs(zi) >= STOP_Z
            timeout    = held >= MAX_HOLD
            if hit_target or hit_stop or timeout:
                exit_spread = spread_full.iloc[i]
                pnl = side * (exit_spread - entry_spread) - cost_per_trade
                trades.append({
                    "entry": idx[entry_idx], "exit": t, "held": held,
                    "side": side, "pnl": pnl,
                    "reason": "target" if hit_target else ("stop" if hit_stop else "time"),
                })
                in_pos = False

    return pd.DataFrame(trades)


def run_wfo(df: pd.DataFrame, pairs: list[str], cost: float):
    all_trades = []
    start = df.index.min() + pd.DateOffset(years=TRAIN_YRS)
    end   = df.index.max()
    cursor = start
    while cursor + pd.DateOffset(years=OOS_YRS) <= end:
        oos_start = cursor
        oos_end   = cursor + pd.DateOffset(years=OOS_YRS)
        train_start = oos_start - pd.DateOffset(years=TRAIN_YRS)
        for pair in pairs:
            t1, t2 = pair.split("-")
            if t1 not in df.columns or t2 not in df.columns:
                continue
            train = df.loc[train_start:oos_start, [t1, t2]].dropna()
            if len(train) < 250:
                continue
            beta = estimate_beta(train[t1], train[t2])
            full = df.loc[train_start:oos_end, [t1, t2]].dropna()
            if len(full) < Z_WIN + 10:
                continue
            sp = spread_series(full, t1, t2, beta)
            tr = backtest_pair_oos(sp, oos_start, oos_end, cost)
            if not tr.empty:
                tr["pair"] = pair
                tr["window_oos_start"] = oos_start
                all_trades.append(tr)
        cursor += pd.DateOffset(months=STEP_MO)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def summarise(tr: pd.DataFrame, label: str):
    if tr.empty:
        print(f"[{label}] NO TRADES")
        return
    pnl = tr["pnl"]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    # daily portfolio chain via group by exit date
    daily = tr.groupby(tr["exit"].dt.date)["pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  Trades:        {len(tr)}")
    print(f"  Win rate:      {(pnl > 0).mean()*100:.1f}%")
    print(f"  Total PnL:     {pnl.sum():+.4f}  (log-spread units)")
    print(f"  Profit factor: {wins/losses if losses>0 else float('inf'):.2f}")
    print(f"  Max DD:        {dd:+.4f}")
    print(f"  Sharpe (daily):{sharpe:+.3f}")
    print(f"  Exit reasons:  {tr['reason'].value_counts().to_dict()}")
    by_pair = tr.groupby("pair")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
    print("\n  Per-pair PnL:")
    print(by_pair.to_string())


def main():
    df, pairs = load_data()
    print(f"Loaded {len(df)} days × {df.shape[1]} tickers; {len(pairs)} pairs")
    print(f"WFO: TRAIN={TRAIN_YRS}y, OOS={OOS_YRS}y, step={STEP_MO}mo, z_win={Z_WIN}d")

    tr_gross = run_wfo(df, pairs, cost=0.0)
    summarise(tr_gross, f"GROSS  (cost=0 bps)")

    # cost_per_trade in log-spread units ≈ 2 * COST_BPS / 1e4 (two legs)
    cost_log = 2 * COST_BPS / 1e4
    tr_net = run_wfo(df, pairs, cost=cost_log)
    summarise(tr_net, f"NET    (cost={COST_BPS} bps/leg)")


if __name__ == "__main__":
    main()
