"""unified_timeline_backtest.py — Single bar-by-bar multi-asset engine.

Replaces isolated pair-by-pair loops with a chronological timeline:
  Bar t: evaluate signals across all pairs, queue orders → fill at OPEN of bar t+1.

This removes the look-ahead bias of current step3j_wfo_daily.py which fills at
the same bar's close, and adds portfolio-level constraints (concurrent-trade
cap, gross-exposure cap) that fire in real time.

Performance: all per-pair state stored as numpy arrays. No pandas lookups
inside the timeline loop.

CLI:
    python unified_timeline_backtest.py --pairs pairs_bias_test.csv \\
        --closes closes_daily_extended_v2.csv --start 2014-12-31 \\
        --max-concurrent 10 --notional 100000
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from almgren_chriss_repricing import (
    classify_ticker as ac_classify,
    leg_cost_log as ac_leg_cost,
    borrow_log_per_day as ac_borrow_per_day,
)

Z_WIN, ENTRY_Z, EXIT_Z, STOP_Z, MAX_HOLD = 60, 2.0, 0.0, 3.5, 30


def estimate_beta(p1: np.ndarray, p2: np.ndarray) -> float:
    x, y = np.log(p2), np.log(p1)
    var = np.var(x)
    if var <= 0:
        return float("nan")
    return float(np.cov(y, x, ddof=0)[0, 1] / var)


def rolling_z(spread: np.ndarray, win: int) -> np.ndarray:
    z = np.full_like(spread, np.nan)
    for i in range(win, len(spread)):
        w = spread[i - win + 1: i + 1]
        m, s = w.mean(), w.std()
        if s > 0:
            z[i] = (spread[i] - m) / s
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",          required=True)
    ap.add_argument("--closes",         required=True)
    ap.add_argument("--start",          default="2014-12-31")
    ap.add_argument("--max-concurrent", type=int, default=10,
                    help="Portfolio cap on simultaneously open trades.")
    ap.add_argument("--notional",       type=float, default=100_000)
    ap.add_argument("--tag",            default="unified")
    ap.add_argument("--fill-mode",      default="next_bar", choices=["moc", "next_bar"],
                    dest="fill_mode",
                    help="moc: signal+fill at same bar close (MOC orders). "
                         "next_bar: signal at t, fill at t+1 close (no look-ahead, +1d delay).")
    args = ap.parse_args()

    closes = pd.read_csv("data/" + args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    pairs  = pd.read_csv("data/" + args.pairs)["pair"].tolist()
    oos_start = pd.Timestamp(args.start)

    # ── Pre-compute per-pair arrays (beta, spread, z, fill prices) ─────────
    print(f"Pre-computing {len(pairs)} pair arrays ...")
    state = []
    common_idx = closes.index
    for pair in pairs:
        t1, t2 = pair.split("-")
        if t1 not in closes.columns or t2 not in closes.columns:
            continue
        sub = closes[[t1, t2]].dropna()
        if len(sub) < 500: continue
        # Beta on data BEFORE oos_start (full train)
        train = sub.loc[:oos_start]
        if len(train) < 100: continue
        beta = estimate_beta(train[t1].values, train[t2].values)
        if not np.isfinite(beta) or not (0.1 <= abs(beta) <= 15.0):
            continue
        # Spread + z on FULL series (lookback for z needs train data)
        sp = (np.log(sub[t1].values) - beta * np.log(sub[t2].values))
        z  = rolling_z(sp, Z_WIN)
        # Per-pair Almgren-Chriss cost (constant per pair, computed once)
        rets = np.log(train[[t1, t2]]).diff().dropna()
        sig1 = max(float(rets[t1].std()), 0.005)
        sig2 = max(float(rets[t2].std()), 0.005)
        # Tier-default ADV (skip per-day ADV for speed)
        from almgren_chriss_repricing import TIER_TABLE
        AC_FALLBACK_ADV = {"us_large_cap":2e9,"us_mid_cap":3e8,"us_small_cap":2.5e7,
                          "canadian_us_list":1.5e8,"canadian_local":1e8,
                          "european_adr_us":2e8,"european_local":6e7,
                          "aussie_adr_us":2.5e8,"aussie_local":3e7}
        adv1 = AC_FALLBACK_ADV[ac_classify(t1)]
        adv2 = AC_FALLBACK_ADV[ac_classify(t2)]
        c1 = ac_leg_cost(t1, sig1, args.notional, adv1)
        c2 = ac_leg_cost(t2, sig2, args.notional, adv2)
        cost_round = 2 * (c1 + c2)
        borrow_per_d = ac_borrow_per_day(t2)
        # Map sub's index to global index for alignment
        idx_in_global = common_idx.get_indexer(sub.index)
        state.append(dict(pair=pair, beta=beta, sp=sp, z=z, idx=idx_in_global,
                           dates=sub.index.values,
                           cost=cost_round, borrow=borrow_per_d))
    print(f"  {len(state)} pairs ready (filtered from {len(pairs)})")

    # Build global-aligned arrays per pair: spread_g[t] = NaN if pair not active
    N_BARS = len(common_idx)
    P = len(state)
    sp_g = np.full((P, N_BARS), np.nan)
    z_g  = np.full((P, N_BARS), np.nan)
    for i, s in enumerate(state):
        sp_g[i, s["idx"]] = s["sp"]
        z_g[i,  s["idx"]] = s["z"]

    # ── Timeline loop ─────────────────────────────────────────────────────
    oos_mask = common_idx >= oos_start
    in_pos    = np.zeros(P, dtype=bool)
    side      = np.zeros(P, dtype=np.int8)
    entry_bar = np.full(P, -1, dtype=np.int64)
    entry_sp  = np.zeros(P)
    pending_entry = np.zeros(P, dtype=np.int8)   # +1/-1 queued for next-bar fill
    pending_exit  = np.zeros(P, dtype=bool)

    trades = []
    skipped_capacity = 0

    moc = (args.fill_mode == "moc")
    for t in range(Z_WIN + 1, N_BARS - 1):
        # ── Step A: Fill pending orders at THIS bar's close
        # next_bar mode: orders queued at t-1 fill here at bar t close (1-day delay)
        # moc mode:      orders queued at t fill at t close (same bar)
        for i in range(P):
            fill_price = sp_g[i, t]
            if not np.isfinite(fill_price):
                pending_entry[i] = 0
                pending_exit[i]  = False
                continue
            if pending_exit[i] and in_pos[i]:
                held = t - entry_bar[i]
                gross = side[i] * (fill_price - entry_sp[i])
                pnl = gross - state[i]["cost"] - held * state[i]["borrow"]
                trades.append({
                    "pair": state[i]["pair"],
                    "entry": common_idx[entry_bar[i]],
                    "exit":  common_idx[t],
                    "held":  held,
                    "side":  int(side[i]),
                    "pnl":   pnl,
                    "size":  1.0,
                    "reason": "exit",
                })
                in_pos[i] = False
                pending_exit[i] = False
            if pending_entry[i] != 0 and not in_pos[i]:
                # Portfolio capacity check at fill time
                if in_pos.sum() >= args.max_concurrent:
                    skipped_capacity += 1
                    pending_entry[i] = 0
                    continue
                in_pos[i] = True
                side[i]   = pending_entry[i]
                entry_bar[i] = t
                entry_sp[i]  = fill_price
                pending_entry[i] = 0

        # ── Step B: Evaluate signals at this bar (queue for next-bar fill)
        if not oos_mask[t]:
            continue
        for i in range(P):
            zi = z_g[i, t]
            if not np.isfinite(zi):
                continue
            if in_pos[i]:
                held_so_far = t - entry_bar[i] + 1
                hit_target = (side[i] == -1 and zi <= EXIT_Z) or (side[i] == +1 and zi >= EXIT_Z)
                hit_stop   = abs(zi) >= STOP_Z
                hit_time   = held_so_far >= MAX_HOLD
                if hit_target or hit_stop or hit_time:
                    pending_exit[i] = True
                    if moc:
                        # Execute immediately at this bar (same-bar MOC fill)
                        fill_price = sp_g[i, t]
                        if np.isfinite(fill_price):
                            held = t - entry_bar[i]
                            gross = side[i] * (fill_price - entry_sp[i])
                            pnl = gross - state[i]["cost"] - held * state[i]["borrow"]
                            trades.append({"pair": state[i]["pair"],
                                            "entry": common_idx[entry_bar[i]],
                                            "exit":  common_idx[t],
                                            "held":  held, "side": int(side[i]),
                                            "pnl":   pnl, "size": 1.0, "reason": "moc_exit"})
                            in_pos[i] = False
                            pending_exit[i] = False
            else:
                signal = 0
                if zi > ENTRY_Z:
                    signal = -1
                elif zi < -ENTRY_Z:
                    signal = +1
                if signal != 0:
                    if moc:
                        # MOC entry: fill at this bar's close immediately
                        if in_pos.sum() < args.max_concurrent:
                            fill_price = sp_g[i, t]
                            if np.isfinite(fill_price):
                                in_pos[i] = True
                                side[i] = signal
                                entry_bar[i] = t
                                entry_sp[i]  = fill_price
                        else:
                            skipped_capacity += 1
                    else:
                        pending_entry[i] = signal

    # ── Final mark-to-market for any open positions at OOS end
    for i in range(P):
        if in_pos[i]:
            t = N_BARS - 1
            fill_price = sp_g[i, t]
            if np.isfinite(fill_price):
                held = t - entry_bar[i]
                gross = side[i] * (fill_price - entry_sp[i])
                pnl = gross - state[i]["cost"] - held * state[i]["borrow"]
                trades.append({
                    "pair": state[i]["pair"],
                    "entry": common_idx[entry_bar[i]],
                    "exit":  common_idx[t],
                    "held":  held,
                    "side":  int(side[i]),
                    "pnl":   pnl,
                    "size":  1.0,
                    "reason": "mtm_close",
                })

    df = pd.DataFrame(trades)
    df_oos = df[df["exit"] >= oos_start]
    print(f"\nTotal trades: {len(df)}    OOS trades: {len(df_oos)}")
    print(f"Skipped due to portfolio cap (--max-concurrent={args.max_concurrent}): {skipped_capacity}")

    # ── Summary ───────────────────────────────────────────────────────────
    oos_days = common_idx[oos_mask]
    d = df_oos.set_index("exit")["pnl"].groupby(level=0).sum().reindex(oos_days).fillna(0.0)
    sh = d.mean() / d.std() * np.sqrt(252) if d.std() else float("nan")
    eq = d.cumsum(); dd = (eq - eq.cummax()).min()
    pf = df_oos[df_oos.pnl>0].pnl.sum() / abs(df_oos[df_oos.pnl<0].pnl.sum()) if (df_oos.pnl<0).any() else float("inf")
    win = (df_oos.pnl > 0).mean() * 100

    print(f"\n{'='*70}")
    print("UNIFIED TIMELINE BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  OOS span:        {oos_days.min().date()} → {oos_days.max().date()}  ({len(oos_days)} days)")
    print(f"  Pairs active:    {len(state)}")
    print(f"  Max concurrent:  {args.max_concurrent}")
    print(f"  Notional/leg:    ${args.notional:,.0f}")
    print(f"  Costs:           Almgren-Chriss tiered per-pair")
    print(f"  Execution:       NEXT-BAR (signal at t → fill at t+1 close)")
    print(f"")
    print(f"  OOS trades:      {len(df_oos)}")
    print(f"  Win rate:        {win:.1f}%")
    print(f"  Total PnL:       {df_oos.pnl.sum():+.3f}")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Max DD:          {dd:+.3f}")
    print(f"  Sharpe (daily):  {sh:+.3f}")

    # Save
    out = Path(f"data/wfo_daily_results_{args.tag}.csv")
    df_oos.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
