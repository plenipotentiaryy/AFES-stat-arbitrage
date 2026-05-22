"""
step4d_pair_grid.py — Per-pair parameter optimisation (train) + validation (test).

Grid (18 combos per pair):
  entry_z : 1.65 / 1.7 / 1.8
  exit_z  : -0.1 / 0.0
  stop_z  : 3.0 / 3.2 / 3.5

Stop ceiling 4.0: beyond Z=4 cointegration is likely broken (0.007% prob of random fluctuation).

Workflow:
  1. Run grid on TRAINING data only  → find best params per pair (by Sharpe)
  2. Validate those params on TEST data (OOS check)
  3. Save per-pair optimal params → data/optimal_params.csv
     (step4_backtest.py reads this and uses per-pair settings)
  4. Heatmaps + equity curves

Why train/test split here:
  We optimise on the first ~55 % of the data and validate on the last 45 %.
  This avoids in-sample overfitting while still customising per pair.
"""

import itertools
import sys
import pandas as pd
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    CLOSES_FILE, COST_TAKER, BORROW_RATE_ANNUAL,
    RTH_START, RTH_END, SIGNAL_START, TRAIN_RATIO,
    BARS_PER_DAY, DATA_DIR, OUTPUT_DIR,
)

ENTRY_Z_GRID = [1.65, 1.7, 1.8]
STOP_Z_GRID  = [3.0, 3.2, 3.5]
EXIT_Z_GRID  = [-0.1, 0.0]
COMBOS       = list(itertools.product(ENTRY_Z_GRID, EXIT_Z_GRID, STOP_Z_GRID))
N_COMBOS     = len(COMBOS)
MIN_TRADES   = 8


# ── Numba JIT grid kernel ─────────────────────────────────────────────────────
# Runs ALL combos in a single pass over the bars.
# Zero-Pandas rule: only float64 / int64 NumPy arrays inside.
#
# results[c] = [total_pnl, n_trades, n_wins, sum_pnl_sq, max_dd, n_stops]

@njit(cache=True)
def _grid_kernel(zscore:    np.ndarray,   # float64[n_bars]
                 spread:    np.ndarray,   # float64[n_bars]
                 t1_price:  np.ndarray,   # float64[n_bars]
                 t2_price:  np.ndarray,   # float64[n_bars]
                 entry_arr: np.ndarray,   # float64[n_combos]
                 exit_arr:  np.ndarray,   # float64[n_combos]
                 stop_arr:  np.ndarray,   # float64[n_combos]
                 beta:           float,
                 cost_taker:     float,
                 borrow_rate:    float,
                 bars_per_day:   float) -> np.ndarray:

    n_bars   = len(zscore)
    n_combos = len(entry_arr)

    # Per-combo mutable state (pre-allocated, no Python lists)
    pos    = np.zeros(n_combos, dtype=np.int64)
    e_sp   = np.zeros(n_combos)
    e_t1   = np.zeros(n_combos)
    e_t2   = np.zeros(n_combos)
    e_bar  = np.zeros(n_combos, dtype=np.int64)

    # Results matrix — 6 stats per combo
    results  = np.zeros((n_combos, 6))
    cum_pnl  = np.zeros(n_combos)
    peak_pnl = np.zeros(n_combos)

    for i in range(n_bars):
        z  = zscore[i]
        s  = spread[i]
        p1 = t1_price[i]
        p2 = t2_price[i]

        for c in range(n_combos):
            ez = entry_arr[c]
            xz = exit_arr[c]
            sz = stop_arr[c]
            pc = pos[c]

            # ── Exit / stop ───────────────────────────────────────────────
            if pc != 0:
                is_exit = (pc == 1 and z >= xz) or (pc == -1 and z <= -xz)
                is_stop = (pc == 1 and z <= -sz) or (pc == -1 and z >= sz)

                if is_exit or is_stop:
                    gross    = pc * (s - e_sp[c])
                    notional = e_t1[c] + beta * e_t2[c]
                    tx       = 2.0 * notional * cost_taker
                    hold_d   = (i - e_bar[c]) / bars_per_day
                    short_n  = (beta * e_t2[c]) if pc == 1 else e_t1[c]
                    borrow   = short_n * borrow_rate * hold_d / 252.0
                    net      = gross - tx - borrow

                    results[c, 0] += net          # total_pnl
                    results[c, 1] += 1.0          # n_trades
                    if net > 0.0:
                        results[c, 2] += 1.0      # n_wins
                    results[c, 3] += net * net    # sum_pnl_sq  (for Sharpe std)
                    if is_stop:
                        results[c, 5] += 1.0      # n_stops

                    cum_pnl[c] += net
                    if cum_pnl[c] > peak_pnl[c]:
                        peak_pnl[c] = cum_pnl[c]
                    dd = cum_pnl[c] - peak_pnl[c]
                    if dd < results[c, 4]:
                        results[c, 4] = dd        # max_dd (negative number)

                    pos[c] = 0

            # ── Entry ─────────────────────────────────────────────────────
            if pos[c] == 0:
                if z < -ez:
                    pos[c]  = 1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i
                elif z > ez:
                    pos[c]  = -1
                    e_sp[c] = s; e_t1[c] = p1; e_t2[c] = p2; e_bar[c] = i

    return results


# ── Python wrapper ────────────────────────────────────────────────────────────

def run_grid_numba(df: pd.DataFrame, t1: str, t2: str,
                   beta: float, combos: list, days: float,
                   pair: str = "") -> pd.DataFrame:
    """
    Run full grid search via the Numba kernel in one pass.

    Parameters
    ----------
    df     : DataFrame with columns zscore, spread, {t1}_close, {t2}_close
    combos : list of (entry_z, exit_z, stop_z) — invalid ones are filtered
    days   : number of calendar days in the period (for annualised Sharpe)

    Returns
    -------
    DataFrame with one row per valid combo, sorted by Sharpe descending.
    """
    # Filter invalid combos (stop must be > entry, exit must be < entry)
    valid = [(ez, xz, sz) for ez, xz, sz in combos if sz > ez and xz < ez]
    if not valid:
        return pd.DataFrame()

    # Pandas → contiguous float64 NumPy arrays (zero-copy where possible)
    zscore   = np.ascontiguousarray(df["zscore"].to_numpy(np.float64))
    spread   = np.ascontiguousarray(df["spread"].to_numpy(np.float64))
    t1_price = np.ascontiguousarray(df[f"{t1}_close"].to_numpy(np.float64))
    t2_price = np.ascontiguousarray(df[f"{t2}_close"].to_numpy(np.float64))

    entry_arr = np.array([c[0] for c in valid], dtype=np.float64)
    exit_arr  = np.array([c[1] for c in valid], dtype=np.float64)
    stop_arr  = np.array([c[2] for c in valid], dtype=np.float64)

    results = _grid_kernel(zscore, spread, t1_price, t2_price,
                           entry_arr, exit_arr, stop_arr,
                           float(beta),
                           float(COST_TAKER),
                           float(BORROW_RATE_ANNUAL),
                           float(BARS_PER_DAY))

    # Convert results matrix → DataFrame
    years = max(days / 365.25, 1e-9)
    rows  = []
    for i, (ez, xz, sz) in enumerate(valid):
        n_trades = int(results[i, 1])
        if n_trades < MIN_TRADES:
            continue

        total_pnl = results[i, 0]
        n_wins    = int(results[i, 2])
        sum_pnl2  = results[i, 3]
        max_dd    = results[i, 4]
        n_stops   = int(results[i, 5])

        mean_pnl  = total_pnl / n_trades
        var_pnl   = max(sum_pnl2 / n_trades - mean_pnl ** 2, 0.0)
        std_pnl   = np.sqrt(var_pnl)
        tpy       = n_trades / years
        sharpe    = mean_pnl / std_pnl * np.sqrt(tpy) if std_pnl > 0.0 else 0.0

        rows.append({
            "pair":      pair,
            "entry_z":   ez,
            "exit_z":    xz,
            "stop_z":    sz,
            "trades":    n_trades,
            "win_rate":  n_wins / n_trades * 100.0,
            "sharpe":    sharpe,
            "total_pnl": total_pnl,
            "avg_pnl":   mean_pnl,
            "max_dd":    max_dd,
            "stop_rate": n_stops / n_trades * 100.0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_closes_split() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Returns (train_closes, test_closes, days_train)."""
    path = DATA_DIR / CLOSES_FILE
    if not path.exists():
        path = DATA_DIR / "closes_15min.csv"
    closes = pd.read_csv(path, index_col=0, parse_dates=True)
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert("US/Eastern")
    closes = closes.between_time(RTH_START, RTH_END)

    pairs_path = DATA_DIR / "pairs_selected.csv"
    if pairs_path.exists():
        meta = pd.read_csv(pairs_path)
        if not meta.empty:
            needed    = {t for p in meta["pair"] for t in p.split("-")}
            available = [t for t in needed if t in closes.columns]
            # Keep ALL rows — each pair drops its own NaN in build_signals
            closes = closes[available]

            if "test_start_date" in meta.columns:
                test_start = pd.Timestamp(meta["test_start_date"].iloc[0]).tz_localize("US/Eastern")
                train = closes[closes.index < test_start]
                test  = closes[closes.index >= test_start]
                if len(train) > 200 and len(test) > 200:
                    # days_train = full calendar span of training data
                    days_train = (train.index[-1] - train.index[0]).days
                    return train, test, float(days_train)

    # Fallback: split by TRAIN_RATIO, no NaN-dropping across all pairs
    n     = len(closes)
    split = int(n * TRAIN_RATIO)
    train = closes.iloc[:split]
    test  = closes.iloc[split:]
    return train, test, float((train.index[-1] - train.index[0]).days)


def build_signals(closes: pd.DataFrame, t1: str, t2: str,
                  beta: float, half_life: float) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread":      spread,
        "zscore":      zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, t1: str, t2: str, beta: float,
             entry_z: float, exit_z: float, stop_z: float) -> list[dict]:
    t1c, t2c = f"{t1}_close", f"{t2}_close"
    pos = es = et1 = et2 = 0.0
    ebar = 0
    trades = []

    for i in range(len(df)):
        z  = df["zscore"].iloc[i]
        s  = df["spread"].iloc[i]
        p1 = df[t1c].iloc[i]
        p2 = df[t2c].iloc[i]

        if pos != 0:
            ex = (pos == 1 and z >= exit_z) or (pos == -1 and z <= -exit_z)
            st = (pos == 1 and z <= -stop_z) or (pos == -1 and z >= stop_z)
            if ex or st:
                gross    = pos * (s - es)
                notional = et1 + beta * et2
                tx       = 2 * notional * COST_TAKER
                hold_d   = (i - ebar) / BARS_PER_DAY
                borrow   = (beta * et2 if pos == 1 else et1) * BORROW_RATE_ANNUAL * hold_d / 252
                trades.append({
                    "net_pnl":      gross - tx - borrow,
                    "gross_pnl":    gross,
                    "exit_reason":  "STOP" if st else "SIGNAL",
                    "holding_bars": i - ebar,
                })
                pos = 0

        if pos == 0:
            if z < -entry_z:   pos = 1
            elif z > entry_z:  pos = -1
            if pos != 0:
                es = s; et1 = p1; et2 = p2; ebar = i

    return trades


def calc_metrics(trades: list[dict], days: float) -> dict | None:
    if len(trades) < MIN_TRADES:
        return None
    pnl   = np.array([t["net_pnl"] for t in trades])
    curve = np.cumsum(pnl)
    wr    = float((pnl > 0).mean() * 100)
    tpy   = len(pnl) / max(days / 365.25, 0.01)
    sh    = float(pnl.mean() / pnl.std() * np.sqrt(tpy)) if pnl.std() > 0 else 0.0
    dd    = float((curve - np.maximum.accumulate(curve)).min())
    wins  = pnl[pnl > 0].sum()
    loss  = abs(pnl[pnl <= 0].sum())
    pf    = float(wins / loss) if loss > 0 else float("inf")
    stops = sum(1 for t in trades if t["exit_reason"] == "STOP")
    return {
        "trades":    len(pnl),
        "win_rate":  wr,
        "sharpe":    sh,
        "total_pnl": float(pnl.sum()),
        "avg_pnl":   float(pnl.mean()),
        "max_dd":    dd,
        "pf":        pf,
        "stops":     stops,
        "stop_rate": stops / len(pnl) * 100,
        "curve":     curve,
    }


# ── Load data ─────────────────────────────────────────────────────────────────

closes_train, closes_test, days_train = load_closes_split()
days_test = (closes_test.index[-1] - closes_test.index[0]).days
pairs = pd.read_csv(DATA_DIR / "pairs_selected.csv")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"Per-pair grid  |  {N_COMBOS} combos  |  optimise on TRAIN → validate on TEST")
print(f"TRAIN: {closes_train.index[0].date()} → {closes_train.index[-1].date()}  "
      f"({days_train:.0f} days)")
print(f"TEST:  {closes_test.index[0].date()}  → {closes_test.index[-1].date()}   "
      f"({days_test} days)")
print(f"Entry : {ENTRY_Z_GRID}")
print(f"Exit  : {EXIT_Z_GRID}")
print(f"Stop  : {STOP_Z_GRID}\n")

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Per-pair grid search ──────────────────────────────────────────────────────

all_best: list[dict] = []
all_pair_train_results: dict = {}

for _, row in pairs.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    beta      = float(row.get("beta_daily", row["beta"]) or row["beta"])
    half_life = float(row["half_life_bars"])

    if t1 not in closes_train.columns or t2 not in closes_train.columns:
        print(f"  SKIP {pair_name}: missing data")
        continue

    # ── Build signals for both splits ────────────────────────────────────────
    sig_train = build_signals(closes_train, t1, t2, beta, half_life)
    sig_test  = build_signals(closes_test,  t1, t2, beta, half_life)

    if len(sig_train) < 200:
        print(f"  SKIP {pair_name}: too few train bars ({len(sig_train)})")
        continue

    print(f"  {pair_name}  train={len(sig_train)} bars  test={len(sig_test)} bars ...",
          end="", flush=True)

    # ── Grid search on TRAIN — single Numba pass over all combos ─────────────
    df_train = run_grid_numba(sig_train, t1, t2, beta, COMBOS, days_train, pair_name)

    if df_train.empty:
        print("  no valid train combos")
        continue

    all_pair_train_results[pair_name] = df_train

    # ── Parameter Plateau: pick robust point, not argmax ─────────────────────
    # For each combo, score = Sharpe × min_ratio_of_neighbors.
    # A pair (e, x, s) with neighbors having Sharpe 0.9× of own = robust plateau.
    # A pair whose neighbors have 0.2× = isolated peak = overfit.
    plateau_scores = []
    g = df_train.set_index(["entry_z", "exit_z", "stop_z"])["sharpe"].to_dict()
    ez_grid = sorted(df_train["entry_z"].unique())
    xz_grid = sorted(df_train["exit_z"].unique())
    sz_grid = sorted(df_train["stop_z"].unique())
    def _next(grid, v, step):
        idx = grid.index(v) + step
        return grid[idx] if 0 <= idx < len(grid) else None
    for _, r in df_train.iterrows():
        own = r["sharpe"]
        if own <= 0:
            plateau_scores.append(-1e9); continue
        neighbors = []
        for de in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                for ds in [-1, 0, 1]:
                    if de == dx == ds == 0: continue
                    ez = _next(ez_grid, r["entry_z"], de)
                    xz = _next(xz_grid, r["exit_z"],  dx)
                    sz = _next(sz_grid, r["stop_z"],  ds)
                    if ez is None or xz is None or sz is None: continue
                    n_sh = g.get((ez, xz, sz))
                    if n_sh is not None: neighbors.append(n_sh)
        if not neighbors:
            plateau_scores.append(own); continue
        # Plateau score: own Sharpe weighted by neighborhood stability.
        # Penalty for cliff (any neighbor < 50% of own) is harsh.
        n_arr = np.array(neighbors)
        stability = (n_arr >= 0.5 * own).mean()   # fraction of neighbors near own
        median_neigh = np.median(n_arr)
        score = own * (0.5 + 0.5 * stability) * (median_neigh / max(own, 0.01))
        plateau_scores.append(score)
    df_train["plateau_score"] = plateau_scores
    best_train = df_train.loc[df_train["plateau_score"].idxmax()]

    # ── Validate best params on TEST — single-combo Numba pass ───────────────
    best_combo = [(float(best_train["entry_z"]),
                   float(best_train["exit_z"]),
                   float(best_train["stop_z"]))]
    df_test_best = run_grid_numba(sig_test, t1, t2, beta, best_combo, days_test)
    if df_test_best.empty:
        test_m = {"trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                  "total_pnl": 0.0, "max_dd": 0.0, "stop_rate": 0.0}
    else:
        test_m = df_test_best.iloc[0].to_dict()

    argmax_train = df_train.iloc[df_train["sharpe"].argmax()]
    print(f"  TRAIN plateau → entry={best_train['entry_z']}  "
          f"exit={best_train['exit_z']:+.1f}  stop={best_train['stop_z']}  "
          f"Sh={best_train['sharpe']:.2f}  (argmax was Sh={argmax_train['sharpe']:.2f})  "
          f"│  TEST Sh={test_m['sharpe']:.2f}  WR={test_m['win_rate']:.0f}%")

    all_best.append({
        "pair":            pair_name,
        "entry_z":         float(best_train["entry_z"]),
        "exit_z":          float(best_train["exit_z"]),
        "stop_z":          float(best_train["stop_z"]),
        # train metrics
        "train_sharpe":    round(float(best_train["sharpe"]), 2),
        "train_wr":        round(float(best_train["win_rate"]), 1),
        "train_trades":    int(best_train["trades"]),
        "train_pnl":       round(float(best_train["total_pnl"]), 4),
        # test (OOS) metrics
        "test_sharpe":     round(float(test_m["sharpe"]), 2),
        "test_wr":         round(float(test_m["win_rate"]), 1),
        "test_trades":     int(test_m["trades"]),
        "test_pnl":        round(float(test_m["total_pnl"]), 4),
    })

print()
if not all_best:
    raise SystemExit("No valid results found.")

df_best = pd.DataFrame(all_best)

# ── Summary table ─────────────────────────────────────────────────────────────

print("=" * 110)
print(f"{'OPTIMAL PARAMS PER PAIR':^110}")
print("=" * 110)
print(f"{'Pair':<10} {'Entry':>6} {'Exit':>6} {'Stop':>6}  "
      f"{'── TRAIN ──':^30}  {'── TEST (OOS) ──':^30}")
print(f"{'':>30}  {'Sh':>8} {'WR':>6} {'Tr':>5} {'P&L':>10}  "
      f"{'Sh':>8} {'WR':>6} {'Tr':>5} {'P&L':>10}")
print("-" * 110)
for _, r in df_best.iterrows():
    oos_flag = " ✓" if r["test_sharpe"] > 0 else " ✗"
    print(f"{r['pair']:<10} {r['entry_z']:>6.1f} {r['exit_z']:>+6.1f} {r['stop_z']:>6.2f}  "
          f"{r['train_sharpe']:>8.2f} {r['train_wr']:>5.1f}% {r['train_trades']:>5} "
          f"{r['train_pnl']:>+10.4f}  "
          f"{r['test_sharpe']:>8.2f} {r['test_wr']:>5.1f}% {r['test_trades']:>5} "
          f"{r['test_pnl']:>+10.4f}{oos_flag}")
print("=" * 110)

save_cols = ["pair", "entry_z", "exit_z", "stop_z",
             "train_sharpe", "train_wr", "train_trades", "train_pnl",
             "test_sharpe", "test_wr", "test_trades", "test_pnl"]
df_best[save_cols].to_csv(DATA_DIR / "optimal_params.csv", index=False)
print(f"\nSaved optimal params → data/optimal_params.csv")
print("step4_backtest.py will use these per-pair parameters automatically.\n")

# ── Per-pair ranked tables (train) ────────────────────────────────────────────
print()
for pair_name, df_t in all_pair_train_results.items():
    top = df_t.head(10)
    print(f"\n{'─'*75}  {pair_name}  top-10 TRAIN combos")
    print(f"{'#':>3} {'entry':>6} {'exit':>6} {'stop':>6} "
          f"{'trades':>7} {'WR':>6} {'Sharpe':>8} {'P&L':>10} {'Stop%':>6}")
    for rank, (_, r) in enumerate(top.iterrows()):
        m = " ◄" if rank == 0 else ""
        print(f"{rank+1:>3} {r['entry_z']:>6.1f} {r['exit_z']:>+6.1f} {r['stop_z']:>6.2f} "
              f"{r['trades']:>7.0f} {r['win_rate']:>5.1f}% {r['sharpe']:>8.2f} "
              f"{r['total_pnl']:>+10.4f} {r['stop_rate']:>5.1f}%{m}")

# ── Per-pair heatmaps (train) ─────────────────────────────────────────────────
for pair_name, df_t in all_pair_train_results.items():
    n_stop = len(STOP_Z_GRID)
    fig, axes = plt.subplots(1, n_stop, figsize=(5 * n_stop, 5))
    if n_stop == 1:
        axes = [axes]

    vmin, vmax = df_t["sharpe"].min(), df_t["sharpe"].max()

    for ax, stop_z in zip(axes, STOP_Z_GRID):
        sub = df_t[df_t["stop_z"] == stop_z]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="entry_z", columns="exit_z", values="sharpe")
        pivot = pivot.reindex(index=sorted(pivot.index, reverse=True),
                               columns=sorted(pivot.columns, reverse=True))

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:+.1f}" for v in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=9)
        ax.set_xlabel("exit_z")
        ax.set_ylabel("entry_z")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    c = "white" if abs(val) > 0.5 * max(abs(vmin), abs(vmax)) else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=c)
        ax.set_title(f"stop={stop_z}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"{pair_name}  TRAIN Sharpe grid (entry × exit per stop)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / f"pair_grid_{pair_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Heatmap → {out}")

# ── Top-5 test equity curves per pair ────────────────────────────────────────
for _, row in df_best.iterrows():
    pair_name = row["pair"]
    t1, t2    = pair_name.split("-")
    beta      = float(pairs.loc[pairs["pair"] == pair_name, "beta_daily"].fillna(
                      pairs.loc[pairs["pair"] == pair_name, "beta"]).iloc[0])
    hl        = float(pairs.loc[pairs["pair"] == pair_name, "half_life_bars"].iloc[0])

    sig_test  = build_signals(closes_test, t1, t2, beta, hl)
    if sig_test.empty or t1 not in closes_test.columns:
        continue

    df_t = all_pair_train_results.get(pair_name)
    if df_t is None:
        continue

    colors_5 = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: top-5 OOS equity curves — use old Python backtest() for curve shape
    ax = axes[0]
    for rank, (_, r) in enumerate(df_t.head(5).iterrows()):
        tr = backtest(sig_test, t1, t2, beta,
                      float(r["entry_z"]), float(r["exit_z"]), float(r["stop_z"]))
        if not tr:
            continue
        curve = np.cumsum([t["net_pnl"] for t in tr])
        lbl   = (f"#{rank+1} e={r['entry_z']} x={r['exit_z']:+.1f} s={r['stop_z']}  "
                 f"[train Sh={r['sharpe']:.2f}]")
        ax.plot(curve, label=lbl, color=colors_5[rank],
                lw=2.5 if rank == 0 else 1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{pair_name}  TEST equity curves (train-optimal params)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative net P&L")
    ax.legend(fontsize=7)

    # Right: train vs test Sharpe scatter — Numba batch for all combos at once
    ax2 = axes[1]
    merged = df_t.copy()
    all_test_combos = list(zip(merged["entry_z"], merged["exit_z"], merged["stop_z"]))
    df_test_all = run_grid_numba(sig_test, t1, t2, beta,
                                  all_test_combos, days_test)
    key_cols = ["entry_z", "exit_z", "stop_z", "sharpe"]
    if not df_test_all.empty:
        df_test_all = df_test_all[key_cols].rename(columns={"sharpe": "test_sharpe"})
        merged = merged.merge(df_test_all, on=["entry_z", "exit_z", "stop_z"], how="left")
        merged["test_sharpe"] = merged["test_sharpe"].fillna(0.0)
    else:
        merged["test_sharpe"] = 0.0

    ax2.scatter(merged["sharpe"], merged["test_sharpe"], alpha=0.5, s=20, c="steelblue")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("TRAIN Sharpe")
    ax2.set_ylabel("TEST Sharpe")
    ax2.set_title(f"{pair_name}  Train vs Test Sharpe (all {len(merged)} combos)")

    corr = merged[["sharpe", "test_sharpe"]].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax2.transAxes,
             fontsize=10, va="top")

    plt.tight_layout()
    out = OUTPUT_DIR / f"pair_traintest_{pair_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Train/Test chart → {out}")
