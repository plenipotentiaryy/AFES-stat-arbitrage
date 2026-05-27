"""pca_residual_backtest.py — Avellaneda-Lee (2010) single-name PCA-residual
mean-reversion test on our S&P universe.

Hypothesis (from §4 of universe-expansion spec): trading the *idiosyncratic
residual* of each stock — after removing common factor returns extracted by
PCA on a sector/cluster — gives cleaner mean-reversion than raw pair-spreads.
Expected gain: +0.2–0.3 Sharpe over the pair-trade baseline (+1.22 net).

Honest test:
    1. Cluster names on TRAIN slice only (2006-12 → 2014-12).
    2. Fit PCA per cluster on TRAIN.  Project TRAIN+OOS returns through the
       train-fitted PCs — never re-fit on OOS data.
    3. Residual_i,t = R_i,t − Σ_k loading_{i,k} · factor_k,t.
    4. Trade rule: z-score of cumulative residual; z>2 short, z<−2 long,
       exit z=0, stop z=3.5, max hold 30d.
    5. Cost model = same as baseline (20 bps round-trip + 80 bps/y borrow
       on short side); no explicit factor hedge in execution (first pass).
    6. Apply maxconc=3 globally to match baseline portfolio constraint.
    7. Compare full-day Sharpe to baseline 1.22.

If Sharpe ≤ 1.3 → hypothesis fails, universe-expansion prompt's main
argument loses force.  If Sharpe ≥ 1.4 → continue with full §4 build.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

DATA = Path("data")

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_END = "2014-12-31"
OOS_START = TRAIN_END
N_CLUSTERS = 10
N_PCS = 10
Z_WIN = 60
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5
MAX_HOLD = 30
COST_RT = 20 / 1e4         # 20 bps round-trip (matches baseline 5+5×2)
BORROW_PER_DAY = 80 / (252 * 1e4)
MAXCONC = 3
MIN_TRAIN_OBS = 1000


# ── Pipeline ────────────────────────────────────────────────────────────────

def load_and_filter(path: Path, train_end: str) -> pd.DataFrame:
    closes = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    train = closes.loc[:train_end]
    keep = [t for t in closes.columns
            if train[t].notna().sum() >= MIN_TRAIN_OBS
            and train[t].dropna().min() >= 5.0]
    return closes[keep]


def cluster_stocks(train_returns: pd.DataFrame, n_clusters: int) -> dict[str, int]:
    """Hierarchical Ward clustering on 1−|correlation|."""
    # Use pairwise correlation with min_periods to handle staggered IPOs.
    corr = train_returns.corr(min_periods=250).fillna(0.0)
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)
    # Symmetrise (defensive).
    dist = (dist + dist.T) / 2.0
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="ward")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    return dict(zip(corr.columns, labels))


def compute_residuals(returns: pd.DataFrame, cluster_members: dict[int, list[str]],
                       n_pcs: int, train_end: str) -> pd.DataFrame:
    """Per-cluster PCA fit on TRAIN, residualise across the full series.

    For each cluster:
        1. Standardise TRAIN returns (z_train = (r − μ_train) / σ_train).
        2. Fit PCA on z_train (centered, no extra centering).
        3. Project FULL standardised returns through PC loadings.
        4. Reconstruct predicted standardised returns; un-standardise.
        5. Residual = raw return − factor-predicted return.
    """
    residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns,
                              dtype=float)
    cluster_meta = {}
    for cid, members in cluster_members.items():
        if len(members) < n_pcs + 2:
            continue
        train_block = returns.loc[:train_end, members]
        train_clean = train_block.dropna()
        if len(train_clean) < 250:
            # Try pairwise availability — accept rows with >=80% coverage.
            train_clean = train_block.dropna(thresh=int(0.8 * len(members)))
            train_clean = train_clean.fillna(0.0)
        mu = train_clean.mean()
        sigma = train_clean.std().replace(0, 1.0)
        z_train = (train_clean - mu) / sigma
        n_use = min(n_pcs, max(1, len(members) - 1))
        pca = PCA(n_components=n_use)
        pca.fit(z_train.values)

        full = returns[members].copy()
        full_z = (full - mu) / sigma
        full_z = full_z.fillna(0.0)
        factors = full_z.values @ pca.components_.T
        recon_z = factors @ pca.components_
        recon = pd.DataFrame(recon_z, index=full.index, columns=members) * sigma + mu
        cluster_residual = full - recon
        residuals.loc[:, members] = cluster_residual.values
        cluster_meta[cid] = {
            "n_members": len(members),
            "n_pcs": n_use,
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
        }
    return residuals, cluster_meta


def build_residual_z(residuals: pd.DataFrame, win: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative residual index + rolling z-score."""
    cum = residuals.fillna(0.0).cumsum()
    mu = cum.rolling(win, min_periods=win).mean()
    sd = cum.rolling(win, min_periods=win).std()
    z = (cum - mu) / sd
    return cum, z


def backtest_pca_residuals(cum: pd.DataFrame, z: pd.DataFrame, oos_start: str,
                            cost_rt: float, borrow_per_day: float) -> pd.DataFrame:
    """Trade single-stock residual mean-reversion on each column independently."""
    oos_ts = pd.Timestamp(oos_start)
    trades = []
    for stock in cum.columns:
        zs = z[stock]
        cs = cum[stock]
        in_pos = False
        side, entry_idx, entry_val = 0, 0, 0.0
        for i in range(len(cs)):
            zi = zs.iloc[i]
            if not np.isfinite(zi):
                continue
            t = cs.index[i]
            if not in_pos:
                if t < oos_ts:
                    continue
                if zi > ENTRY_Z:
                    side, in_pos = -1, True
                    entry_idx, entry_val = i, cs.iloc[i]
                elif zi < -ENTRY_Z:
                    side, in_pos = +1, True
                    entry_idx, entry_val = i, cs.iloc[i]
            else:
                held = i - entry_idx
                hit_t = (side == -1 and zi <= EXIT_Z) or (side == +1 and zi >= EXIT_Z)
                hit_s = abs(zi) >= STOP_Z
                tm = held >= MAX_HOLD
                if hit_t or hit_s or tm:
                    exit_val = cs.iloc[i]
                    gross = side * (exit_val - entry_val)
                    pnl = gross - cost_rt - held * borrow_per_day
                    trades.append({
                        "stock": stock,
                        "entry": cs.index[entry_idx],
                        "exit":  t,
                        "side":  side,
                        "held":  held,
                        "z_entry": zs.iloc[entry_idx],
                        "z_exit":  zi,
                        "gross_pnl": gross,
                        "pnl":   pnl,
                        "reason": "target" if hit_t else ("stop" if hit_s else "time"),
                    })
                    in_pos = False
    return pd.DataFrame(trades)


def apply_maxconc(trades: pd.DataFrame, cap: int) -> pd.DataFrame:
    if trades.empty or cap <= 0:
        return trades
    ts = trades.sort_values("entry").reset_index(drop=False)
    open_exits, keep = [], []
    for _, row in ts.iterrows():
        open_exits = [e for e in open_exits if e > row["entry"]]
        if len(open_exits) >= cap:
            continue
        open_exits.append(row["exit"])
        keep.append(row["index"])
    return trades.loc[keep].sort_values("entry").reset_index(drop=True)


def report(trades: pd.DataFrame, oos_index: pd.DatetimeIndex, label: str):
    print(f"\n=== {label} ===")
    if trades.empty:
        print("  NO TRADES")
        return
    pnl = trades["pnl"]
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")

    daily_exit = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    daily_full = daily_exit.reindex(oos_index.normalize(), fill_value=0.0)
    fsh = (daily_full.mean() / daily_full.std() * np.sqrt(252)
            if daily_full.std() > 0 else 0.0)

    print(f"  trades        = {len(trades)}")
    print(f"  win%          = {(pnl > 0).mean()*100:.1f}")
    print(f"  PnL           = {pnl.sum():+.4f}")
    print(f"  DD            = {dd:+.4f}")
    print(f"  PF            = {pf:.2f}")
    print(f"  full_Sharpe   = {fsh:+.3f}")
    print(f"  DD/PnL        = {abs(dd)/abs(pnl.sum())*100:.1f}%"
          if pnl.sum() != 0 else "  DD/PnL        = n/a")
    print(f"  exit reasons  = {trades['reason'].value_counts().to_dict()}")
    print(f"  avg held      = {trades['held'].mean():.1f} bars  "
          f"(median {trades['held'].median():.0f})")


def main():
    print("Loading data ...")
    closes = load_and_filter(DATA / "closes_daily.csv", TRAIN_END)
    print(f"  {closes.shape[0]} days × {closes.shape[1]} tickers "
          f"(after train-history filter)")

    returns = np.log(closes).diff()
    train_returns = returns.loc[:TRAIN_END]

    # Drop columns with very sparse train coverage even after first filter.
    keep_cols = train_returns.columns[train_returns.notna().sum() >= 500].tolist()
    returns = returns[keep_cols]
    train_returns = train_returns[keep_cols]
    print(f"  {len(keep_cols)} tickers usable for clustering")

    print(f"Clustering ({N_CLUSTERS} clusters, Ward on 1−|corr|) ...")
    labels = cluster_stocks(train_returns, N_CLUSTERS)
    sizes = Counter(labels.values())
    print(f"  Cluster sizes: {sorted(sizes.items())}")
    cluster_members = {}
    for stock, c in labels.items():
        cluster_members.setdefault(c, []).append(stock)

    print(f"Computing per-cluster PCA residuals ({N_PCS} PCs) ...")
    residuals, meta = compute_residuals(returns, cluster_members, N_PCS, TRAIN_END)
    if meta:
        avg_var = np.mean([m["variance_explained"] for m in meta.values()])
        print(f"  Avg variance explained by first {N_PCS} PCs: {avg_var*100:.1f}%")

    print(f"Building cumulative residual z-scores (window={Z_WIN}d) ...")
    cum, z = build_residual_z(residuals, Z_WIN)

    print(f"Backtesting OOS from {OOS_START} ...")
    trades = backtest_pca_residuals(cum, z, OOS_START, COST_RT, BORROW_PER_DAY)

    oos_index = closes.index[closes.index >= pd.Timestamp(OOS_START)]

    report(trades, oos_index, "Raw PCA-residual (no concurrency cap)")
    trades_cap = apply_maxconc(trades, MAXCONC)
    report(trades_cap, oos_index, f"With maxconc={MAXCONC}")

    print("\n=== Baseline reference (pair-trade combo) ===")
    print("  trades = 360, full_Sharpe = +1.224, DD/PnL = 7.2%")

    out = DATA / "pca_residual_trades.csv"
    trades_cap.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
