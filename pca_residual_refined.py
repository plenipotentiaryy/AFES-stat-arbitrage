"""pca_residual_refined.py — Avellaneda-Lee 2010 with OU calibration.

Refines the failed first-pass PCA-residual experiment by:
    1. Using the wider S&P 1500 universe (~788 names with train history).
    2. Per-cluster PCA with K=10 factors (vs K=2 in first pass).
    3. Fitting an Ornstein-Uhlenbeck process to each stock's cumulative
       residual on TRAIN, and trading ONLY residuals whose mean-reversion
       speed κ exceeds a threshold.
    4. Reporting the equivalent of an explicit factor-hedged execution
       (residual-return = idiosyncratic component, already factor-neutral
       in PnL when the basket hedge is equal-weighted within the cluster).

Honest setup:
    - Cluster + PCA fit on 2006-2014 (TRAIN) only.
    - OU κ-fit on TRAIN residuals only.
    - Backtest on 2015-2026 (OOS).
    - Cost model identical to baseline: 20 bps round-trip + 80 bps/y borrow.
    - Survivorship caveat applies (universe = current S&P 1500).

Compare to baseline +2.64.  If refined PCA cannot exceed +1.5 on this
larger universe with OU filter, hypothesis is closed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import statsmodels.api as sm

DATA = Path("data")

# Config defaults
TRAIN_END         = "2014-12-31"
N_CLUSTERS        = 15
N_PCS             = 10
Z_WIN             = 60
ENTRY_Z           = 2.0
EXIT_Z            = 0.0
STOP_Z            = 3.5
MAX_HOLD          = 30
COST_RT           = 20 / 1e4
BORROW_PER_DAY    = 80 / (252 * 1e4)
MAXCONC           = 3
MIN_TRAIN_COV     = 0.70    # require ≥70% of train days covered
MIN_PRICE         = 5.0
KAPPA_MIN         = 0.025   # ~28-day half-life ceiling


# ── OU fit per stock ───────────────────────────────────────────────────────

def ou_fit_ar1(x: pd.Series) -> tuple[float, float, float]:
    """Fit AR(1): X_t = c + φ·X_{t-1} + ε. Returns (φ, κ=-ln φ, half_life days)."""
    s = x.dropna()
    if len(s) < 100:
        return (np.nan, np.nan, np.nan)
    lag = s.shift(1).dropna()
    cur = s.iloc[1:]
    try:
        res = sm.OLS(cur.values, sm.add_constant(lag.values)).fit()
        phi = float(res.params[1])
    except Exception:
        return (np.nan, np.nan, np.nan)
    if not (0 < phi < 1):
        return (phi, np.nan, np.nan)
    kappa = -np.log(phi)
    hl = np.log(2.0) / kappa
    return (phi, kappa, hl)


# ── Pipeline ───────────────────────────────────────────────────────────────

def filter_universe(closes: pd.DataFrame, train_end: str,
                     min_cov: float, min_price: float) -> pd.DataFrame:
    train = closes.loc[:train_end]
    keep = [t for t in closes.columns
            if train[t].notna().sum() >= min_cov * len(train)
            and train[t].dropna().min() >= min_price]
    return closes[keep]


def cluster_stocks(train_returns: pd.DataFrame, n_clusters: int) -> dict[str, int]:
    corr = train_returns.corr(min_periods=250).fillna(0.0)
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="ward")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    return dict(zip(corr.columns, labels))


def compute_residuals(returns: pd.DataFrame, cluster_members: dict[int, list[str]],
                       n_pcs: int, train_end: str) -> tuple[pd.DataFrame, dict]:
    residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns,
                              dtype=float)
    meta = {}
    for cid, members in cluster_members.items():
        if len(members) < n_pcs + 2:
            continue
        train_block = returns.loc[:train_end, members]
        train_clean = train_block.dropna()
        if len(train_clean) < 250:
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
        residuals.loc[:, members] = (full - recon).values
        meta[cid] = {
            "n_members": len(members),
            "n_pcs": n_use,
            "var_explained": float(pca.explained_variance_ratio_.sum()),
        }
    return residuals, meta


def build_cum_z(residuals: pd.DataFrame, win: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cum = residuals.fillna(0.0).cumsum()
    mu = cum.rolling(win, min_periods=win).mean()
    sd = cum.rolling(win, min_periods=win).std()
    z = (cum - mu) / sd
    return cum, z


def screen_by_ou(cum: pd.DataFrame, train_end: str,
                  kappa_min: float) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for col in cum.columns:
        train_x = cum.loc[:train_end, col]
        phi, kappa, hl = ou_fit_ar1(train_x)
        rows.append({"stock": col, "phi": phi, "kappa": kappa, "half_life": hl})
    df = pd.DataFrame(rows)
    sel = df[(df["kappa"].notna()) & (df["kappa"] >= kappa_min)]
    return sel["stock"].tolist(), df


# ── Backtest ───────────────────────────────────────────────────────────────

def backtest(cum: pd.DataFrame, z: pd.DataFrame, oos_start: str,
              tradable_stocks: list[str],
              cost_rt: float, borrow_per_day: float) -> pd.DataFrame:
    oos_ts = pd.Timestamp(oos_start)
    trades = []
    for stock in tradable_stocks:
        if stock not in cum.columns:
            continue
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
                    gross = side * (cs.iloc[i] - entry_val)
                    pnl = gross - cost_rt - held * borrow_per_day
                    trades.append({
                        "stock": stock, "entry": cs.index[entry_idx],
                        "exit": t, "side": side, "held": held,
                        "z_entry": zs.iloc[entry_idx], "z_exit": zi,
                        "pnl": pnl,
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
    dd = float((eq - eq.cummax()).min())
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    daily = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["pnl"].sum()
    daily_full = daily.reindex(oos_index.normalize(), fill_value=0.0)
    fsh = (daily_full.mean() / daily_full.std() * np.sqrt(252)
            if daily_full.std() > 0 else 0.0)
    print(f"  trades        = {len(trades)}")
    print(f"  win%          = {(pnl > 0).mean()*100:.1f}")
    print(f"  PnL           = {pnl.sum():+.4f}")
    print(f"  DD            = {dd:+.4f}")
    print(f"  PF            = {pf:.2f}")
    print(f"  full_Sharpe   = {fsh:+.3f}")
    print(f"  DD/PnL        = {abs(dd)/abs(pnl.sum())*100:.1f}%"
           if pnl.sum() != 0 else "  DD/PnL n/a")
    print(f"  active stocks = {trades['stock'].nunique()}")
    print(f"  exit reasons  = {trades['reason'].value_counts().to_dict()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--closes",    default="closes_daily_r3k.csv")
    p.add_argument("--train-end", default=TRAIN_END)
    p.add_argument("--clusters",  type=int, default=N_CLUSTERS)
    p.add_argument("--pcs",       type=int, default=N_PCS)
    p.add_argument("--kappa-min", type=float, default=KAPPA_MIN)
    args = p.parse_args()

    print(f"Loading {args.closes} …")
    closes = pd.read_csv(DATA / args.closes,
                          parse_dates=["Date"]).set_index("Date").sort_index()
    print(f"  raw: {closes.shape[0]} days × {closes.shape[1]} tickers")

    closes = filter_universe(closes, args.train_end, MIN_TRAIN_COV, MIN_PRICE)
    print(f"  after universe filter (≥{int(MIN_TRAIN_COV*100)}% train, price≥${MIN_PRICE}): "
           f"{closes.shape[1]} tickers")

    returns = np.log(closes).diff()
    train_returns = returns.loc[:args.train_end]

    print(f"\nClustering ({args.clusters} clusters, Ward on 1−|corr|) …")
    labels = cluster_stocks(train_returns, args.clusters)
    sizes = Counter(labels.values())
    print(f"  Cluster sizes (top 5):  {sorted(sizes.items(), key=lambda x:-x[1])[:5]}")
    cluster_members: dict[int, list[str]] = {}
    for s, c in labels.items():
        cluster_members.setdefault(c, []).append(s)

    print(f"\nComputing per-cluster PCA residuals (K={args.pcs}) …")
    residuals, meta = compute_residuals(returns, cluster_members, args.pcs,
                                          args.train_end)
    avg_var = np.mean([m["var_explained"] for m in meta.values()])
    print(f"  Avg variance explained by first {args.pcs} PCs: {avg_var*100:.1f}%")

    print(f"\nBuilding cumulative residual z-scores (window={Z_WIN}d) …")
    cum, z = build_cum_z(residuals, Z_WIN)

    print(f"\nOU κ-screening (κ_min={args.kappa_min}, "
           f"half-life ≤ {np.log(2)/args.kappa_min:.0f} days) …")
    tradable, ou_df = screen_by_ou(cum, args.train_end, args.kappa_min)
    print(f"  fit on train: {ou_df['kappa'].notna().sum()}/{len(ou_df)} stocks usable")
    print(f"  passing κ≥{args.kappa_min}: {len(tradable)}")
    if not ou_df.empty:
        kappa_dist = ou_df["kappa"].dropna()
        print(f"  κ distribution: min={kappa_dist.min():.4f}  med={kappa_dist.median():.4f}  "
               f"max={kappa_dist.max():.4f}")

    ou_df.to_csv(DATA / "pca_refined_ou_screen.csv", index=False)

    print(f"\nBacktesting OOS from {args.train_end} on {len(tradable)} κ-screened stocks …")
    trades = backtest(cum, z, args.train_end, tradable, COST_RT, BORROW_PER_DAY)
    oos_index = closes.index[closes.index >= pd.Timestamp(args.train_end)]
    report(trades, oos_index, "PCA-refined raw (no maxconc)")
    trades_cap = apply_maxconc(trades, MAXCONC)
    report(trades_cap, oos_index, f"PCA-refined + maxconc={MAXCONC}")

    out = DATA / "pca_refined_trades.csv"
    trades_cap.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(f"\n=== Reference: baseline+ADR = Sharpe +2.64 (95% CI [+2.03, +3.26]) ===")


if __name__ == "__main__":
    main()
