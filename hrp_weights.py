"""Hierarchical Risk Parity weights for the daily WFO.

Implements de Prado 2016 (Building Diversified Portfolios that Outperform Out
of Sample).  Input: per-pair daily P&L series harvested from the TRAIN slice
of a WFO window.  Output: a weight per pair in [0,1] that sums to ~1.

Algorithm:
    1. corr   = pair daily-PnL correlation matrix
    2. dist   = sqrt(0.5 * (1 - corr))                 (de Prado distance)
    3. link   = scipy.cluster.hierarchy.linkage(dist, 'single')
    4. order  = quasi-diagonalised pair ordering from the dendrogram
    5. recursive bisection allocates capital inversely-proportional to
       each cluster's variance.

If a pair has no train PnL data, it gets a fallback weight of 1/N (equal).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Sort cluster leaves by hierarchical clustering output."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _inv_var_weights(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = _inv_var_weights(sub).reshape(-1, 1)
    return float((w.T @ sub @ w).item())


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[start:end] for c in clusters
                     for start, end in [(0, len(c) // 2), (len(c) // 2, len(c))]
                     if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha  = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w.loc[c0] *= alpha
            w.loc[c1] *= (1.0 - alpha)
    return w


def hrp_weights(pnl_panel: pd.DataFrame) -> pd.Series:
    """Compute HRP weights for the pairs (columns of pnl_panel).

    pnl_panel: index=dates, columns=pair names, values=daily PnL.
    Returns a Series of weights ∈ [0,1] summing to 1, indexed by pair name.
    Pairs with zero variance or all-NaN get equal-weight fallback.
    """
    pairs = list(pnl_panel.columns)
    n = len(pairs)
    # Pairs with usable variance:
    var = pnl_panel.var(axis=0)
    usable_mask = var.fillna(0) > 0
    if usable_mask.sum() < 2:
        return pd.Series(1.0 / n, index=pairs)

    sub = pnl_panel.loc[:, usable_mask].fillna(0.0)
    cov  = np.array(sub.cov().values,  copy=True)
    corr = np.array(sub.corr().values, copy=True)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    try:
        cond = squareform(dist, checks=False)
        link = linkage(cond, method="single")
        order = _quasi_diag(link)
        w_sub = _recursive_bisection(cov, order)
        w_sub = w_sub.sort_index()
        w_sub.index = sub.columns[w_sub.index]
    except Exception:
        # Degenerate clustering — fall back to inverse-variance.
        ivw = _inv_var_weights(cov)
        w_sub = pd.Series(ivw, index=sub.columns)

    # Build full-pair weight vector (unused pairs get small equal share).
    w_full = pd.Series(0.0, index=pairs)
    w_full.loc[w_sub.index] = w_sub.values
    leftover = 1.0 - w_full.sum()
    unused = [p for p in pairs if p not in w_sub.index]
    if unused and leftover > 1e-9:
        w_full.loc[unused] = leftover / len(unused)
    return w_full
