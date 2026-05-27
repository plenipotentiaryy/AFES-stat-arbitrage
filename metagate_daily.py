"""metagate_daily.py — daily-timeframe MetaGate.

Lightweight ML classifier that scores each potential entry by P(Win) and
either blocks (P < θ) or scales position size by (P − θ)/(1 − θ).

Feature set is deliberately minimal — only signals that have a clean
daily-bar interpretation, no intraday-specific features:

    abs_z       : |z_score| at entry bar
    side        : +1 long-spread / -1 short-spread
    vol_20      : rolling 20-day std of spread (fast)
    vol_60      : rolling 60-day std of spread (slow)
    vol_ratio   : vol_20 / vol_60 (regime-of-volatility proxy)
    hurst_60d   : rolling Hurst exponent estimate (mean-revert if < 0.5)

θ is calibrated by 5-fold time-series CV maximising the cumulative EV of
the accepted train trades, swept over θ ∈ [0.50, 0.65].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit


FEATURE_NAMES = ("abs_z", "side", "vol_20", "vol_60", "vol_ratio", "hurst_60d")
THETA_GRID    = np.linspace(0.50, 0.65, 16)


# ── Feature computation ─────────────────────────────────────────────────────

def rolling_hurst(s: pd.Series, window: int = 60, lag: int = 10) -> pd.Series:
    """Cheap rolling Hurst via variance-of-aggregates (a.k.a. R/S-lite)."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    arr = s.values
    for i in range(window, len(arr)):
        w = arr[i - window + 1: i + 1]
        # Variance-based estimator: log-variance at multiple lags, slope/2 is H.
        try:
            lags = np.arange(2, min(lag + 1, len(w)))
            taus = [np.std(np.subtract(w[k:], w[:-k])) for k in lags]
            if min(taus) <= 0:
                continue
            slope = np.polyfit(np.log(lags), np.log(taus), 1)[0]
            out.iloc[i] = slope
        except Exception:
            pass
    return out


def entry_features(spread: pd.Series, z: pd.Series, idx: int,
                    side: int, hurst_series: pd.Series) -> np.ndarray:
    """Compute the 6-feature vector at a single entry bar (causal)."""
    zi = z.iloc[idx]
    vol_20 = spread.iloc[max(0, idx - 19): idx + 1].std()
    vol_60 = spread.iloc[max(0, idx - 59): idx + 1].std()
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    h = hurst_series.iloc[idx] if idx < len(hurst_series) else np.nan
    if not np.isfinite(h):
        h = 0.5
    return np.array([abs(zi), float(side), vol_20, vol_60, vol_ratio, h],
                     dtype=np.float64)


# ── Model + calibration ─────────────────────────────────────────────────────

class MetaGateDaily:
    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.clf = None
        self.theta: float = 0.55
        self.usable: bool = False
        self.kind: str = "off"
        self.n_train: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, pnl: np.ndarray):
        """Fit + calibrate θ. Falls back to neutral if too little data."""
        n = len(X)
        self.n_train = n
        if n < 80 or y.sum() < 20 or (n - y.sum()) < 20:
            self.usable = False
            self.kind = "fallback"
            return self

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        if n >= 300:
            self.clf = HistGradientBoostingClassifier(
                max_iter=120, max_depth=4, min_samples_leaf=20,
                random_state=0,
            )
            self.kind = "gbm"
        else:
            self.clf = LogisticRegression(max_iter=400, C=1.0)
            self.kind = "logit"
        self.clf.fit(Xs, y)

        # Calibrate θ on out-of-fold predictions.
        oof_p = np.zeros(n)
        tscv  = TimeSeriesSplit(n_splits=5)
        for tr_idx, te_idx in tscv.split(Xs):
            if y[tr_idx].sum() < 5 or (len(tr_idx) - y[tr_idx].sum()) < 5:
                continue
            if self.kind == "gbm":
                fold = HistGradientBoostingClassifier(
                    max_iter=120, max_depth=4, min_samples_leaf=20,
                    random_state=0,
                )
            else:
                fold = LogisticRegression(max_iter=400, C=1.0)
            fold.fit(Xs[tr_idx], y[tr_idx])
            oof_p[te_idx] = fold.predict_proba(Xs[te_idx])[:, 1]

        best_theta, best_ev = self.theta, -np.inf
        for theta in THETA_GRID:
            ev = float(np.sum(pnl * (oof_p >= theta)))
            if ev > best_ev:
                best_ev, best_theta = ev, float(theta)
        self.theta = best_theta
        self.usable = True
        return self

    def predict_proba(self, feat: np.ndarray) -> float:
        if not self.usable:
            return 0.5
        try:
            x = self.scaler.transform(feat.reshape(1, -1))
            return float(self.clf.predict_proba(x)[0, 1])
        except Exception:
            return 0.5

    def size_multiplier(self, p: float, mode: str = "binary") -> float:
        """mode='binary' → 0/1 gate; mode='soft' → (P-θ)/(1-θ) sizing."""
        if not self.usable:
            return 1.0
        if p < self.theta:
            return 0.0
        if mode == "binary":
            return 1.0
        denom = 1.0 - self.theta
        if denom <= 0:
            return 1.0
        return float(min(1.0, (p - self.theta) / denom))


# ── Train-trade harvest (run baseline on TRAIN to collect features+labels) ──

def harvest_train_trades(closes_train: pd.DataFrame, pairs: list[str],
                          tr_s, tr_e,
                          z_win: int, entry_z: float, exit_z: float,
                          stop_z: float, max_hold: int,
                          cost_per_trade: float, borrow_per_day: float,
                          estimate_beta_fn, spread_series_fn,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Walk train data; for every entry made by the baseline rules, capture
    (features at entry, win-label, net PnL).

    Also build a (dates × pairs) daily-PnL panel: each trade's PnL is
    booked on its exit date.  This panel is used by HRP weighting.

    Returns (X, y, pnl, pnl_panel).
    """
    X_rows, y_lbl, pnl_lbl = [], [], []
    pnl_by_date_pair: dict[tuple[pd.Timestamp, str], float] = {}
    for pair in pairs:
        t1, t2 = pair.split("-")
        if t1 not in closes_train.columns or t2 not in closes_train.columns:
            continue
        full = closes_train[[t1, t2]].dropna()
        if len(full) < z_win + 30:
            continue
        beta = estimate_beta_fn(full[t1], full[t2])
        if not (0.1 <= abs(beta) <= 15.0):
            continue
        sp = spread_series_fn(full, t1, t2, beta)
        mu  = sp.rolling(z_win).mean()
        sig = sp.rolling(z_win).std()
        z   = (sp - mu) / sig
        hurst = rolling_hurst(sp, window=z_win)

        in_pos = False
        side = 0
        entry_idx, entry_spread = 0, 0.0
        feat_entry: np.ndarray | None = None
        for i in range(len(sp)):
            zi = z.iloc[i]
            if pd.isna(zi):
                continue
            if not in_pos:
                if zi > entry_z:
                    side, in_pos = -1, True
                    entry_idx, entry_spread = i, sp.iloc[i]
                    feat_entry = entry_features(sp, z, i, side, hurst)
                elif zi < -entry_z:
                    side, in_pos = +1, True
                    entry_idx, entry_spread = i, sp.iloc[i]
                    feat_entry = entry_features(sp, z, i, side, hurst)
            else:
                held = i - entry_idx
                hit_target = (side == -1 and zi <= exit_z) or (side == +1 and zi >= exit_z)
                hit_stop   = abs(zi) >= stop_z
                timeout    = held >= max_hold
                if hit_target or hit_stop or timeout:
                    gross = side * (sp.iloc[i] - entry_spread)
                    pnl = gross - cost_per_trade - held * borrow_per_day
                    X_rows.append(feat_entry)
                    y_lbl.append(1 if pnl > 0 else 0)
                    pnl_lbl.append(pnl)
                    exit_date = sp.index[i].normalize()
                    key = (exit_date, pair)
                    pnl_by_date_pair[key] = pnl_by_date_pair.get(key, 0.0) + pnl
                    in_pos = False

    # Build daily-PnL panel from the dict-of-(date,pair).
    if pnl_by_date_pair:
        df = pd.DataFrame(
            [(d, p, v) for (d, p), v in pnl_by_date_pair.items()],
            columns=["date", "pair", "pnl"],
        )
        panel = df.pivot(index="date", columns="pair", values="pnl").fillna(0.0)
        for p in pairs:
            if p not in panel.columns:
                panel[p] = 0.0
        panel = panel[pairs].sort_index()
    else:
        panel = pd.DataFrame(columns=pairs)

    if not X_rows:
        return (np.zeros((0, len(FEATURE_NAMES))),
                np.zeros(0, dtype=int), np.zeros(0),
                panel)
    return (np.vstack(X_rows),
            np.asarray(y_lbl, dtype=int),
            np.asarray(pnl_lbl, dtype=float),
            panel)
