from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".mplconfig-validation"))
sys.path.insert(0, str(ROOT / "signal"))

from tail_ev import TailEVConfig
from tail_ev.tail_data_layer import TailDataLayer
from tail_ev.tail_ev_gate import TailEVDecision
from tail_ev.tail_ev_gate import TailEVDecisionGate
from tail_ev.tail_ev_layer import TailEVLayer


def simulate_merton_jump_diffusion(
    n: int = 2200,
    dt: float = 1.0 / 252.0,
    mu: float = 0.00,
    sigma: float = 0.18,
    lam: float = 8.0,
    mu_j: float = 0.00,
    sigma_j: float = 0.12,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate a Merton jump-diffusion price path and convert it to z-scores.

    The jump component creates clustered overshoots that are benign for small
    moves but punishing in the far tail, which is exactly the regime where the
    POT gate should trade less aggressively than a legacy empirical EV model.
    """

    rng = np.random.default_rng(seed)
    stress = np.zeros(n, dtype=int)
    stress_sign = 1
    for i in range(1, n):
        if stress[i - 1] == 1:
            stress[i] = int(rng.random() < 0.90)
        else:
            stress[i] = int(rng.random() < 0.03)
        if stress[i] == 1 and stress[i - 1] == 0:
            stress_sign = 1 if rng.random() > 0.5 else -1

    jumps = np.empty(n, dtype=float)
    for i in range(n):
        lam_t = lam * (4.5 if stress[i] else 1.0)
        mu_j_t = (0.09 * stress_sign) if stress[i] else mu_j
        sigma_j_t = sigma_j * (1.8 if stress[i] else 1.0)
        jump_count = rng.poisson(lam_t * dt)
        jumps[i] = rng.normal(mu_j_t, sigma_j_t) * jump_count

    diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    log_returns = diffusion + jumps
    log_price = np.cumsum(log_returns)
    price = np.exp(log_price)
    spread = pd.Series(price).diff().fillna(0.0)
    z = (spread - spread.rolling(40, min_periods=20).mean()) / spread.rolling(40, min_periods=20).std()
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z.index = pd.date_range("2018-01-01", periods=n, freq="B")
    stress_s = pd.Series(stress, index=z.index, dtype=float)
    z = z + stress_s * np.sign(z.replace(0.0, 1.0)) * 1.8
    return z.astype(float), stress_s


def build_context(z: pd.Series, stress: pd.Series) -> pd.DataFrame:
    tail = z.abs()
    return pd.DataFrame(
        {
            "z_velocity": z.diff().fillna(0.0),
            "vol_ratio": (
                tail.rolling(21, min_periods=5).std()
                / tail.rolling(252, min_periods=40).std().replace(0.0, np.nan)
            ).fillna(1.0),
            "OFI": np.sign(z).rolling(5, min_periods=1).mean().fillna(0.0),
            "macro_regime": np.where(stress > 0, 2.0, 1.0),
            "hurst_exp": np.where(stress > 0, 0.72, 0.38),
            "coint_score": np.where(stress > 0, -1.1 - 0.10 * tail, -2.4 - 0.05 * tail),
        },
        index=z.index,
    )


def build_legacy_gate(
    train_z: pd.Series,
    config: TailEVConfig,
    context: pd.DataFrame,
) -> "LegacyGaussianGate":
    data_layer = TailDataLayer(config)
    dataset = data_layer.fit_transform(train_z, context=context)
    threshold = dataset.threshold
    labels = []
    gains = []
    losses = []
    tail = train_z.abs()
    for idx in dataset.exceedance_index:
        loc = tail.index.get_loc(idx)
        entry = float(tail.iloc[loc])
        u = float(dataset.threshold_series.iloc[loc])
        target = u * config.revert_target_fraction
        stop = max(entry * config.stop_multiple, u + config.min_stop_excess)
        future = tail.iloc[loc + 1 : loc + 1 + config.max_holding_period]
        if future.empty:
            continue
        fv = future.to_numpy(dtype=float)
        hit_revert = np.where(fv <= target)[0]
        hit_stop = np.where(fv >= stop)[0]
        first_revert = hit_revert[0] if hit_revert.size else np.inf
        first_stop = hit_stop[0] if hit_stop.size else np.inf
        success = int(first_revert < first_stop)
        labels.append(success)
        gains.append(max(entry - target, 0.0))
        losses.append(max(stop - entry, 0.0) * 0.55)

    p_revert = float(np.mean(labels)) if labels else 0.0
    mean_gain = float(np.mean(gains)) if gains else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0

    return LegacyGaussianGate(
        threshold=threshold,
        p_revert=p_revert,
        mean_gain=mean_gain,
        mean_loss=mean_loss,
    )


class LegacyGaussianGate:
    """
    Legacy Gaussian-like gate with static empirical odds and naive loss severity.

    This intentionally mimics the old failure mode: it uses one tail-wide
    average revert probability and a local, under-scaled loss estimate instead
    of a state-conditioned ES. That makes it useful as a comparison baseline
    for the POT gate on jump-diffusion stress data.
    """

    def __init__(self, threshold: float, p_revert: float, mean_gain: float, mean_loss: float):
        self.threshold = float(threshold)
        self.p_revert = float(p_revert)
        self.mean_gain = float(mean_gain)
        self.mean_loss = float(mean_loss)

    def evaluate(self, z: float, context=None, threshold: float | None = None) -> TailEVDecision:
        live_threshold = float(self.threshold if threshold is None else threshold)
        z_mag = abs(float(z))
        if z_mag <= live_threshold:
            return TailEVDecision(
                signal=False,
                regime="legacy_center",
                ev=0.0,
                ratio=np.nan,
                threshold=live_threshold,
                empirical_ev=0.0,
            )

        expected_gain = max(z_mag - live_threshold * 0.50, self.mean_gain)
        naive_loss = max(self.mean_loss, (z_mag - live_threshold) * 0.35)
        ev = self.p_revert * expected_gain - (1.0 - self.p_revert) * naive_loss
        return TailEVDecision(
            signal=bool(ev > 0.0),
            regime="legacy_tail",
            ev=float(ev),
            ratio=float(ev / naive_loss) if naive_loss > 0 else np.nan,
            threshold=live_threshold,
            p_revert=self.p_revert,
            expected_shortfall=naive_loss,
            expected_gain=expected_gain,
        )


def realised_tail_trade_pnl(
    z: pd.Series,
    start_loc: int,
    threshold_series: pd.Series,
    config: TailEVConfig,
) -> tuple[int, float]:
    tail = z.abs()
    entry = float(tail.iloc[start_loc])
    threshold = float(threshold_series.iloc[start_loc])
    target = threshold * config.revert_target_fraction
    stop = max(entry * config.stop_multiple, threshold + config.min_stop_excess)
    future = tail.iloc[start_loc + 1 : start_loc + 1 + config.max_holding_period]
    if future.empty:
        return 0, -(stop - entry)

    fv = future.to_numpy(dtype=float)
    hit_revert = np.where(fv <= target)[0]
    hit_stop = np.where(fv >= stop)[0]
    first_revert = hit_revert[0] if hit_revert.size else np.inf
    first_stop = hit_stop[0] if hit_stop.size else np.inf
    if first_revert < first_stop:
        return 1, entry - target
    return 0, -(stop - entry)


def evaluate_gate(
    gate: TailEVDecisionGate,
    z_test: pd.Series,
    context_test: pd.DataFrame,
    config: TailEVConfig,
) -> pd.DataFrame:
    threshold_series = (
        z_test.abs()
        .rolling(config.rolling_window, min_periods=config.min_periods)
        .quantile(config.threshold_quantile)
        .shift(1)
        .fillna(z_test.abs().expanding(min_periods=max(5, config.min_periods // 4)).quantile(config.threshold_quantile).shift(1))
        .ffill()
        .bfill()
    )
    records: list[dict[str, float | str]] = []
    for i, ts in enumerate(z_test.index):
        z_value = float(z_test.iloc[i])
        threshold = float(threshold_series.iloc[i])
        if abs(z_value) <= threshold:
            continue
        decision = gate.evaluate(z_value, context=context_test.loc[ts], threshold=threshold)
        if not decision.signal:
            continue
        hit, pnl = realised_tail_trade_pnl(z_test, i, threshold_series, config)
        records.append(
            {
                "timestamp": ts,
                "hit": float(hit),
                "pnl": float(pnl),
                "ev": float(decision.ev),
                "threshold": threshold,
                "regime": decision.regime,
            }
        )
    return pd.DataFrame(records)


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if not drawdown.empty else 0.0


def main():
    config = TailEVConfig(
        rolling_window=120,
        min_periods=60,
        min_exceedances=35,
        max_holding_period=30,
        min_train_size=80,
        ad_bootstrap_samples=80,
        risk_appetite_k=0.25,
    )
    z, stress = simulate_merton_jump_diffusion()
    context = build_context(z, stress)
    split = int(len(z) * 0.70)
    train_z, test_z = z.iloc[:split], z.iloc[split:]
    train_context, test_context = context.iloc[:split], context.iloc[split:]

    legacy_gate = build_legacy_gate(train_z, config, train_context)
    new_layer = TailEVLayer(config)
    new_layer.fit(train_z, context=train_context)

    threshold_train = new_layer.dataset_.threshold

    def empirical_ev(z_value: float) -> float:
        if abs(z_value) <= threshold_train:
            return 0.0
        return 0.01

    new_gate = TailEVDecisionGate(new_layer, empirical_ev_fn=empirical_ev, config=config)

    legacy_trades = evaluate_gate(legacy_gate, test_z, test_context, config)
    new_trades = evaluate_gate(new_gate, test_z, test_context, config)

    comparison = pd.DataFrame(
        [
            {
                "model": "legacy_gaussian_like",
                "tail_trades": int(len(legacy_trades)),
                "hit_rate": float(legacy_trades["hit"].mean()) if not legacy_trades.empty else 0.0,
                "avg_pnl": float(legacy_trades["pnl"].mean()) if not legacy_trades.empty else 0.0,
                "max_drawdown": max_drawdown(legacy_trades["pnl"]) if not legacy_trades.empty else 0.0,
            },
            {
                "model": "pot_conditional_tail_ev",
                "tail_trades": int(len(new_trades)),
                "hit_rate": float(new_trades["hit"].mean()) if not new_trades.empty else 0.0,
                "avg_pnl": float(new_trades["pnl"].mean()) if not new_trades.empty else 0.0,
                "max_drawdown": max_drawdown(new_trades["pnl"]) if not new_trades.empty else 0.0,
            },
        ]
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "tail_ev_validation.csv"
    comparison.to_csv(out_path, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved validation report to {out_path}")


if __name__ == "__main__":
    main()
