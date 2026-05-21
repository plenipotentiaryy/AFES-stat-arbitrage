from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters import HurstFilter
from kalman import kalman_hedge

from .config import RegimeMemoryConfig
from .decayed_density_estimator import DecayedDensityEstimator, LegacyExpandingWindowProfiler
from .profile_manager import RegimeAwareProfileManager
from .regime_classifier import RegimeClassifier
from .regime_memory_bank import ObservationRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DiagnosticRecord:
    test: str
    variant: str
    metric: str
    value: float


class RegimeMemoryDiagnostics:
    """
    Validation suite for the regime-aware memory bank profiler.

    The tests focus on the two failure modes the module is built to solve:
    stale-regime contamination and delayed profile invalidation when
    mean-reversion speed shifts materially.
    """

    def __init__(self, config: RegimeMemoryConfig):
        self.config = config

    def run_all(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        records.extend(self.regime_contamination_audit())
        records.extend(self.profile_drift_detection_accuracy())
        records.extend(self.ev_surface_quality_by_regime())
        records.extend(self.effective_sample_size_monitor())
        report = pd.DataFrame(records)
        out_path = Path(self.config.validation_output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        return report

    def regime_contamination_audit(self) -> list[dict[str, object]]:
        """
        Measure how much active profile weight comes from recent versus old data.

        In a sustained SLOW regime the weighted estimator should lean heavily on
        the most recent 90 days. If old fast-regime rows still dominate weight,
        the temporal decay is too weak and the memory bank is just another
        expanding window in disguise.
        """

        observations = self._build_historical_observations().sort_values("timestamp")
        slow = observations[observations["regime_label"] == "SLOW"].copy()
        if slow.empty:
            return []

        estimator = DecayedDensityEstimator(self.config)
        estimator.fit(slow.tail(max(120, self.config.min_regime_samples)))
        wf = estimator.weight_frame()
        latest = pd.Timestamp(wf["timestamp"].max())
        age_days = (latest - pd.to_datetime(wf["timestamp"], utc=True)).dt.days
        weight = wf["weight"].to_numpy(dtype=float)
        total = float(weight.sum())
        recent_90 = float(wf.loc[age_days <= 90, "weight"].sum() / total) if total > 0 else 0.0
        old_360 = float(wf.loc[age_days > 360, "weight"].sum() / total) if total > 0 else 0.0
        return [
            {
                "test": "regime_contamination",
                "variant": "slow_regime",
                "metric": "recent_90d_weight_share",
                "value": recent_90,
            },
            {
                "test": "regime_contamination",
                "variant": "slow_regime",
                "metric": "older_360d_weight_share",
                "value": old_360,
            },
        ]

    def profile_drift_detection_accuracy(self) -> list[dict[str, object]]:
        """
        Simulate a half-life jump and measure rebuild lag.

        The manager is considered fast enough if it rebuilds within a handful of
        bars after the half-life breach because that is the whole point of
        replacing the stale expanding window logic.
        """

        obs = self._simulate_half_life_transition()
        manager = RegimeAwareProfileManager(self.config)
        breach_bar = None
        rebuild_bar = None

        recent_hl: list[float] = []
        q = self.config.invalidation_q()
        for i, record in enumerate(obs):
            recent_hl.append(record.hl_estimate)
            if len(recent_hl) > self.config.stale_lookback_observations:
                recent_hl.pop(0)
            if len(recent_hl) >= self.config.stale_lookback_observations:
                hl_median = float(np.median(recent_hl[:-1]))
                if breach_bar is None and hl_median > 0 and record.hl_estimate > q * hl_median:
                    breach_bar = i
            result = manager.update(record)
            if breach_bar is not None and result.profile_rebuilt and rebuild_bar is None:
                rebuild_bar = i

        lag = float((rebuild_bar - breach_bar) if breach_bar is not None and rebuild_bar is not None else np.nan)
        return [
            {
                "test": "profile_drift_detection",
                "variant": "manager",
                "metric": "bars_to_rebuild_after_breach",
                "value": lag,
            }
        ]

    def ev_surface_quality_by_regime(self) -> list[dict[str, object]]:
        """
        Compare weighted and unweighted profiles on OOS regime-transition data.

        Brier score tests reversion calibration directly. Density MAE against a
        held-out KDE tests whether the profile shape itself tracks the current
        regime more accurately than a raw expanding window.
        """

        observations = self._build_historical_observations().sort_values("timestamp")
        if observations.empty:
            return []

        split = int(len(observations) * 0.7)
        train = observations.iloc[:split].copy()
        test = observations.iloc[split:].copy()
        results: list[dict[str, object]] = []

        for regime in ["FAST", "MEDIUM", "SLOW", "UNSTABLE"]:
            test_regime = test[test["regime_label"] == regime].copy()
            if len(test_regime) < 20:
                results.extend(
                    [
                        {
                            "test": "ev_surface_quality",
                            "variant": regime,
                            "metric": "sample_count",
                            "value": float(len(test_regime)),
                        },
                        {
                            "test": "ev_surface_quality",
                            "variant": regime,
                            "metric": "brier_weighted",
                            "value": float("nan"),
                        },
                        {
                            "test": "ev_surface_quality",
                            "variant": regime,
                            "metric": "brier_legacy",
                            "value": float("nan"),
                        },
                        {
                            "test": "ev_surface_quality",
                            "variant": regime,
                            "metric": "density_mae_weighted",
                            "value": float("nan"),
                        },
                        {
                            "test": "ev_surface_quality",
                            "variant": regime,
                            "metric": "density_mae_legacy",
                            "value": float("nan"),
                        },
                    ]
                )
                continue

            weighted = DecayedDensityEstimator(self.config)
            legacy = LegacyExpandingWindowProfiler(self.config)
            weighted.fit(pd.concat([train, test_regime.head(1)], ignore_index=True))
            legacy.fit(train)

            y_true = test_regime["reverted"].to_numpy(dtype=float)
            p_weighted = np.array([weighted.p_revert(z) for z in test_regime["z_score"]], dtype=float)
            p_legacy = np.array([legacy.p_revert(z) for z in test_regime["z_score"]], dtype=float)
            brier_weighted = float(np.mean((p_weighted - y_true) ** 2))
            brier_legacy = float(np.mean((p_legacy - y_true) ** 2))

            kde = gaussian_kde(test_regime["z_score"].to_numpy(dtype=float))
            density_true = kde.evaluate(test_regime["z_score"].to_numpy(dtype=float))
            density_weighted = np.array([weighted.density(z) for z in test_regime["z_score"]], dtype=float)
            density_legacy = np.array([legacy.density(z) for z in test_regime["z_score"]], dtype=float)
            mae_weighted = float(np.mean(np.abs(density_weighted - density_true)))
            mae_legacy = float(np.mean(np.abs(density_legacy - density_true)))

            results.extend(
                [
                    {
                        "test": "ev_surface_quality",
                        "variant": regime,
                        "metric": "sample_count",
                        "value": float(len(test_regime)),
                    },
                    {
                        "test": "ev_surface_quality",
                        "variant": regime,
                        "metric": "brier_weighted",
                        "value": brier_weighted,
                    },
                    {
                        "test": "ev_surface_quality",
                        "variant": regime,
                        "metric": "brier_legacy",
                        "value": brier_legacy,
                    },
                    {
                        "test": "ev_surface_quality",
                        "variant": regime,
                        "metric": "density_mae_weighted",
                        "value": mae_weighted,
                    },
                    {
                        "test": "ev_surface_quality",
                        "variant": regime,
                        "metric": "density_mae_legacy",
                        "value": mae_legacy,
                    },
                ]
            )
        return results

    def effective_sample_size_monitor(self) -> list[dict[str, object]]:
        """
        Track effective sample size through sequential updates and flag blind spots.

        Low ``n_eff`` is the observable signature of memory uncertainty. Mapping
        it to realized outcomes checks whether blind spots line up with signal
        degradation and drawdown risk in practice.
        """

        observations = self._build_historical_observations().sort_values("timestamp")
        manager = RegimeAwareProfileManager(self.config)
        blind_spot_pnls: list[float] = []
        blind_spots = 0
        total = 0

        for _, row in observations.iterrows():
            rec = ObservationRecord(
                timestamp=pd.Timestamp(row["timestamp"]),
                z_score=float(row["z_score"]),
                hl_estimate=float(row["hl_estimate"]),
                vol_ratio=float(row["vol_ratio"]),
                hurst=float(row["hurst"]),
                macro_regime=int(row["macro_regime"]),
                outcome_pnl=float(row["outcome_pnl"]),
                reverted=int(row["reverted"]),
                break_score=float(row.get("break_score", 0.0)),
                regime_label=str(row["regime_label"]),
            )
            result = manager.update(rec)
            total += 1
            if result.n_eff < self.config.min_effective_sample_size:
                blind_spots += 1
                blind_spot_pnls.append(float(row["outcome_pnl"]))

        return [
            {
                "test": "effective_sample_size",
                "variant": "all_regimes",
                "metric": "blind_spot_count",
                "value": float(blind_spots),
            },
            {
                "test": "effective_sample_size",
                "variant": "all_regimes",
                "metric": "blind_spot_rate",
                "value": float(blind_spots / total) if total > 0 else 0.0,
            },
            {
                "test": "effective_sample_size",
                "variant": "all_regimes",
                "metric": "avg_outcome_pnl_when_neff_lt_30",
                "value": float(np.mean(blind_spot_pnls)) if blind_spot_pnls else 0.0,
            },
        ]

    def _build_historical_observations(self) -> pd.DataFrame:
        closes = pd.read_csv(ROOT / "data" / "closes_daily.csv", index_col=0, parse_dates=True)
        closes.index = pd.to_datetime(closes.index, utc=True)
        pairs = pd.read_csv(ROOT / "data" / "pairs_selected.csv").head(self.config.validation_pairs)

        classifier = RegimeClassifier(self.config)
        rows: list[dict[str, object]] = []
        for _, pair_row in pairs.iterrows():
            pair = pair_row["pair"]
            t1, t2 = pair.split("-")
            if t1 not in closes.columns or t2 not in closes.columns:
                continue
            px = closes[[t1, t2]].dropna()
            if len(px) < 400:
                continue

            beta_init = float(pair_row.get("beta_daily", pair_row["beta"]))
            _, beta_arr, innov, _ = kalman_hedge(px[t1].to_numpy(dtype=float), px[t2].to_numpy(dtype=float), beta_init=beta_init)
            spread = pd.Series(innov, index=px.index)
            z = ((spread - spread.rolling(60).mean()) / spread.rolling(60).std()).replace([np.inf, -np.inf], np.nan)
            z = z.dropna()
            spread = spread.loc[z.index]
            beta_series = pd.Series(beta_arr, index=px.index).loc[z.index]

            hl = self._rolling_half_life(spread, self.config.local_half_life_window).loc[z.index]
            hurst = HurstFilter(window=60, hurst_lag=10).compute_hurst(spread).loc[z.index]
            short_vol = spread.diff().rolling(20, min_periods=10).std()
            long_vol = spread.diff().rolling(252, min_periods=60).std()
            vol_ratio = (short_vol / long_vol.replace(0.0, np.nan)).loc[z.index].fillna(1.0)
            macro_regime = np.where(vol_ratio > 1.4, 2, np.where(vol_ratio < 0.8, 0, 1))

            for i, ts in enumerate(z.index):
                z_val = float(z.iloc[i])
                if abs(z_val) < self.config.z_near_threshold:
                    continue
                hl_val = float(hl.iloc[i]) if np.isfinite(hl.iloc[i]) else self.config.hl_break_threshold * 2
                hurst_val = float(hurst.iloc[i]) if np.isfinite(hurst.iloc[i]) else 0.5
                vol_ratio_val = float(vol_ratio.iloc[i]) if np.isfinite(vol_ratio.iloc[i]) else 1.0
                macro_val = int(macro_regime[i])
                reverted, pnl = self._forward_outcome(z.to_numpy(dtype=float), i)
                row = {
                    "timestamp": pd.Timestamp(ts),
                    "z_score": z_val,
                    "hl_estimate": hl_val,
                    "vol_ratio": vol_ratio_val,
                    "hurst": hurst_val,
                    "macro_regime": macro_val,
                    "outcome_pnl": pnl,
                    "reverted": reverted,
                    "break_score": 0.0,
                }
                row["regime_label"] = classifier.classify_rule_based(row)
                rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("timestamp").reset_index(drop=True)

    def _rolling_half_life(self, spread: pd.Series, window: int) -> pd.Series:
        theta = pd.Series(np.nan, index=spread.index, dtype=float)
        for end in range(window, len(spread) + 1):
            chunk = spread.iloc[end - window : end]
            lag = chunk.shift(1).dropna()
            dx = chunk.diff().dropna()
            aligned = lag.index.intersection(dx.index)
            if len(aligned) < 5:
                continue
            y = dx.loc[aligned].to_numpy(dtype=float)
            x = lag.loc[aligned].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            theta.iloc[end - 1] = float(-coeffs[1])
        return pd.Series(np.where(theta > 0.0, np.log(2.0) / theta, self.config.hl_break_threshold * 2), index=spread.index)

    def _forward_outcome(self, z_values: np.ndarray, idx: int) -> tuple[int, float]:
        entry = float(abs(z_values[idx]))
        stop = entry * self.config.stop_multiple
        future = z_values[idx + 1 : idx + 1 + self.config.outcome_horizon_bars]
        if future.size == 0:
            return 0, -entry

        hit_zero = np.where(np.sign(future) != np.sign(z_values[idx]))[0]
        hit_stop = np.where(np.abs(future) >= stop)[0]
        first_zero = hit_zero[0] if hit_zero.size else np.inf
        first_stop = hit_stop[0] if hit_stop.size else np.inf
        if first_zero < first_stop:
            return 1, entry
        return 0, -(stop - entry)

    def _simulate_half_life_transition(self) -> list[ObservationRecord]:
        rng = np.random.default_rng(123)
        observations: list[ObservationRecord] = []
        idx = pd.date_range("2022-01-01", periods=160, freq="D", tz="UTC")
        hl_values = np.r_[np.full(80, 12.0), np.linspace(12.0, 32.0, 15), np.full(65, 32.0)]
        z = 0.0
        for i, ts in enumerate(idx):
            theta = np.log(2.0) / max(hl_values[i], 1e-8)
            z = (1.0 - theta) * z + rng.normal(0.0, 0.7)
            observations.append(
                ObservationRecord(
                    timestamp=ts,
                    z_score=float(z),
                    hl_estimate=float(hl_values[i]),
                    vol_ratio=float(1.0 + 0.1 * rng.normal()),
                    hurst=float(0.42 if hl_values[i] < 20 else 0.48),
                    macro_regime=1,
                    outcome_pnl=float(rng.normal()),
                    reverted=int(rng.random() > 0.4),
                    break_score=0.0,
                )
            )
        return observations
