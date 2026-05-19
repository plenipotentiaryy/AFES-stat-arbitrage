from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from filters import HurstFilter
from kalman import kalman_hedge

from .break_composite_scorer import BreakCompositeScorer
from .break_data_layer import BreakDetectorDataLayer
from .break_risk_gate import BreakRiskGate
from .break_statistics import (
    BetaVelocityMonitor,
    CUSUMDetector,
    KalmanInnovationRatioDetector,
    LocalHalfLifeExplosionDetector,
)
from .config import BreakDetectorConfig


@dataclass(slots=True)
class StrategyValidationResult:
    strategy: str
    total_pnl: float
    max_drawdown: float
    calmar_ratio: float
    abort_triggers: int


class BreakDiagnostics:
    """
    Validation suite for the structural-break speed detector.

    The goal is not just numerical correctness. The diagnostics verify the
    economic reason the detector exists: it must react materially faster than
    rolling Hurst, remain quiet on stationary OU data, and improve drawdown
    control on historical OOS spreads.
    """

    def __init__(self, config: BreakDetectorConfig):
        self.config = config
        self.data_layer = BreakDetectorDataLayer(config)
        self.cusum = CUSUMDetector(config)
        self.innovation = KalmanInnovationRatioDetector(config)
        self.half_life = LocalHalfLifeExplosionDetector(config)
        self.beta_monitor = BetaVelocityMonitor(config)
        self.composite = BreakCompositeScorer(config)
        self.risk_gate = BreakRiskGate(config)

    def run_all(self) -> pd.DataFrame:
        """
        Run all required validation tests and persist a CSV report.

        The report is saved in long form so every metric remains easy to inspect
        and compare across tests without a custom parser.
        """

        records: list[dict[str, object]] = []
        records.extend(self.detection_lag_benchmark())
        records.extend(self.false_positive_rate_test())
        records.extend(self.pnl_impact_analysis())
        report = pd.DataFrame(records)
        out_path = Path(self.config.validation_output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        return report

    def detection_lag_benchmark(self) -> list[dict[str, object]]:
        """
        Benchmark break-score lag versus rolling Hurst lag on a known break.

        The synthetic process starts as OU and switches to a locally explosive
        AR regime with larger innovations and beta drift. That setup isolates
        the exact problem this module is built to solve: stale stationary
        history delays Hurst while fast break statistics should react quickly.
        """

        spread, innovations, beta, t_break = self._simulate_ou_to_explosive()
        stats = self._score_break_process(spread, innovations, beta)

        hurst_filter = HurstFilter(
            h_max=self.config.hurst_threshold,
            window=self.config.hurst_window,
            hurst_lag=self.config.hurst_lag,
        )
        hurst_series = hurst_filter.compute_hurst(spread).fillna(0.5)
        hurst_alarm_idx = hurst_series[hurst_series > self.config.hurst_threshold].index
        break_alarm_idx = stats[stats["break_level"].isin(["ALARM", "ABORT"])].index

        hurst_lag = self._bars_after_break(hurst_alarm_idx, spread.index[t_break])
        break_lag = self._bars_after_break(break_alarm_idx, spread.index[t_break])
        head_start = hurst_lag - break_lag
        return [
            {
                "test": "detection_lag",
                "variant": "hurst",
                "metric": "bars_after_break",
                "value": float(hurst_lag),
            },
            {
                "test": "detection_lag",
                "variant": "break_score",
                "metric": "bars_after_break",
                "value": float(break_lag),
            },
            {
                "test": "detection_lag",
                "variant": "head_start",
                "metric": "bars",
                "value": float(head_start),
            },
        ]

    def false_positive_rate_test(self) -> list[dict[str, object]]:
        """
        Estimate average run length on a stationary OU process.

        A risk gate that fires too often is worse than useless because it
        mechanically kills capacity. ARL is therefore the right calibration
        metric for the false-positive side of this detector.
        """

        spread, innovations, beta = self._simulate_stationary_ou()
        stats = self._score_break_process(spread, innovations, beta)
        alarm_idx = np.flatnonzero(stats["break_level"].isin(["ALARM", "ABORT"]).to_numpy())
        if alarm_idx.size < 2:
            arl = float(len(stats))
        else:
            arl = float(np.diff(alarm_idx).mean())
        return [
            {
                "test": "false_positive_rate",
                "variant": "stationary_ou",
                "metric": "average_run_length_bars",
                "value": arl,
            }
        ]

    def pnl_impact_analysis(self) -> list[dict[str, object]]:
        """
        Compare Hurst-only versus Hurst-plus-break-risk gating on OOS history.

        A lightweight daily backtest is used here because the purpose is to
        quantify drawdown control from faster regime recognition, not to
        reproduce every microstructure detail of the full execution engine.
        """

        closes = pd.read_csv("data/closes_daily.csv", index_col=0, parse_dates=True)
        closes.index = pd.to_datetime(closes.index, utc=True)
        pairs = pd.read_csv("data/pairs_selected.csv").head(self.config.validation_pairs)
        opt = pd.read_csv("data/optimal_params.csv")
        opt_map = {row["pair"]: row for _, row in opt.iterrows()}

        base_daily_pnl = pd.Series(dtype=float)
        break_daily_pnl = pd.Series(dtype=float)
        abort_count = 0
        abort_timestamps: list[pd.Timestamp] = []

        for _, row in pairs.iterrows():
            pair = row["pair"]
            t1, t2 = pair.split("-")
            if t1 not in closes.columns or t2 not in closes.columns:
                continue
            pair_px = closes[[t1, t2]].dropna()
            if pair_px.empty:
                continue

            test_start = pd.Timestamp(row["test_start_date"]).tz_localize("UTC")
            pair_px = pair_px[pair_px.index >= test_start]
            if len(pair_px) < 200:
                continue

            beta_init = float(row.get("beta_daily", row["beta"]))
            alpha, beta_arr, innov, _ = kalman_hedge(
                pair_px[t1].to_numpy(dtype=float),
                pair_px[t2].to_numpy(dtype=float),
                beta_init=beta_init,
            )
            spread = pd.Series(innov, index=pair_px.index)
            beta_series = pd.Series(beta_arr, index=pair_px.index)
            z_window = self.config.validation_z_window
            z = ((spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            spread = spread.loc[z.index]
            beta_series = beta_series.loc[z.index]
            innovations = spread.copy()

            stats = self._score_break_process(z, innovations, beta_series)
            base_res = self._run_simple_strategy(
                spread=spread,
                zscore=z,
                break_stats=stats,
                pair_name=pair,
                use_break_gate=False,
                opt_row=opt_map.get(pair),
            )
            break_res = self._run_simple_strategy(
                spread=spread,
                zscore=z,
                break_stats=stats,
                pair_name=pair,
                use_break_gate=True,
                opt_row=opt_map.get(pair),
            )
            base_daily_pnl = base_daily_pnl.add(base_res["daily_pnl"], fill_value=0.0)
            break_daily_pnl = break_daily_pnl.add(break_res["daily_pnl"], fill_value=0.0)
            abort_count += int(break_res["abort_triggers"])
            abort_timestamps.extend(break_res["abort_timestamps"])

        base_result = self._summarize_strategy("hurst_only", base_daily_pnl, 0)
        break_result = self._summarize_strategy("hurst_plus_break_gate", break_daily_pnl, abort_count)
        pnl_saved = self._average_saved_per_abort(base_daily_pnl, break_daily_pnl, abort_timestamps)
        return [
            {
                "test": "pnl_impact",
                "variant": base_result.strategy,
                "metric": "max_drawdown",
                "value": base_result.max_drawdown,
            },
            {
                "test": "pnl_impact",
                "variant": base_result.strategy,
                "metric": "calmar_ratio",
                "value": base_result.calmar_ratio,
            },
            {
                "test": "pnl_impact",
                "variant": break_result.strategy,
                "metric": "max_drawdown",
                "value": break_result.max_drawdown,
            },
            {
                "test": "pnl_impact",
                "variant": break_result.strategy,
                "metric": "calmar_ratio",
                "value": break_result.calmar_ratio,
            },
            {
                "test": "pnl_impact",
                "variant": break_result.strategy,
                "metric": "abort_triggers",
                "value": float(break_result.abort_triggers),
            },
            {
                "test": "pnl_impact",
                "variant": break_result.strategy,
                "metric": "avg_pnl_saved_per_abort",
                "value": float(pnl_saved),
            },
        ]

    def _score_break_process(
        self,
        spread: pd.Series,
        innovations: pd.Series,
        beta: pd.Series,
    ) -> pd.DataFrame:
        frame = self.data_layer.prepare(spread, innovations, beta)
        pieces = [frame]
        pieces.append(self.cusum.compute(frame))
        pieces.append(self.innovation.compute(frame))
        pieces.append(self.half_life.compute(frame["spread"]))
        pieces.append(self.beta_monitor.compute(frame))
        merged = pd.concat(pieces, axis=1).dropna()
        merged = pd.concat([merged, self.composite.compute(merged)], axis=1)
        return merged

    def _simulate_ou_to_explosive(self, n: int = 700, t_break: int = 430):
        rng = np.random.default_rng(42)
        spread = np.zeros(n, dtype=float)
        beta = np.zeros(n, dtype=float)
        innovations = np.zeros(n, dtype=float)
        beta[0] = 1.0
        for t in range(1, n):
            if t < t_break:
                spread[t] = 0.88 * spread[t - 1] + 0.35 * rng.standard_normal()
                beta[t] = beta[t - 1] + 0.002 * rng.standard_normal()
                innovations[t] = spread[t] - 0.88 * spread[t - 1]
            else:
                spread[t] = 1.045 * spread[t - 1] + 0.90 * rng.standard_normal() + 0.12
                beta[t] = beta[t - 1] + 0.04 * rng.standard_normal() + 0.01
                innovations[t] = spread[t] - 1.045 * spread[t - 1]
        idx = pd.RangeIndex(n)
        return (
            pd.Series(spread, index=idx),
            pd.Series(innovations, index=idx),
            pd.Series(beta, index=idx),
            t_break,
        )

    def _simulate_stationary_ou(self, n: int = 2500):
        rng = np.random.default_rng(7)
        spread = np.zeros(n, dtype=float)
        beta = np.zeros(n, dtype=float)
        innovations = np.zeros(n, dtype=float)
        beta[0] = 1.0
        for t in range(1, n):
            spread[t] = 0.92 * spread[t - 1] + 0.28 * rng.standard_normal()
            beta[t] = beta[t - 1] + 0.0015 * rng.standard_normal()
            innovations[t] = spread[t] - 0.92 * spread[t - 1]
        idx = pd.RangeIndex(n)
        return (
            pd.Series(spread, index=idx),
            pd.Series(innovations, index=idx),
            pd.Series(beta, index=idx),
        )

    @staticmethod
    def _bars_after_break(alarm_index, break_index) -> int:
        future = [idx for idx in alarm_index if idx >= break_index]
        if not future:
            return int(1e9)
        first = future[0]
        if isinstance(first, (int, np.integer)) and isinstance(break_index, (int, np.integer)):
            return int(first - break_index)
        alarm_ts = pd.Timestamp(first)
        break_ts = pd.Timestamp(break_index)
        return int(np.searchsorted(pd.Index([alarm_ts]), break_ts))

    def _run_simple_strategy(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        break_stats: pd.DataFrame,
        pair_name: str,
        use_break_gate: bool,
        opt_row,
    ) -> dict[str, object]:
        entry_z = float(opt_row["entry_z"]) if opt_row is not None else 1.8
        exit_z = float(opt_row["exit_z"]) if opt_row is not None else 0.0
        stop_z = float(opt_row["stop_z"]) if opt_row is not None else 3.5

        hurst_filter = HurstFilter(
            h_max=self.config.hurst_threshold,
            window=self.config.hurst_window,
            hurst_lag=self.config.hurst_lag,
        )

        idx = spread.index.intersection(break_stats.index).intersection(zscore.index)
        spread = spread.loc[idx]
        zscore = zscore.loc[idx]
        break_stats = break_stats.loc[idx]

        position = 0
        size = 0.0
        entry_spread = 0.0
        last_spread = None
        realized = pd.Series(0.0, index=idx)
        abort_triggers = 0
        abort_timestamps: list[pd.Timestamp] = []

        for i, ts in enumerate(idx):
            s = float(spread.iloc[i])
            z = float(zscore.iloc[i])
            level = str(break_stats["break_level"].iloc[i])
            action = self.risk_gate.action_for_level(level) if use_break_gate else self.risk_gate.action_for_level("NORMAL")

            if position != 0 and action.reduce_existing_size_factor > 0.0:
                reduction = min(max(action.reduce_existing_size_factor, 0.0), 1.0)
                if reduction > 0.0 and last_spread is not None:
                    realized.iloc[i] += position * (s - last_spread) * size
                size *= (1.0 - reduction)
                if level == "ABORT":
                    abort_triggers += 1
                    abort_timestamps.append(pd.Timestamp(ts))
                if size <= 1e-8:
                    position = 0
                    size = 0.0
                    entry_spread = 0.0
                    last_spread = s
                    continue

            if position != 0 and last_spread is not None:
                realized.iloc[i] += position * (s - last_spread) * size

            if position != 0:
                should_exit = (position == 1 and z >= exit_z) or (position == -1 and z <= -exit_z)
                should_stop = (position == 1 and z <= -stop_z) or (position == -1 and z >= stop_z)
                if should_exit or should_stop:
                    position = 0
                    size = 0.0
                    entry_spread = 0.0
                    last_spread = s
                    continue

            if position == 0:
                blocked_by_hurst, _ = hurst_filter.should_block(spread.loc[:ts], ts)
                if blocked_by_hurst:
                    last_spread = s
                    continue
                if action.cancel_entries:
                    last_spread = s
                    continue
                if z <= -entry_z:
                    position = 1
                    size = 1.0
                    entry_spread = s
                elif z >= entry_z:
                    position = -1
                    size = 1.0
                    entry_spread = s

            last_spread = s

        return {
            "pair": pair_name,
            "daily_pnl": realized,
            "abort_triggers": abort_triggers,
            "abort_timestamps": abort_timestamps,
        }

    @staticmethod
    def _summarize_strategy(strategy: str, daily_pnl: pd.Series, abort_triggers: int) -> StrategyValidationResult:
        daily_pnl = daily_pnl.sort_index()
        equity = daily_pnl.cumsum()
        max_drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
        total_pnl = float(daily_pnl.sum())
        years = max(len(daily_pnl) / 252.0, 1e-9)
        annual_return = total_pnl / years
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        return StrategyValidationResult(strategy, total_pnl, max_drawdown, calmar, abort_triggers)

    def _average_saved_per_abort(
        self,
        base_daily_pnl: pd.Series,
        break_daily_pnl: pd.Series,
        abort_timestamps: list[pd.Timestamp],
    ) -> float:
        """
        Estimate avoided forward loss after each ABORT trigger.

        This metric is more faithful than whole-period P&L attribution because
        an ABORT is meant to prevent the immediate next leg of damage, not to
        guarantee higher total return over the entire backtest.
        """

        if not abort_timestamps:
            return 0.0

        base_cum = base_daily_pnl.sort_index().cumsum()
        break_cum = break_daily_pnl.sort_index().cumsum()
        common = base_cum.index.intersection(break_cum.index)
        base_cum = base_cum.loc[common]
        break_cum = break_cum.loc[common]
        if common.empty:
            return 0.0

        savings: list[float] = []
        for ts in abort_timestamps:
            if ts not in common:
                continue
            start_loc = common.get_loc(ts)
            if isinstance(start_loc, slice):
                start_loc = start_loc.start
            end_loc = min(int(start_loc) + self.config.validation_time_stop, len(common) - 1)
            base_path = base_cum.iloc[int(start_loc) : end_loc + 1] - base_cum.iloc[int(start_loc)]
            break_path = break_cum.iloc[int(start_loc) : end_loc + 1] - break_cum.iloc[int(start_loc)]
            base_worst = float(base_path.min())
            break_worst = float(break_path.min())
            savings.append(break_worst - base_worst)
        return float(np.mean(savings)) if savings else 0.0
