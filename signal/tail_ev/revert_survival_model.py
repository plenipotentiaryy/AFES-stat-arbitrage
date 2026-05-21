from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import TailEVConfig


@dataclass(slots=True)
class SurvivalFitReport:
    train_size: int
    calibration_size: int
    test_size: int
    base_rate: float
    brier_score: float
    log_loss: float
    roc_auc: float
    walk_forward: pd.DataFrame


class RevertSurvivalModel:
    """
    Time-aware conditional revert classifier for tail entries.

    Logistic regression is used here because the objective is probability
    quality, not raw classification accuracy. A simple, calibrated model is
    preferred over a high-variance learner because the probability feeds
    directly into an EV calculation and unstable probabilities create unstable
    position decisions.
    """

    def __init__(self, config: TailEVConfig):
        self.config = config
        self.feature_columns = list(config.feature_columns)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.random_state,
        )
        self.calibrator = LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
        )
        self.constant_probability_: float | None = None
        self.report_: SurvivalFitReport | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        timestamps: pd.Index | None = None,
    ) -> SurvivalFitReport:
        """
        Fit the revert model using a chronological split only.

        The data is partitioned into train/calibration/test segments in time
        order. This avoids leakage from future regimes into the tail
        probabilities, which would be especially dangerous when the model is
        later used during stress periods.
        """

        Xf = self._prepare_frame(X)
        yv = pd.Series(y, index=Xf.index if timestamps is None else timestamps).astype(int)
        n = len(Xf)
        if n < self.config.min_train_size:
            raise ValueError(f"need at least {self.config.min_train_size} rows, got {n}")

        test_size = max(1, int(n * self.config.test_fraction))
        cal_size = max(1, int(n * self.config.calibration_fraction))
        train_size = n - test_size - cal_size
        if train_size < max(10, self.config.min_train_size // 2):
            raise ValueError("not enough rows left for chronological train segment")

        X_train = Xf.iloc[:train_size]
        y_train = yv.iloc[:train_size]
        X_cal = Xf.iloc[train_size : train_size + cal_size]
        y_cal = yv.iloc[train_size : train_size + cal_size]
        X_test = Xf.iloc[train_size + cal_size :]
        y_test = yv.iloc[train_size + cal_size :]

        if y_train.nunique() < 2:
            self.constant_probability_ = float(y_train.mean())
            preds = np.full(len(y_test), self.constant_probability_, dtype=float)
            report = self._build_report(
                train_size,
                cal_size,
                len(X_test),
                float(yv.mean()),
                y_test,
                preds,
                Xf,
                yv,
            )
            self.report_ = report
            self.is_fitted_ = True
            return report

        X_train_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_s, y_train)

        cal_scores = self.model.decision_function(self.scaler.transform(X_cal))
        if y_cal.nunique() < 2:
            self.constant_probability_ = float(y_train.mean())
        else:
            self.calibrator.fit(cal_scores.reshape(-1, 1), y_cal)
            self.constant_probability_ = None

        self.is_fitted_ = True
        preds = self.predict_proba(X_test)
        report = self._build_report(
            train_size,
            cal_size,
            len(X_test),
            float(yv.mean()),
            y_test,
            preds,
            Xf,
            yv,
        )
        self.report_ = report
        return report

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return calibrated revert probabilities.

        The output is the probability of reverting to the target before the
        stop. Calibration is applied on top of the base logistic score because
        the EV gate is sensitive to probability distortion in the tails.
        """

        if not self.is_fitted_ and self.constant_probability_ is None:
            raise ValueError("model is not fitted")

        Xf = self._prepare_frame(X)
        if self.constant_probability_ is not None:
            return np.full(len(Xf), self.constant_probability_, dtype=float)

        raw_scores = self.model.decision_function(self.scaler.transform(Xf))
        if hasattr(self.calibrator, "coef_"):
            return self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return self.model.predict_proba(self.scaler.transform(Xf))[:, 1]

    def walk_forward_validate(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """
        Evaluate the classifier in expanding-window folds.

        Walk-forward validation is required because random shuffles would mix
        calm and stressed regimes and produce artificially smooth tail
        probabilities.
        """

        Xf = self._prepare_frame(X)
        yv = np.asarray(y, dtype=int)
        n = len(Xf)
        if n < self.config.min_train_size + 10:
            return pd.DataFrame(columns=["fold", "train_end", "test_end", "brier", "roc_auc"])

        test_block = max(5, n // (self.config.walk_forward_splits + 2))
        rows: list[dict[str, float]] = []
        fold = 0
        for train_end in range(self.config.min_train_size, n - test_block + 1, test_block):
            test_end = min(train_end + test_block, n)
            fold += 1
            X_train = Xf.iloc[:train_end]
            y_train = yv[:train_end]
            X_test = Xf.iloc[train_end:test_end]
            y_test = yv[train_end:test_end]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue

            scaler = StandardScaler()
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.config.random_state,
            )
            model.fit(scaler.fit_transform(X_train), y_train)
            preds = model.predict_proba(scaler.transform(X_test))[:, 1]
            rows.append(
                {
                    "fold": float(fold),
                    "train_end": float(train_end),
                    "test_end": float(test_end),
                    "brier": float(brier_score_loss(y_test, preds)),
                    "roc_auc": float(roc_auc_score(y_test, preds)),
                }
            )
        return pd.DataFrame(rows)

    def _prepare_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_columns)
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = 0.0
        return frame[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _build_report(
        self,
        train_size: int,
        calibration_size: int,
        test_size: int,
        base_rate: float,
        y_test: pd.Series,
        preds: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SurvivalFitReport:
        if len(y_test) == 0:
            brier = logloss = roc_auc = np.nan
        else:
            preds = np.clip(np.asarray(preds, dtype=float), 1e-6, 1 - 1e-6)
            brier = float(brier_score_loss(y_test, preds))
            logloss = (
                float(log_loss(y_test, preds, labels=[0, 1]))
                if y_test.nunique() > 1
                else np.nan
            )
            roc_auc = float(roc_auc_score(y_test, preds)) if y_test.nunique() > 1 else np.nan
        return SurvivalFitReport(
            train_size=train_size,
            calibration_size=calibration_size,
            test_size=test_size,
            base_rate=base_rate,
            brier_score=brier,
            log_loss=logloss,
            roc_auc=roc_auc,
            walk_forward=self.walk_forward_validate(X, y),
        )
