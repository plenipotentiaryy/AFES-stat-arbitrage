import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.linear_model import LogisticRegression

class TailAdjustedEVProfiler:
    """
    Tail-Adjusted Expected Value (Criticality 9 upgrade).
    Uses Peaks-Over-Threshold (POT) with Generalized Pareto Distribution (GPD)
    to model the conditional loss in the tail.
    
    Interface compatible with Step 4 backtest calls.
    """

    def __init__(self, 
                 tail_threshold: float = 3.0,
                 confidence_level: float = 0.95,
                 rr_threshold: float = 2.0,
                 exit_z: float = 0.0,
                 tail_refit_freq: int = 30,
                 gain_col: str = "expected_gain",
                 u_threshold: float = None):
        # Support both u_threshold and tail_threshold for compatibility
        self.u_threshold = u_threshold if u_threshold is not None else tail_threshold
        self.confidence_level = confidence_level
        self.rr_threshold = rr_threshold
        self.exit_z = exit_z
        self.gain_col = gain_col
        
        self.gpd_params = (0.1, 1.0) # xi, beta
        self.revert_model = LogisticRegression()
        
    def fit(self, training_frame: pd.DataFrame, sample_weights: np.ndarray = None):
        """
        Fits both the Logistic Reversion model and the GPD tail model.
        """
        # 1. Fit GPD to tail exceedances
        z = np.abs(training_frame["zscore"].values)
        exceedances = z[z > self.u_threshold] - self.u_threshold
        
        if len(exceedances) >= 10:
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            self.gpd_params = (shape, scale)
            
        # 2. Fit Logistic Reversion Model
        if "revert_label" in training_frame.columns:
            X = training_frame[["zscore", "velocity"]].values
            y = training_frame["revert_label"].values
            self.revert_model.fit(X, y, sample_weight=sample_weights)

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Tail-Adjusted Expected Value (EV) for each row.
        Handles NaN values robustly.
        """
        X_df = df[["zscore", "velocity"]].fillna(0)
        X = X_df.values
        
        try:
            p_revert = self.revert_model.predict_proba(X)[:, 1]
        except Exception:
            p_revert = np.full(len(df), 0.5)
            
        z = np.abs(df["zscore"].fillna(0).values)
        gain = df[self.gain_col].fillna(0).values
        
        # Expected Shortfall from GPD
        xi, beta = self.gpd_params
        x_tail = np.maximum(0, z - self.u_threshold)
        
        if xi < 1.0:
            es_exceedance = (beta + xi * x_tail) / (1 - xi)
        else:
            es_exceedance = x_tail * 2.0
            
        # Conditional EV
        # If in tail: use GPD ES for loss
        # If not: use empirical 0.5*z for loss
        es_loss = np.where(z > self.u_threshold, es_exceedance, z * 0.5)
        
        df["tail_ev"] = p_revert * gain - (1 - p_revert) * es_loss
        df["tail_signal_ok"] = df["tail_ev"] > self.rr_threshold
        return df

    def fit_tail(self, pair_name: str, z_series: pd.Series):
        """Deprecated: use fit() instead. Kept for manual calls."""
        z = np.abs(z_series.dropna().values)
        exceedances = z[z > self.u_threshold] - self.u_threshold
        if len(exceedances) >= 10:
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            self.gpd_params = (shape, scale)

    def calculate_ev(self, pair_name: str, current_z: float, 
                     p_revert: float, expected_gain: float) -> float:
        """Deprecated: use predict_ev() instead. Kept for manual calls."""
        z = abs(current_z)
        if z > self.u_threshold:
            xi, beta = self.gpd_params
            x_tail = z - self.u_threshold
            es = (beta + xi * x_tail) / (1 - xi) if xi < 1.0 else x_tail * 2.0
            return p_revert * expected_gain - (1 - p_revert) * es
        return p_revert * expected_gain - (1 - p_revert) * z * 0.5
