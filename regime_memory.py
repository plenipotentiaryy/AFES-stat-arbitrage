import numpy as np
import pandas as pd

class RegimeMemoryWeighter:
    """
    State-Weighted Memory (Criticality 8 upgrade).
    Weights historical observations based on recency and regime similarity.
    """
    
    def __init__(self, age_lambda: float = 0.02, hl_gamma: float = 1.0):
        self.age_lambda = age_lambda
        self.hl_gamma = hl_gamma
        
    def compute_weights(self, 
                        df: pd.DataFrame, 
                        now_index: pd.Timestamp,
                        hl_now: float,
                        state_now: int) -> np.ndarray:
        """
        Computes weights for rows in df based on distance from now_index and regime similarity.
        """
        # 1. Recency weights
        ages = (now_index - df.index).total_seconds() / (3600 * 24) # days
        w_age = np.exp(-self.age_lambda * ages)
        
        # 2. HL Regime similarity weights
        # We assume df has 'half_life' column
        hl_hist = df["half_life"].values if "half_life" in df.columns else np.full(len(df), hl_now)
        hl_diff = np.abs(hl_hist - hl_now) / (hl_now + 1e-9)
        w_hl = np.exp(-self.hl_gamma * hl_diff)
        
        # 3. Macro state match (binary boost)
        # We assume df has 'macro_state' column
        state_hist = df["macro_state"].values if "macro_state" in df.columns else np.full(len(df), state_now)
        w_state = np.where(state_hist == state_now, 1.0, 0.5)
        
        # Combined
        w = w_age * w_hl * w_state
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(len(w)) / len(w)

    def get_weights(self, timestamps: pd.DatetimeIndex, historical_hl: np.ndarray, current_hl: float) -> np.ndarray:
        """Backward compatibility helper."""
        now = timestamps[-1]
        ages = (now - timestamps).total_seconds() / (3600 * 24)
        w_recency = np.exp(-self.age_lambda * ages)
        hl_diff = np.abs(historical_hl - current_hl) / (current_hl + 1e-9)
        w_similarity = np.exp(-self.hl_gamma * hl_diff)
        w = w_recency * w_similarity
        return w / np.sum(w)
