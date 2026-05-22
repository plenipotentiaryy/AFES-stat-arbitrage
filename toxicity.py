import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _hawkes_intensity(shock_values: np.ndarray, alpha: float,
                      beta_decay: float, threshold: float) -> np.ndarray:
    n = len(shock_values)
    out = np.zeros(n)
    lam = 0.0
    decay = np.exp(-beta_decay)
    for i in range(1, n):
        lam *= decay
        if abs(shock_values[i - 1]) > threshold:
            lam += alpha * abs(shock_values[i - 1])
        out[i] = lam
    return out


class HawkesToxicityFilter:
    """
    Simplified Hawkes Process for Order Flow Toxicity (Criticality 9 upgrade).
    Estimates self-exciting intensity of 'price shocks' to detect adverse selection.
    
    lambda(t) = lambda_0 + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    """
    
    def __init__(self, alpha: float = 0.8, beta: float = 1.2, threshold: float = 2.5):
        self.alpha = alpha
        self.beta  = beta
        self.threshold = threshold
        self.intensity = 0.0
        
    def compute_intensity(self, shocks: pd.Series) -> pd.Series:
        """Compute rolling Hawkes intensity (numba-accelerated)."""
        vals = np.ascontiguousarray(shocks.fillna(0).values, dtype=np.float64)
        out  = _hawkes_intensity(vals, self.alpha, self.beta, 1.5)
        return pd.Series(out, index=shocks.index)

    def get_execution_penalty(self, intensity: float) -> float:
        """
        Returns a cost multiplier based on toxicity.
        If intensity > threshold, execution is 'toxic' (adverse selection).
        """
        if intensity > self.threshold:
            # Linear increase in slippage beyond threshold
            return 1.0 + (intensity - self.threshold) * 0.5
        return 1.0
