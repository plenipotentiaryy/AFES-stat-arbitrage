import numpy as np
import pandas as pd

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
        """
        Compute rolling Hawkes intensity on a series of shocks (e.g., abs(z_diff)).
        """
        n = len(shocks)
        intensities = np.zeros(n)
        curr_lambda = 0.0
        
        # We assume shocks are values > some quantile (e.g. 1.5 sigma moves)
        shock_values = shocks.values
        
        for i in range(1, n):
            # Decay from previous step
            # dt = 1 unit (5-min bar)
            curr_lambda *= np.exp(-self.beta)
            
            # Add new shock contribution
            if abs(shock_values[i-1]) > 1.5: # Threshold for a 'shock event'
                curr_lambda += self.alpha * abs(shock_values[i-1])
            
            intensities[i] = curr_lambda
            
        return pd.Series(intensities, index=shocks.index)

    def get_execution_penalty(self, intensity: float) -> float:
        """
        Returns a cost multiplier based on toxicity.
        If intensity > threshold, execution is 'toxic' (adverse selection).
        """
        if intensity > self.threshold:
            # Linear increase in slippage beyond threshold
            return 1.0 + (intensity - self.threshold) * 0.5
        return 1.0
