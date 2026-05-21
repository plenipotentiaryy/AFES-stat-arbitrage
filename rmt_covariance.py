import numpy as np
import pandas as pd

def clean_covariance_rmt(returns: pd.DataFrame, q: float = None) -> pd.DataFrame:
    """
    Cleans the covariance matrix using Random Matrix Theory (Marchenko-Pastur).
    
    Parameters:
    - returns: T x N dataframe of returns
    - q: N/T ratio. If None, calculated from returns.
    
    Returns:
    - Cleaned covariance matrix as a DataFrame
    """
    T, N = returns.shape
    if q is None:
        q = N / T
        
    # 1. Compute empirical correlation matrix
    corr = returns.corr().fillna(0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    
    # 2. Marchenko-Pastur boundaries
    # sigma^2 is the variance of the bulk (usually 1.0 for correlation matrix)
    sigma2 = 1.0 
    lambda_plus = sigma2 * (1 + np.sqrt(q))**2
    
    # 3. Identify the bulk (noise eigenvalues)
    bulk_indices = eigvals <= lambda_plus
    
    # 4. Clean eigenvalues: shrink bulk towards its mean
    cleaned_eigvals = eigvals.copy()
    if np.any(bulk_indices):
        bulk_mean = eigvals[bulk_indices].mean()
        cleaned_eigvals[bulk_indices] = bulk_mean
    
    # 5. Reconstruct correlation matrix
    cleaned_corr = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    
    # 6. Ensure diagonal is 1s
    np.fill_diagonal(cleaned_corr, 1.0)
    
    # 7. Convert back to covariance
    stds = returns.std()
    cleaned_cov = np.outer(stds, stds) * cleaned_corr
    
    return pd.DataFrame(cleaned_cov, index=returns.columns, columns=returns.columns)

def marchenko_pastur_pdf(sigma2, q, n_pts=100):
    lambda_minus = sigma2 * (1 - np.sqrt(q))**2
    lambda_plus = sigma2 * (1 + np.sqrt(q))**2
    x = np.linspace(lambda_minus, lambda_plus, n_pts)
    pdf = q / (2 * np.pi * sigma2 * x) * np.sqrt((lambda_plus - x) * (x - lambda_minus))
    return x, pdf
