import re

def update_filters():
    with open('filters.py', 'r') as f:
        content = f.read()

    kde_code = """
from scipy.stats import gaussian_kde

def validate_kde_density(z_series: pd.Series, entry_z: float, threshold_ratio: float = 0.5) -> bool:
    \"\"\"
    KDE Structural Filter.
    Checks if the historical density at entry_z is at least `threshold_ratio` of the 
    theoretical Gaussian density at entry_z.
    If KDE(entry_z) < Gaussian(entry_z) * threshold_ratio, it's a Low Density Node (void).
    Returns True if valid (HVN or normal), False if invalid (LDN / void).
    \"\"\"
    z = z_series.dropna().values
    if len(z) < 100:
        return True # Not enough data
        
    try:
        kde = gaussian_kde(z)
        
        # We evaluate empirical density at entry_z and -entry_z
        d_pos = kde(entry_z)[0]
        d_neg = kde(-entry_z)[0]
        d_empirical = (d_pos + d_neg) / 2.0
        
        # Theoretical Gaussian PDF at entry_z
        # phi(z) = (1 / sqrt(2*pi)) * e^(-0.5 * z^2)
        d_theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (entry_z ** 2))
        
        return bool(d_empirical >= (d_theoretical * threshold_ratio))
    except Exception:
        # e.g., singular matrix if variance is zero
        return True
"""
    if 'validate_kde_density' not in content:
        content += kde_code
        with open('filters.py', 'w') as f:
            f.write(content)

def update_wfo():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()

    # Fix potential missed volumes arg in build_signals
    content = re.sub(r'build_signals\(closes_test,\s*t1,\s*t2,', r'build_signals(closes_test, volumes, t1, t2,', content)

    # Inject KDE Filter
    old_code = """            if best is None:
                continue

            # 5. Trade OOS with the best params found on TRAIN"""
            
    new_code = """            if best is None:
                continue

            # ── KDE Structural Filter (Quality Check) ─────────────
            from filters import validate_kde_density
            is_kde_valid = validate_kde_density(sig_train["zscore"], best["entry_z"], threshold_ratio=0.5)
            if not is_kde_valid:
                print(f"  {pair_name:<10} TRAIN: REJECTED by KDE (Low Density Node at Z={best['entry_z']})")
                continue

            # 5. Trade OOS with the best params found on TRAIN"""
            
    if 'KDE Structural Filter' not in content:
        content = content.replace(old_code, new_code)
        
    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_filters()
    update_wfo()
    print("V6 patches (KDE Structural Filter) applied successfully.")
