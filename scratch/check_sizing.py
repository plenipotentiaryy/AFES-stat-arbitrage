import pandas as pd
import numpy as np
from config import DATA_DIR
from step3e_sizing import (
    load_regimes, load_iv, load_global_hmm, load_mc_confidence,
    load_corr_throttle, load_hrp_weights, iv_multiplier_series
)

regimes = load_regimes()
vix, macro_alert_s = load_iv()
global_hmm_s = load_global_hmm()
mc_conf = load_mc_confidence()
corr_throttle_s = load_corr_throttle()
hrp_w = load_hrp_weights()

print("MC Conf:", mc_conf)
print("HRP weights:", hrp_w)
if macro_alert_s is not None:
    print("Macro Alert counts:\n", macro_alert_s.value_counts())
if corr_throttle_s is not None:
    print("Corr Throttle range:", corr_throttle_s.min(), "to", corr_throttle_s.max())
