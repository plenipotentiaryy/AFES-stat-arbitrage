"""
step6_paper_trade.py — Forward Testing (Live Simulation)
Fetches recent 15m data via yfinance, loads ML/MVO state from WFO,
and logs virtual trades.
"""

import time
import datetime
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from pathlib import Path

from config import (
    DATA_DIR, OUTPUT_DIR, SIGNAL_START, RTH_END,
    ADAPT_ENABLED, ADAPT_DIAG_CSV, ADAPT_CUSUM_K, ADAPT_CUSUM_H,
    POSTTRADE_ENABLED,
)
from step3j_wfo import build_signals, _precompute_hurst_series, _precompute_adf_pvalues
from filters import validate_kde_density, OnlineParameterAdapter
from metagate import (
    FEATURE_NAMES, build_score_frame, MetaGateModel,
    rolling_cusum, simple_break_score, append_diagnostics,
)
from regime import RegimeState, compute_effective_trade_knobs
from post_trade import ShadowBuffer, PairPerformanceTracker
from feedback import PerformanceFeedbackTracker

PAPER_LOG  = OUTPUT_DIR / "paper_trades_live.csv"
ADAPT_LOG  = OUTPUT_DIR / ADAPT_DIAG_CSV

# Pair-keyed adapter registry. Survives across run_iteration() calls so
# the EWMA state and rolling-vol buffers persist between 15-minute bars.
_ADAPTERS: dict[str, OnlineParameterAdapter] = {}


def _get_or_create_adapter(pair: str, params: dict) -> OnlineParameterAdapter:
    """Lazy per-pair adapter construction with sane WFO-baseline defaults."""
    adapter = _ADAPTERS.get(pair)
    if adapter is None:
        hl = params.get("dynamic_hl", 60.0)
        w_base = int(max(20, min(int(hl), 200)))
        adapter = OnlineParameterAdapter(
            w_base=w_base,
            entry_z_base=float(params.get("entry_z", 1.8)),
            stop_z_base=float(params.get("stop_z",  3.5)),
        )
        _ADAPTERS[pair] = adapter
    return adapter

def fetch_recent_data(tickers):
    """Fetch last 5 days of 15-minute data from yfinance."""
    if not tickers: return None, None
    print(f"Fetching 15m data for {len(tickers)} tickers via yfinance...")
    
    # Download in bulk
    df = yf.download(tickers, period="5d", interval="15m", progress=False)
    if df.empty:
        return None, None
        
    closes = df['Close'].copy()
    volumes = df['Volume'].copy()
    
    # yfinance returns tz-aware local or UTC depending on version. Ensure US/Eastern
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize('UTC')
        volumes.index = volumes.index.tz_localize('UTC')
        
    closes.index = closes.index.tz_convert('US/Eastern')
    volumes.index = volumes.index.tz_convert('US/Eastern')
    
    # Align to market hours
    closes = closes.between_time(SIGNAL_START, RTH_END)
    volumes = volumes.between_time(SIGNAL_START, RTH_END)
    
    return closes, volumes

def run_iteration():
    state_file = OUTPUT_DIR / "live_state.pkl"
    if not state_file.exists():
        print(f"Error: {state_file} not found. Run step3j_wfo.py first.")
        return
        
    state = joblib.load(state_file)
    window_best = state["window_best"]
    weights = state["optimal_weights"]

    # Pair-level performance feedback tracker. Restore from disk if it was
    # written by step3j_wfo.py; otherwise spin up a fresh instance so live
    # trades start populating its FIFO immediately.
    pfb_state = state.get("performance_feedback")
    if pfb_state is not None:
        try:
            performance_feedback = PerformanceFeedbackTracker.from_dict(pfb_state)
        except Exception as e:
            print(f"performance_feedback restore failed ({e}); using a fresh tracker")
            performance_feedback = PerformanceFeedbackTracker()
    else:
        performance_feedback = PerformanceFeedbackTracker()
    state["performance_feedback_obj"] = performance_feedback
    
    active_pairs = [p for p, w in weights.items() if w > 1e-4]
    if not active_pairs:
        print("No active pairs with MVO weight > 0.")
        return
        
    print(f"\n--- Live Iteration: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Loaded {len(active_pairs)} active pairs from live state.")
    
    # Get all unique tickers
    tickers = set()
    for p in active_pairs:
        t1, t2 = p.split("-")
        tickers.add(t1); tickers.add(t2)
        
    closes, volumes = fetch_recent_data(list(tickers))
    if closes is None:
        print("Failed to fetch data.")
        return
        
    trades_taken = []
    
    for pair in active_pairs:
        t1, t2 = pair.split("-")
        params = window_best[pair]
        beta = params.get("dynamic_beta", 1.0)
        hl = params.get("dynamic_hl", 60)
        
        # Build signals for the last 5 days
        try:
            sig = build_signals(closes, volumes, t1, t2, beta, hl)
        except Exception as e:
            print(f"Error building signals for {pair}: {e}")
            continue
            
        if len(sig) < 2:
            continue
            
        # Get the latest bar
        last_bar = sig.iloc[-1]
        z = last_bar["zscore"]
        vr = last_bar["vr"]
        ez = params["entry_z"]
        sz = params.get("stop_z", 3.5)

        try:
            daily_spread = (closes[t1] - beta * closes[t2]).resample("1D").last().dropna()
        except Exception:
            daily_spread = pd.Series(dtype=float)
        hurst_daily = _precompute_hurst_series(daily_spread) if not daily_spread.empty else pd.Series(dtype=float)
        adf_pvals   = _precompute_adf_pvalues(daily_spread)  if not daily_spread.empty else pd.Series(dtype=float)
        scores_live = build_score_frame(sig, hurst_daily, adf_pvals, hmm_series=None)
        regime_state = RegimeState.from_score_row(sig.index[-1], scores_live.iloc[-1])

        # 1. Microstructure Filter (VW-Z)
        if vr < 0.5:
            continue

        # ── Online Parameter Adaptation ─────────────────────────────────
        # Feed the per-pair adapter EVERY bar of the recent window so its
        # EWMA + buffers warm up correctly, then override (ez, sz) with
        # adaptive values for the latest bar.
        adapt_enabled = bool(params.get("adaptive", ADAPT_ENABLED))
        if adapt_enabled:
            knobs = compute_effective_trade_knobs(
                regime_state,
                entry_z_base=float(ez),
                stop_z_base=float(sz),
                max_hold_base=int(max(1, round(float(hl) * 2))),
            )
            ez_adapt = knobs.entry_z_eff
            sz_adapt = knobs.stop_z_eff
            print(f"[{pair}] adaptive  max_hold={knobs.max_hold_eff:3d} "
                  f"entry_z={ez_adapt:.2f} (base {ez:.2f})  "
                  f"stop_z={sz_adapt:.2f} (base {sz:.2f})")
            # Adaptive-params trace (one row per live bar inspected).
            append_diagnostics(ADAPT_LOG, [{
                "timestamp":        sig.index[-1],
                "pair":             pair,
                "raw_spread":       float(sig["spread"].iloc[-1]),
                "break_score":      float(regime_state.break_score),
                "cusum":            float(regime_state.cusum_val),
                "adaptive_window":  np.nan,
                "adaptive_entry_z": ez_adapt,
                "adaptive_stop_z":  sz_adapt,
                "adaptive_max_hold": knobs.max_hold_eff,
                "realized_zscore":  float(z),
            }])
            ez = ez_adapt
            sz = sz_adapt

        # 2. Trigger
        if abs(z) >= ez:
            # 3. Macrostructure Filter (KDE)
            is_kde_valid = validate_kde_density(sig["zscore"], ez, threshold_ratio=0.5)
            if not is_kde_valid:
                print(f"[{pair}] KDE Rejected (LDN) at Z={z:.2f}")
                continue
                
            # 4. MetaGate gating (probabilistic ensemble of risk filters)
            mg: MetaGateModel | None = params.get("metagate_model")
            ml_mult = 1.0
            prob = 0.5
            decision = "ENTER"

            # Post-trade learning: per-pair shadow buffer + perf tracker.
            # Persisted inside `params` so they survive across iterations
            # and accumulate state from prior live bars.
            pt_enabled = bool(params.get("post_trade", POSTTRADE_ENABLED))
            shadow_buf = params.get("shadow_buffer") if pt_enabled else None
            perf_trk   = params.get("perf_tracker")  if pt_enabled else None
            if pt_enabled:
                if shadow_buf is None:
                    shadow_buf = ShadowBuffer()
                    params["shadow_buffer"] = shadow_buf
                if perf_trk is None:
                    perf_trk = PairPerformanceTracker()
                    params["perf_tracker"] = perf_trk

            if mg is not None and mg.usable:
                feat = regime_state.to_features(abs(z), vr)
                # Shadow buffer supersedes the frozen WFO model once it has
                # warmed up past `min_buffer` and fitted a classifier.
                if pt_enabled and shadow_buf is not None and shadow_buf.classifier is not None:
                    prob = shadow_buf.predict_proba(feat)
                    ml_mult = shadow_buf.size_multiplier(prob)
                    decision = f"ENTER(shadow θ={shadow_buf.theta:.2f})" if ml_mult > 0 else "BLOCK(shadow)"
                else:
                    prob = mg.predict_proba(feat)
                    ml_mult = mg.size_multiplier(prob)
                    decision = "ENTER" if ml_mult > 0 else "BLOCK"

                # Apply S_perf (pair-level rolling-Sharpe scaler).
                if pt_enabled and perf_trk is not None:
                    ml_mult *= perf_trk.s_perf
                # Apply pair-level RoN feedback (independent loop with
                # Bayesian shrinkage). Multiplier stays in [0.2, 1.2].
                pfb_mult = performance_feedback.get_multiplier(pair)
                ml_mult *= pfb_mult
            else:
                # Fallback: legacy AND-gate (clf-style 2*(p-0.5)).
                clf = params.get("ml_model")
                if clf is not None:
                    spread_std = last_bar["spread_std"]
                    hour = sig.index[-1].hour + sig.index[-1].minute / 60.0
                    features = np.array([[abs(z), vr, spread_std, hour]])
                    try:
                        prob = clf.predict_proba(features)[0][1]
                        ml_mult = max(0.0, 2.0 * (prob - 0.5))
                        decision = "ENTER" if ml_mult > 0 else "BLOCK"
                    except Exception:
                        pass

            if ml_mult <= 0:
                print(f"[{pair}] {decision} (P(Win)={prob:.2%}) at Z={z:.2f}")
                continue
                
            # EXECUTE VIRTUAL TRADE
            weight = weights[pair]
            direction = "LONG" if z < 0 else "SHORT"
            trades_taken.append({
                "time": sig.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": pair,
                "direction": direction,
                "zscore": round(z, 2),
                "prob_win": round(prob, 3),
                "size_mult": round(ml_mult * weight * len(active_pairs), 3)
            })
            print(f"🚀 [{pair}] SIGNAL {direction} Z={z:.2f} | P(Win)={prob:.0%} | Size={ml_mult:.2f}x")
            
    if trades_taken:
        df_trades = pd.DataFrame(trades_taken)
        hdr = not PAPER_LOG.exists()
        df_trades.to_csv(PAPER_LOG, mode='a', header=hdr, index=False)
        print(f"Logged {len(trades_taken)} trades to {PAPER_LOG.name}")
    else:
        print("No active entry signals on the last bar.")

    # Persist updated state back to live_state.pkl so shadow buffers /
    # perf trackers / performance-feedback FIFOs survive across iterations.
    state["performance_feedback"] = performance_feedback.to_dict()
    state.pop("performance_feedback_obj", None)
    try:
        joblib.dump(state, state_file)
    except Exception as e:
        print(f"Warning: could not persist live_state.pkl: {e}")

if __name__ == "__main__":
    run_iteration()
