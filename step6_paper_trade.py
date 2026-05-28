"""
step6_paper_trade.py — Forward Testing (Live Simulation)
Fetches recent 15m data via yfinance, loads ML/MVO state from WFO,
and logs virtual trades.
"""

import time
import datetime
import os
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
from risk import ExposureCapsController, TailEVRiskGovernor

PAPER_LOG  = OUTPUT_DIR / "paper_trades_live.csv"
ADAPT_LOG  = OUTPUT_DIR / ADAPT_DIAG_CSV
TAIL_RISK_LOG = OUTPUT_DIR / "tail_risk_diagnostics.csv"
LIVE_STATE_FILE = Path(os.environ.get("AFES_LIVE_STATE", str(OUTPUT_DIR / "live_state.pkl")))
PRODUCTION_UNIVERSE_CSV = Path(
    os.environ.get("AFES_PRODUCTION_UNIVERSE", str(DATA_DIR / "production_universe.csv"))
)

SYNTHETIC_USD_LEGS = {
    "BHP.AX_USD": ("BHP.AX", "AUDUSD=X"),
    "RIO.AX_USD": ("RIO.AX", "AUDUSD=X"),
    "SHEL.L_USD": ("SHEL.L", "GBPUSD=X"),
    "BP.L_USD": ("BP.L", "GBPUSD=X"),
    "SAP.DE_USD": ("SAP.DE", "EURUSD=X"),
    "SAN.PA_USD": ("SAN.PA", "EURUSD=X"),
    "NOVN.SW_USD": ("NOVN.SW", "CHFUSD=X"),
}

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


def split_pair_name(pair: str) -> tuple[str, str]:
    """Split an AFES pair key into two trade legs."""
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot split pair name {pair!r}; expected exactly one '-'")
    return parts[0], parts[1]


def load_production_pair_filter(path: Path = PRODUCTION_UNIVERSE_CSV) -> set[str] | None:
    """Load the optional production-universe whitelist.

    When `data/production_universe.csv` exists, paper trading is restricted to
    that file. This prevents research artifacts such as PCA residual baskets,
    permissive ETF experiments, or broad current-constituent scans from leaking
    into live simulation.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "pair" not in df.columns:
        raise ValueError(f"{path} must contain a 'pair' column")
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("production")]
    pairs = {str(p) for p in df["pair"].dropna().unique()}
    print(f"Production universe loaded: {len(pairs)} whitelisted pairs from {path}")
    return pairs


def restore_risk_controllers(state: dict) -> tuple[ExposureCapsController, TailEVRiskGovernor, dict[str, str]]:
    """Restore exposure/tail governors from live_state with safe defaults."""
    exp_state = state.get("exposure_caps")
    tail_state = state.get("tail_ev_governor")
    exposure_caps = (
        ExposureCapsController.from_dict(exp_state)
        if isinstance(exp_state, dict)
        else ExposureCapsController()
    )
    tail_governor = (
        TailEVRiskGovernor.from_dict(tail_state)
        if isinstance(tail_state, dict)
        else TailEVRiskGovernor(max_portfolio_es_pct=1.0)
    )
    pair_sector_map = {
        str(k): str(v) for k, v in state.get("pair_sector_map", {}).items()
    }
    return exposure_caps, tail_governor, pair_sector_map


def append_tail_risk_diagnostics(rows: list[dict]) -> None:
    """Append tail-risk diagnostics to CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not TAIL_RISK_LOG.exists()
    df.to_csv(TAIL_RISK_LOG, mode="a", header=header, index=False)


def expand_yfinance_tickers(tickers: list[str]) -> list[str]:
    """Map AFES synthetic USD legs into raw Yahoo tickers needed to build them."""
    raw = set()
    for ticker in tickers:
        if ticker in SYNTHETIC_USD_LEGS:
            local, fx = SYNTHETIC_USD_LEGS[ticker]
            raw.add(local)
            raw.add(fx)
        else:
            raw.add(ticker)
    return sorted(raw)


def add_synthetic_usd_legs(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Add local-listing USD synthetic columns to a close-like DataFrame."""
    out = df.copy()
    for synthetic in requested:
        if synthetic not in SYNTHETIC_USD_LEGS:
            continue
        local, fx = SYNTHETIC_USD_LEGS[synthetic]
        if local not in out.columns or fx not in out.columns:
            print(f"Warning: cannot build {synthetic}; missing {local} or {fx}")
            continue
        out[synthetic] = out[local].astype(float) * out[fx].astype(float)
    return out


def fetch_recent_data(tickers):
    """Fetch last 5 days of 15-minute data from yfinance."""
    if not tickers: return None, None
    raw_tickers = expand_yfinance_tickers(list(tickers))
    print(
        f"Fetching 15m data for {len(tickers)} AFES legs "
        f"({len(raw_tickers)} Yahoo tickers) via yfinance..."
    )
    
    # Download in bulk
    df = yf.download(raw_tickers, period="5d", interval="15m", progress=False)
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

    closes = add_synthetic_usd_legs(closes, list(tickers))
    # Synthetic FX-normalized local legs do not have a real consolidated volume.
    # Reuse the local listing volume where possible, otherwise leave NaN.
    for synthetic in tickers:
        if synthetic not in SYNTHETIC_USD_LEGS:
            continue
        local, _ = SYNTHETIC_USD_LEGS[synthetic]
        if local in volumes.columns:
            volumes[synthetic] = volumes[local]
    
    return closes, volumes

def run_iteration():
    state_file = LIVE_STATE_FILE
    if not state_file.exists():
        print(
            f"Error: {state_file} not found. Run step3j_wfo.py first, "
            "or run build_production_live_state.py and set AFES_LIVE_STATE."
        )
        return
        
    state = joblib.load(state_file)
    window_best = state["window_best"]
    weights = state["optimal_weights"]
    production_pairs = load_production_pair_filter()
    current_capital = float(state.get("current_capital", 10_000.0))
    open_positions = list(state.get("open_positions", []))
    exposure_caps, tail_governor, pair_sector_map = restore_risk_controllers(state)

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
    if production_pairs is not None:
        before = len(active_pairs)
        active_pairs = [p for p in active_pairs if p in production_pairs]
        skipped = before - len(active_pairs)
        if skipped:
            print(f"Production whitelist skipped {skipped} non-production live_state pairs.")
    if not active_pairs:
        print("No active pairs with MVO weight > 0.")
        return
        
    print(f"\n--- Live Iteration: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Loaded {len(active_pairs)} active pairs from live state.")
    try:
        portfolio_safe, derisk_order = tail_governor.evaluate_portfolio_risk(
            open_positions,
            current_capital,
        )
        portfolio_es = tail_governor.calculate_portfolio_es_usd(open_positions)
    except Exception as e:
        portfolio_safe, derisk_order, portfolio_es = True, [], 0.0
        print(f"Tail governor portfolio evaluation skipped: {e}")
    append_tail_risk_diagnostics([{
        "timestamp": datetime.datetime.now().isoformat(),
        "scope": "portfolio",
        "pair": "",
        "portfolio_es_usd": portfolio_es,
        "max_es_usd": tail_governor.max_portfolio_es_pct * current_capital,
        "is_safe": portfolio_safe,
        "derisk_order": ",".join(derisk_order),
        "open_positions": len(open_positions),
    }])
    if not portfolio_safe:
        print(
            "Tail Governor breach: blocking new entries. "
            f"portfolio_ES=${portfolio_es:.2f}, "
            f"cap=${tail_governor.max_portfolio_es_pct * current_capital:.2f}, "
            f"de-risk order={derisk_order}"
        )
    
    # Get all unique tickers
    tickers = set()
    for p in active_pairs:
        t1, t2 = split_pair_name(p)
        tickers.add(t1); tickers.add(t2)
        
    closes, volumes = fetch_recent_data(list(tickers))
    if closes is None:
        print("Failed to fetch data.")
        return
        
    trades_taken = []
    
    for pair in active_pairs:
        t1, t2 = split_pair_name(pair)
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

            weight = weights[pair]
            nominal_size_usd = current_capital * abs(float(weight)) * abs(float(ml_mult))
            allowed_exp, exp_reason = exposure_caps.check_entry_allowed(
                pair_name=pair,
                direction="LONG" if z < 0 else "SHORT",
                nominal_size_usd=nominal_size_usd,
                open_positions=open_positions,
                current_capital=current_capital,
                pair_sector_map=pair_sector_map,
            )
            if not allowed_exp:
                print(f"[{pair}] exposure blocked: {exp_reason}")
                append_tail_risk_diagnostics([{
                    "timestamp": sig.index[-1],
                    "scope": "candidate",
                    "pair": pair,
                    "decision": "BLOCKED_EXPOSURE",
                    "reason": exp_reason,
                    "nominal_size_usd": nominal_size_usd,
                    "zscore": float(z),
                }])
                continue

            candidate_position = {
                "pair": pair,
                "nominal_size_usd": nominal_size_usd,
                "current_z": float(z),
                "spread_std_usd": float(params.get("spread_std_usd", 1.0)),
                "position_size_units": float(params.get("position_size_units", max(nominal_size_usd, 1.0))) * abs(float(ml_mult)),
                "p_revert": float(prob),
                "expected_gain_usd": max(abs(float(z)) - tail_governor.u_threshold, 0.0)
                    * float(params.get("spread_std_usd", 1.0))
                    * float(params.get("position_size_units", max(nominal_size_usd, 1.0)))
                    * abs(float(ml_mult)),
                "sector": pair_sector_map.get(pair, params.get("sector", "UNKNOWN")),
            }
            allowed_tail, tail_reason, projected_es = tail_governor.check_candidate_entry(
                candidate_position,
                open_positions,
                current_capital,
            )
            append_tail_risk_diagnostics([{
                "timestamp": sig.index[-1],
                "scope": "candidate",
                "pair": pair,
                "decision": "ALLOW" if allowed_tail else "BLOCKED_TAIL_GOVERNOR",
                "reason": tail_reason,
                "nominal_size_usd": nominal_size_usd,
                "zscore": float(z),
                "projected_portfolio_es_usd": projected_es,
                "max_es_usd": tail_governor.max_portfolio_es_pct * current_capital,
                "spread_std_usd": candidate_position["spread_std_usd"],
                "position_size_units": candidate_position["position_size_units"],
            }])
            if not allowed_tail or not portfolio_safe:
                print(f"[{pair}] tail governor blocked: {tail_reason}")
                continue
                
            # EXECUTE VIRTUAL TRADE
            direction = "LONG" if z < 0 else "SHORT"
            trades_taken.append({
                "time": sig.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": pair,
                "direction": direction,
                "zscore": round(z, 2),
                "prob_win": round(prob, 3),
                "size_mult": round(ml_mult * weight * len(active_pairs), 3),
                "nominal_size_usd": round(nominal_size_usd, 2),
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
