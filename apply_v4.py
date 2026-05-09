import re
import os

def update_config():
    with open('config.py', 'r') as f:
        content = f.read()

    # Update SIGNAL_START to 09:30
    if 'SIGNAL_START = "10:00"' in content:
        content = content.replace('SIGNAL_START = "10:00"', 'SIGNAL_START = "09:30"  # Opened up to catch Price Discovery (Morning Gaps)')

    # Add Tail Hedge params
    if 'TAIL_HEDGE_DRAG_ANNUAL' not in content:
        tail_params = """
# ── The Black Swan Hedge (Tail Risk Convexity) ───────────────────────────────
TAIL_HEDGE_DRAG_ANNUAL = 0.015  # 1.5% annual drag on portfolio (buying far OTM Puts)
TAIL_HEDGE_PAYOUT_MULT = 10.0   # Convexity multiplier when HMM detects Panic
"""
        content += tail_params

    with open('config.py', 'w') as f:
        f.write(content)

def update_step3j():
    with open('step3j_wfo.py', 'r') as f:
        content = f.read()

    if 'TAIL_HEDGE_DRAG_ANNUAL' not in content:
        # Import tail hedge params
        content = content.replace('from config import (', 'from config import (\n    TAIL_HEDGE_DRAG_ANNUAL, TAIL_HEDGE_PAYOUT_MULT, INITIAL_CAPITAL,')

    # Inject hedge logic at the end of OOS calculation
    #             oos_rows.append({...})
    #             cum_pnl += window_oos_pnl
    
    old_oos_summary = """        # ── End of Window Summary ──────────────────────────────────────────────
        m_wr = (oos_win / window_oos_trades * 100) if window_oos_trades > 0 else 0.0
        
        # WFO OOS Sharpe calculation
        y_frac = max((win_end - oos_start).days / 365.25, 1e-9)"""
        
    new_oos_summary = """        # ── The Black Swan Hedge Simulator (Portfolio Level) ─────────────────────
        oos_days = max((win_end - oos_start).days, 1)
        # Approximate trading days in this specific OOS window (excluding weekends)
        oos_trading_days = int(oos_days * 252 / 365.25)
        
        daily_drag = INITIAL_CAPITAL * (TAIL_HEDGE_DRAG_ANNUAL / 252.0)
        hedge_pnl = 0.0
        
        if global_hmm is not None:
            # We check how many days in the OOS window were Panic (Regime 1)
            win_hmm = global_hmm.loc[oos_start:win_end]
            panic_days = int(win_hmm.sum())
            calm_days = len(win_hmm) - panic_days
            
            # Pay insurance drag on calm days
            hedge_pnl -= calm_days * daily_drag
            # Receive Convexity Payout on panic days
            hedge_pnl += panic_days * daily_drag * TAIL_HEDGE_PAYOUT_MULT
        else:
            hedge_pnl -= oos_trading_days * daily_drag
            
        window_oos_pnl += hedge_pnl
        window_gross   += hedge_pnl # Add to gross to reflect in Sharpe

        # ── End of Window Summary ──────────────────────────────────────────────
        m_wr = (oos_win / window_oos_trades * 100) if window_oos_trades > 0 else 0.0
        
        # WFO OOS Sharpe calculation
        y_frac = max((win_end - oos_start).days / 365.25, 1e-9)"""
        
    if 'The Black Swan Hedge Simulator' not in content:
        content = content.replace(old_oos_summary, new_oos_summary)
        
    # Inject hedge PnL info into the console print
    #         print(f"    Gross:  {window_gross:>+10.4f}  |  Tx:     {window_tx:>+10.4f}")
    old_print = """        print(f"    Gross:  {window_gross:>+10.4f}  |  Tx:     {window_tx:>+10.4f}")"""
    new_print = """        print(f"    Gross:  {window_gross:>+10.4f}  |  Tx:     {window_tx:>+10.4f}")
        print(f"    Hedge:  {hedge_pnl:>+10.4f}  |  Net:    {window_oos_pnl:>+10.4f}")"""
        
    if 'Hedge:' not in content:
        content = content.replace(old_print, new_print)

    with open('step3j_wfo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_config()
    update_step3j()
    print("V4 patches (09:30 open + Tail Hedge) applied successfully.")
