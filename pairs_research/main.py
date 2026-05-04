import sys
import time
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from src.data_loader import generate_sp500_pairs, load_all_prices, load_all_volumes
from src.metrics import compute_all_metrics
from src.filters import compute_adv, structural_break_check
from src.scoring import calculate_composite_score
from src.backtest import run_mini_backtest

def process_pair(pair: dict, prices: pd.DataFrame, volumes: pd.DataFrame) -> dict:
    """Process a single pair: calculate metrics, filters, and score."""
    t1 = pair["ticker_1"]
    t2 = pair["ticker_2"]
    
    # 1. Check liquidity
    adv_1 = compute_adv(volumes[[t1]] if t1 in volumes.columns else pd.DataFrame()).iloc[0] if t1 in volumes.columns else 0.0
    adv_2 = compute_adv(volumes[[t2]] if t2 in volumes.columns else pd.DataFrame()).iloc[0] if t2 in volumes.columns else 0.0
    
    if pd.isna(adv_1) or pd.isna(adv_2) or adv_1 < 50_000_000 or adv_2 < 50_000_000:
        pair["is_liquid"] = False
        pair["is_broken"] = False
        pair["broken_reasons"] = "Illiquid"
        pair["score"] = 0.0
        return pair
        
    pair["is_liquid"] = True
    pair["adv_1"] = adv_1
    pair["adv_2"] = adv_2
    
    # 2. Compute metrics
    metrics = compute_all_metrics(prices, t1, t2)
    pair.update(metrics)
    
    # 3. Structural breaks
    is_broken, reasons = structural_break_check(
        prices, t1, t2, metrics["half_life"], metrics["eg_pvalue"]
    )
    pair["is_broken"] = is_broken
    pair["broken_reasons"] = " | ".join(reasons) if is_broken else ""
    
    # 4. Composite score
    if is_broken:
        pair["score"] = 0.0
    else:
        pair["score"] = calculate_composite_score(
            eg_pvalue=metrics["eg_pvalue"],
            eg_pvalue_std=metrics["eg_pvalue_std"],
            half_life=metrics["half_life"],
            adv_1=adv_1,
            adv_2=adv_2,
            corr_std=metrics["corr_120d_std"],
            hurst=metrics["hurst"],
            priority=pair["priority"]
        )
        
    return pair

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 1. Load or Generate Pairs
    pairs_csv = data_dir / "pairs_input.csv"
    pairs_df = generate_sp500_pairs(pairs_csv)
    
    unique_tickers = list(set(pairs_df["ticker_1"].tolist() + pairs_df["ticker_2"].tolist()))
    
    # 2. Download/Load Market Data
    prices = load_all_prices(unique_tickers, data_dir, years=5)
    volumes = load_all_volumes(unique_tickers, data_dir)
    
    if prices.empty:
        print("No price data loaded. Exiting.")
        return
        
    # 3. Process pairs in parallel
    print(f"\nProcessing {len(pairs_df)} pairs (metrics, filtering, scoring)...")
    processed_pairs = []
    
    start_time = time.time()
    
    # ProcessPoolExecutor for CPU-bound pandas/statsmodels calculations
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_pair, row.to_dict(), prices, volumes): i 
            for i, row in pairs_df.iterrows()
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            processed_pairs.append(future.result())
            if i % 100 == 0:
                print(f"  Processed {i}/{len(pairs_df)} pairs...")
                
    print(f"Processing complete in {time.time() - start_time:.1f}s")
    
    # 4. Filter and sort results
    results_df = pd.DataFrame(processed_pairs)
    
    # Save full results
    results_df.to_csv(results_dir / "full_pairs_results.csv", index=False)
    
    # Save broken pairs
    broken_df = results_df[results_df["is_broken"]]
    broken_df.to_csv(results_dir / "broken_pairs.csv", index=False)
    
    # Save top 20 and 50
    valid_df = results_df[results_df["is_liquid"] & ~results_df["is_broken"]]
    valid_df = valid_df.sort_values("score", ascending=False)
    
    top50 = valid_df.head(50)
    top50.to_csv(results_dir / "top50_pairs.csv", index=False)
    
    top20 = valid_df.head(20)
    top20.to_csv(results_dir / "top20_pairs.csv", index=False)
    
    print("\n" + "="*80)
    print("TOP 20 PAIRS SUMMARY")
    print("="*80)
    
    # 5. Mini-backtest for Top 20
    print("\nRunning Mini-Backtest for Top 20 Pairs (0bps and 5bps costs)...\n")
    
    print(f"{'Pair':<10} | {'Score':<6} | {'HL':<5} | {'Hurst':<6} | {'Corr252':<7} | {'0bps Shrp':<10} | {'5bps Shrp':<10} | {'Rationale'}")
    print("-" * 100)
    
    for _, row in top20.iterrows():
        t1, t2 = row["ticker_1"], row["ticker_2"]
        pair_name = f"{t1}-{t2}"
        score = row["score"]
        hl = row["half_life"]
        hurst = row["hurst"]
        corr = row["corr_252d"]
        beta = row["hedge_ratio"]
        rationale = row["economic_rationale"]
        
        # Backtest 0bps
        bt_0 = run_mini_backtest(prices, t1, t2, beta, transaction_bps=0.0)
        shrp_0 = bt_0.get("sharpe", 0.0)
        
        # Backtest 5bps
        bt_5 = run_mini_backtest(prices, t1, t2, beta, transaction_bps=5.0)
        shrp_5 = bt_5.get("sharpe", 0.0)
        
        print(f"{pair_name:<10} | {score:>5.1f} | {hl:>5.1f} | {hurst:>6.2f} | {corr:>7.2f} | {shrp_0:>10.2f} | {shrp_5:>10.2f} | {rationale}")
        
    print("\nPipeline execution complete. Check 'results/' directory for CSVs.")

if __name__ == "__main__":
    main()
