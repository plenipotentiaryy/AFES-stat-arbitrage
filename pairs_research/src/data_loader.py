import os
import time
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

# Exclude AI mega-caps to avoid structurally broken or overly dominant pairs
EXCLUDED_TICKERS = {"AAPL", "MSFT", "GOOG", "GOOGL", "META", "NVDA", "AMD"}

def get_sp500_constituents() -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    
    # Rename columns to standard names
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry"
    })
    
    # Clean tickers (e.g., BRK.B -> BRK-B for yfinance)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    
    # Exclude mega-caps
    df = df[~df["ticker"].isin(EXCLUDED_TICKERS)]
    return df[["ticker", "company", "sector", "sub_industry"]]

def generate_sp500_pairs(output_csv: Path) -> pd.DataFrame:
    """Generate Priority 1 and Priority 2 pairs and save to CSV."""
    if output_csv.exists():
        print(f"Loading existing pairs from {output_csv}")
        return pd.read_csv(output_csv)
        
    print("Fetching S&P 500 constituents from Wikipedia...")
    df = get_sp500_constituents()
    
    pairs = []
    seen_pairs = set()
    
    def add_pairs(group_df, priority, rationale_col):
        for name, group in group_df.groupby(rationale_col):
            tickers = group["ticker"].tolist()
            companies = group["company"].tolist()
            sectors = group["sector"].tolist()
            
            # Generate all unique combinations
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, c1, s1 = tickers[i], companies[i], sectors[i]
                    t2, c2, s2 = tickers[j], companies[j], sectors[j]
                    
                    # Ensure consistent ordering (alphabetical by ticker)
                    if t1 > t2:
                        t1, t2 = t2, t1
                        c1, c2 = c2, c1
                    
                    pair_id = f"{t1}-{t2}"
                    if pair_id not in seen_pairs:
                        seen_pairs.add(pair_id)
                        pairs.append({
                            "ticker_1": t1,
                            "company_1": c1,
                            "ticker_2": t2,
                            "company_2": c2,
                            "sector": s1,  # Since they are matched by group, sectors match
                            "sub_industry_or_group": name,
                            "economic_rationale": f"Same {rationale_col.replace('_', ' ').title()}",
                            "priority": priority
                        })
                        
    # Priority 1: Exact GICS Sub-Industry match
    add_pairs(df, priority=1, rationale_col="sub_industry")
    
    # Priority 2: Exact GICS Sector match (Macro-driver)
    # This will skip any pairs already added in Priority 1
    add_pairs(df, priority=2, rationale_col="sector")
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Generated {len(pairs_df)} pairs ({len(pairs_df[pairs_df['priority']==1])} Priority 1, {len(pairs_df[pairs_df['priority']==2])} Priority 2)")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_csv, index=False)
    return pairs_df

def download_ticker(ticker: str, cache_dir: Path, start_date: str) -> bool:
    """Download daily prices for a single ticker and cache to CSV."""
    cache_path = cache_dir / f"{ticker}.csv"
    if cache_path.exists():
        # Check if the cache is recent enough. For simplicity, if it exists, we skip it.
        # A more robust system would check the last date and backfill, but for this exercise 
        # downloading fresh or relying on cache is fine.
        # We will assume if file exists and has > 100 rows, it's good.
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 100:
                return True
        except Exception:
            pass

    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty:
            print(f"  [{ticker}] No data found")
            return False
            
        # yfinance sometimes returns MultiIndex columns if multiple tickers are passed, 
        # or single level if one. Let's ensure it's flat.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # Save to CSV
        df.to_csv(cache_path)
        return True
    except Exception as e:
        print(f"  [{ticker}] Error: {e}")
        return False

def load_all_prices(tickers: list[str], data_dir: Path, years: int = 5) -> pd.DataFrame:
    """Ensure all tickers are downloaded, then load into a single DataFrame of Adj Close prices."""
    cache_dir = data_dir / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    
    print(f"Ensuring data for {len(tickers)} tickers (since {start_date})...")
    
    # Download missing data in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_ticker, t, cache_dir, start_date): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            t = futures[future]
            if i % 50 == 0:
                print(f"  Processed {i}/{len(tickers)} tickers...")
                
    # Load all prices into a single DataFrame
    print("Loading prices into memory...")
    series_list = []
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # yfinance returns 'Adj Close', fallback to 'Close'
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                if col in df.columns:
                    s = df[col]
                    s.name = ticker
                    series_list.append(s)
            except Exception:
                pass
                
    if not series_list:
        return pd.DataFrame()
        
    prices_df = pd.concat(series_list, axis=1)
    prices_df.sort_index(inplace=True)
    return prices_df

def load_all_volumes(tickers: list[str], data_dir: Path) -> pd.DataFrame:
    """Load Volume and Close prices to calculate Dollar Volume."""
    cache_dir = data_dir / "prices"
    series_list = []
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if "Volume" in df.columns and "Close" in df.columns:
                    s = df["Volume"] * df["Close"]
                    s.name = ticker
                    series_list.append(s)
            except Exception:
                pass
                
    if not series_list:
        return pd.DataFrame()
        
    vol_df = pd.concat(series_list, axis=1)
    vol_df.sort_index(inplace=True)
    return vol_df
