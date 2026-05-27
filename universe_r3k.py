"""universe_r3k.py — Fetch S&P 1500 constituent universe and download history.

Sources (Wikipedia, current snapshot):
    S&P 500 (large-cap)   ~500 names
    S&P 400 mid-cap        ~400 names
    S&P 600 small-cap      ~600 names
    Total                  ~1500 unique names

⚠ SURVIVORSHIP BIAS ⚠
Yahoo and Wikipedia both give the CURRENT constituent lists. Any 20-year
backtest on this universe systematically excludes companies that were
delisted, acquired, or removed from the index. This inflates measured
Sharpe by a meaningful amount — literature estimates +0.2 to +0.5 Sharpe
for survivor-only universes vs full historical.

The downloaded universe is suitable for:
    - PCA factor-decomposition stability tests (need wide universe)
    - Sector clustering experiments
    - Sanity checks on whether mid/small-cap mean-reversion exists
NOT suitable for:
    - Production deployment claims of Sharpe (use bias_test instead)
    - Replicating Avellaneda-Lee 2010 academic claims rigorously

Run:
    python universe_r3k.py
        → fetches S&P 1500 tickers
        → downloads daily history from yfinance
        → saves data/closes_daily_r3k.csv
"""
from __future__ import annotations
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests
import yfinance as yf

DATA = Path("data")
START = "2006-01-01"
END   = "2026-05-01"

WIKI_PAGES = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400":  "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600":  "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def fetch_wiki_constituents(name: str, url: str) -> list[str]:
    """Pull ticker column from a Wikipedia constituent table."""
    print(f"  {name}: fetching {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    for tbl in tables:
        for col in tbl.columns:
            if str(col).lower() in ("symbol", "ticker", "ticker symbol"):
                tickers = (tbl[col].astype(str)
                            .str.replace(".", "-", regex=False)   # BRK.B → BRK-B (yfinance style)
                            .str.upper().str.strip()
                            .tolist())
                tickers = [t for t in tickers if t and t != "NAN"]
                print(f"    {len(tickers)} tickers")
                return tickers
    raise RuntimeError(f"No ticker column found in {url}")


def batched_download(tickers: list[str], start: str, end: str,
                      batch: int = 30, sleep: float = 1.5) -> pd.DataFrame:
    frames = []
    n_batches = (len(tickers) + batch - 1) // batch
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        bn = i // batch + 1
        try:
            df = yf.download(chunk, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            print(f"    [{bn}/{n_batches}] FAIL: {e}")
            continue
        if df.empty:
            print(f"    [{bn}/{n_batches}] empty")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"] if "Close" in df.columns.levels[0] else df.iloc[:, 0:len(chunk)]
        else:
            close = df["Close"] if "Close" in df.columns else df
            if isinstance(close, pd.Series):
                close = close.to_frame(name=chunk[0])
        frames.append(close)
        if bn % 10 == 0:
            print(f"    [{bn}/{n_batches}] cumulative cols: "
                   f"{sum(f.shape[1] for f in frames)}")
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def main():
    print("=" * 70)
    print("Fetching S&P 1500 constituent lists from Wikipedia")
    print("=" * 70)
    all_tickers: set[str] = set()
    for name, url in WIKI_PAGES.items():
        try:
            all_tickers.update(fetch_wiki_constituents(name, url))
        except Exception as e:
            print(f"    {name} failed: {e}")
    tickers = sorted(all_tickers)
    print(f"\nTotal unique tickers across S&P 500/400/600: {len(tickers)}")

    # Filter out anything already in our extended closes file
    existing_path = DATA / "closes_daily_extended.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path, nrows=1).columns.tolist()
    else:
        existing = pd.read_csv(DATA / "closes_daily.csv", nrows=1).columns.tolist()
    new = [t for t in tickers if t not in existing]
    print(f"  already have: {len(tickers) - len(new)}")
    print(f"  to download:  {len(new)}")
    print(f"\n⚠ NOTE: Yahoo provides current constituents only — survivorship bias.")

    print(f"\nDownloading {len(new)} new tickers (batches of 30) …")
    df_new = batched_download(new, START, END)
    print(f"\nDownloaded shape: {df_new.shape}")
    if df_new.empty:
        print("Nothing downloaded; abort.")
        return

    # Normalize index
    if df_new.index.tz is not None:
        df_new.index = df_new.index.tz_localize(None)

    # Load existing data
    existing_df = pd.read_csv(existing_path if existing_path.exists()
                                else DATA / "closes_daily.csv",
                               parse_dates=["Date"]).set_index("Date").sort_index()
    df_new = df_new.reindex(existing_df.index)

    # Coverage report — how many new tickers have ≥ 70% of train history
    train_idx = existing_df.index[existing_df.index <= "2014-12-31"]
    new_train = df_new.loc[train_idx]
    cov = (new_train.notna().sum() / len(train_idx) * 100).round(1)
    full = cov[cov >= 70].index.tolist()
    short = cov[cov < 70].index.tolist()
    print(f"\nTrain-window coverage (2006-01 → 2014-12, {len(train_idx)} days):")
    print(f"  ≥70% history:  {len(full)} tickers   ← usable for honest screening")
    print(f"  <70% history:  {len(short)} tickers  ← post-2014 IPOs / sparse")

    # Merge and save
    combined = pd.concat([existing_df, df_new], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    out = DATA / "closes_daily_r3k.csv"
    combined.to_csv(out)
    print(f"\nSaved → {out}")
    print(f"  total shape: {combined.shape}  ({combined.shape[1] - existing_df.shape[1]} new cols)")
    print(f"  file size:   {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
