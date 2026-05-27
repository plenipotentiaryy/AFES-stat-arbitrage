"""universe_expand.py — Download extended universe (§2 of spec).

Tier A — sub-sector / thematic / international ETFs (clean, liquid, 2006+):
    sub-sector:    KRE, SMH, IBB, OIH, ITA, IYR, VNQ, IAI, KIE, XME, XHB, XRT
    international: EWZ, EWG, EWJ, FXI, EEM, EFA
    cross-asset:   TLT, IEF, HYG, JNK, GLD, SLV, USO, UNG, DBC
    factor:        IWM, MDY, IWB, SPY, QQQ

Tier B — ADR vs local share (currency-normalised to USD):
    BHP    vs BHP.AX  (AUD)
    RIO    vs RIO.AX  (AUD)
    SHEL   vs SHEL.L  (GBP)
    BP     vs BP.L    (GBP)
    SAP    vs SAP.DE  (EUR)
    HSBC   vs HSBC.L  (GBP)
    SNY    vs SAN.PA  (EUR)
    NVS    vs NOVN.SW (CHF)
    Plus FX series for currency normalisation.

Output: data/closes_daily_extended.csv merged with existing closes_daily.
"""
from __future__ import annotations
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path("data")
START = "2006-01-01"
END   = "2026-05-01"

TIER_A_ETFS = [
    # Sub-sector
    "KRE", "SMH", "IBB", "OIH", "ITA", "IYR", "VNQ", "IAI", "KIE",
    "XME", "XHB", "XRT",
    # International
    "EWZ", "EWG", "EWJ", "FXI", "EEM", "EFA",
    # Cross-asset
    "TLT", "IEF", "HYG", "JNK", "GLD", "SLV", "USO", "UNG", "DBC",
    # Factor / broad
    "IWM", "MDY", "IWB", "QQQ",
]

# ADR (US) + local listings + FX series (Yahoo tickers).
TIER_B_ADRS = {
    "BHP":  ("BHP.AX",  "AUDUSD=X"),
    "RIO":  ("RIO.AX",  "AUDUSD=X"),
    "SHEL": ("SHEL.L",  "GBPUSD=X"),
    "BP":   ("BP.L",    "GBPUSD=X"),
    "SAP":  ("SAP.DE",  "EURUSD=X"),
    "HSBC": ("HSBC.L",  "GBPUSD=X"),
    "SNY":  ("SAN.PA",  "EURUSD=X"),
    "NVS":  ("NOVN.SW", "CHFUSD=X"),
}


def batched_download(tickers: list[str], start: str, end: str,
                      batch: int = 25, sleep: float = 1.0) -> pd.DataFrame:
    """Yahoo batch download with throttling. Returns DataFrame of Close prices."""
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  batch {i//batch + 1}: {chunk[:5]}{' ...' if len(chunk) > 5 else ''}")
        try:
            df = yf.download(chunk, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            print(f"    ✗ download error: {e}")
            continue
        if df.empty:
            continue
        # Multi-index → take "Close" level
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.levels[0]:
                close = df["Close"]
            else:
                close = df.iloc[:, 0:len(chunk)]
        else:
            close = df["Close"] if "Close" in df.columns else df
            if isinstance(close, pd.Series):
                close = close.to_frame(name=chunk[0])
        frames.append(close)
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    # Drop duplicate columns (multi-batch overlaps)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def align_to_us_calendar(df: pd.DataFrame, ref_index: pd.DatetimeIndex,
                          max_stale: int = 3) -> pd.DataFrame:
    """Reindex foreign-market data to US trading calendar.
    Forward-fill up to `max_stale` days for foreign holidays / late prints."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.reindex(ref_index).ffill(limit=max_stale)
    return df


def main():
    print(f"Loading existing closes_daily.csv …")
    existing = pd.read_csv(DATA / "closes_daily.csv",
                            parse_dates=["Date"]).set_index("Date").sort_index()
    print(f"  existing: {existing.shape[0]} days × {existing.shape[1]} tickers")
    ref_index = existing.index

    # ── Tier A: extended ETFs ────────────────────────────────────────────
    new_a = [t for t in TIER_A_ETFS if t not in existing.columns]
    print(f"\nTier A — downloading {len(new_a)} ETFs not in existing data: {new_a}")
    tier_a = batched_download(new_a, START, END)
    if not tier_a.empty:
        tier_a = align_to_us_calendar(tier_a, ref_index)
        # Drop rows with all NaN
        coverage_a = tier_a.notna().sum()
        print(f"  Tier A coverage (days with price):")
        for t in tier_a.columns:
            print(f"    {t:<8s} {coverage_a[t]:>5} / {len(ref_index)}  "
                  f"({coverage_a[t]/len(ref_index)*100:.0f}%)")
    else:
        print("  Tier A download empty.")

    # ── Tier B: ADR / local + FX ─────────────────────────────────────────
    print(f"\nTier B — ADR/local pairs ({len(TIER_B_ADRS)} pairs)")
    local_tickers = list(set(loc for loc, _ in TIER_B_ADRS.values()))
    fx_tickers    = list(set(fx for _, fx in TIER_B_ADRS.values()))
    print(f"  local: {local_tickers}")
    print(f"  FX:    {fx_tickers}")
    tier_b_local = batched_download(local_tickers, START, END)
    tier_b_fx    = batched_download(fx_tickers, START, END)
    print(f"  Local shape: {tier_b_local.shape}, FX shape: {tier_b_fx.shape}")

    # Currency-normalise local prices to USD.
    tier_b_local_usd = pd.DataFrame(index=ref_index)
    fx_aligned = align_to_us_calendar(tier_b_fx, ref_index) if not tier_b_fx.empty else None
    if fx_aligned is not None:
        local_aligned = align_to_us_calendar(tier_b_local, ref_index)
        for adr, (loc, fx) in TIER_B_ADRS.items():
            if loc not in local_aligned.columns or fx not in fx_aligned.columns:
                continue
            # Local price × (1 / FX-rate) → USD. YF FX uses local/USD direction
            # e.g. AUDUSD=X = AUD-quoted-in-USD → multiply by it to get USD.
            usd = local_aligned[loc] * fx_aligned[fx]
            tier_b_local_usd[f"{loc}_USD"] = usd
        print(f"\n  USD-normalised local listings: "
              f"{[c for c in tier_b_local_usd.columns]}")

    # Make sure ADR US tickers themselves are downloaded too (some may be in
    # existing data, some not).
    adr_us = [a for a in TIER_B_ADRS.keys() if a not in existing.columns]
    if adr_us:
        print(f"\n  Downloading missing ADR US tickers: {adr_us}")
        adr_us_df = batched_download(adr_us, START, END)
        if not adr_us_df.empty:
            adr_us_df = align_to_us_calendar(adr_us_df, ref_index)
        else:
            adr_us_df = pd.DataFrame(index=ref_index)
    else:
        adr_us_df = pd.DataFrame(index=ref_index)

    # ── Merge everything ─────────────────────────────────────────────────
    parts = [existing]
    if not tier_a.empty:           parts.append(tier_a)
    if not adr_us_df.empty:        parts.append(adr_us_df)
    if not tier_b_local_usd.empty: parts.append(tier_b_local_usd)
    extended = pd.concat(parts, axis=1).loc[:, ~pd.concat(parts, axis=1).columns.duplicated()]
    out_path = DATA / "closes_daily_extended.csv"
    extended.to_csv(out_path)
    print(f"\nSaved extended universe → {out_path}")
    print(f"  shape: {extended.shape}  ({extended.shape[1] - existing.shape[1]} new tickers)")

    new_cols = [c for c in extended.columns if c not in existing.columns]
    cov = (extended[new_cols].notna().sum() / len(extended) * 100).round(1)
    full_history = cov[cov >= 70].index.tolist()
    print(f"\n  {len(full_history)} new tickers with ≥70% history "
          f"(usable for screening on 2006-2014 train slice)")


if __name__ == "__main__":
    main()
