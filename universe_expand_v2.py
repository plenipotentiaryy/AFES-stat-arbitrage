"""universe_expand_v2.py — Second-wave universe extension (§3 of expansion plan).

Adds 20 new candidates on top of the existing 28-pair production universe:

Layer C — Canadian cross-listings (5 pairs, USD-normalized via USDCAD=X):
    RY / RY.TO, TD / TD.TO, BNS / BNS.TO, ENB / ENB.TO, SU / SU.TO

Layer D — Canadian intra-sector pairs (5 pairs, same currency, no FX):
    RY-TD, BNS-RY, ENB-TRP, SU-CNQ, MFC-SLF  (US-listings of Canadian banks/energy/insurance)

Layer E — Style ETF pairs (5 pairs, US session, structural factor exposure):
    SPLV-SPHB, IWD-IWF, VLUE-MTUM, USMV-SPHB, EFV-EFG

Layer F — Cross-asset (5 pairs, structural relationships):
    GLD-SLV, TLT-IEI, HYG-LQD, USO-UNG, XLU-XLY

Output:
    data/closes_daily_extended_v2.csv      — extended price panel
    data/production_universe_v2.csv        — extended candidate definitions (48 rows)
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

# ─────────────────────── Layer C: Canadian cross-listings ───────────────────────
CANADIAN_DUAL = {
    "RY":  ("RY.TO",  "USDCAD=X"),
    "TD":  ("TD.TO",  "USDCAD=X"),
    "BNS": ("BNS.TO", "USDCAD=X"),
    "ENB": ("ENB.TO", "USDCAD=X"),
    "SU":  ("SU.TO",  "USDCAD=X"),
}

# ─────────────────────── Layer D: Canadian intra-sector US-listings ─────────────
CANADIAN_INTRA = [
    ("RY",  "TD",  "ca_banks"),
    ("BNS", "RY",  "ca_banks"),
    ("ENB", "TRP", "ca_pipelines"),
    ("SU",  "CNQ", "ca_oil_sands"),
    ("MFC", "SLF", "ca_insurance"),
]

# ─────────────────────── Layer E: Style ETFs ────────────────────────────────────
STYLE_ETF_PAIRS = [
    ("SPLV", "SPHB", "low_vol_vs_high_beta"),
    ("IWD",  "IWF",  "value_vs_growth_lc"),
    ("VLUE", "MTUM", "value_vs_momentum"),
    ("USMV", "SPHB", "minvol_vs_high_beta"),
    ("EFV",  "EFG",  "intl_value_vs_growth"),
]

# ─────────────────────── Layer F: Cross-asset ───────────────────────────────────
CROSS_ASSET_PAIRS = [
    ("GLD", "SLV", "precious_metals"),
    ("TLT", "IEI", "curve_long_vs_belly"),
    ("HYG", "LQD", "credit_hy_vs_ig"),
    ("USO", "UNG", "oil_vs_gas"),
    ("XLU", "XLY", "defensive_vs_cyclical"),
]


def batched_download(tickers: list[str], start: str, end: str,
                      batch: int = 15, sleep: float = 1.0) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  batch {i//batch + 1}: {chunk}")
        try:
            df = yf.download(chunk, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            print(f"    download error: {e}")
            continue
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"] if "Close" in df.columns.levels[0] else df.iloc[:, :len(chunk)]
        else:
            close = df["Close"] if "Close" in df.columns else df
            if isinstance(close, pd.Series):
                close = close.to_frame(name=chunk[0])
        frames.append(close)
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def align(df: pd.DataFrame, ref_index: pd.DatetimeIndex, max_stale: int = 3) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.reindex(ref_index).ffill(limit=max_stale)


def main():
    existing = (pd.read_csv(DATA / "closes_daily_extended.csv", parse_dates=["Date"])
                  .set_index("Date").sort_index())
    print(f"Existing extended panel: {existing.shape}")
    ref_index = existing.index

    # Collect all tickers we need.
    us_tickers = set()
    for adr in CANADIAN_DUAL.keys():
        us_tickers.add(adr)
    for a, b, _ in CANADIAN_INTRA:
        us_tickers.update([a, b])
    for a, b, _ in STYLE_ETF_PAIRS + CROSS_ASSET_PAIRS:
        us_tickers.update([a, b])

    local_tickers = set(v[0] for v in CANADIAN_DUAL.values())
    fx_tickers    = set(v[1] for v in CANADIAN_DUAL.values())

    # Filter to what's missing.
    us_missing    = sorted([t for t in us_tickers if t not in existing.columns])
    local_missing = sorted([t for t in local_tickers if t not in existing.columns])
    fx_missing    = sorted([t for t in fx_tickers if t not in existing.columns])

    print(f"\nUS tickers needed: {len(us_tickers)}; missing: {len(us_missing)} → {us_missing}")
    print(f"Local tickers needed: {len(local_tickers)}; missing: {len(local_missing)} → {local_missing}")
    print(f"FX tickers needed: {len(fx_tickers)}; missing: {len(fx_missing)} → {fx_missing}")

    parts = [existing]

    if us_missing:
        print("\nDownloading US tickers …")
        us_df = batched_download(us_missing, START, END)
        if not us_df.empty:
            us_df = align(us_df, ref_index)
            cov = (us_df.notna().sum() / len(ref_index) * 100).round(1)
            for t in us_df.columns:
                print(f"  {t:<8s} {cov[t]:>5.1f}% history")
            parts.append(us_df)

    local_usd_df = pd.DataFrame(index=ref_index)
    if local_missing or fx_missing:
        print("\nDownloading local + FX …")
        local_df = batched_download(local_missing, START, END) if local_missing else pd.DataFrame()
        fx_df    = batched_download(fx_missing, START, END) if fx_missing else pd.DataFrame()
        if not fx_df.empty:
            fx_df = align(fx_df, ref_index)
        if not local_df.empty:
            local_df = align(local_df, ref_index)
            for adr, (loc, fx) in CANADIAN_DUAL.items():
                if loc not in local_df.columns or fx not in fx_df.columns:
                    print(f"  skip {adr}: missing {loc} or {fx}")
                    continue
                # USDCAD=X is "1 USD = X CAD", so CAD-price-in-USD = CAD / USDCAD.
                usd_price = local_df[loc] / fx_df[fx]
                local_usd_df[f"{loc}_USD"] = usd_price
                cov = local_usd_df[f"{loc}_USD"].notna().mean() * 100
                print(f"  {loc}_USD: {cov:.1f}% history")
            parts.append(local_usd_df)

    merged = pd.concat(parts, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    out = DATA / "closes_daily_extended_v2.csv"
    merged.to_csv(out)
    print(f"\nSaved → {out}  shape={merged.shape}")

    # ── Build extended production universe rows ──────────────────────────────
    existing_univ = pd.read_csv(DATA / "production_universe.csv")
    new_rows = []

    # Layer C: Canadian cross-listings
    for adr, (loc, fx) in CANADIAN_DUAL.items():
        loc_usd = f"{loc}_USD"
        new_rows.append({
            "pair": f"{adr}-{loc_usd}",
            "leg1": adr, "leg2": loc_usd,
            "source_layer": "canadian_dual_listing",
            "status": "candidate_v2",
            "market_overlap": "us_us",  # same NA timezone
            "structure_type": "pair",
            "requires_fx": True, "local_leg": loc, "fx_ticker": fx, "currency": "CAD",
            "rationale": f"Canadian dual listing {adr}/{loc}, CAD-normalized via USDCAD.",
        })

    # Layer D: Canadian intra-sector
    for a, b, theme in CANADIAN_INTRA:
        new_rows.append({
            "pair": f"{a}-{b}", "leg1": a, "leg2": b,
            "source_layer": "canadian_intra_sector",
            "status": "candidate_v2",
            "market_overlap": "us_us", "structure_type": "pair",
            "requires_fx": False, "currency": "USD",
            "rationale": f"Canadian intra-sector ({theme}) US-listing pair.",
        })

    # Layer E: Style ETFs
    for a, b, theme in STYLE_ETF_PAIRS:
        new_rows.append({
            "pair": f"{a}-{b}", "leg1": a, "leg2": b,
            "source_layer": "style_etf",
            "status": "candidate_v2",
            "market_overlap": "us_us", "structure_type": "pair",
            "requires_fx": False, "currency": "USD",
            "rationale": f"Style ETF pair ({theme}).",
        })

    # Layer F: Cross-asset
    for a, b, theme in CROSS_ASSET_PAIRS:
        new_rows.append({
            "pair": f"{a}-{b}", "leg1": a, "leg2": b,
            "source_layer": "cross_asset",
            "status": "candidate_v2",
            "market_overlap": "us_us", "structure_type": "pair",
            "requires_fx": False, "currency": "USD",
            "rationale": f"Cross-asset structural pair ({theme}).",
        })

    new_df = pd.DataFrame(new_rows)
    full = pd.concat([existing_univ, new_df], axis=0, ignore_index=True)
    out_univ = DATA / "production_universe_v2.csv"
    full.to_csv(out_univ, index=False)
    print(f"\nSaved universe → {out_univ}  rows={len(full)} (existing {len(existing_univ)} + new {len(new_df)})")

    # Also dump a flat candidate-pair list for bias_test.py.
    pair_list = full["pair"].tolist()
    pd.DataFrame({"pair": pair_list}).to_csv(DATA / "pairs_v2_candidates.csv", index=False)
    print(f"Saved candidate pair list → data/pairs_v2_candidates.csv ({len(pair_list)} pairs)")


if __name__ == "__main__":
    main()
