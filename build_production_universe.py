"""Build the AFES production shortlist for paper/live forward testing.

The research tree contains several universes: broad S&P/Russell scans,
ETF experiments, PCA residual experiments, and ADR/local structural pairs.
This script writes a deliberately narrow paper/live candidate list:

1. The official baseline same-sector pairs that survived the baseline audit.
2. Only ADR/local structural pairs with meaningful Europe/US market overlap.

Australia/US ADR-local pairs such as BHP/BHP.AX and RIO/RIO.AX are retained
as research rows but are not production rows by default.  Their daily backtest
edge is vulnerable to non-synchronous trading bias because Sydney is closed
when the NY ADR signal is observed, so simultaneous execution is not available
for a normal retail/paper setup.

PCA baskets, broad current-constituent scans, permissive ETF/stock correlation
pairs, and rejected candidate-score experiments are intentionally excluded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA = Path("data")
BASELINE_PAIR_SUMMARY = Path("output/baseline/pair_summary_baseline.csv")
OUT_FILE = DATA / "production_universe.csv"


ADR_PRODUCTION_PAIRS = {
    "BHP-BHP.AX_USD": {
        "adr": "BHP",
        "local": "BHP.AX",
        "fx": "AUDUSD=X",
        "currency": "AUD",
        "status": "research_async",
        "market_overlap": "none",
        "rationale": "ADR/local dual listing, AUD-normalized local leg.",
    },
    "RIO-RIO.AX_USD": {
        "adr": "RIO",
        "local": "RIO.AX",
        "fx": "AUDUSD=X",
        "currency": "AUD",
        "status": "research_async",
        "market_overlap": "none",
        "rationale": "ADR/local dual listing, AUD-normalized local leg.",
    },
    "SAP-SAP.DE_USD": {
        "adr": "SAP",
        "local": "SAP.DE",
        "fx": "EURUSD=X",
        "currency": "EUR",
        "status": "production",
        "market_overlap": "europe_us",
        "rationale": "ADR/local dual listing, EUR-normalized local leg.",
    },
    "SNY-SAN.PA_USD": {
        "adr": "SNY",
        "local": "SAN.PA",
        "fx": "EURUSD=X",
        "currency": "EUR",
        "status": "production",
        "market_overlap": "europe_us",
        "rationale": "ADR/local dual listing, EUR-normalized local leg.",
    },
    "NVS-NOVN.SW_USD": {
        "adr": "NVS",
        "local": "NOVN.SW",
        "fx": "CHFUSD=X",
        "currency": "CHF",
        "status": "production",
        "market_overlap": "europe_us",
        "rationale": "ADR/local dual listing, CHF-normalized local leg.",
    },
}


def split_pair(pair: str) -> tuple[str, str]:
    """Split AFES pair names into their two legs.

    AFES pair names use a single hyphen as the pair separator. Synthetic ADR
    local legs use dots and underscores, not hyphens, so the simple split is
    still unambiguous for the current production universe.
    """
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot split pair name {pair!r}; expected exactly one '-'")
    return parts[0], parts[1]


def load_baseline_pairs() -> pd.DataFrame:
    """Load audited baseline pairs and format them as production rows."""
    if not BASELINE_PAIR_SUMMARY.exists():
        raise FileNotFoundError(
            f"{BASELINE_PAIR_SUMMARY} not found. Run audit_baseline.py first."
        )
    df = pd.read_csv(BASELINE_PAIR_SUMMARY)
    required = {"pair", "trades", "win_rate", "total_pnl", "max_dd", "sharpe"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{BASELINE_PAIR_SUMMARY} missing columns: {sorted(missing)}")

    rows = []
    for row in df.sort_values("total_pnl", ascending=False).itertuples(index=False):
        leg1, leg2 = split_pair(row.pair)
        rows.append({
            "pair": row.pair,
            "leg1": leg1,
            "leg2": leg2,
            "source_layer": "baseline_same_sector",
            "status": "production",
            "market_overlap": "us_us",
            "structure_type": "pair",
            "requires_fx": False,
            "local_leg": "",
            "fx_ticker": "",
            "currency": "USD",
            "baseline_trades": int(row.trades),
            "baseline_win_rate": float(row.win_rate),
            "baseline_total_pnl": float(row.total_pnl),
            "baseline_max_dd": float(row.max_dd),
            "baseline_sharpe": float(row.sharpe),
            "rationale": "Audited baseline same-sector relationship.",
        })
    return pd.DataFrame(rows)


def build_adr_rows() -> pd.DataFrame:
    """Return production ADR/local rows with FX metadata for live synthesis."""
    rows = []
    for pair, meta in ADR_PRODUCTION_PAIRS.items():
        leg1, leg2 = split_pair(pair)
        rows.append({
            "pair": pair,
            "leg1": leg1,
            "leg2": leg2,
            "source_layer": "adr_local_structural",
            "status": meta["status"],
            "market_overlap": meta["market_overlap"],
            "structure_type": "pair",
            "requires_fx": True,
            "local_leg": meta["local"],
            "fx_ticker": meta["fx"],
            "currency": meta["currency"],
            "baseline_trades": pd.NA,
            "baseline_win_rate": pd.NA,
            "baseline_total_pnl": pd.NA,
            "baseline_max_dd": pd.NA,
            "baseline_sharpe": pd.NA,
            "rationale": meta["rationale"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_pairs()
    adr = build_adr_rows()
    out = pd.concat([baseline, adr], ignore_index=True)
    if out["pair"].duplicated().any():
        dupes = out.loc[out["pair"].duplicated(), "pair"].tolist()
        raise ValueError(f"Duplicate production pairs: {dupes}")
    out.to_csv(OUT_FILE, index=False)

    print(f"Saved production universe -> {OUT_FILE}")
    print(f"  total pairs: {len(out)}")
    print(out["source_layer"].value_counts().to_string())
    print("\nADR/local pairs:")
    print(out[out["source_layer"].eq("adr_local_structural")]
          [["pair", "local_leg", "fx_ticker", "currency"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
