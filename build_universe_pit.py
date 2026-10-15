"""build_universe_pit.py — the point-in-time pairs universe (audit T5).

The headline daily WFO run reads only the file this script writes,
data/universe_pit_<asof>.csv, built from data up to the as-of date and frozen
before any OOS run (step3j_wfo_daily.py marks a run HEADLINE only on such a
file whose as-of date is before the run's first OOS date).

    python build_universe_pit.py --asof 2014-12-31 --constituents <csv> --closes <csv>
                                 [--cross-listings <csv>] [--out <csv>]

Candidates are the index constituents as of the as-of date, from a
historical-constituents file.  Source: this repo has none (the current pairs
were mined from today's index members, with hindsight and survivorship); the
owner provides the file and records its source (vendor, index, download date)
in AUDIT_FIX_LOG.md next to the universe's SHA-256.  File paths are relative to
the working directory.

--constituents (one row per ticker and index-membership spell):
    ticker    the ticker's column name in the closes file
    start     first date of the spell, YYYY-MM-DD (required)
    end       the date the ticker left the index (not a member on that date);
              empty = still a member
    sector    its classification as of the as-of date
    industry  optional: a finer classification as of the as-of date; when the
              column exists the pairs are formed on it instead of `sector`
  The as-of members are the rows with start ≤ asof < end, or end empty; a
  member with two different classifications at the as-of date is an error.

--closes: the repo's closes format (a Date column, one column per ticker,
  NaN = no print).  It is cut to the dates ≤ asof before anything is computed.

--cross-listings (optional): the only way a cross-listing enters — a rule
  fixed as of the as-of date, one row per pair: leg1, leg2, start, end (the
  row applies if start ≤ asof < end, or end empty).  The pair gets the sector
  "cross-listing" and the same screens; the CLI's headline policy (audit T2)
  then decides whether it can be traded.  No such rule exists in the repo yet.

Pairs: every unordered pair of as-of members with the same industry (if the
column exists) else the same sector — a mechanical rule; leg1 < leg2 in string
order, pair = "leg1-leg2".  Screens: `bias_test.screen_pair` — its minimum
history (500 rows with both legs), return correlation ≥ CORR_MIN, Johansen at
JOH_CRIT with 0.1 ≤ |β| ≤ 15 and half-life ≤ HL_MAX, unchanged — on the
prices up to the as-of date only.

Output (--out, default data/universe_pit_<asof>.csv): pair, leg1, leg2,
sector (the classification the pair was formed on: the industry when the
constituents file has that column; "cross-listing"), corr, beta, hl_days,
johansen_stat (the Johansen trace statistic for r = 0 of the same test), asof;
sorted by pair, floats written as %.12g; its SHA-256 is printed.  The sidecar
<out stem>.missing.csv lists the as-of members without a close on or before the
as-of date (ticker, sector): the survivorship gap, printed as a count and a
share of the members.  Truncating the closes after the as-of date gives
byte-identical files.

The CLI parses pair names with pair.split("-"), so an as-of member or a
cross-listing leg whose ticker contains "-" (BRK-B, BF-B) is refused with an
error naming it; mapping such tickers is the owner's decision.  A spell or a
rule row without a start date is refused too.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from bias_test import JOH_DET_ORDER, JOH_K_AR_DIFF, screen_pair
from config import DATA_DIR

COLUMNS = ["pair", "leg1", "leg2", "sector", "corr", "beta", "hl_days", "johansen_stat", "asof"]
PAIR_COLUMNS = ["pair", "leg1", "leg2", "sector"]
FLOAT_FORMAT = "%.12g"
CROSS_LISTING = "cross-listing"


def default_out(asof: str) -> Path:
    return DATA_DIR / f"universe_pit_{asof}.csv"


def _require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} file lacks the column(s) {missing}")


def _in_force(rows: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """start ≤ asof < end, or end empty.  A row without a start date is an
    error (it would otherwise drop out silently, outside the survivorship gap)."""
    start, end = pd.to_datetime(rows["start"]), pd.to_datetime(rows["end"])
    if start.isna().any():
        raise ValueError(f"no start date in row(s) {rows[start.isna()].to_dict('records')}")
    return (start <= asof) & (end.isna() | (asof < end))


def asof_members(constituents: pd.DataFrame, asof) -> pd.DataFrame:
    """The index members as of `asof` (ticker, sector), sorted by ticker;
    `sector` holds the pairing classification (the industry when the file has
    that column)."""
    _require(constituents, ["ticker", "start", "end", "sector"], "constituents")
    asof = pd.Timestamp(asof)
    group = "industry" if "industry" in constituents.columns else "sector"
    m = (constituents.loc[_in_force(constituents, asof), ["ticker", group]]
         .rename(columns={group: "sector"}).drop_duplicates())
    twice = sorted(set(m["ticker"][m["ticker"].duplicated()]))
    if twice:
        raise ValueError(f"members with two {group} values as of {asof.date()}: {twice}")
    return m.sort_values("ticker").reset_index(drop=True)


def same_group_pairs(members: pd.DataFrame) -> pd.DataFrame:
    """All unordered pairs of members with the same (non-empty) classification."""
    classified = members[members["sector"].fillna("") != ""]
    rows = [(f"{a}-{b}", a, b, group)
            for group, g in classified.groupby("sector")
            for a, b in itertools.combinations(sorted(g["ticker"]), 2)]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values("pair").reset_index(drop=True)


def cross_listing_pairs(rules: pd.DataFrame, asof) -> pd.DataFrame:
    """The cross-listed pairs whose rule row is in force on `asof`."""
    _require(rules, ["leg1", "leg2", "start", "end"], "cross-listings")
    r = rules.loc[_in_force(rules, pd.Timestamp(asof))]
    pairs = pd.DataFrame({"pair": r["leg1"] + "-" + r["leg2"], "leg1": r["leg1"],
                          "leg2": r["leg2"], "sector": CROSS_LISTING}, columns=PAIR_COLUMNS)
    return pairs.drop_duplicates("pair").sort_values("pair").reset_index(drop=True)


def johansen_trace(closes: pd.DataFrame, t1: str, t2: str) -> float:
    """Trace statistic (r = 0) of bias_test.johansen_coint's test on the pair:
    the same input (log prices, both legs present) and the same test
    specification (bias_test's JOH_DET_ORDER / JOH_K_AR_DIFF constants)."""
    slc = np.log(closes[[t1, t2]].dropna())
    return float(coint_johansen(slc, det_order=JOH_DET_ORDER, k_ar_diff=JOH_K_AR_DIFF).lr1[0])


def build_universe(constituents: pd.DataFrame, closes: pd.DataFrame, asof,
                   cross_listings: pd.DataFrame | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(universe, missing): the screened pairs of the as-of members (and of
    the cross-listing rule), and the members without a close up to `asof`."""
    asof = pd.Timestamp(asof)
    asof_str = asof.strftime("%Y-%m-%d")
    # Cut first: nothing below sees a price after the as-of date.
    closes = closes.sort_index().loc[:asof]
    assert (closes.index <= asof).all()

    members = asof_members(constituents, asof)
    if members.empty:
        raise ValueError(f"no index members as of {asof_str}")
    cross = None if cross_listings is None else cross_listing_pairs(cross_listings, asof)
    legs = set(members["ticker"]) | (set() if cross is None else set(cross["leg1"]) | set(cross["leg2"]))
    hyphen = sorted(t for t in legs if "-" in t)
    if hyphen:
        raise ValueError(f"tickers containing '-': {hyphen}. The WFO CLI splits pair names on '-', "
                         "so they cannot be legs of a pair; mapping them to other symbols (in the "
                         "constituents and the closes) is the owner's decision when the dataset "
                         "arrives.")
    has_px = closes.reindex(columns=members["ticker"]).notna().any().to_numpy()
    missing = members[~has_px].reset_index(drop=True)
    print(f"as-of members ({asof_str}): {len(members)}")
    print(f"survivorship gap: {len(missing)} of {len(members)} as-of members "
          f"({len(missing) / len(members):.1%}) have no close on or before {asof_str}")

    candidates = same_group_pairs(members)
    if cross is not None:
        candidates = pd.concat([candidates, cross], ignore_index=True).drop_duplicates("pair")
    rows = []
    for c in candidates.itertuples(index=False):
        screened = screen_pair(closes, c.leg1, c.leg2)
        if screened is not None:
            rows.append({"pair": c.pair, "leg1": c.leg1, "leg2": c.leg2, "sector": c.sector,
                         **screened, "johansen_stat": johansen_trace(closes, c.leg1, c.leg2),
                         "asof": asof_str})
    universe = pd.DataFrame(rows, columns=COLUMNS).sort_values("pair").reset_index(drop=True)
    print(f"candidate pairs: {len(candidates)}; passed the screens: {len(universe)}")
    return universe, missing


def write_csv(df: pd.DataFrame, path) -> str:
    """Write `df` in the fixed format; return the SHA-256 of the file's bytes."""
    path = Path(path)
    df.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description="Point-in-time pairs universe (audit T5)")
    p.add_argument("--asof", required=True, help="As-of date, YYYY-MM-DD.")
    p.add_argument("--constituents", required=True,
                   help="Historical index constituents: ticker, start, end, sector[, industry].")
    p.add_argument("--closes", required=True, help="Daily closes CSV (Date + one column per ticker).")
    p.add_argument("--cross-listings", default=None, dest="cross_listings",
                   help="Cross-listing rule fixed as of the as-of date: leg1, leg2, start, end.")
    p.add_argument("--out", default=None,
                   help="Output CSV (default data/universe_pit_<asof>.csv).")
    args = p.parse_args()

    asof = pd.Timestamp(args.asof).strftime("%Y-%m-%d")
    out = Path(args.out) if args.out else default_out(asof)
    read_rules = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)  # noqa: E731
    constituents = read_rules(args.constituents)
    cross = read_rules(args.cross_listings) if args.cross_listings else None
    closes = pd.read_csv(args.closes, parse_dates=["Date"]).set_index("Date").sort_index()
    print(f"POINT-IN-TIME UNIVERSE as of {asof}: constituents {args.constituents}, "
          f"closes {args.closes} (cut to dates ≤ {asof}), cross-listings "
          f"{args.cross_listings or 'none'}")

    universe, missing = build_universe(constituents, closes, asof, cross)
    sidecar = out.with_suffix(".missing.csv")
    digest = write_csv(universe, out)
    digest_missing = write_csv(missing, sidecar)
    print(f"universe → {out}  ({len(universe)} pairs)  SHA-256 {digest}")
    print(f"missing  → {sidecar}  ({len(missing)} members)  SHA-256 {digest_missing}")


if __name__ == "__main__":
    main()
