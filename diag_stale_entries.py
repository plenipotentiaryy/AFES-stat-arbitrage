"""diag_stale_entries.py — entries on a stale print (audit task T2).

For a trades file of step3j_wfo_daily.py, prints the share (and count) of
entries where either leg had an exactly zero return on the signal day
(`entry_signal`; `entry` in old files, whose fills were on the signal day),
overall and per pair.  A leg's return on day d is its close on d against its
previous non-NaN close; a NaN close on d counts as stale (no print).

With --local-closes (raw local-currency closes on their own calendar), an
"X_USD" leg is judged on its local series X reindexed to the panel calendar
with forward fill — the price the legacy panel actually used — so an entry
on a local holiday shows as a zero return.  On USD-converted legs a stale
local price is not an exact zero, because the FX rate moves.

    python diag_stale_entries.py --trades data/wfo_daily_results.csv \\
        --closes data/closes_daily_extended_v2.csv [--local-closes local.csv]

After the T2 fix the share is 0.
"""
from __future__ import annotations

import argparse

import pandas as pd

from cross_listing import repeats_previous_print


def read_frame(path: str) -> pd.DataFrame:
    """A CSV with dates in its first column, as the repo writes them."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def leg_series(leg: str, closes: pd.DataFrame, local_closes: pd.DataFrame | None) -> pd.Series:
    """The prices a leg is judged on: its panel column, or for an "X_USD" leg
    with X in local_closes, X forward-filled onto the panel calendar."""
    local = leg[:-len("_USD")] if leg.endswith("_USD") else None
    if local_closes is not None and local in local_closes.columns:
        return local_closes[local].reindex(closes.index).ffill()
    return closes[leg]


def stale_entries(trades: pd.DataFrame, closes: pd.DataFrame,
                  local_closes: pd.DataFrame | None = None) -> pd.Series:
    """Per trade (trades' index): True if either leg's close on the signal day
    is missing or equals that leg's previous non-NaN close (the engine's
    zero-return rule, cross_listing.repeats_previous_print)."""
    signal = pd.to_datetime(trades["entry_signal"] if "entry_signal" in trades.columns
                            else trades["entry"])
    stale = pd.Series(False, index=trades.index)
    for pair, rows in trades.groupby("pair").groups.items():
        for leg in pair.split("-"):
            px = leg_series(leg, closes, local_closes)
            no_fresh_print = px.isna() | repeats_previous_print(px)
            stale.loc[rows] |= no_fresh_print.reindex(signal[rows], fill_value=True).to_numpy()
    return stale


def report(trades: pd.DataFrame, closes: pd.DataFrame,
           local_closes: pd.DataFrame | None = None) -> pd.DataFrame:
    """entries / stale / share per pair."""
    stale = stale_entries(trades, closes, local_closes)
    per_pair = stale.groupby(trades["pair"]).agg(entries="size", stale="sum")
    per_pair["share"] = per_pair["stale"] / per_pair["entries"]
    return per_pair


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trades", required=True, help="trades CSV of step3j_wfo_daily.py")
    p.add_argument("--closes", required=True, help="the close panel the run used")
    p.add_argument("--local-closes", default=None,
                   help="raw local-currency closes (own calendar) of the X_USD legs")
    args = p.parse_args(argv)

    trades = pd.read_csv(args.trades)
    closes = read_frame(args.closes)
    local = read_frame(args.local_closes) if args.local_closes else None
    per_pair = report(trades, closes, local)
    n, k = int(per_pair["entries"].sum()), int(per_pair["stale"].sum())
    print(f"trades: {args.trades}")
    print(f"closes: {args.closes}" + (f"   local closes: {args.local_closes}" if local is not None else ""))
    print(f"stale entries: {k} of {n} ({k / n if n else 0.0:.2%})  "
          "— either leg's close on the signal day missing or equal to its previous print")
    print(per_pair.assign(share=per_pair["share"].map("{:.2%}".format)).to_string())


if __name__ == "__main__":
    main()
