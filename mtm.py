"""mtm.py — daily mark-to-market PnL of the daily WFO trades (audit task T4).

PnL comes from positions (hard rule 3).  A trade holds q1 = 1/P1 of leg 1 and
q2 = β/P2 of leg 2 (per $1 of leg-1 notional, times its size), fixed at the
entry fill.  Each leg is marked from its entry fill price to its close on
every date it prints, up to its exit fill, where the last step ends at the
exit fill price (a VWAP fill is marked to the close on its fill date):

    PnL_d = side·size·(q1·(P1_d − P1_{d−1}) − q2·(P2_d − P2_{d−1})),

so a trade's daily PnL sums to its gross PnL; a leg without a print on a
date books nothing that date (its next print carries the move, no price is
filled).  Costs are charged on the fill dates, borrow on each of the pair's
rows with both legs in (entry fill, exit fill].  Audit T8: each fill costs
rate_leg × |leg notional at the fill| summed over the legs (`fill_cost`;
the trade's cost_entry on the entry fill date, cost_exit on the exit fill
date), and each held row's borrow is rate × |q·P| of each short leg at that
row's close (`borrow_rows`; the trade's borrow_rate1 / borrow_rate2, 0 for a
long leg).  legacy_flat_cost trades (borrow rates NaN): half the round trip
on each fill date and borrow / held on each held row, as before T8.

Capital (ruling, audit T4): a constant, the number of pairs in the run ×
CAPITAL_PER_PAIR = 2.0 USD per $1 of leg-1 notional (a pair with |β| = 1
holds $1 long and $1 short).  Daily return = daily MTM PnL / capital.  The
one Sharpe (hard rule 4) is mean / std · √252 of the daily returns over
every business day of the OOS span, zeros included; it does not depend on
the constant.  MaxDD is the deepest fall of the date-ordered equity curve
(cumulative daily returns, starting at 0) below its running peak, as a
fraction of capital.

The CLI (`step3j_wfo_daily.py`) writes the run's series to
wfo_daily_mtm<tag>.csv (Date, pnl, ret) and the per-pair daily PnL to
wfo_daily_mtm_pairs<tag>.csv (Date × pairs, the same dates; its columns sum
to the portfolio pnl exactly) next to wfo_daily_results<tag>.csv.  The
downstream scripts read the portfolio pnl column with `add_series_args` /
`mtm_pnl` (the trades' PnL units, which their tables and formats are made
for; ret = pnl / capital, so Sharpe ratios, t-stats and R² are the same on
either column), checked against the trades file; dashboard_daily.py also
reads the per-pair file (`load_mtm_pairs`).

Audit T10: every script reads a run through `load_mtm` (or, LEGACY,
`legacy_exit_date_pnl`), which asserts that the files and the date range
match the run's meta sidecar wfo_daily_meta<tag>.json (`check_run`: the
trades, MTM and closes file names; the series' dates are exactly the run's
span, rebuilt from the sidecar's config and the trades (`run_span`), and lie
before the final holdout unless it is a --final-holdout run) and prints the
run's span and HEADLINE / NON-HEADLINE status.  A legacy exit-date series
covers the closes days of the run's span only (no zero-padding past it) and
its ratio is labelled "not a Sharpe".
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CAPITAL_PER_PAIR = 2.0     # USD of capital per pair, per $1 of leg-1 notional

LEGACY_LABEL = ("LEGACY exit-date booking — not a Sharpe: each trade's PnL on its exit date "
                "(not a mark-to-market series; comparison only)")
LEGACY_RATIO = "LEGACY exit-date ratio (not a Sharpe)"   # mean / std · √252 of that series
MTM_LABEL = "daily mark-to-market PnL, USD per $1 of leg-1 notional"
SAMPLING_LABEL = "sampling uncertainty only — does not detect bias"     # every bootstrap (T10)


# ── Costs that scale with the position (audit T8) ───────────────────────────
# Shared by the engines (wfo_daily_core) and the daily schedule below, so a
# trade's cost and borrow and its daily charges are the same arithmetic.

def fill_cost(rate1: float, rate2: float, q1: float, q2: float, p1: float, p2: float) -> float:
    """Cost of one fill of both legs per unit size: Σ rate_leg · |q_leg · P_leg|,
    the rates per leg per fill as fractions of the leg's notional at the fill
    price P (per $1 of leg-1 notional the entry notionals are 1 and |β|)."""
    return rate1 * abs(q1 * p1) + rate2 * abs(q2 * p2)


def borrow_rows(rate1: float, rate2: float, q1: float, q2: float, p1, p2) -> np.ndarray:
    """Borrow of each held row per unit size: Σ rate_leg · |q_leg · P_leg| at the
    row's closes p1, p2 (arrays), with rate_leg the leg's borrow per day when
    it is short and 0 when it is long."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    return rate1 * np.abs(q1 * p1) + rate2 * np.abs(q2 * p2)


# ── Daily MTM ───────────────────────────────────────────────────────────────

def _leg_steps(vals: np.ndarray, lo: int, hi: int, p_in: float,
               p_out: float) -> tuple[np.ndarray, np.ndarray]:
    """(rows, price changes) of a leg filled on row lo at p_in and on row hi
    at p_out: on each row in [lo, hi] where the leg prints, its close minus
    the previous mark (the first mark is p_in); the change on row hi ends at
    p_out instead of the close."""
    rows = lo + np.flatnonzero(~np.isnan(vals[lo:hi + 1]))
    assert rows.size and rows[0] == lo and rows[-1] == hi, "a leg prints on both fill dates"
    marks = vals[rows]
    marks[-1] = p_out
    return rows, np.diff(marks, prepend=p_in)


def _trade_arrays(v1: np.ndarray, v2: np.ndarray, lo: int, hi: int,
                  t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gross, cost, borrow) per unit size of trade `t` on rows lo..hi (its
    fill rows) of the legs' closes v1, v2."""
    n = hi - lo + 1
    gross = np.zeros(n)
    rows, dp = _leg_steps(v1, lo, hi, t.p1_entry, t.p1_exit)
    gross[rows - lo] += t.side * t.q1 * dp
    rows, dp = _leg_steps(v2, lo, hi, t.p2_entry, t.p2_exit)
    gross[rows - lo] -= t.side * t.q2 * dp
    # Costs on the fill dates: the entry fill's at the entry, the exit fill's
    # at the exit (legacy_flat_cost: half the round trip each).
    cost = np.zeros(n)
    cost[0] += t.cost_entry
    cost[-1] += t.cost_exit
    # Borrow on each of the pair's rows (both legs print) in (entry, exit]:
    # `held` of them on the engine's panel.
    both = ~(np.isnan(v1[lo + 1:hi + 1]) | np.isnan(v2[lo + 1:hi + 1]))
    assert both.sum() == t.held, "the trade's held rows are the pair's rows of this panel"
    borrow = np.zeros(n)
    if t.held:
        if np.isnan(t.borrow_rate1):           # legacy_flat_cost: per unit size per held row
            borrow[1:][both] = t.borrow / t.held
        else:                                  # audit T8: the short legs' notional at the row's close
            rows = lo + 1 + np.flatnonzero(both)
            borrow[rows - lo] = borrow_rows(t.borrow_rate1, t.borrow_rate2, t.q1, t.q2,
                                            v1[rows], v2[rows])
    return gross, cost, borrow


def trade_mtm(trade, closes: pd.DataFrame) -> pd.DataFrame:
    """One trade's daily mark-to-market per unit size: columns gross, cost,
    borrow on the dates from its entry fill to its exit fill.  Its daily net
    PnL is size·(gross − cost − borrow).  `trade`: a row of the trades table
    (a Series or dict); `closes`: the price panel the engine ran on (stale
    prints masked)."""
    t = SimpleNamespace(**dict(trade))
    t1, t2 = t.pair.split("-")
    lo, hi = closes.index.get_indexer([t.entry, t.exit])
    assert 0 <= lo <= hi
    g, c, b = _trade_arrays(closes[t1].to_numpy(dtype=float), closes[t2].to_numpy(dtype=float),
                            lo, hi, t)
    return pd.DataFrame({"gross": g, "cost": c, "borrow": b}, index=closes.index[lo:hi + 1])


def daily_mtm_pairs(trades: pd.DataFrame, closes: pd.DataFrame, index: pd.DatetimeIndex,
                    pairs=()) -> pd.DataFrame:
    """Per-pair daily MTM PnL: for each pair, Σ over its trades of
    size·(gross − cost − borrow) per date (`trade_mtm`), on `index` with
    zeros on the other days.  Columns: the pairs with trades, in order of
    first appearance in `trades`, then the other `pairs` (all zeros).
    trades: the final trades table (after every filter; per-unit gross, cost,
    borrow, cost_entry, cost_exit, borrow_rate1, borrow_rate2; pnl =
    size·(gross − cost − borrow)); closes: the engine's panel.
    Asserts that no date with PnL falls outside `index` and that each pair's
    series sums to its trades' PnL."""
    cols = {}
    if not trades.empty:
        lo = closes.index.get_indexer(trades["entry"])
        hi = closes.index.get_indexer(trades["exit"])
        assert (lo >= 0).all() and (hi >= lo).all()
        recs = list(trades.itertuples(index=False))
        for pair, pos in trades.groupby("pair", sort=False).indices.items():
            t1, t2 = pair.split("-")
            v1, v2 = closes[t1].to_numpy(dtype=float), closes[t2].to_numpy(dtype=float)
            total = np.zeros(len(closes.index))
            for k in pos:
                t = recs[k]
                g, c, b = _trade_arrays(v1, v2, lo[k], hi[k], t)
                total[lo[k]:hi[k] + 1] += t.size * (g - c - b)
            assert closes.index[total != 0.0].isin(index).all(), "a day with PnL lies outside the index"
            col = pd.Series(total, index=closes.index).reindex(index, fill_value=0.0)
            assert math.isclose(col.sum(), trades["pnl"].iloc[pos].sum(), rel_tol=1e-9, abs_tol=1e-9)
            cols[pair] = col.to_numpy()
    for pair in pairs:
        cols.setdefault(pair, np.zeros(len(index)))
    return pd.DataFrame(cols, index=index, columns=list(cols))


def portfolio_pnl(by_pair: pd.DataFrame) -> pd.Series:
    """Σ over the columns of a per-pair daily PnL frame, added column by
    column in its order (so the per-pair file sums to the portfolio file
    exactly)."""
    total = np.zeros(len(by_pair.index))
    for pair in by_pair.columns:
        total += by_pair[pair].to_numpy()
    return pd.Series(total, index=by_pair.index, name="pnl")


def daily_mtm(trades: pd.DataFrame, closes: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Portfolio daily MTM PnL on `index`: the sum over pairs of
    `daily_mtm_pairs`.  Asserts that it sums to the trades' PnL."""
    out = portfolio_pnl(daily_mtm_pairs(trades, closes, index))
    assert math.isclose(out.sum(), trades["pnl"].sum() if not trades.empty else 0.0,
                        rel_tol=1e-9, abs_tol=1e-9), "the daily series sums to the trades' PnL"
    return out


# ── Statistics on constant capital ──────────────────────────────────────────

def mtm_stats(daily_pnl: pd.Series, capital: float) -> dict:
    """Risk statistics of a daily MTM PnL series on constant `capital`:
    sharpe = mean / std · √252 of the daily returns (sample std, zeros
    included; NaN if the std is 0), ann_return = mean · 252, ann_vol =
    std · √252, max_dd = the deepest fall of the date-ordered equity curve
    (cumulative returns, starting at 0) below its running peak, as a fraction
    of capital (≤ 0), max_dd_peak / max_dd_trough its dates (peak = the last
    date at the running peak before the trough)."""
    nan = float("nan")
    ret = daily_pnl / capital
    out = {"capital": capital, "sharpe": nan, "ann_return": nan, "ann_vol": nan,
           "max_dd": nan, "max_dd_peak": pd.NaT, "max_dd_trough": pd.NaT}
    if ret.empty:
        return out
    sd = ret.std()
    eq = ret.cumsum()
    peak = eq.cummax().clip(lower=0.0)
    dd = eq - peak
    trough = dd.idxmin()
    at_peak = eq.loc[:trough] >= peak.loc[trough]
    out.update({
        "sharpe": ret.mean() / sd * math.sqrt(252) if sd > 0 else nan,
        "ann_return": ret.mean() * 252,
        "ann_vol": sd * math.sqrt(252),
        "max_dd": dd.loc[trough],
        "max_dd_peak": at_peak.index[at_peak.to_numpy()][-1] if at_peak.any() else eq.index[0],
        "max_dd_trough": trough,
    })
    return out


# ── The MTM files, for the downstream scripts ───────────────────────────────

def mtm_path(trades_path: str, kind: str = "wfo_daily_mtm") -> str:
    """An MTM file the CLI writes next to a trades file: wfo_daily_results<tag>.csv
    → <kind><tag>.csv (kind wfo_daily_mtm: the portfolio; wfo_daily_mtm_pairs:
    per pair)."""
    p = Path(trades_path)
    if "wfo_daily_results" not in p.name:
        raise SystemExit(f"cannot derive the {kind} file from {trades_path}: pass its path")
    return str(p.with_name(p.name.replace("wfo_daily_results", kind)))


def read_mtm(path: str) -> pd.DataFrame:
    """A daily MTM file (index Date; portfolio: columns pnl, ret; per pair:
    one column per pair), its floats read back exactly as written."""
    if not Path(path).exists():
        raise SystemExit(f"{path} missing: step3j_wfo_daily.py writes it next to the trades "
                         "file (pass --legacy-exit-date-pnl for the old exit-date booking)")
    return pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip").set_index("Date")


def _same_total(what: str, path: str, total: float, expected: float, against: str) -> None:
    """SystemExit unless `total` (from `path`) equals `expected` (from `against`)."""
    if not math.isclose(total, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise SystemExit(f"{path} does not match {against}: {what} {total:+.10f} vs "
                         f"{expected:+.10f} (a stale MTM file from another run? re-run "
                         "step3j_wfo_daily.py with the same --tag)")


def file_sha256(path) -> str:
    """SHA-256 of a file's bytes (the run's files in its meta sidecar, T10)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_from(index: pd.DatetimeIndex, date, option: str) -> None:
    """Audit T10 (R46): a --split / --start after the first date of the run's
    span would make a script report a sub-span as if it were the run —
    refused (fail closed); a date on or before it changes nothing."""
    if date is not None and len(index) and pd.Timestamp(date) > index.min():
        raise SystemExit(f"{option} {date} falls inside the run's span ({index.min().date()} → "
                         f"{index.max().date()}): the scripts report the whole run, not a sub-span; "
                         f"omit it or pass a date on or before {index.min().date()}")


def trades_span_days(days: pd.DatetimeIndex, trades: pd.DataFrame, split=None) -> pd.DatetimeIndex:
    """The calendar of a LEGACY exit-date series in the research scripts
    (almgren_chriss_repricing, audit_baseline, hmm_modes_proper; audit T10,
    R45 / R47): the `days` from max(`split`, the trades' first OOS start — or
    first entry without an `oos_start` column) to their last exit, not to the
    end of the closes file.  Trades that exit in the final holdout are refused."""
    from wfo_daily_core import HOLDOUT_START     # not at the top: wfo_daily_core imports this module
    exits = pd.to_datetime(trades["exit"])
    if exits.max() >= HOLDOUT_START:
        raise SystemExit(f"exits up to {exits.max().date()} reach the final holdout (from "
                         f"{HOLDOUT_START.date()}); use a development-period run (--end 2023-12-31)")
    first = pd.to_datetime(trades["oos_start"] if "oos_start" in trades.columns else trades["entry"]).min()
    if split is not None:
        first = max(first, pd.Timestamp(split))
    days = pd.DatetimeIndex(days)
    return days[(days >= first) & (days <= exits.max())]


def meta_path(trades_path: str) -> str:
    """The run's meta sidecar the CLI writes next to its trades file (audit
    T8): wfo_daily_results<tag>.csv → wfo_daily_meta<tag>.json."""
    return str(Path(mtm_path(trades_path, "wfo_daily_meta")).with_suffix(".json"))


def run_meta(args) -> dict:
    """The run's meta sidecar (--meta, else derived from --trades), as JSON."""
    return json.loads(Path(args.meta or meta_path(args.trades)).read_text())


def run_end(meta: dict) -> pd.Timestamp | None:
    """The last date a run's series may carry: the run's --end and, unless it
    is a --final-holdout run, the day before the final holdout."""
    from wfo_daily_core import HOLDOUT_START     # not at the top: wfo_daily_core imports this module
    cfg = meta["config"]
    ends = [pd.Timestamp(cfg["end"])] if cfg.get("end") else []
    if not cfg.get("final_holdout"):
        ends.append(HOLDOUT_START - pd.Timedelta(days=1))
    return min(ends) if ends else None


def span_line(index: pd.DatetimeIndex, unit: str) -> str:
    return f"span: {index.min().date()} → {index.max().date()} ({len(index)} {unit})"


def run_span(meta: dict, trades_path: str) -> pd.DatetimeIndex:
    """The run's MTM dates (`wfo_daily_core.mtm_span`), rebuilt from its meta
    sidecar: the windows of its config (start, end, train, oos, step; without
    --start or --end, also the dates of its closes file, which the CLI reads
    from the directory it writes the trades file to) and its trades' last
    exit fill."""
    from wfo_daily_core import make_windows, mtm_span   # not at the top: it imports this module
    cfg = meta["config"]
    dates = pd.DatetimeIndex([])
    if not (cfg.get("start") and cfg.get("end")):
        closes = Path(trades_path).parent / str(meta.get("closes_file"))
        if not closes.exists():
            raise SystemExit(f"date range: cannot rebuild the run's windows without its closes file "
                             f"{closes} (next to the trades file)")
        dates = pd.DatetimeIndex(pd.read_csv(closes, usecols=["Date"], parse_dates=["Date"])["Date"])
    windows = make_windows(pd.DataFrame(index=dates.sort_values()), cfg["train"], cfg["oos"],
                           cfg["step"], start=cfg.get("start"), end=cfg.get("end"))
    if not windows:
        raise SystemExit(f"date range: the run's config gives no OOS window ({trades_path})")
    return mtm_span(windows, pd.read_csv(trades_path, usecols=["exit"], parse_dates=["exit"]))


def check_dates(index: pd.DatetimeIndex, meta: dict, meta_file: str,
                expected: pd.DatetimeIndex | None = None) -> None:
    """SystemExit unless the series dates `index` end on or before the run's
    --end and before the final holdout (`run_end`) and, if given, are exactly
    `expected` (the run's span, `run_span`)."""
    if len(index) == 0:
        raise SystemExit(f"date range: the series of the run {meta_file} is empty")
    cfg = meta["config"]
    first, last, end = index.min(), index.max(), run_end(meta)
    wrong = []
    if end is not None and last > end:
        why = (f"the run's --end {cfg['end']}" if cfg.get("end") and pd.Timestamp(cfg["end"]) == end
               else "the final holdout starts the next day")
        wrong.append(f"ends {last.date()}, after {end.date()} ({why})")
    if expected is not None and not index.equals(expected):
        wrong.append(f"covers {first.date()} → {last.date()} ({len(index)} days), the run's span is "
                     f"{expected.min().date()} → {expected.max().date()} ({len(expected)} business days)")
    if wrong:
        raise SystemExit(f"date range does not match the run's meta sidecar {meta_file}: the series "
                         + "; ".join(wrong))


def check_run(trades_path: str, meta_file: str | None = None, *, mtm_file: str | None = None,
              closes: str | None = None, index: pd.DatetimeIndex | None = None) -> dict:
    """The run's meta sidecar (audit T8; default: derived from the trades
    path), asserted to match what a script reads (audit T10): the trades file,
    the MTM file and the closes file by name, the series dates `index`
    (exactly the run's span, `run_span`; `check_dates`), and the files'
    SHA-256 recorded by the CLI (R44; a sidecar written before they were
    recorded: a warning, contents not verified).  SystemExit on a mismatch or
    a missing sidecar.  Prints the run's HEADLINE / NON-HEADLINE status;
    returns the sidecar."""
    meta_file = meta_file or meta_path(trades_path)
    if not Path(meta_file).exists():
        raise SystemExit(f"{meta_file} missing: step3j_wfo_daily.py writes the run's meta sidecar next "
                         "to its trades file; a series that cannot be checked against it is refused "
                         "(audit T10)")
    meta = json.loads(Path(meta_file).read_text())
    files = meta.get("files") or {}
    for what, given, recorded in [("trades file", trades_path, files.get("trades")),
                                  ("MTM file", mtm_file, files.get("mtm")),
                                  ("closes file", closes, meta.get("closes_file"))]:
        if given is not None and Path(given).name != recorded:
            raise SystemExit(f"{given} does not match the run's meta sidecar {meta_file}: its {what} "
                             f"is {recorded}")
    if index is not None:
        check_dates(index, meta, meta_file, run_span(meta, trades_path))
    check_hashes(meta, meta_file, trades=trades_path, mtm=mtm_file, closes=closes)
    # Final review R49: the headline line names the run's cost model (an
    # older sidecar without it: the line as before).
    label = (meta.get("cost_model") or {}).get("label")
    status = (("HEADLINE RUN" + (f" ({label})" if label else "")) if meta.get("headline") is True
              else "NON-HEADLINE RUN (not a frozen point-in-time universe, legacy options, costs "
                   "below the defaults or a comparison cost model: see the run's banner)")
    print(f"run: {meta_file} (config {str(meta.get('config_hash'))[:12]}…): {status}")
    return meta


def check_hashes(meta: dict, meta_file: str, **files) -> None:
    """SystemExit unless each given file (keyword = its key in the sidecar's
    `sha256`: trades, mtm, mtm_pairs, closes; None = not read) has the SHA-256
    the CLI recorded.  A sidecar without hashes (written before audit T10):
    a loud warning — the contents are not verified."""
    hashes = meta.get("sha256")
    if hashes is None:
        print(f"WARNING: no file hashes recorded in {meta_file} (a sidecar written before audit "
              "T10) — contents not verified")
        return
    for key, path in files.items():
        if path is not None and file_sha256(path) != hashes.get(key):
            raise SystemExit(f"{path} does not match the run's meta sidecar {meta_file}: its SHA-256 "
                             f"is not the {key} file's the run wrote (edited, or from another run)")


def add_series_args(parser) -> None:
    """--mtm, --meta and --legacy-exit-date-pnl of the downstream scripts."""
    parser.add_argument("--mtm", default=None,
                        help="Daily MTM file of the run (Date,pnl,ret); default: the --trades "
                             "path with wfo_daily_results → wfo_daily_mtm.")
    parser.add_argument("--meta", default=None,
                        help="The run's meta sidecar; default: the --trades path with "
                             "wfo_daily_results<tag>.csv → wfo_daily_meta<tag>.json.  The files and "
                             "the date range must match it (audit T10).")
    parser.add_argument("--legacy-exit-date-pnl", action="store_true", dest="legacy_exit_date_pnl",
                        help="LEGACY: book each trade's PnL on its exit date (the pre-audit "
                             "series) instead of reading the MTM file. Comparison only.")


def load_mtm(args, closes: str | None = None) -> pd.DataFrame:
    """The run's daily MTM file (--mtm, else derived from --trades), checked
    against the trades file (its PnL must sum to the trades' PnL) and the
    run's meta sidecar (`check_run`: file names — `closes` too, if the script
    reads it — and the date range)."""
    path = args.mtm or mtm_path(args.trades)
    m = read_mtm(path)
    trades = pd.read_csv(args.trades, usecols=["pnl"])
    _same_total("Σ pnl", path, m["pnl"].sum(), trades["pnl"].sum(), args.trades)
    check_run(args.trades, args.meta, mtm_file=path, closes=closes, index=m.index)
    print(f"series: {MTM_LABEL} ({path})")
    print(span_line(m.index, "business days"))
    return m


def legacy_exit_date_pnl(args, closes: str, start=None, option: str = "--split") -> pd.Series:
    """LEGACY: each trade's PnL on its exit date, zeros on the other closes
    days of the run's span (`run_span`; the pre-T10 series ran from `start`
    to the closes file's last date, zero-padded past the run), checked against
    the run's meta sidecar (trades and closes files; never past the run's end
    or into the final holdout).  A `start` inside the span is refused
    (`check_from`)."""
    meta_file = args.meta or meta_path(args.trades)
    meta = check_run(args.trades, meta_file, closes=closes)
    span = run_span(meta, args.trades)
    days = pd.DatetimeIndex(pd.read_csv(closes, parse_dates=["Date"], usecols=["Date"])["Date"]).normalize()
    days = days[(days >= span.min()) & (days <= span.max())]
    check_from(days, start, option)
    check_dates(days, meta, meta_file)
    tr = pd.read_csv(args.trades, parse_dates=["exit"])
    exits = tr["exit"].dt.normalize()
    print(f"series: {LEGACY_LABEL}")
    print(span_line(days, "closes days"))
    return tr.groupby(exits)["pnl"].sum().reindex(days, fill_value=0.0).rename_axis("Date")


def mtm_pnl(args) -> pd.Series:
    """The run's daily MTM PnL (column pnl of `load_mtm`)."""
    return load_mtm(args)["pnl"]


def load_mtm_pairs(path: str, trades: pd.DataFrame, portfolio: pd.Series,
                   meta: dict | None = None) -> pd.DataFrame | None:
    """The run's per-pair daily MTM file, or None if it does not exist.
    Checked: the name the run's meta sidecar records (if given), the
    portfolio file's dates, the columns sum to the portfolio pnl, and each
    traded pair's column sums to its trades' PnL."""
    if not Path(path).exists():
        return None
    if meta is not None and Path(path).name != (meta.get("files") or {}).get("mtm_pairs"):
        raise SystemExit(f"{path} does not match the run's meta sidecar: its per-pair MTM file is "
                         f"{(meta.get('files') or {}).get('mtm_pairs')}")
    if meta is not None and meta.get("sha256") is not None:
        check_hashes(meta, "(the run's meta sidecar)", mtm_pairs=path)
    by_pair = read_mtm(path)
    if not by_pair.index.equals(portfolio.index):
        raise SystemExit(f"{path} does not match the portfolio MTM file: different dates")
    gap = float(np.abs(portfolio_pnl(by_pair).to_numpy() - portfolio.to_numpy()).max(initial=0.0))
    _same_total("max |Σ pairs − portfolio|", path, gap, 0.0, "the portfolio MTM file")
    for pair, pnl in trades.groupby("pair")["pnl"].sum().items():
        _same_total(f"Σ pnl of {pair}", path,
                    by_pair[pair].sum() if pair in by_pair.columns else float("nan"), pnl, "the trades")
    print(f"per-pair series: {MTM_LABEL} ({path})")
    return by_pair
