"""step3j_wfo_daily.py — Walk-Forward Optimization on the daily timeframe.

Built on top of the sanity-tested engine in `daily_sanity.py`.  The intraday
`step3j_wfo.py` carries a stack of filters (Johansen on rolling 3y, MetaGate
trained on intraday features, limit-order matching, VW-zscore, grid search
on TRAIN) which all degrade on daily bars — the result on baseline 14 pairs
was Sharpe ≈ 0 with 57 trades vs. sanity's Sharpe ≈ +1.59 with 4564 trades.

This file is the daily replacement: expanding-window WFO with the simple,
robust pieces only.  Filters and gates (HMM / Hurst / MetaGate / sizing /
ticker-overlap) will be layered back in subsequent phases on top of this
backbone, each justified by an A/B comparison vs. this raw baseline.

The engine lives in `wfo_daily_core.py` (`run_wfo_daily`); this file is the
CLI: it reads data/, builds the `WfoConfig`, applies the headline-universe
policy to the pair list (audit T2), prints the summary and the separate
cross-listed table, and writes the trades CSV, the daily mark-to-market
series (wfo_daily_mtm<tag>.csv: Date, pnl, ret — audit T4; the input of
bootstrap_sharpe / dashboard_daily / stress_crisis_analysis /
factor_attribution), the per-pair daily MTM PnL (wfo_daily_mtm_pairs<tag>.csv,
read by dashboard_daily) and the equity chart.  A run is HEADLINE only on a
frozen point-in-time universe (audit T5, `non_headline_reasons`); every other
run prints a NON-HEADLINE banner in its header and summary, and
`WfoResult.stats["headline"]` records the verdict.

CLI:
    python step3j_wfo_daily.py --pairs pairs_baseline14.csv
    python step3j_wfo_daily.py --pairs pairs_selected.csv --cost 10
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, BAR_TIMEFRAME
from build_universe_pit import COLUMNS as PIT_COLUMNS
from cross_listing import close_time
from wfo_daily_core import (
    TRAIN_INIT_MO, OOS_MO, STEP_MO,
    WfoConfig, make_windows, run_wfo_daily, summarise, trade_stats,
)


def save_equity_chart(tr: pd.DataFrame, path: Path, daily_returns: pd.Series | None = None):
    """The date-ordered daily mark-to-market equity (cumulative daily returns
    on capital, audit T4); without daily_returns (legacy exit-date booking)
    the cumulative trade PnL by exit date."""
    if tr.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    if daily_returns is None:
        eq = tr.sort_values("exit")["pnl"].cumsum()
        ax.plot(tr.sort_values("exit")["exit"].values, eq.values)
        ax.set_title("Daily WFO — cumulative log-spread PnL")
    else:
        ax.plot(daily_returns.index, daily_returns.cumsum().to_numpy() * 100)
        ax.set_title("Daily WFO — daily mark-to-market equity (cumulative return, % of capital)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── Headline universe (audit T2) ────────────────────────────────────────────

def headline_exclusions(pairs: list[str], unknown_ok: bool = False) -> pd.DataFrame:
    """Pairs the headline run excludes, with their legs' exchanges and the
    reasons: a leg is fx_unaligned (a "*_USD" leg: a local leg converted
    without an FX rate observed at its own close — this repo has no intraday
    FX), trades on an exchange without NYSE session overlap (ASX, TSE: both
    legs can never fill together), or trades on a European exchange (allowed
    only with synchronous 09:30–11:30 ET snapshots, which this repo lacks).
    A leg with an unknown exchange suffix raises KeyError naming the ticker
    (cross_listing.close_time) unless `unknown_ok` (the --legacy-cross-listed
    run), where the pair is listed with the reason "unknown exchange"."""
    rows = {}
    for pair in pairs:
        legs = pair.split("-")
        names, reasons = [], []
        if any(t.endswith("_USD") for t in legs):
            reasons.append("fx_unaligned")
        for t in legs:
            try:
                ex = close_time(t)
            except KeyError:
                if not unknown_ok:
                    raise
                names.append("unknown")
                reasons.append(f"unknown exchange ({t})")
                continue
            seen = ex.name in names
            names.append(ex.name)
            if seen:                                    # one reason per exchange
                continue
            if not ex.overlaps_nyse:
                reasons.append(f"{ex.name}: no session overlap with NYSE")
            if ex.region == "Europe":
                reasons.append(f"{ex.name}: no synchronous 09:30-11:30 ET snapshots")
        if reasons:
            rows[pair] = {"exchanges": " + ".join(names), "reasons": "; ".join(reasons)}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["exchanges", "reasons"])


def cross_listed_table(excluded: pd.DataFrame, trades: pd.DataFrame | None = None) -> str:
    """The separate table of the pairs the headline excludes.  With `trades`
    (a --legacy-cross-listed run) it adds their numbers, labelled LEGACY —
    NOT VALID."""
    title = f"CROSS-LISTED PAIRS (not in headline): {len(excluded)}"
    table = excluded.copy()
    if trades is not None:
        title += "  — numbers: LEGACY — NOT VALID (--legacy-cross-listed)"
        mine = trades[trades["pair"].isin(excluded.index)].groupby("pair")["pnl"]
        table["trades"] = mine.count().reindex(table.index, fill_value=0)
        table["win_rate"] = (mine.apply(lambda p: (p > 0).mean()).reindex(table.index)
                             .map(lambda w: "—" if pd.isna(w) else f"{w:.1%}"))
        table["total_pnl"] = mine.sum().reindex(table.index, fill_value=0.0).round(4)
    bar = "=" * 70
    return f"{bar}\n{title}\n{bar}\n{table.to_string() if len(table) else '(none)'}"


# ── Headline run (audit T5) ─────────────────────────────────────────────────

PIT_UNIVERSE_FILE = re.compile(r"universe_pit_(\d{4}-\d{2}-\d{2})\.csv")
NON_HEADLINE_BANNER = ("NON-HEADLINE RUN: pairs file is not a frozen point-in-time universe "
                       "(hindsight / survivorship)")


def non_headline_reasons(pairs_file: str, pairs_df: pd.DataFrame, first_oos,
                         legacy: list[str]) -> list[str]:
    """Why a run is not a headline run; empty = HEADLINE.  A headline run
    reads a frozen point-in-time universe written by build_universe_pit.py — a
    file named universe_pit_<asof>.csv with the builder's columns, a value in
    each of them on every row, and that one as-of date in the `asof` column
    of every row (a hand-appended pair is not frozen) — whose as-of date is
    strictly before the run's first OOS date (`first_oos`, None if the run has
    no window), and uses no legacy option (`legacy`, the --legacy-* options
    given; hard rule 5).  Manual pair lists, step7's OOS-PnL curation and every
    other pairs file are not headline."""
    reasons = []
    name = PIT_UNIVERSE_FILE.fullmatch(Path(pairs_file).name)
    frozen = (name is not None and len(pairs_df) > 0
              and all(c in pairs_df.columns for c in PIT_COLUMNS)
              and bool(pairs_df[PIT_COLUMNS].notna().all().all())
              and bool((pairs_df["asof"].astype(str) == name.group(1)).all()))
    if not frozen:
        reasons.append(NON_HEADLINE_BANNER)
    elif first_oos is None or not pd.Timestamp(name.group(1)) < first_oos:
        first = "none (no window)" if first_oos is None else first_oos.date()
        reasons.append(f"{NON_HEADLINE_BANNER}: as of {name.group(1)}, not before the first "
                       f"OOS date {first}")
    if legacy:
        reasons.append("NON-HEADLINE RUN: legacy options " + ", ".join(legacy))
    return reasons


def headline_block(reasons: list[str], pairs_file: str, first_oos) -> str:
    """The banner of a non-headline run, or the headline line."""
    if reasons:
        bar = "!" * 70
        return "\n".join([bar, *reasons, bar])
    return (f"HEADLINE RUN: frozen point-in-time universe {Path(pairs_file).name}, "
            f"first OOS date {first_oos.date()}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily Walk-Forward Optimization")
    p.add_argument("--pairs",  type=str, default="pairs_selected.csv",
                   help="CSV in data/ with a 'pair' column (e.g. AMT-AWK).")
    p.add_argument("--start",  type=str, default=None)
    p.add_argument("--end",    type=str, default=None)
    p.add_argument("--cost",   type=float, default=5.0,
                   help="Commission + half-spread, bps per leg (default 5).")
    p.add_argument("--slip",   type=float, default=5.0,
                   help="Slippage, bps per leg (default 5).")
    p.add_argument("--borrow", type=float, default=80.0,
                   help="Borrow rate on short leg, bps/year (default 80).")
    p.add_argument("--train",  type=int, default=TRAIN_INIT_MO,
                   help=f"Initial train length in months (default {TRAIN_INIT_MO}).")
    p.add_argument("--oos",    type=int, default=OOS_MO,
                   help=f"OOS length in months (default {OOS_MO}).")
    p.add_argument("--step",   type=int, default=STEP_MO,
                   help=f"Walk-forward step in months (default {STEP_MO}).")
    p.add_argument("--kalman", action="store_true",
                   help="Use Kalman dynamic β (innovations as spread, e/√S as z).")
    p.add_argument("--vwz",    action="store_true",
                   help="Use volume-weighted z + VR≥0.5 liquidity gate.")
    p.add_argument("--metagate", action="store_true",
                   help="Per-window MetaGate ML entry classifier "
                        "(features: abs_z, side, vol_20/60, vol_ratio, hurst).")
    p.add_argument("--hrp", action="store_true",
                   help="Per-window HRP weights scale each pair's trade size. "
                        "Weights renormalized so MEAN weight = 1 (low-corr "
                        "pairs scaled up, high-corr pairs scaled down).")
    p.add_argument("--ivol", action="store_true",
                   help="Inverse-volatility sizing: weight ∝ 1/std(spread) on train, "
                        "mean-normalized to 1. Simpler/more robust than HRP.")
    p.add_argument("--volvol", action="store_true",
                   help="Dynamic vol-targeting: per-entry size ∝ 1/recent_20d_spread_vol, "
                        "normalised against window mean. Pro-cyclical.")
    p.add_argument("--pairmom", action="store_true",
                   help="Rolling-Sharpe pair momentum: hot pair → up-weight, cold → down. "
                        "Causal: only uses pair's executed OOS trades so far.")
    p.add_argument("--maxconc", type=int, default=0,
                   help="Hard cap on concurrent open trades across all pairs "
                        "(post-hoc filter, 0 = no cap).")
    p.add_argument("--ticker-overlap", action="store_true",
                   help="Block new entry if either of its tickers is already "
                        "exposed in another open trade.")
    p.add_argument("--per-pair-quarter-cap", type=int, default=0,
                   help="Max trades per pair per calendar quarter (0 = no cap).")
    p.add_argument("--vol-tight-stop", action="store_true",
                   help="Tighten stop_z to 2.5 when entry's 20d/60d vol ratio > 1.5.")
    p.add_argument("--exec", type=str, default="next_close",
                   choices=["next_close", "vwap"], dest="exec_mode",
                   help="Fill of an order decided at close t: next_close = close "
                        "t+1 (default); vwap = next-day VWAP (signal still computed "
                        "on close). vwap requires data/vwap_daily.csv.")
    p.add_argument("--vwap-exec", action="store_const", const="vwap", dest="exec_mode",
                   help="Alias of --exec vwap.")
    p.add_argument("--legacy-same-close", action="store_true", dest="legacy_same_close",
                   help="LEGACY: restore the pre-audit fills at the close that made "
                        "the signal (impossible fills). For before/after comparison "
                        "only; never use for headline results.")
    p.add_argument("--legacy-spread-pnl", action="store_true", dest="legacy_spread_pnl",
                   help="LEGACY: book trade PnL as the spread change between the fills "
                        "instead of from the legs' fill prices. Comparison only; never "
                        "for headline results.")
    p.add_argument("--legacy-stale-prints", action="store_true", dest="legacy_stale_prints",
                   help="LEGACY: keep exactly unchanged closes with zero/missing volume "
                        "as prints and allow entries on a zero-return signal day. "
                        "Comparison only; never for headline results.")
    p.add_argument("--legacy-window-reset", action="store_true", dest="legacy_window_reset",
                   help="LEGACY: reset each pair's position state per walk-forward window "
                        "(positions still open at a window end are dropped; the training "
                        "slice includes the first OOS day). Needed by --legacy-same-close "
                        "and --legacy-spread-pnl. Comparison only; never for headline results.")
    p.add_argument("--legacy-exit-date-pnl", action="store_true", dest="legacy_exit_date_pnl",
                   help="LEGACY: book each trade's PnL on its exit date (daily PnL, Sharpe(full), "
                        "Sharpe(exit)) and MaxDD over trades in table order, instead of the daily "
                        "mark-to-market on constant capital; no MTM file is written. Needed by "
                        "--legacy-spread-pnl. Comparison only; never for headline results.")
    p.add_argument("--legacy-cross-listed", action="store_true", dest="legacy_cross_listed",
                   help="LEGACY: keep the cross-listed pairs (fx_unaligned, ASX/TSE or "
                        "European legs) in the run; their numbers are NOT VALID. "
                        "Comparison only; never for headline results.")
    p.add_argument("--rotation", action="store_true",
                   help="Quarterly per-pair S_perf rescaling (§5): trades sized by "
                        "rolling pair-Sharpe with Bayesian shrinkage. Drop pairs "
                        "with SR_6m < 0 for the current quarter.")
    p.add_argument("--voltarget", type=float, default=0.0,
                   help="Portfolio vol target (annualised, e.g. 0.10 = 10%%). "
                        "Applied as causal leverage scaling. 0 = off.")
    p.add_argument("--closes",   type=str, default="closes_daily.csv",
                   help="Closes file in data/ (extended universe support).")
    p.add_argument("--hmm",    type=str, default="off",
                   choices=["off", "block-panic", "block-calm",
                            "size-panic", "extreme-3state"],
                   help="HMM mode: off / block-panic / block-calm (invert) / "
                        "size-panic (50%% size in panic) / extreme-3state "
                        "(3-state, block only highest-vol).")
    p.add_argument("--cusum",  type=float, default=None,
                   help="CUSUM break threshold h (e.g. 5.0). If unset, CUSUM off.")
    p.add_argument("--tag",    type=str, default="",
                   help="Suffix for output filenames (e.g. 'kalman_cusum').")
    p.add_argument("--aussie-filter", action="store_true", dest="aussie_filter",
                   help="Conditional gate for Aussie/JP ADR pairs: entry_z>=2.5, "
                        "blackout when 5d realized spread vol > 95th-pct of trailing 252d, "
                        "require rolling 60d AR(1) half-life < 5 days.")
    p.add_argument("--execution-model", type=str, default="almgren",
                   choices=["flat", "almgren"], dest="execution_model",
                   help="flat = commission+slip+borrow scalars (--cost --slip --borrow). "
                        "almgren = per-pair tiered cost: half_spread + κ·σ·√(notional/ADV), "
                        "plus tier-specific borrow on short leg.  Default: almgren.")
    p.add_argument("--notional", type=float, default=100_000,
                   dest="notional",
                   help="USD notional per leg, used by Almgren-Chriss impact term. "
                        "Higher notional → larger impact cost.  Default $100k.")
    return p


def main():
    args = build_parser().parse_args()

    if BAR_TIMEFRAME != "daily":
        raise SystemExit(
            f"BAR_TIMEFRAME = {BAR_TIMEFRAME!r}; this script is for daily mode. "
            "Set BAR_TIMEFRAME='daily' in config.py."
        )

    closes_path = DATA_DIR / args.closes
    closes = pd.read_csv(closes_path, parse_dates=["Date"]).set_index("Date").sort_index()
    pairs_df = pd.read_csv(DATA_DIR / args.pairs)
    pairs  = pairs_df["pair"].tolist()

    vwap_daily = None
    if args.exec_mode == "vwap":
        vp = DATA_DIR / "vwap_daily.csv"
        if not vp.exists():
            raise SystemExit("vwap_daily.csv missing — run build_vwap_daily.py")
        vwap_daily = pd.read_csv(vp, parse_dates=["Date"]).set_index("Date").sort_index()

    # volumes_daily.csv: required by --vwz; if present, used for the stale-print
    # mask (audit T2) and by almgren for the $ ADV.
    volumes = None
    vol_path = DATA_DIR / "volumes_daily.csv"
    if args.vwz:
        if not vol_path.exists():
            raise SystemExit("volumes_daily.csv missing — run build_volumes_daily.py")
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()
    elif vol_path.exists():
        volumes = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # Headline universe (audit T2).  Every *_USD leg is a local leg converted
    # with a daily FX close (no intraday FX at its close time exists here):
    # fx_unaligned.  The legacy run lists unclassifiable legs instead of stopping.
    excluded = headline_exclusions(pairs, unknown_ok=args.legacy_cross_listed)
    run_pairs = pairs if args.legacy_cross_listed else [p for p in pairs if p not in excluded.index]
    print("HEADLINE UNIVERSE POLICY: excludes pairs with an fx_unaligned leg (*_USD), an ASX/TSE leg "
          "(no NYSE overlap) or a European leg (no synchronous 09:30-11:30 ET snapshots)")
    if args.legacy_cross_listed:
        print(f"  LEGACY (--legacy-cross-listed): all {len(pairs)} pairs run; the "
              f"{len(excluded)} cross-listed pairs' numbers are NOT VALID")
    else:
        print(f"  headline pairs: {len(run_pairs)} / {len(pairs)}  "
              f"({len(excluded)} cross-listed pairs excluded, listed at the end)")

    cfg = WfoConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WfoConfig)})

    # Audit T5: HEADLINE only on a frozen point-in-time universe.
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step, start=cfg.start, end=cfg.end)
    first_oos = windows[0][2] if windows else None
    legacy = [f"--{k.replace('_', '-')}" for k, v in vars(args).items() if k.startswith("legacy_") and v]
    reasons = non_headline_reasons(args.pairs, pairs_df, first_oos, legacy)
    digest = hashlib.sha256((DATA_DIR / args.pairs).read_bytes()).hexdigest()
    print(f"UNIVERSE: {DATA_DIR / args.pairs}  SHA-256 {digest}")
    banner = headline_block(reasons, args.pairs, first_oos)
    print(banner)

    result = run_wfo_daily(closes, run_pairs, cfg, volumes=volumes, vwap=vwap_daily)
    result.stats["headline"] = not reasons
    verdict = f"{banner}\nheadline={result.stats['headline']}"
    if result.trades.empty:
        print("\nNo trades.")
        print("\n" + cross_listed_table(excluded))
        print("\n" + verdict)
        return result
    trades = result.trades

    legacy_pnl = args.legacy_exit_date_pnl
    summarise(trades, label=f"NET {args.cost} bps/leg",
              n_windows=result.stats["n_windows"], n_pairs=len(run_pairs),
              oos_index=result.daily_pnl.index, stats=None if legacy_pnl else result.stats)
    if args.legacy_cross_listed:
        print("\n" + cross_listed_table(excluded, trades))
        cross = trades[trades["pair"].isin(excluded.index)]
        if not cross.empty:
            st = trade_stats(cross, result.stats["n_windows"], len(excluded), result.daily_pnl.index)
            # The exit-date Sharpe(full) only with the legacy booking (rule 4: one Sharpe).
            print(f"  cross-listed block (LEGACY — NOT VALID): {st['n_trades']} trades, "
                  f"win {st['win_rate']:.1%}, total PnL {st['total_pnl']:+.4f}"
                  + (f", Sharpe(full) {st['sharpe_full']:+.3f}" if legacy_pnl else ""))
    else:
        print("\n" + cross_listed_table(excluded))
    print("\n" + verdict)

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = DATA_DIR / f"wfo_daily_results{suffix}.csv"
    trades.to_csv(out_csv, index=False)
    chart = OUTPUT_DIR / f"wfo_daily_equity{suffix}.png"
    save_equity_chart(trades, chart, None if legacy_pnl else result.daily_returns)
    print(f"\nSaved → {out_csv} ({len(trades)} trades)")
    if not legacy_pnl:
        # Audit T4: the daily MTM series, the downstream scripts' input.
        out_mtm = DATA_DIR / f"wfo_daily_mtm{suffix}.csv"
        pd.DataFrame({"pnl": result.daily_pnl, "ret": result.daily_returns}).rename_axis("Date") \
            .to_csv(out_mtm)
        out_pairs = DATA_DIR / f"wfo_daily_mtm_pairs{suffix}.csv"
        result.daily_pnl_pairs.rename_axis("Date").to_csv(out_pairs)
        print(f"MTM    → {out_mtm} ({len(result.daily_pnl)} business days), per pair → {out_pairs}")
    print(f"Chart  → {chart}")
    return result


if __name__ == "__main__":
    main()
