# Tests

Run everything from the repo root (pytest.ini sets `testpaths = tests`):

    python -m pytest -q

The suite runs offline and never reads `data/`. Wall time ≈ 50–90 s on a 10-core
Mac (the limit is ~2 minutes). The older unittest modules (`test_lookahead.py`,
`test_risk_governor.py`, …) run under pytest unchanged.

## Placebo harness (audit task T0)

The daily walk-forward backtest (`wfo_daily_core.run_wfo_daily`) is run on
synthetic data whose true edge is known.

| file | role |
|---|---|
| `synthetic.py` | generators: `independent_rw`, `async_same_asset` (+ `build_async_panel`, legacy; `build_async_panel_fixed`, corrected, T2), `ou_positive_control` |
| `placebo_harness.py` | variant table, datasets, seeds, statistics, process-pool runner with per-session cache |
| `test_placebo.py` | null tests, positive control, truncation invariance; the expected-outcome table `XFAIL` |
| `placebo_report.py` | prints the report table for `AUDIT_FIX_LOG.md` |
| `test_synthetic.py` | generator sanity checks |
| `test_wfo_core_identity.py` + `fixtures/t0_golden/` | the core reproduces the pre-refactor CLI byte for byte |
| `test_next_close_fills.py` | fills on the next bar (T1), on hand-made 10-day spreads |
| `test_cross_listing.py` | exchange close times, stale-print mask, FX at the local close, union-calendar panel (T2) |
| `test_leg_fills.py` | PnL from leg fills; no signal or fill on a stale print, no entry on a zero-return day (T2) |
| `test_diag_stale_entries.py` | `diag_stale_entries.py` on hand-made prices and the async holiday dataset (T2) |
| `test_headline_universe.py` | the CLI's headline-universe filter and cross-listed table (T2) |
| `test_continuous_positions.py` | one position per pair across windows, β per window for new entries, end-of-run close (T3) |
| `test_daily_mtm.py` | daily mark-to-market PnL, capital, the one Sharpe, date-ordered MaxDD; the downstream scripts read the MTM file (T4) |

### Variants

`placebo_harness.VARIANTS`: `legacy` = the gross config with every `legacy_*`
flag on (T1: `legacy_same_close`, the pre-audit same-close fills; T2:
`legacy_spread_pnl`, `legacy_stale_prints`; T3: `legacy_window_reset`, the
per-window engine; T4: `legacy_exit_date_pnl`, PnL booked on exit dates) and
the legacy async panel builder
`build_async_panel` (local prints forward-filled onto the US calendar,
converted with the daily FX close); `fixed` = the gross config with the
corrected defaults (T1: an order decided at close t fills at close t+1; T2:
stale prints masked, no entry on a zero-return day, PnL from the leg fills;
T3: one continuous position per pair across windows; T4: `daily_pnl` = the
daily mark-to-market of the trades over every business day) and
`build_async_panel_fixed` (each local print converted with the grid FX at
its own close, on the union calendar, NaN on local holidays).

### What it checks

- **Null tests** (no edge exists: log prices are martingales and fills are
  causal, so no rule can forecast them).  In USD the legs, held in quantities
  fixed at the entry fill, are not exact martingales: a zero-log-drift price
  has E[ΔP/P] = e^{σ²h/2} − 1 over h days, so a trade's expected gross is
  side·(1 − β)·(e^{σ²h/2} − 1) — on `independent_rw` ≈ +25 bp per long-spread
  and ≈ −25 bp per short-spread trade (seeds 0–19 at T3: measured +23.9 bp /
  −23.3 bp).  The z-score is sign-symmetric, so the two sides cancel and the
  expected pooled gross PnL is ≈ 0 (measured +0.3 bp per trade), which is
  what the null tests check. Datasets: `independent_rw` (20 pairs,
  two independent zero-drift random walks each) and `async_same_asset` (one
  efficient price per pair printed by a US and a local exchange) with close lag
  4.5 h or 15 h × 0 or 5 local holidays per year. Gross config: flat model with
  cost = slip = borrow = 0. Seeds 0–19 per variant. Both statistics need at
  least `MIN_POOLED_TRADES` (1000) pooled trades, so no null test passes on
  (almost) no trades; the floor fails through `pytest.fail`, so an xfail mark
  (`raises=AssertionError`) cannot absorb it.
  - mean over seeds of the annualised gross Sharpe ∈ [−0.2, +0.2];
  - pooled per-trade t-stat: |t| < 3.

  The async datasets go through the variant's panel builder (the repo's
  `cross_listing` code): `build_async_panel` for `legacy`,
  `build_async_panel_fixed` for `fixed`.
- **Positive control**: `ou_positive_control` (log p1 = β·log p2 + s, s an OU
  spread with half-life 5 d and sd 0.02), zero costs, variant `fixed`,
  seeds 0–9: mean Sharpe > 1.0. If it fails, the pipeline is broken —
  investigate; never change the strategy's parameters to make it pass.
- **Truncation invariance**: seed 0 of `independent_rw`, `async_lag15_hol5`
  and `ou_positive_control`, run on the panel cut at its middle date D
  (inclusive) and on the full panel.  With C = min(D, the truncated run's
  last OOS end): every entry decided before C has the same `entry_signal`,
  `entry` (fill), `side`, `size`, `beta` and entry leg prices, and every exit
  decided before C is identical in all columns (exit fill, `reason`, exit leg
  prices, `pnl`, …) (T3, the plan's full form; at T0 whole windows were
  compared, Ruling R4, at T1/T2 the trades of the truncated run's windows).
  Decisions at or after C are excluded, and so are the truncated run's
  `end_of_data` closes: it decides nothing after its last OOS end and closes
  what is open then at the next available close.
- **Cross-listed data layer** (`test_cross_listing.py`, T2): `close_time`
  by ticker suffix (unknown suffix → `KeyError`); `repeats_previous_print`
  (the zero-return rule shared by the mask, the entry guard and the
  diagnostic); `close_timestamps` in New
  York time, DST-correct (ASX 02:00 / 01:00 / 00:00 ET, Xetra 11:30 / 12:30
  ET); `mask_stale_prints` (an unchanged close with zero, NaN or no volume
  is NaN; nothing filled); the corrected builder converts each print with the
  last FX quote at or before its own close (not the daily close; too old →
  NaN), keeps the local dates (holidays NaN on the union calendar), and
  without the FX series writes no column and flags the leg `fx_unaligned`; a
  tz-naive or unsorted FX index raises `ValueError`;
  `assemble_panel` = union calendar, no fill.  `test_synthetic.py` checks
  `build_async_panel_fixed` against the raw grid.
- **Leg fills** (`test_leg_fills.py`, T2): hand-made two-leg trades book
  side·(q1·ΔP1 − q2·ΔP2) with q1 = 1/P1 and q2 = β/P2 fixed at the entry
  fill (β = 2 and β = −1.5; a flat spread books nothing by itself), costs and
  size as before, β of the entry decision bar (Kalman), VWAP leg fills with
  a per-leg fallback to the close at exits, `end_of_data` and CUSUM exits.
  On an OU panel whose leg closes exactly unchanged every 4th day: with
  volume, no entry is decided on those days (the unguarded legacy run does
  enter there); with zero volume, no decision or fill falls on them.
- **Stale-entry diagnostic** (`test_diag_stale_entries.py`, T2): on the
  async dataset (lag 15 h, 10 holidays a year) the legacy run's stale share
  with the raw local closes equals its entries signalled on a local holiday
  (> 0); the fixed run's share is exactly 0 with and without them, and none
  of its decisions or fills is on a local holiday.
- **Headline universe** (`test_headline_universe.py`, T2): pairs with an
  `fx_unaligned` leg (any `*_USD` leg), an ASX/TSE leg or a European leg
  (`Exchange.overlaps_nyse` / `Exchange.region`) are excluded and
  listed in the "CROSS-LISTED PAIRS (not in headline)" table; with
  `--legacy-cross-listed` they run and their numbers read "LEGACY — NOT
  VALID" (in-process CLI run on a synthetic panel); an unknown exchange
  suffix stops the headline run and is listed as "unknown exchange" in the
  legacy run.
- **Next-bar fills** (`test_next_close_fills.py`, `backtest_pair` level): the
  acceptance case (the spread jumps on day k and reverts on day k+1: legacy
  books the jump, the default enters at k+1 and books ≈ 0), holding period and
  borrow from the fill, `end_of_data` exits, an entry decided on the last bar
  (not filled), `decision_end` (no decision on fill-only bars) and the VWAP path.
- **Identity** (`test_wfo_core_identity.py`): `run_wfo_daily` with
  `LEGACY_FLAGS` (`legacy_same_close`, T1; `legacy_spread_pnl` and
  `legacy_stale_prints`, T2; `legacy_window_reset`, T3; `legacy_exit_date_pnl`,
  T4) reproduces the
  trades CSV of `step3j_wfo_daily.py` at commit 46ffdfb on 26 small synthetic
  cases that cover every CLI option, compared on the fixture's columns (later
  tasks add columns; the legacy `entry_signal` / `exit_signal` equal `entry` /
  `exit`); without the flag the `flat` and `vwap_exec` trades differ.
  `fixtures/t0_golden/make_golden.py` made the fixtures by running that CLI
  with its helper modules, all extracted from 46ffdfb, and it refuses to
  write them unless every case except the default run changes the trades of
  its panel's default run (the option binds). The
  fixtures are the permanent "legacy = pre-audit code" anchor. They were
  generated with Python 3.14.3, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1,
  statsmodels 0.14.6, scikit-learn 1.8.0 and hmmlearn 0.3.3; other versions
  may change the last digits of the floats (and then the byte comparison).
  `--metagate` / `--hrp` keep their pre-existing `UnboundLocalError` (Ruling R7).
- **Default mode on the identity inputs** (`test_wfo_core_identity.py`, T1,
  T2, T3): all 26 cases run with the corrected defaults (`DEFAULTS` = every
  legacy flag off, i.e. the continuous engine); every fill is the pair's
  next row with genuine prints of both legs after its decision (never after
  `cfg.end`; on the decision date only for an `end_of_data` close with no
  such row left), a pair's positions never overlap (each entry fill after
  the previous exit fill), every entry lies in the OOS of its
  `window_entered`, `window_exited` is the latest-starting window whose OOS
  has begun on the exit decision date (≥ `window_entered`), an `end_of_data`
  close is decided on the pair's last row up to the last OOS end, and
  `daily_pnl` holds every trade's PnL — since T4 it is the daily
  mark-to-market of the final trades over `pd.bdate_range(first OOS start,
  max(last OOS end, last exit fill))`, each trade's daily gross summing to
  its `gross` and its daily net to its `pnl`, and pnl = size·(gross − cost −
  borrow) after every filter. With `cfg.end` set, the run on the
  full panel equals the run on the panel cut at `cfg.end` (the fill-only bar
  never lies after it, Ruling R15). With zero costs, each trade's PnL is
  rebuilt from its fill prices, in `next_close` and `vwap` mode: by default
  from the legs (q1 = 1/P1, q2 = β/P2 at the entry fill, β from the entry
  window's training rows before its OOS start; a leg without a VWAP at its
  close), with `legacy_spread_pnl` (on the per-window engine) as the spread
  (or VWAP spread) change between the fill dates.
- **Continuous positions** (`test_continuous_positions.py`, T3): run level,
  on a hand-made two-leg panel (zero costs): the acceptance case — a short
  spread decided on the last days of window 1, open at the cut, stopped out
  in window 2 at a loss: it is in `trades` (`window_entered` 1,
  `window_exited` 2, PnL from the leg fills) and in `daily_pnl` (since T4
  marked to market: most of the loss is booked before the exit); it keeps
  β of window 1 (β1 ≠ β2) and exits on the rolling z of log PA − β1·log PB;
  window 2's z is above ENTRY_Z while it is open, yet no second position
  opens (the per-window reset opens one and drops the window-1 trade); new
  window-2 entries use β2; the end-of-run close at the next row with both
  legs (a missing print skips a day; with `cfg.end` at the last OOS end, at
  the last decision row's close); β from the training rows strictly before
  the OOS start (T9.1; the per-window reset keeps the slice that includes
  it); `legacy_same_close` / `legacy_spread_pnl` without
  `legacy_window_reset` raise `ValueError`.  Engine level
  (`backtest_pair_continuous` on 32 hand-made rows, a hand-made z per
  window): exits on the entry window's z whatever window the row is in;
  no entries in an ineligible window or a gap, the latest-starting window
  when OOS periods overlap; an entry decided on the last decision row is
  closed on the fill-only row at once (held 0, costs charged); cost and
  borrow of the entry window; CUSUM (a break in a window's OOS force-closes,
  one in its training rows only blocks its entries); Kalman β of the entry
  decision row and the exit z of its own spread; a pair whose rows end
  before the last window: its end-of-run close is labelled with the window
  of its last decision row.

- **Daily mark-to-market** (`test_daily_mtm.py`, T4).  Trade level
  (`mtm.trade_mtm` / `mtm.daily_mtm`, hand-made two-leg trades): the
  acceptance case — long the spread PA − PB with β = 1 (2.0 USD of capital,
  one pair), −5% of capital at mid-life (PA −5%, PB +5%), exit at +1%:
  `mtm_stats` MaxDD −5% (peak and trough dates), while the exit-date
  booking's trade-order MaxDD is 0; the equity curve starts at 0 (a loss on
  the first day is a drawdown); half the round-trip cost on each fill date,
  borrow on each held row, size scaling, Σ daily gross = gross and Σ daily
  net = pnl; VWAP fills (entry day marked from the VWAP to the close, exit
  day from the close to the exit VWAP); a leg without a print on a date
  books only the other leg that date, no borrow, and its next print carries
  its move; a held-0 round trip books only its costs; the portfolio is the
  sum of the trades, and a date with PnL outside the index or a panel whose
  rows do not match the trade's `held` is refused.  Run level (OU panel,
  flat costs): `daily_pnl` = `daily_mtm` of the final trades on the business
  days of the span, `daily_returns` = `daily_pnl` / capital, `stats`
  (capital, sharpe, ann_return, ann_vol, max_dd and its dates; no
  `sharpe_full` / `sharpe_exit`), `daily_pnl_pairs` (every pair of the run;
  its columns, added in order, are `daily_pnl` exactly);
  `legacy_exit_date_pnl` restores the exit-date `daily_pnl` and
  `trade_stats` exactly; `legacy_spread_pnl`
  without it raises `ValueError`; the summary prints the MTM figures and no
  exit-day Sharpe.  Downstream (the CLI in a temp run dir on a synthetic
  panel): it writes `wfo_daily_mtm_<tag>.csv` (Date, pnl, ret) and
  `wfo_daily_mtm_pairs_<tag>.csv` (Date × pairs, summing to the portfolio
  pnl exactly), and `bootstrap_sharpe.py`, `stress_crisis_analysis.py`,
  `factor_attribution.py` and `dashboard_daily.py` read them (point Sharpe,
  VaR, crisis drawdowns, regression input, dashboard Sharpe / maxDD); the
  dashboard's per-pair and sub-block Sharpe ratios come from the per-pair
  file (no exit-date Sharpe appears; without the file "—"); without the MTM
  file each script stops (`SystemExit`), with `--legacy-exit-date-pnl` each
  runs and labels its output "LEGACY exit-date booking"; an MTM file that
  does not sum to the trades file (or a per-pair file that does not match)
  stops them ("does not match").

### Statistic definitions

- **Sharpe per seed** = `daily_pnl.mean() / daily_pnl.std() * sqrt(252)` on
  `WfoResult.daily_pnl` (pandas std, ddof = 1; 0.0 if the std is 0 or NaN).
  Until T3 `daily_pnl` booked each trade's PnL on its exit date, zeros on the
  other OOS trading days of the panel (since T1 `exit` is the exit fill date,
  `exit_signal` the decision date).  Since T4 it is the daily mark-to-market
  PnL of the positions (`mtm.daily_mtm`) on every business day from the first
  OOS start to max(last OOS end, last exit fill), zeros on empty days and
  holidays — the one Sharpe definition (hard rule 4; `legacy` keeps the
  exit-date booking, `legacy_exit_date_pnl`).
- **Capital** (T4 ruling): constant, number of pairs in the run ×
  `mtm.CAPITAL_PER_PAIR` = 2.0 USD per $1 of leg-1 notional (a pair with
  |β| = 1 holds $1 long and $1 short).  `WfoResult.daily_returns` =
  `daily_pnl` / capital; `stats["sharpe"]` = mean / std · √252 of the daily
  returns (the same number as on `daily_pnl`: Sharpe does not depend on the
  constant); `stats["max_dd"]` = deepest fall of the date-ordered equity
  curve (cumulative daily returns, starting at 0) below its running peak, as
  a fraction of capital; `ann_return` = mean · 252, `ann_vol` = std · √252.
- **Mean Sharpe ± SE**: average over seeds; SE = sample std over seeds / √(seeds).
- **Per-trade t-stat** = mean / (std / √n) (sample std) of `trades["pnl"]`
  pooled over all seeds of a dataset. It is a t-statistic, not a Sharpe.
- **Win rate**: share of pooled trades with pnl > 0.

### Expected outcomes (measured at T4)

| test (null: mean Sharpe and per-trade t-stat) | `legacy` | `fixed` |
|---|---|---|
| null, `independent_rw` | strict xfail: pre-T1 behaviour (bug named in the reason) | pass (since T3) |
| null, `async_lag4.5_hol0` | strict xfail | pass (since T3; passed at T1, failed at T2) |
| null, `async_lag15_hol0` | strict xfail | pass (since T1) |
| null, `async_lag4.5_hol5`, `async_lag15_hol5` | strict xfail | pass (since T2) |
| OU positive control | — | pass (+6.62; +6.68 at T3 on exit-date booking) |
| truncation invariance (3 datasets) | pass | pass |

The measured numbers are in `AUDIT_FIX_LOG.md` (T0–T4). Until T2 the
per-window engine dropped every position still open at a window end: on
`independent_rw` those were losers (5541 positions, mean mark-to-market at
the cut −195 bp, t = −25.3), which made the closed trades show +0.76 Sharpe
from nothing; on the async datasets they were winners (+37 to +65 bp), which
biased the closed trades negative (`async_lag4.5_hol0` −0.247, t = −3.4).
With one continuous position per pair (T3) `independent_rw` gives +0.006
(t = +0.0) and `async_lag4.5_hol0` −0.166 (t = −2.3).

### How later tasks update the harness

1. `placebo_harness.VARIANTS` — the one place that defines the variants: add the
   task's `legacy_*` WfoConfig fields to `legacy`, switch the defaults for
   `fixed` (T2 gave `fixed` the corrected async panel builder).
2. `test_placebo.XFAIL` — remove the entries a fix makes pass (a strict xfail
   that passes fails the run, so stale entries cannot go unnoticed). Add entries
   only for failures that were measured (Ruling R5 — never force a failure).
   The marks expect `AssertionError` only; any other exception is an error.
3. `test_wfo_core_identity.LEGACY_FLAGS` — add the task's legacy flag, so the
   legacy path keeps reproducing the T0 fixtures.
4. Run `python tests/placebo_report.py` (≈ 50 s) and paste the table into the
   task's `AUDIT_FIX_LOG.md` entry.

### Commands

    python -m pytest -q tests/test_placebo.py -rxX    # placebo tests with xfail reasons
    python tests/placebo_report.py                    # report table
    python -m pytest -q tests/test_wfo_core_identity.py

Runtime at T0: full suite ≈ 30 s wall, of which the placebo pool (110 runs,
8 workers) ≈ 17 s and the identity cases ≈ 9 s.  At T1 ≈ 67 s: `legacy` and
`fixed` now differ, so the pool runs 210 seeds (≈ 37 s) and the truncation
test 12 runs instead of 6; the default-mode runs on the identity inputs
add ≈ 13 s.  At T2 ≈ 74–87 s: a default run costs ≈ 18% more (stale-print
mask, entry guard, leg prices), the pool ≈ 51 s; the T2 tests add ≈ 6 s.
At T3 ≈ 50–55 s: the continuous engine makes one pass per pair instead of one
per pair and window (`independent_rw` seed 0, best of 5: 0.29 s vs 1.16 s
for the per-window engine), the pool ≈ 35 s; the T3 tests add ≈ 1 s.
At T4 ≈ 50 s: the daily mark-to-market adds ≈ 0.02 s per run (`independent_rw`
seed 0, best of 5: 0.297 s vs 0.276 s with `legacy_exit_date_pnl`); the T4
tests add ≈ 3 s.
