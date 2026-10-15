# Tests

Run everything from the repo root (pytest.ini sets `testpaths = tests`):

    python -m pytest -q

The suite runs offline and never reads `data/`. Wall time ≈ 75–90 s on a 10-core
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

### Variants

`placebo_harness.VARIANTS`: `legacy` = the gross config with every `legacy_*`
flag on (T1: `legacy_same_close`, the pre-audit same-close fills; T2:
`legacy_spread_pnl`, `legacy_stale_prints`) and the legacy async panel builder
`build_async_panel` (local prints forward-filled onto the US calendar,
converted with the daily FX close); `fixed` = the gross config with the
corrected defaults (T1: an order decided at close t fills at close t+1; T2:
stale prints masked, no entry on a zero-return day, PnL from the leg fills)
and `build_async_panel_fixed` (each local print converted with the grid FX at
its own close, on the union calendar, NaN on local holidays).

### What it checks

- **Null tests** (no edge exists, so with causal fills every leg is a martingale
  and the expected gross PnL is exactly 0). Datasets: `independent_rw` (20 pairs,
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
  and `ou_positive_control`, panel cut at its middle date D (inclusive). Among
  the trades of the windows the truncated run has (windows still reset per
  pair until T3), every trade whose exit was decided before D is identical in
  all columns, and every entry decided before D has the same `entry_signal`,
  `entry` (fill), `side` and `size` (T1; at T0 whole windows were compared,
  Ruling R4). Decisions at or after D are excluded: an exit decided on D fills
  on D+1 in the full run and at D's close (`end_of_data`) in the truncated one.
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
  `legacy_stale_prints`, T2) reproduces the
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
  T2): all 26 cases run with the corrected defaults (`DEFAULTS` = every
  legacy flag off); every fill is the pair's next row with genuine prints of
  both legs after its decision (never after `cfg.end`; `end_of_data` only
  where no such row is left), a pair's positions never overlap, and
  `daily_pnl` holds every trade's PnL. With `cfg.end` set, the run on the
  full panel equals the run on the panel cut at `cfg.end` (the fill-only bar
  never lies after it, Ruling R15). With zero costs, each trade's PnL is
  rebuilt from its fill prices, in `next_close` and `vwap` mode: by default
  from the legs (q1 = 1/P1, q2 = β/P2 at the entry fill; a leg without a VWAP
  at its close), with `legacy_spread_pnl` as the spread (or VWAP spread)
  change between the fill dates.

### Statistic definitions

- **Sharpe per seed** = `daily_pnl.mean() / daily_pnl.std() * sqrt(252)` on
  `WfoResult.daily_pnl` (pandas std, ddof = 1; 0.0 if the std is 0 or NaN).
  At T0 `daily_pnl` books each trade's PnL on its exit date, zeros on the other
  OOS trading days (since T1 `exit` is the exit fill date, `exit_signal` the
  decision date); T4 replaces it with daily mark-to-market over all business days.
- **Mean Sharpe ± SE**: average over seeds; SE = sample std over seeds / √(seeds).
- **Per-trade t-stat** = mean / (std / √n) (sample std) of `trades["pnl"]`
  pooled over all seeds of a dataset. It is a t-statistic, not a Sharpe.
- **Win rate**: share of pooled trades with pnl > 0.

### Expected outcomes (measured at T2)

| test (null: mean Sharpe and per-trade t-stat) | `legacy` | `fixed` |
|---|---|---|
| null, `independent_rw` | strict xfail: pre-T1 behaviour (bug named in the reason) | strict xfail: "T3 pending: positions still open at a window end are dropped" |
| null, `async_lag4.5_hol0` | strict xfail | strict xfail: "T3 pending: …" (since T2; passed at T1) |
| null, `async_lag15_hol0` | strict xfail | pass (since T1) |
| null, `async_lag4.5_hol5`, `async_lag15_hol5` | strict xfail | pass (since T2) |
| OU positive control | — | pass (+6.26) |
| truncation invariance (3 datasets) | pass | pass |

The measured numbers are in `AUDIT_FIX_LOG.md` (T0, T1, T2). On
`independent_rw` the whole bias (+0.76 Sharpe, t = +10.9 for `fixed`) comes
from positions still open at a window end being dropped (T3): adding their
mark-to-market at the cut brings the pooled t-stat to +0.04. On the async
datasets the dropped positions are winners instead (+37 to +65 bp at the
cut), which biases the closed trades negative: over seeds 20–79 the closed
trades of `async_lag4.5_hol0` give −0.08 Sharpe and the marks bring it to
0.00; on seeds 0–19 the closed trades give −0.247 (t = −3.4), with the marks
−0.165 (t = −2.3).

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
