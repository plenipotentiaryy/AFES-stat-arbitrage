# AUDIT_FIX_LOG

Fixes of the daily backtest after the audit of commit `b7440fb`
(plan: `AFES_AGENT_TASKS.md`). One entry per task, appended in order.

## T0 — Make the backtest testable; add the placebo harness
- Commit: fix(T0): Make the backtest testable; add the placebo harness
- Files changed:
  - `wfo_daily_core.py` (new) — the engine moved out of `step3j_wfo_daily.py`: constants, `is_aussie_jp_pair`, `aussie_blackout_series`, `estimate_beta`, `spread_series`, `backtest_pair`, `fit_macro_hmm`, `vwz_spread_z`, `kalman_spread_z`, `make_windows`, `summarise`, `apply_quarterly_rotation`, `apply_vol_target`, and `main()`'s post-IO body as `run_wfo_daily(closes, pairs, cfg, volumes=None, vwap=None) -> WfoResult`. New: `WfoConfig` (frozen dataclass; fields = the CLI's argparse dests minus `pairs/closes/tag`, same defaults), `WfoResult(trades, daily_pnl, stats)`, `trade_stats` (the numbers `summarise` prints) and `exit_date_daily_pnl`. `backtest_pair`'s per-bar loop reads numpy arrays taken once before the loop (speed only).
  - `step3j_wfo_daily.py` — now the CLI only: `build_parser()`, `BAR_TIMEFRAME` check, all file IO and missing-file `SystemExit`s, `WfoConfig` from the args, "No trades.", `summarise`, CSV, chart.
  - `cross_listing.py` (new) — `align_to_ref_calendar`, `to_usd`, `build_cross_listed_usd`: the date alignment + daily-FX conversion that was inline in `universe_expand.main()` / `universe_expand_v2.main()`, unchanged (T2 fixes it).
  - `universe_expand.py`, `universe_expand_v2.py` — call `cross_listing`; `align_to_us_calendar` / `align` are thin wrappers.
  - `tests/synthetic.py`, `tests/placebo_harness.py`, `tests/placebo_report.py`, `tests/test_placebo.py`, `tests/test_synthetic.py`, `tests/test_wfo_core_identity.py`, `tests/fixtures/t0_golden/make_golden.py` + 25 fixture CSVs, `tests/README.md`.
  - `pytest.ini` (`testpaths = tests`), `requirements-dev.txt` (`pytest`), `.gitignore` (`!tests/fixtures/**/*.csv`), `AUDIT_FIX_LOG.md`.
- Tests added (66 items: 46 pass, 20 strict xfail; full suite 138 passed + 20 xfailed in ≈ 30 s wall):
  - `test_wfo_core_identity.py` — 26 cases (every CLI option: flat, flat zero-cost, almgren with/without volumes, kalman, vwz, hmm block-panic / size-panic / block-calm / extreme-3state, cusum 5 / 10 / 30, vol-tight-stop, vwap-exec, aussie-filter, ivol, volvol, pairmom, rotation, voltarget 0.10, maxconc 3, ticker-overlap, per-pair-quarter-cap 2, start/end, train 12 / oos 6 / step 6): `run_wfo_daily` reproduces the 46ffdfb CLI's trades CSV byte for byte (`cusum 5` has no trades → no fixture, empty result expected; `make_golden.py` refuses to write fixtures if any case other than the default run gives the same trades as its panel's default run, i.e. every option binds); `--metagate` / `--hrp` still raise `UnboundLocalError` (R7); `vwz` / `vwap_exec` without their frames raise `ValueError`; CLI dests == `WfoConfig` fields with equal defaults; `daily_pnl` / `stats` agree with the trades.
  - `test_placebo.py` — null mean gross Sharpe ∈ [−0.2, +0.2] and pooled per-trade |t| < 3, each on at least `MIN_POOLED_TRADES` = 1000 pooled trades (5 null datasets × 2 variants × 2 statistics, all `xfail(strict=True, raises=AssertionError)` at T0); OU positive control mean Sharpe > 1.0; truncation invariance (3 datasets × 2 variants).
  - `test_synthetic.py` — determinism by seed, calendars and holidays (no local print on a local holiday), FX sample time, OU half-life ≈ 5 d and sd ≈ 0.02, zero-drift random walks, degenerate async case (identical legs, no PnL).
- Synthetic before → after: there was no harness before T0; the table is the current (pre-audit) behaviour, i.e. the **legacy numbers**. At T0 `legacy` == `fixed` (no `--legacy-*` flag exists yet). Gross = flat, cost = slip = borrow = 0; 10-year panels from 2005-01-03; null seeds 0–19, control seeds 0–9. Sharpe per seed = `daily_pnl.mean()/daily_pnl.std()·√252` on `WfoResult.daily_pnl` (exit-date booking at T0), mean ± SE over seeds; t = mean/(sd/√n) of all trades' pnl pooled over seeds. Output of `python tests/placebo_report.py`:

  | variant | dataset | seeds | mean Sharpe ± SE | per-trade t | trades | win rate |
  |---|---|---|---|---|---|---|
  | legacy | independent_rw | 20 | +0.750 ± 0.066 | +10.6 | 13391 | 59.5% |
  | legacy | async_lag4.5_hol0 | 20 | +9.702 ± 0.037 | +593.0 | 18729 | 100.0% |
  | legacy | async_lag4.5_hol5 | 20 | +7.768 ± 0.065 | +354.8 | 18286 | 99.3% |
  | legacy | async_lag15_hol0 | 20 | +9.756 ± 0.045 | +591.2 | 18651 | 100.0% |
  | legacy | async_lag15_hol5 | 20 | +9.206 ± 0.055 | +506.5 | 18668 | 99.6% |
  | fixed | independent_rw | 20 | +0.750 ± 0.066 | +10.6 | 13391 | 59.5% |
  | fixed | async_lag4.5_hol0 | 20 | +9.702 ± 0.037 | +593.0 | 18729 | 100.0% |
  | fixed | async_lag4.5_hol5 | 20 | +7.768 ± 0.065 | +354.8 | 18286 | 99.3% |
  | fixed | async_lag15_hol0 | 20 | +9.756 ± 0.045 | +591.2 | 18651 | 100.0% |
  | fixed | async_lag15_hol5 | 20 | +9.206 ± 0.055 | +506.5 | 18668 | 99.6% |
  | fixed | ou_positive_control | 10 | +6.701 ± 0.042 | +166.9 | 5053 | 96.3% |

  - Every null test fails under the current code (all 20 marked `xfail(strict=True)`: reason "pre-T1 behaviour: …" for `legacy`, "T1–T4 pending" for `fixed`); no null test passes, so none is left unmarked. The positive control passes (+6.70 > 1.0). Truncation invariance passes.
  - `independent_rw` has no microstructure effect, yet shows Sharpe +0.75: the cause is T3. Diagnostic (scratch run, not committed): recording each position still open at a window end at its mark-to-market gives 5452 such positions with mean MTM −1.90% (t = −25.3); closed trades have mean +0.79% (t = +10.6); together mean +0.01% (t = +0.2).
- Real data (2014-12-31 → 2023-12-31, `pairs_bias_test.csv`, `closes_daily_extended_v2.csv`) before → after: **identical**. Post-refactor CLI in `output/audit_runs/t0_post/`: both trades CSVs `cmp`-identical to `output/audit_golden/data/wfo_daily_results_audit_pre_{flat,almgren}.csv`, and the summary block identical to the golden logs.
  - flat: 34 windows, 1901 trades, win 78.1%, total PnL +28.1218, MaxDD −0.5701, Sharpe(full) +3.853, Sharpe(exit) +5.671 → same.
  - almgren (default): 34 windows, 1901 trades, win 77.9%, total PnL +24.1792, MaxDD −0.6051, Sharpe(full) +3.392, Sharpe(exit) +4.956 → same.
  - Extra check (scratch): the 46ffdfb CLI and the new CLI on the same real data for 23 option sets (every option above plus a combination of all post-filters): all trades CSVs byte-identical and summary blocks identical; `--metagate` and `--hrp` fail with the same `UnboundLocalError` in both.
  - Run time of one CLI run: 4.9–5.5 s before → 2.5–3.1 s after (numpy per-bar loop).
- Rulings / conservative interpretations:
  - R1: `WfoConfig` = frozen dataclass of scalars named after the argparse dests; volumes / VWAP are keyword arguments of `run_wfo_daily`.
  - R2: at T0 `daily_pnl` = exit-date-booked PnL over the panel's OOS trading days (what `summarise` uses for Sharpe(full)); T4 switches to daily MTM over all business days.
  - R3: "gross" = flat model with cost = slip = borrow = 0.
  - R4: truncation test at T0 compares all trades of the windows the truncated run has; T3 tightens it.
  - R5: xfail marks only where measured to fail (all 20 null tests).
  - R6: fixtures from the 46ffdfb CLI (`git show`), committed via a `.gitignore` negation. Its helper modules are extracted from 46ffdfb too and are the subprocess's only PYTHONPATH entry (controller ruling in review, superseding the brief's "PYTHONPATH = repo root"), so later changes to the helpers cannot reach the fixtures; the library versions they were generated with are in `tests/README.md`.
  - R7: `--metagate` / `--hrp` `UnboundLocalError` preserved (see limitations).
  - R9: positive control uses seeds 0–9; R10: `independent_rw` has zero log drift; R11: async FX is USD per local unit, the daily FX close is sampled at local close + `fx_lag_hours`.
  - R13 (CLI stdout): the run header line `pairs: N from <file>` became `pairs: N` — the pairs file name is not an input of `run_wfo_daily` (`WfoConfig` excludes the file-IO options); and a missing vwap / volumes file now stops the CLI before the `hmm mode` print instead of after it. Written files and the summary block are identical.
  - R12: the brief's sanity check "lag 0, no holidays, no FX lag, no noise → identical legs and zero trades" is impossible with floating point: the converted leg equals the US leg only to 1 ulp (≈ 87% of days bitwise), and even bitwise-identical legs give an OLS β of 1 ± 1–2 ulp in ≈ 72% of windows, so the z-score of a ~1e-16 spread still crosses ±2 (≈ 30 trades, each |pnl| ≤ 8.9e-16). The test therefore asserts identical legs (rtol 1e-12) and no trade PnL above 1e-12.
  - Synthetic calendars cover whole calendar years (`n_years=10` → 2005-01-03 … 2014-12-31) so every year has its `holidays_per_year` local holidays.
  - Identity cases beyond the brief's list (hmm block-calm / extreme-3state, cusum 10 / 30, a SPY column, a late-listed ticker with gaps, a ticker missing from the panel, a ticker without volumes, a losing momentum pair) were added so the moved code's branches run: `--cusum 5` alone breaks inside the training slice and never trades.
  - `universe_expand.main()`: when the Yahoo local download comes back as an empty, column-less frame while FX is not empty, the old code raised `AttributeError` (`RangeIndex.tz`); `build_cross_listed_usd` now returns no local columns instead. Every other path gives identical frames (scratch check against the old inline code: 90 cases, incl. tz-aware indexes, missing columns and gaps). Not run: those scripts download from Yahoo.
- Known limitations / missing data:
  - Defect (R7), pre-existing and kept: with `--metagate` or `--hrp`, `run_wfo_daily` (formerly `main()`) passes `cost_log` / `borrow_per_day` to the train-trade harvest before they are assigned → `UnboundLocalError` in window 1. Must be fixed before either flag is used.
  - At T0 no corrected behaviour exists, so the harness only proves that it detects the current bugs; the `fixed` column turns green as T1–T4 land.
  - `daily_pnl` is still exit-date booked (T4).
  - The rotation drop rule is not exercised by the synthetic fixtures (it is in the real-data sweep: 1875 of 1901 trades kept).
