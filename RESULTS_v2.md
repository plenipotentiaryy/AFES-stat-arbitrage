# RESULTS_v2 — AFES daily pairs backtest after the audit fixes (T0–T10)

> **THE HEADLINE CANNOT BE PRODUCED.** The plan's T11 headline needs a frozen point-in-time
> universe (`universe_pit_<D>.csv`, built by `build_universe_pit.py`, T5), and none exists: the repo has
> no historical index-constituents data. Missing (owner): membership spells (start / end dates) of the
> index or indices to trade over 2006–2014 at least, each member's sector / industry **as of the as-of
> date**, the source of that data, and daily price histories of the members delisted, merged or acquired
> after the as-of date. Until they exist, **no number in this file is a headline result or a valid
> performance estimate.**

Plan definition of done: the placebo nulls, the positive control and the truncation-invariance tests
pass (section 3). The rest cannot be met without the universe, so it is met only in the sense that no
headline is reported: the only real-data run below uses pairs selected after the fact, which the plan
forbids for a reported result. The final holdout (dates from 2024-01-01) was not touched: every run ends
at 2023-12-31, and `HOLDOUT_USES.md` has no entry.

## 1. Non-headline diagnostic — hindsight universe (pairs_bias_test.csv, US-only after the T2 policy); NOT a valid performance estimate

**UPPER BOUND.** `pairs_bias_test.csv` comes from a candidate list built with hindsight (hand-added
layers, some added after earlier results had been seen) and from today's names (survivorship), T5. It
can only overstate what a point-in-time universe would give. The run uses the corrected defaults: one
continuous position per pair across windows, fills at the next close, PnL from the leg fills, daily
mark-to-market (MTM), the entry guard, flat costs per fill per leg, and no `--legacy-*` or optional
flag. The CLI prints the `NON-HEADLINE RUN` banner and `headline=False`.

### 1.1 Run identification

| item | value |
|---|---|
| command (cwd `output/audit_runs/t11/`, `data/` = symlinks to the owner's `data/`) | `python step3j_wfo_daily.py --pairs pairs_bias_test.csv --closes closes_daily_extended_v2.csv --start 2014-12-31 --end 2023-12-31 --tag t11_default` (the default model is flat) |
| comparison run | the same + `--execution-model almgren --tag t11_almgren` |
| git | both runs on `f5af875` (T10), clean tree (sidecar `git_revision` `f5af875f379d…` without `+dirty`). This file is added by the T11 commit on top of `f5af875`; no code changed in T11. The final-review commit `fix(review): Final whole-branch review fixes` changes code (R49–R52: the HEADLINE marker's cost rule, `--exec vwap` entries, labels, cut tests) but no default output: the default configuration re-run once on it has the same config hash and byte-identical trades, MTM and per-pair MTM files (`output/audit_runs/final/`, one more `TRIALS.csv` row, N unchanged; `AUDIT_FIX_LOG.md`, "Final review fixes"). The next commit (R55, R56: cut tests at four dates; a stable, causal tie order in three non-default filters, see section 3) changes no code the default run executes; the default configuration re-run once more on it (`output/audit_runs/final_r56/`, sidecar `339cf48…` without `+dirty`) has the same config hash and byte-identical trades, MTM and per-pair MTM files (one more `TRIALS.csv` row, N unchanged). |
| universe SHA-256 | **n/a — no PIT universe** |
| pairs file | `data/pairs_bias_test.csv`, 36 pairs → 26 US-only pairs after the T2 policy; SHA-256 `1790dd9ee45fed48e32385d0e3dc8755c5ba2bb34fcb55d5f34bcbb808ccce91` (equal to the value recorded at T5; a hindsight file, not a frozen universe) |
| closes / volumes | `closes_daily_extended_v2.csv` SHA-256 `b4a47bc6995c21a9527fcdfc3919d472a8dfc1dbc68f5eab1a3b3186af2a662e` (also in the sidecar); `volumes_daily.csv` SHA-256 `c2b798c60c5eb1693d591f91179567fa5f4065046622930c26b38ea83199bc6a` (stale-print mask; almgren ADV) |
| period | 34 walk-forward windows (training anchored at 2014-12-31, 3-month OOS periods); MTM span 2015-06-30 → 2023-12-29 = 2219 business days |
| config hash | default `c418c56c3e71a1729f54b2f1257e647c7c2969bdb00456faa5e756fcda4b2e57` = the T9 R40 row of `TRIALS.csv` (same configuration, so this run leaves N unchanged at 120); almgren `fe931f887fcbb6583b5d8c2e5206805b36d3ccdaa3a7f41d8e634c7901c0b15d`, a new configuration (≠ T8's `fe48158e…` because T9 added three `WfoConfig` fields) → N = 121 |
| files | trades, MTM and per-pair MTM files of both runs are byte-identical to `t9_flat_r40` (default) and `t8_almgren_r38` (almgren) |

Cost model (from the meta sidecar `wfo_daily_meta_t11_default.json`, label "NET flat 10.0 bp/leg/fill +
FX 2.0 bp"): per fill per leg, commission 5 + slippage 5 = 10 bp × the leg's notional at the fill (round
trip at unchanged prices 2·10·(1 + |β|) bp per $1 of leg-1 notional); FX 2 bp per fill of a `*_USD` leg
(none among the 26 pairs); borrow on each short leg × its notional at each held row's close, 80 bp/yr for
US listings (300 bp/yr non-US). Almgren (comparison): per fill per leg the leg's tier half-spread +
κ·σ·√(leg notional / ADV), leg 1 at $100,000, leg 2 at |β|·$100,000, borrow at the leg's tier rate.
Rule R25 (T8): the default is the model with the lower development Sharpe (flat), never the higher one.

### 1.2 Metrics

Units: PnL in USD per $1 of leg-1 notional per trade; capital = 26 pairs × 2.0 = 52 in the same unit,
constant; daily return = daily MTM PnL / 52.

| metric | definition | default (flat) | almgren (comparison) |
|---|---|---|---|
| **Sharpe** | mean / sample sd of the daily MTM return over all 2219 business days (zero days included) × √252 | **+0.278** | +0.367 |
| PSR(0) | Φ((SR̂ − 0)·√(T − 1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4·SR̂²)), SR̂ per day, T = 2219, γ₃ skew, γ₄ raw kurtosis of the daily returns | **0.800** (SR̂ +0.0175/day, γ₃ +2.55, γ₄ 50.1) | 0.868 |
| DSR, N = 121 | PSR(SR₀), SR₀ = the expected maximum Sharpe of N no-skill trials with Sharpe variance V (section 1.4) | **4.35e-12** | 2.05e-11 |
| MaxDD | largest fall of the date-ordered daily MTM equity (starting at 0), % of capital | **−4.50%** (peak 2016-07-15 → trough 2018-10-02) | −4.09% (same dates) |
| turnover | annualised traded notional / capital: Σ over fills of \|leg notional\| ÷ capital ÷ years | **9.52 × capital per year** | 9.52 (same trades) |
| ann. return / ann. vol | mean daily return × 252 / sd × √252 | +0.56% / 2.03% | +0.74% / 2.03% |
| trades, win rate, profit factor | trades of the run; share with net PnL > 0; Σ wins / \|Σ losses\| | 1166, 57.2%, 1.10 | 1166, 58.0%, 1.13 |
| Σ gross / Σ cost / Σ borrow | over the trades | 7.648 / 4.358 / 0.710 | 7.648 / 3.399 / 0.844 |
| exit reasons | target / time / stop / end of run | 585 / 447 / 126 / 8 | same |

Turnover in detail: leg notional at a fill = size·|q·P| at the fill price (entry: size·1 and size·|β|;
exit: size·|q1·P1_exit| and size·|q2·P2_exit|); years = 2219 business days / 252 = 8.806. Σ over the 4664
fills (1166 trades × 2 legs × 2 fills; size 1 on every trade) = 4358.15, ÷ 52 = 83.81, ÷ 8.806 = 9.52.
Check: the flat model charges 10 bp × each |leg notional|, and 0.001 × 4358.15 = 4.358 = Σ cost.

```python
tr = pd.read_csv("data/wfo_daily_results_t11_default.csv", float_precision="round_trip")
notional = tr["size"] * ((tr.q1 * tr.p1_entry).abs() + (tr.q2 * tr.p2_entry).abs()
                         + (tr.q1 * tr.p1_exit).abs() + (tr.q2 * tr.p2_exit).abs())
turnover = notional.sum() / 52.0 / (2219 / 252)       # 9.52
```

Where the PnL comes from (flat): the 88 trades entered in window 19 (OOS 2019-12-30 → 2020-03-30) net
+2.578 of the run's +2.580; the trades of the other 33 windows net +0.002 (almgren: +2.647 of +3.404).
Daily MTM PnL by calendar year, % of capital: 2015 −0.12, 2016 +0.82, 2017 −1.14, 2018 −2.17,
2019 +2.45, 2020 +4.38, 2021 −1.39, 2022 +2.19, 2023 −0.06.

Analyses of the default run (no new trial):
- Block bootstrap (10 000 replicates, mean block 5 days; **sampling uncertainty only — does not detect
  bias**): Sharpe 95% CI [−0.349, +0.850], p(Sharpe < 0) = 0.188.
- Factor attribution (daily MTM return on 12 same-day ETF log returns, 2141 aligned days): R² 0.171;
  HYG loading t +4.62, TLT t −2.78, XLU t +2.70, XLP t −2.31; alpha t +0.47; factor-neutral
  information ratio +0.147 (alpha + residual, × √252; not a Sharpe).
- Stress: 2020 COVID crash window (43 business days) PnL +2.35 with drawdown −1.099; worst day
  2020-03-18, −0.603 (−1.16% of capital); 2017 calm bull −0.60. Daily VaR 1% −0.156, CVaR 1% −0.227
  (USD per $1; ÷ 52 for % of capital).
- Per pair (dashboard, each pair's daily MTM on 2.0 of capital): Sharpe from −0.513 (PNC-SNA) to +0.774
  (HD-SHW); 9 of the 26 pairs negative.

### 1.3 Per-block table

US-only block: no as-of sector classification exists in the repo (T5), so the 26 pairs form one block.
(The `source_layer` labels of `production_universe*.csv` are not sectors: "baseline_same_sector" also
covers QCOM-TROW, BAC-CRH, ADSK-BX.) Block Sharpe and MaxDD from the per-pair MTM file on block
capital = pairs × 2.0.

| block | pairs | capital | trades | win rate | total PnL | Sharpe (daily MTM) | MaxDD |
|---|---|---|---|---|---|---|---|
| US-only, flat (default) | 26 | 52 | 1166 | 57.2% | +2.580 | +0.278 | −4.50% |
| US-only, almgren | 26 | 52 | 1166 | 58.0% | +3.404 | +0.367 | −4.09% |

Cross-listed pairs: all 10 are **excluded** from the run (T2 policy; reasons as the T11 CLI prints
them). The data to run them correctly do not exist (section 5). The numbers on the right are from T2's
`--legacy-cross-listed` flat run (T2 code: per-window engine, exit-date booking, no entry guard, pre-T8
costs) and are **LEGACY — NOT VALID**: they only show that the block did not carry an edge once fills
stopped using stale prices.

| pair | exchanges | reasons for exclusion | LEGACY — NOT VALID: trades, win rate, total PnL (T2 flat) |
|---|---|---|---|
| SU-SU.TO_USD | NYSE/Nasdaq + TSX | fx_unaligned | 89, 41.6%, −0.1366 |
| TD-TD.TO_USD | NYSE/Nasdaq + TSX | fx_unaligned | 91, 30.8%, −0.2222 |
| RY-RY.TO_USD | NYSE/Nasdaq + TSX | fx_unaligned | 94, 29.8%, −0.2250 |
| BNS-BNS.TO_USD | NYSE/Nasdaq + TSX | fx_unaligned | 89, 38.2%, −0.1822 |
| ENB-ENB.TO_USD | NYSE/Nasdaq + TSX | fx_unaligned | 90, 35.6%, −0.2062 |
| SNY-SAN.PA_USD | NYSE/Nasdaq + Euronext Paris | fx_unaligned; Euronext Paris: no synchronous 09:30-11:30 ET snapshots | 91, 42.9%, −0.0204 |
| SAP-SAP.DE_USD | NYSE/Nasdaq + Xetra | fx_unaligned; Xetra: no synchronous 09:30-11:30 ET snapshots | 98, 34.7%, −0.1716 |
| NVS-NOVN.SW_USD | NYSE/Nasdaq + SIX | fx_unaligned; SIX: no synchronous 09:30-11:30 ET snapshots | 88, 34.1%, −0.2225 |
| BHP-BHP.AX_USD | NYSE/Nasdaq + ASX | fx_unaligned; ASX: no session overlap with NYSE | 99, 41.4%, −0.2473 |
| RIO-RIO.AX_USD | NYSE/Nasdaq + ASX | fx_unaligned; ASX: no session overlap with NYSE | 76, 59.2%, +0.6452 |
| block | | | 905, 38.5%, −0.9888; LEGACY exit-date ratio −0.600 (not a Sharpe). Pre-audit code on the same block: 916 trades, 97.9% wins, ratio +6.054 |

### 1.4 Trials: N, V, SR₀ and the DSR

- **N = 121** distinct configuration hashes in `TRIALS.csv` (123 rows) after T11: 107 back-filled
  pre-audit trials (T6), 12 CLI configurations of T6–T11 and 2 registered diagnostics (T9). N is a lower
  bound (trials never saved to disk are not counted).
- **V = 0.00377**: sample variance of one recorded per-day Sharpe per configuration (77 of the 121; legacy
  rows carry none), sd 0.0614 per day = 0.975 annualised. 66 of the 77 are back-filled exit-date ratios
  of the pre-audit trials, which carry the defects T1–T4 removed, so V is larger than corrected re-runs
  would give.
- **SR₀** = √V·((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))), γ = 0.5772 → 0.1595 per day = **2.53
  annualised**. Any corrected strategy near the expected Sharpe of 0–0.5 gets DSR ≈ 0 against it.
- DSR with the registry's N and V (the reported value): flat **4.35e-12**, almgren 2.05e-11. At run time
  the default run printed DSR 3.86e-12 (N = 120, V = 0.0038 from 76, before the almgren row); the almgren
  run printed the final values.
- **Sensitivity.** No V gives DSR > 0.95: SR₀ ≥ 0 for every V ≥ 0, so DSR ≤ PSR(0) = 0.800 (flat; almgren
  0.868), with equality at V = 0 (SR₀ = 0). DSR > 0.5 (SR₀ < SR̂) would need V < 4.55e-5, 1.2% of the
  registry's V (trial Sharpe sd ≤ 0.107 annualised). With the registry's N and V, DSR = 0.95 would need a
  per-day Sharpe of 0.194 (3.08 annualised) with the same T, skew and kurtosis.

## 2. What changed and why (before → after on real data)

`pairs_bias_test.csv`, `closes_daily_extended_v2.csv`, 2014-12-31 → 2023-12-31, 34 windows. Up to T3
the statistic is the pre-audit exit-date ratio (PnL booked on exit dates, panel days, × √252),
**not a Sharpe**; from T4 on it is the one daily-MTM Sharpe. All rows are NON-HEADLINE (hindsight universe).

| step | pairs | trades | flat | almgren | statistic | MaxDD flat (date-ordered MTM) |
|---|---|---|---|---|---|---|
| pre-audit code (`46ffdfb`, T0 golden runs) | 36 | 1901 | +3.853 | +3.392 | LEGACY exit-date ratio (full days) — not a Sharpe | — |
| T1 fills at the next close | 36 | 1898 | +0.809 | +0.211 | same | — |
| T2 cross-listed pairs out of the headline (26 US-only); stale prints masked; PnL from leg fills | 26 | 979 | +1.008 | +0.817 | same | — |
| T3 positions kept across window ends | 26 | 1202 | +0.641 | +0.425 | same | — |
| T4 daily MTM over all business days | 26 | 1202 | +0.455 | +0.302 | daily MTM Sharpe | −3.64% |
| T5 (PIT builder, NON-HEADLINE marker), T6 (holdout, DSR) | 26 | 1202 | +0.455 | +0.302 | unchanged | −3.64% |
| T7 entry guard entry_z < \|z\| < stop_z | 26 | 1166 | +0.493 | +0.338 | daily MTM Sharpe | −3.67% |
| T8 costs per fill per leg; default model → flat | 26 | 1166 | +0.278 | +0.367 | daily MTM Sharpe | −4.50% |
| T9 remaining look-ahead (default trades unchanged), T10 outputs (no engine change) | 26 | 1166 | +0.278 | +0.367 | unchanged | −4.50% |
| **T11 final runs (this file)** | 26 | 1166 | **+0.278** | +0.367 | daily MTM Sharpe | −4.50% |

The documented 3.73 is `wfo_daily_results_fullrun.csv`: flat costs over the full span, which runs into
what is now the final holdout; its first 34 windows are the +3.853 above (the pre-audit code's golden
run; the T8 code with every legacy flag reproduces it cell for cell), and the documented command, which
then defaulted to almgren, gives +3.392 (`AUDIT_FIX_LOG.md` T8, reproduction note; nothing was run on
dates from 2024).

Steps that lowered the number:
- **T1**: the pre-audit code filled at the close that made the signal (3.853 → 0.809 flat).
- **T3**: positions open at a window end used to be dropped; on these pairs they lost on average
  (marked at the cut: −58.1 bp, t −2.08, 335 positions), so keeping them lowers the ratio (1.008 → 0.641).
- **T4**: daily MTM over every business day instead of exit-date booking (+0.641 → +0.629 from the
  calendar alone, → +0.455 with the marks); the MaxDD is now date-ordered.
- **T8**: the old flat model charged 20 bp per round trip whatever |β| (5 bp per leg per fill); per fill
  per leg on each leg's notional the mean cost per trade is 37.4 bp (0.493 → 0.278 flat).

Steps that raised it:
- **T2**: the 10 cross-listed pairs, which lost money once filled at the next close (at T1 their block had
  an exit-date ratio of −0.645 flat / −2.236 almgren), left the headline; on the 26 US pairs alone the T2
  corrections moved the ratio from +1.018 to +1.008.
- **T7**: 99 of the pre-guard entries had been decided at |z| ≥ 3.5; the guard refuses them. With the
  changed position paths, 103 pre-guard entries disappeared (net +0.747 flat) and 67 new entries took
  their place (net +0.930), so the Sharpe rose (0.455 → 0.493). The rule stays because it is the
  specified one, not because of the result.
- **T8, almgren only** (0.338 → 0.367; 0.369 before fix R38): leg 2 is now charged on |β| (mean |β| 0.857)
  and borrow only on the short leg. The default became flat, the lower of the two.

Not in the default path: `--kalman` +0.562 and `--hmm block-panic` +0.113 (T9). They are
separate trials (counted in N), not reasons to change the default.

## 3. Placebo, positive control and truncation tests

`python tests/placebo_report.py` on the final code (gross: flat, all costs 0; null seeds 0–19, control
seeds 0–9; Sharpe = the daily Sharpe of `WfoResult.daily_pnl` per seed, mean ± SE over seeds; t = pooled
per-trade t-statistic, not a Sharpe). For `legacy` the series is booked on exit dates, so its column is
a LEGACY exit-date ratio, not a Sharpe. Pass: null mean Sharpe in [−0.2, +0.2] and |t| < 3 (each on
≥ 1000 pooled trades); control mean Sharpe > 1.0. `legacy` = every `--legacy-*` behaviour with the
pre-audit async panel builder; `fixed` = the corrected defaults.

| variant | dataset | seeds | mean Sharpe ± SE | per-trade t | trades | win rate |
|---|---|---|---|---|---|---|
| legacy | independent_rw | 20 | +0.750 ± 0.066 | +10.6 | 13391 | 59.5% |
| legacy | async_lag4.5_hol0 | 20 | +9.702 ± 0.037 | +593.0 | 18729 | 100.0% |
| legacy | async_lag4.5_hol5 | 20 | +7.768 ± 0.065 | +354.8 | 18286 | 99.3% |
| legacy | async_lag15_hol0 | 20 | +9.756 ± 0.045 | +591.2 | 18651 | 100.0% |
| legacy | async_lag15_hol5 | 20 | +9.206 ± 0.055 | +506.5 | 18668 | 99.6% |
| fixed | independent_rw | 20 | +0.005 ± 0.070 | +0.1 | 16671 | 56.9% |
| fixed | async_lag4.5_hol0 | 20 | −0.118 ± 0.056 | −2.3 | 19062 | 49.4% |
| fixed | async_lag4.5_hol5 | 20 | −0.008 ± 0.059 | −0.1 | 18681 | 50.1% |
| fixed | async_lag15_hol0 | 20 | +0.024 ± 0.056 | +0.5 | 18967 | 49.7% |
| fixed | async_lag15_hol5 | 20 | +0.039 ± 0.049 | +0.8 | 18563 | 49.7% |
| fixed | ou_positive_control | 10 | +6.613 ± 0.038 | +125.1 | 5721 | 94.9% |

- The `legacy` rows show what the harness catches. The same asset printed on two exchanges, with no edge,
  gives a ratio of +7.8 to +9.8 and 99–100% wins under same-close fills on forward-filled, daily-FX-converted
  prints. Two independent random walks give +0.75 when positions open at a window end are dropped. All 10
  legacy null tests fail and are strict `xfail`, so they would turn the suite red if they ever passed.
- Every `fixed` null passes. The closest to a bound is `async_lag4.5_hol0` (−0.118, t −2.3); fresh seeds
  20–79 gave +0.015 ± 0.046 there (T3 code). The OU positive control (true edge) passes at +6.613.
- Truncation invariance: `test_truncation_invariance` runs both variants on `independent_rw`,
  `async_lag15_hol5` and `ou_positive_control`, cut at the middle date. Every entry and exit decided
  before the cut is identical to the full run's, and all six cases pass. The per-flag tests of T9 (cut and different future)
  pass for `--ivol --volvol --pairmom --rotation --voltarget --cusum --aussie-filter --kalman --hmm block-panic`,
  and since the final review (R50) for the six options a headline run may use: `--maxconc`, `--ticker-overlap`,
  `--per-pair-quarter-cap`, `--vol-tight-stop`, `--vwz` and `--exec vwap` (the cut and the different future
  apply to the closes, the volumes and the VWAPs alike). Since R55 they run at four cut dates (2005-10-03,
  2006-12-25, 2008-07-07, 2009-06-08) and also compare every decision made before the date; nine look-ahead
  mutations of these options (a few rows ahead in `--vol-tight-stop` / `--vwz`, a VWAP two rows ahead, outcome-
  or exit-ordered filters) all fail them. The four-date test found one real look-ahead (R56, non-default only):
  `--maxconc` and `--ticker-overlap` could keep a different one of two entries filled on the same day depending on
  later trades, because the post-hoc filters took same-day entries in an unstable sort order over the whole trades
  table; the three filters now use (fill date, pair name) (`--per-pair-quarter-cap` was not affected: it counts per
  pair, and a pair has at most one entry per day). `--metagate` and `--hrp` stay without a test and are refused
  in a headline run. Also allowed in a headline run without a per-flag cut test (re-review): the window geometry
  `--train` / `--oos` / `--step` (the final review's 20-seed null probes cover it) and the `--hmm` modes other than
  `block-panic` (the same forward filter as the tested mode).
- `--exec vwap` (final review R51): an entry decided on a row no longer requires a VWAP on the next row; a
  leg without one fills at its close, as at the exit. Gross null probe (20 seeds, VWAP = close × 1 bp noise),
  mean Sharpe ± SE before → after: `independent_rw` +0.005 ± 0.071 → +0.005 ± 0.071, `async_lag15_hol5`
  +0.033 ± 0.049 → +0.033 ± 0.049 (identical per seed: that VWAP is missing only where the close is); with
  10% of the VWAP cells also removed: −0.023 ± 0.068 → +0.005 ± 0.071 and +0.056 ± 0.048 → +0.032 ± 0.049
  (the entries the old rule dropped are back). All inside the bounds, |t| ≤ 1.2.
- Full suite on the final code: `python -m pytest -q` → **479 passed, 10 xfailed** in 92.1 s (93.0 s
  wall, load average 5–7); the placebo report took 25.5 s. After the final-review commit: **493 passed,
  10 xfailed** in 95.6 s (96.6 s wall, load average 4–7); after R55 / R56: **493 passed, 10 xfailed** in
  101.1 s (101.9 s wall, load average 2.2–2.4); after the re-review follow-ups (one more test): **494 passed,
  10 xfailed** in 114.7 s (115.6 s wall, load average 4.0–6.7); the placebo table above is unchanged.

## 4. Phase 2 (intraday) — out of scope

The intraday path (`step4a_backtest.py`, `step3j_wfo.py`, `step4b_*`, `step4c_*`, `step4d_*`, the grids
and profilers) was not audited or fixed here. Its numbers, "Sharpe 4.20, WR 83.9%"
(`step4b_backtest_strict.py:13`) and "honest OOS Sharpe 2.06 → 2.63" (`step4d_backtest_full.py:281`),
are **not validated**. Phase 2 waits for the owner's decision: B0 (deprecate) or B1–B7 (fix).

## 5. Open items for the owner

1. **Point-in-time universe (T5)**: historical constituents with as-of sectors / industries, their source,
   and the delisted members' price histories. Then: `python build_universe_pit.py --asof 2014-12-31
   --constituents <file> --closes data/closes_daily_r3k.csv`; record the printed SHA-256 in
   `AUDIT_FIX_LOG.md`; run the headline with `--pairs universe_pit_2014-12-31.csv --start 2014-12-31
   --end 2023-12-31`. The CLI-printed SHA-256 must equal the recorded one.
2. **Ticker mapping**: hyphenated tickers (`BRK-B`, `BF-B` in `closes_daily_r3k.csv`) are refused by the
   builder, because pair names are split on "-".
3. **Cross-listed pairs (T2)**: raw local closes with exchange timestamps, intraday FX at each local
   close, and synchronous 09:30–11:30 ET snapshots for European pairs. Without them all 10 stay excluded.
4. **Live signal (T7)**: `live/daily_signal.py` (entry at :280-283) still enters at any |z| > entry_z, so a
   live entry can be sent on a 5–10σ jump. Other copies of the rule are unguarded too: `metagate_daily.py`,
   `daily_sanity.py` and the research scripts listed in the T7 log entry.
5. **Phase 2 decision**: B0 or B1–B7.
6. **`--metagate` / `--hrp` (R7)**: still crash with `UnboundLocalError`. Their train harvest reads the
   window including `oos_s` and would see zero costs under T8. Both are refused in headline runs.
7. **`edge_monitor.py` / `compare_results.py`** accept only daily WFO runs with a meta sidecar (R43);
   using them on a live ledger needs a decision.
8. **`bias_test.py` / `run_*.sh`** now pass `--end 2023-12-31`. The scripts still `cd` into the owner's
   checkout and were not run.
9. **Off the audited path**: the same unstable same-day order (R56) is still in the concurrency caps of
   `pca_residual_backtest.apply_maxconc`, `pca_residual_refined.apply_maxconc` and the intraday
   `step4b_concurrent_cap.py` / `step4d_backtest_full.py` (Phase 2); none of them writes a headline sidecar.
   **Labels**: `daily_sanity.py`, `pca_residual_backtest.py`, and the intraday
   `step3j_wfo.py` / `step4c_bootstrap_sharpe.py` (Phase 2) still print exit-date ratios as "Sharpe".
   Runs with `--legacy-*` options that are not exit-date booked still print an MTM Sharpe under the
   LEGACY banner (never in V or a DSR).
10. **Production universe**: `audit_baseline.py` refuses trades files with exits on or after 2024-01-01,
    so the production universe summary can only be rebuilt from a development-period trades file.
    `baseline_sharpe` in the production universe holds a LEGACY exit-date ratio; renaming it is the
    owner's decision.
11. **Assumptions in the costs (T8)**: no borrow / locate or FX execution data; 80 / 300 bp/yr borrow and
    2 bp per FX conversion are assumptions.
12. **Trials (T6)**: V comes from the pre-audit trial results. Re-running those configurations under the
    corrected engine would give a smaller V; it was out of scope. Add any trials that were never saved
    to disk to `TRIALS.csv`. The config hash names the closes file, not its content: a regenerated
    `closes_daily_extended_v2.csv` under the same name counts as the same trial; the meta sidecar's closes
    SHA-256 makes every run traceable to the exact file. Hashing the content is an owner decision (it
    would change every future hash).
13. **Headline cost rule (final review R49)**: a run is HEADLINE only with `--cost`, `--slip`, `--borrow`,
    `--borrow-nonus` and `--fx-cost` not below their defaults (5 / 5 bp per leg per fill, 80 / 300 bp/yr,
    2 bp; higher costs are allowed) and the default `--execution-model flat` (almgren is the comparison
    model, R25). An owner who documents execution costs below 5 / 5 bp, or wants almgren as the headline
    model, must change this rule in `step3j_wfo_daily.non_headline_reasons` explicitly.

## 6. Sources

| section / table | source |
|---|---|
| 1.1 commands, hashes, files | `output/audit_runs/t11/run_all.sh`; CLI logs `run_t11_default.log`, `run_t11_almgren.log`; sidecars `data/wfo_daily_meta_t11_{default,almgren}.json`; `shasum -a 256` of the input files; `cmp` against `output/audit_runs/t9/data/*_t9_flat_r40*` and `output/audit_runs/t8/data/*_t8_almgren_r38*`; `TRIALS.csv` (last 3 rows); `AUDIT_FIX_LOG.md` T5 (pairs SHA-256) |
| 1.2 Sharpe, return, vol, MaxDD, trades, win, PF, exit reasons, window-19 PnL | CLI logs (summary block, window lines) |
| 1.2 PSR(0) (4 decimals), DSR at N = 121, skew / kurtosis, turnover, Σ gross / cost / borrow, PnL by window and year | `tools/results_v2_derived_stats.py` (committed in the final review from `output/audit_runs/t11/derived_stats.py`, logic unchanged; run with cwd `output/audit_runs/t11/`) → `derived_stats.log` there (`perf_stats`, `mtm`, `step3j_wfo_daily.registry_trials` on the run files; reads only, no registry row; re-run on the final-review code before its re-run: identical output) |
| 1.2 bootstrap, factor, stress, per-pair | `output/audit_runs/t11/run_downstream.sh` → `ds_bootstrap.log`, `ds_factor.log`, `ds_stress.log`, `ds_dashboard.log` (identical to T10's logs of the same series, except for file paths) |
| 1.3 US block | `derived_stats.log` (per-pair MTM files), CLI logs |
| 1.3 cross-listed exclusions and reasons | CLI log `run_t11_default.log` (CROSS-LISTED PAIRS table) |
| 1.3 LEGACY — NOT VALID numbers | `AUDIT_FIX_LOG.md` T2 (cross-listed table; pre-audit block +6.054 from T1/T2) |
| 1.4 N, V, SR₀, DSR, sensitivity | `derived_stats.log`; `TRIALS.csv`; CLI logs (run-time N, V, DSR) |
| 2 before → after chain | `AUDIT_FIX_LOG.md` T0–T10 real-data tables; T11 = CLI logs |
| 3 placebo table | `output/audit_runs/t11/placebo_report.log` (identical to the T7–T10 tables) |
| 3 suite result | `output/audit_runs/t11/pytest.log` |
| 4, 5 | the plan (`AFES_AGENT_TASKS.md`), `AUDIT_FIX_LOG.md` known limitations of T2, T5–T10 |
