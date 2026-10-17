# Audit rulings (R1–R57)

Interpretive decisions taken while fixing the daily backtest (`AFES_AGENT_TASKS.md`, T0–T11, and the final
whole-branch review), where the plan was ambiguous, silent, or needed a concrete rule. Per the plan (§2) each
ambiguity took the more conservative reading (the one that gives lower performance). "Cost if wrong" is given
where it was assessed. The per-task details, numbers and tests are in `AUDIT_FIX_LOG.md`; the results in
`RESULTS_v2.md`. Wording is kept as recorded at the time ("implementer", "controller", "reviewer" are the roles
of the agents that did the work; "amend" refers to a task's own commit before the next task started).

## T0

- **R1** — WfoConfig = frozen dataclass, field names = argparse dests minus pairs/closes/tag; volumes/vwap are kwargs — keeps cfg hashable for T6 — cost if wrong: a later signature change.
- **R2** — T0 daily_pnl = exit-date booking over panel trading days (identity with summarise); T4 switches to MTM over pd.bdate_range (conservative: holidays as zero days) — cost if wrong: ~1-2% Sharpe difference.
- **R3** — "gross" for placebo = flat model with cost=slip=borrow=0 — almgren would classify synthetic tickers and add fallback ADV costs.
- **R4** — T0 truncation test compares trades from windows present in the truncated run; T3 tightens to decisions before D — cost if wrong: weaker T0 test only.
- **R5** — null tests xfail(strict=True) only where measured to fail; passing ones stay unmarked and are logged — never force failures.
- **R6** — synthetic golden fixtures from the pre-refactor CLI (git show 46ffdfb) committed under tests/fixtures/t0_golden (.gitignore negation) — permanent legacy anchor.
- **R7** — --metagate/--hrp UnboundLocalError (cost_log before assignment) is pre-existing; preserved in T0, logged as known defect.
- **R8** — all real-data numbers end 2023-12-31 (holdout protection ahead of T6).
- **R9** — positive control uses 10 seeds (plan gives no count); nulls use 20.
- **R10** — independent_rw keeps literal zero log drift (price drift σ²/2 is side-symmetric, negligible).
- **R11** — async FX convention USD-per-local (AUDUSD=X style, universe_expand multiply path); FX daily close sampled at local close + fx_lag.
- **R12** — degenerate async check asserts identical legs and |pnl| ≤ 1e-12 instead of "zero trades" — OLS β = 1 ± ulp makes z of a 1e-16 spread cross ±2 — cost if wrong: none economically.
- **R13** — accept CLI stdout deltas (header "pairs: N" without file name; missing-file SystemExit now before hmm/vwap prints); written files + summary identical — cost if wrong: cosmetic.
- **R14** — fix T0 minors now as an amendment of the T0 commit (one commit per task), because the harness gates every later task: xfail raises=AssertionError, null trade-count floor, hmm_block_panic case on a panel where the gate binds (+ make_golden asserts each flag case differs from its baseline), make_golden extracts helper modules from 46ffdfb too (+ README records library versions), delete false DST note in AUDIT_FIX_LOG, single GROSS, drop trailing blank line — cost if wrong: ~30 min.

## T1

- **R15** — fill-only bar capped at cfg.end (≡ cutting closes at cfg.end; holdout protection) — accepted — cost if wrong: none.
- **R16** — daily_pnl index extends to the last exit fill (else a booked trade is dropped) — accepted; T4 replaces daily_pnl anyway.
- **R17** — time exits record held = MAX_HOLD+1 (decision at held ≥ 30 counted from the fill, fill next bar) — parameter unchanged, follows "holding period starts at the fill" — cost if wrong: one extra day of borrow per time exit.
- **R18** — fix T1 minors 1–4 now by amending the T1 commit: commit a default-mode all-cases test (fill = next both-legs row, end_of_data ⇒ exit==exit_signal, no overlap, PnL rebuilt from fill-date spreads, daily sum == pnl sum), an end-cap test, and the after_oos speed-up — the default path is what T2–T4 modify — cost if wrong: ~30 min.

## T4 (drafted early)

- **R19** — (T4 draft) capital = n_pairs × 2.0 USD per $1 leg-1 notional; Sharpe invariant to it.

## T2

- **R20** — fixed async_lag4.5_hol0 strict-xfail attributed to T3 accepted; if T3 does not bring it inside the bounds it is a defect to investigate, NOT another xfail.
- **R21** — fix T2 minors 2–6 now by amending the T2 commit (registry/policy robustness, FX-index asserts, one shared previous-print helper, Exchange attributes instead of name strings, cosmetics).
- **R22** — keep requires_fx=True for v2 Canadian rows (the pair needs FX; fx_unaligned records it is unavailable).

## T5

- **R23** — T5 = code path + synthetic test unless owner provides constituents

## T6

- **R24** — T6 back-fill counts every data/wfo_daily_results_<tag>.csv as a trial (47) + run_*.sh + flags + FULL_DAILY_RUN rows, conservative larger N

## T8

- **R25** — T8 default model = whichever gives LOWER dev Sharpe

## T9

- **R26** — T9 HMM gate shifted one session unless proxy provably printed by 16:00

## T11

- **R27** — T11 without PIT universe → RESULTS_v2.md states headline cannot be produced, diagnostic labelled upper bound

## T3

- **R28** — Kalman path — entry z (dynamic-β spread) and exit z (fixed β_e spread) disagree → mostly 1-day round trips; fix in T9.2: entry z computed on the spread definition the trade will carry (β_t applied to the trailing window) so entry and exit z agree on the entry row. → T9 brief.
- **R29** — CUSUM force-close only for breaks inside the window's OOS (implementer's reading, matches legacy "training break = block only"); verify in T9 cusum truncation test.
- **R30** — end_of_data in VWAP mode fills at the next available CLOSE (plan's literal text) — accepted.
- **R31** — fix now (amend T3): share the entry-gate chain and the target/stop/time exit rule between both engines (T7/T8 will edit them — drift risk), window_ctx → NamedTuple, window_exited label for a pair whose data ends early = window containing its last decision row, README martingale/convexity note. Module split optional. Outputs must stay byte-identical.

## T4

- **R32** — fix the dashboard properly, not by deleting labels: the CLI writes per-pair daily MTM (dates × pairs, same span/index as the portfolio file) next to the portfolio MTM file; dashboard per-pair ranking and sub-block series/table use it (block Sharpe = block daily PnL on the one definition; block capital = n_pairs_block × 2.0); missing file → no Sharpe figures, only trades/PnL. This also feeds T11's per-block table.
- **R33** — (accepted concerns) downstream reads pnl (Sharpe/t/R² invariant; T10 switches factor regression to ret); mtm.py module; legacy_spread_pnl requires legacy_exit_date_pnl; default stats drop sharpe_full/sharpe_exit; borrow per unit per held row until T8.

## T5

- **R34** — accept the stricter HEADLINE rule (no --legacy-*; filename date == asof column).
- **R35** — fix now (amend T5): every row must carry the one asof (no dropna); empty `start` raises; builder refuses tickers containing "-" with a clear message (CLI splits pair names on "-"; owner decides a ticker mapping when the dataset arrives) — conservative, logged as known limitation; headline marker also requires the builder's columns; commit the builder→CLI headline test; rename T2's "HEADLINE UNIVERSE:" header line to "HEADLINE UNIVERSE POLICY:"; comment tying johansen_trace to johansen_coint's parameters.

## T6

- **R36** — N = number of distinct config_hash (incl. this run); V = one sr_daily per hash (latest row); hash covers everything that changes the run (WfoConfig minus final_holdout, run_pairs, pairs-file SHA, closes name, volumes presence); hash excludes code revision (documented: same config under different engine versions = one trial). Back-fill hash = sha256(source + "\n" + description) → 107 distinct.
- **R37** — legacy runs (any legacy_* flag) are registered (they are evaluations; conservative larger N) with sr_daily EMPTY (not in V; non-MTM Sharpes never enter the MTM column); TRIALS.csv/HOLDOUT_USES.md excluded from the +dirty check; DSR printed with .3g; holdout test parametrised over hmm and almgren-with-volumes.

## T8

- **R38** — fix it (conservative): leg-2 almgren participation uses the leg's own notional |β_e|·N at the entry decision (Kalman: β at entry); exit uses the same per-leg rate on exit notional (second-order); legacy_flat_cost unchanged. Re-run the almgren real run ONCE (new TRIALS row) and record before/after. Minors: summary label "NET 5.0 bps/leg" → cost-model label (keep old label in legacy-flat-cost runs); log note: mg_harvest would get zero costs under T8 (dormant behind R7); note CSV round-trip precision for cost == cost_entry + cost_exit.

## T9

- **R39** — register both diagnostic variants now (V2 prior+dynamic-z is causal → sr_daily; V1 posterior+R28 → LEGACY blank Sharpe); log note that the DSRs stored on the two T9 kalman/hmm rows used N two short (rows are not edited).
- **R40** — eligibility must count only rows < oos_s (causal, conservative): require ≥ Z_WIN+10 both-legs rows in [tr_s, oos_s) in the continuous engine (legacy engine unchanged); re-run the default flat real run once; if the trades change, record before/after; if identical, record that.
- **R41** — --kalman with --hmm in the continuous engine → ValueError (the original code also ignored the HMM gate on the kalman path; make it explicit, not silent); move the HEADLINE_FORBIDDEN check above the holdout recording; fix pair_momentum comment; drop the wasted rolling z if trivial.

## T10

- **R42** — the real-data FDR screen (data ≤ 2014-12-31, no PnL, hindsight list labelled NON-HEADLINE) evaluates no strategy → not a TRIALS row.
- **R43** — compare_results/edge_monitor accept only daily WFO runs with a meta sidecar (fail closed); neither is called by live/; a live-ledger use is an owner follow-up (report it).
- **R44** — meta sidecar records SHA-256 of trades/mtm/mtm_pairs/closes; check_run verifies when present, loud "no file hashes" line when absent (pre-T10 sidecars) — no re-run of old runs; test with a Σ-preserving edit → SystemExit.
- **R45** — relabel every legacy exit-date ratio in the CLI/summarise as "LEGACY exit-date ratio (full|exit days) — not a Sharpe" (no stdout identity test pins them); same for the 4 out-of-scope daily-path scripts (label, and cap almgren_chriss_repricing's span at the trades' run span / ≤ 2023-12-31); step4c intraday left for Phase 2 (note).
- **R46** — refuse --split/--start inside the run's MTM span (fail closed) in the five downstream scripts; add the stress positive-branch test and the closes-mismatch test; FDR warns when a non-PIT cut is after 2014-12-31 (dev OOS years); dedupe the Sharpe wrapper.
- **R47** — (controller, before re-review) build_production_universe.py (production path; live/daily_signal.py reads its output) must accept BOTH the old `sharpe` and the new `legacy_exit_date_ratio` per-pair column (an existing baseline summary must not break the owner's universe rebuild); output key `baseline_sharpe` unchanged (renaming is an owner decision); cap audit_baseline.py and hmm_modes_proper.py calendars at the trades' span / ≤ 2023-12-31 like almgren.

## T11

- **R48** — T11 is doc-only; its task review is folded into the final whole-branch review (brief item 7 checks RESULTS_v2.md number by number) — cost if wrong: one review pass less on a doc.

## final review

- **R49** — HEADLINE marker refuses cost parameters below defaults and any non-default execution model (R25; T11 ruling 1) — cost if wrong: owner must explicitly change the marker for documented lower costs / almgren headline.
- **R50** — cut/different-future tests for --maxconc, --ticker-overlap, --per-pair-quarter-cap, --vol-tight-stop, --vwz, --exec vwap.
- **R51** — --exec vwap entry decision independent of row i+1 VWAP availability; fill falls back to next close; re-run vwap 20-seed null before→after (stop if outside SE).
- **R52** — "Sharpe" → "per-trade ratio" in sizing-input labels/help.
- **R53** — docs M3/M4/M6/M7/M8; derived_stats committed as tools/results_v2_derived_stats.py.
- **R54** — (advisor) final-review round = two new commits (A code/docs, B TRIALS row + re-run numbers), never amend after the real run (sidecar must name a reachable commit); deferred minors per reviewer verdicts accepted/owner list.
- **R55** — strengthen the per-flag cut/different-future test with several cut dates spread over the panel (all 15 flags), proven by the four scratch mutations now failing; test-only commit C `fix(review): Stronger per-flag truncation tests` (no code change → B's re-run still valid for HEAD) — cost if wrong: ~+20 s suite time. (As committed: the four-date test found R56, so commit C `339cf48` also carries that engine change in three non-default filters; the default was re-run on C and recorded by `5f02f19`; suite +5 s.)
- **R56** — post-hoc filters (--ticker-overlap, --per-pair-quarter-cap, --maxconc) take same-day entries in (entry, pair) stable order (`entry_order`); legacy_sizing_lookahead keeps the old quicksort order (fixtures unchanged) — cost if wrong: none for the default path (filters off); a different same-day trade kept under the three options.
- **R57** — controller finishes the stopped implementer's work itself (small, verified diff) and records it as commit C; the default re-run on clean C is recorded by commit D (R54 pattern); then one scoped re-review of 3886de9..D on a non-Opus model (the Opus subagent quota is exhausted).
