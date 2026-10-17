"""Remaining look-ahead (audit task T9); see tests/README.md.

T9.1  Every per-window training computation of the default engine — β,
      eligibility, the cost rates (almgren σ and ADV), the ivol weights, the
      macro-HMM fit — reads rows strictly before the window's OOS start:
      prices changed from oos_s(w) on leave window w's training outputs
      unchanged.  legacy_window_reset keeps the overlap (train ends on oos_s).
T9.2  Kalman: β_{t|t−1}, the state before the update with y_t
      (`kalman_hedge(return_prior=True)`), builds the daily spread and sizes
      the trade; the entry z is the z of the spread the trade carries
      (Ruling R28); legacy_kalman_posterior restores the posterior path.
T9.3  Macro HMM: a causal forward filter (never hmmlearn's predict /
      predict_proba / decode / score_samples, which decode or smooth the whole
      window) on an EW proxy of US closes only; legacy_hmm_viterbi restores
      the Viterbi path.
T9.4  Every optional flag of the plan's list (and, final review R50, the six
      options a headline run may use: --maxconc, --ticker-overlap,
      --per-pair-quarter-cap, --vol-tight-stop, --vwz, --exec vwap): its
      decisions before a cut do not depend on the data after it, on the data
      (closes, volumes, VWAPs) cut at D and on the data with a different
      future after D.  --volvol, --pairmom and --voltarget
      size from data at the entry decision only (legacy_sizing_lookahead
      restores the pre-T9 sizing); --metagate / --hrp are refused in a
      headline run; VWAP fills on the --kalman / --vwz paths are refused.
"""
import contextlib
import functools
import io
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kalman  # noqa: E402
import step3j_wfo_daily as cli  # noqa: E402
import synthetic  # noqa: E402
import wfo_daily_core as core  # noqa: E402
from build_universe_pit import COLUMNS as PIT_COLUMNS  # noqa: E402
from cross_listing import mask_stale_prints  # noqa: E402
from test_wfo_core_identity import as_cli_reads, cli_config, make_golden  # noqa: E402
from wfo_daily_core import ENTRY_Z, STOP_Z, WfoConfig, make_windows, run_wfo_daily  # noqa: E402

ENTRY_COLS = ["pair", "window_entered", "entry_signal", "entry", "side", "size", "beta",
              "p1_entry", "p2_entry"]
DECISION_ENTRY_COLS = ENTRY_COLS[:7]
DECISION_EXIT_COLS = ["pair", "entry_signal", "exit_signal", "exit", "reason"]


# ── Runs on the golden cases' inputs ─────────────────────────────────────────

def case_cfg(case: str, **overrides) -> WfoConfig:
    """The WfoConfig the CLI builds for a golden case (the corrected defaults)."""
    return replace(cli_config(make_golden.case_inputs(case)["args"]), **overrides)


def run_inputs(inp: dict, cfg: WfoConfig, closes: pd.DataFrame | None = None,
               transform=None) -> core.WfoResult:
    """run_wfo_daily on a golden case's inputs as the CLI reads them (`closes`
    replaces the panel; `transform` — a cut or a different future — is applied
    to the closes, the volumes and the VWAPs alike)."""
    frames = [as_cli_reads(inp["closes"]) if closes is None else closes,
              as_cli_reads(inp["volumes"]), as_cli_reads(inp["vwap"])]
    if transform is not None:
        frames = [None if f is None else transform(f) for f in frames]
    px, volumes, vwap = frames
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(px, inp["pairs"], cfg, volumes=volumes, vwap=vwap)


def future_changed(closes: pd.DataFrame, d) -> pd.DataFrame:
    """`closes` with a different future: after d every log return from the last
    print on or before d is multiplied by −3 (reversed, three times as
    volatile); prices on or before d untouched, the NaN pattern kept."""
    ref = np.log(closes.loc[:d].ffill().iloc[-1])
    out = closes.copy()
    after = out.index > d
    out.loc[after] = np.exp(ref - 3 * (np.log(closes.loc[after]) - ref))
    return out


def cut_date(closes: pd.DataFrame) -> pd.Timestamp:
    """D: 30 rows after the panel's middle date — on the golden panels
    2008-02-13, six weeks into window 11's OOS (2008-01-03 → 2008-04-03)."""
    return closes.index[len(closes) // 2 + 30]


@functools.lru_cache(maxsize=None)
def base_run(panel: str) -> pd.DataFrame:
    """Trades of a golden panel's run without options (the corrected defaults)."""
    inp = make_golden.inputs({"args": [], "panel": panel})
    return run_inputs(inp, cli_config(inp["args"])).trades


# Cut dates D of the per-flag tests (final review R55; the T9 / R50 tests had
# the one date cut_date() = 2008-02-13): in the OOS of windows 2, 6, 13 and
# 16 of the golden panels.  A look-ahead of a few rows only shows in the
# decisions just before a D, so the dates are spread over the panel; at these
# four, every look-ahead mutation of the R55 check is caught (AUDIT_FIX_LOG,
# "Final review fixes").
CUT_DATES = tuple(pd.Timestamp(d) for d in ("2005-10-03", "2006-12-25", "2008-07-07",
                                             "2009-06-08"))


@functools.lru_cache(maxsize=None)
def flag_runs(case: str, overrides: tuple = ()) -> dict:
    """A golden case run on its full panel and, for each D of CUT_DATES, on
    the panel cut at D (inclusive) and on the panel with a different future
    after D (closes, volumes and VWAPs alike, R50), with C = min(D, the cut
    run's last OOS end) as in test_placebo's truncation test; and its panel's
    run without the case's options."""
    inp = make_golden.case_inputs(case)
    cfg = case_cfg(case, **dict(overrides))
    closes = as_cli_reads(inp["closes"])
    cuts = []
    for d in CUT_DATES:
        last_oos_e = make_windows(closes.loc[:d], cfg.train, cfg.oos, cfg.step,
                                  start=cfg.start, end=cfg.end)[-1][3]
        cuts.append({"d": d, "c": min(d, last_oos_e),
                     "cut": run_inputs(inp, cfg, transform=lambda f, d=d: f.loc[:d]).trades,
                     "future": run_inputs(inp, cfg,
                                          transform=lambda f, d=d: future_changed(f, d)).trades})
    return {"full": run_inputs(inp, cfg).trades, "cuts": cuts,
            "base": base_run(make_golden.CASES[case].get("panel", "base"))}


def filled_before(tr: pd.DataFrame, c) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, exits) decided and filled before c: an entry's decision,
    fill, side, size, β and entry prices; an exit in all columns.  (R55: a
    fill on or after c — the cut run's last rows — can meet the cut run's
    end-of-run closes there, e.g. a --maxconc slot freed by one; the decisions
    on those rows are compared by the different-future test, `decided_by`.)"""
    ent = tr[tr["entry"] < c].sort_values(["pair", "entry_signal"])[ENTRY_COLS]
    ex = tr[(tr["exit"] < c) & (tr["reason"] != "end_of_data")]
    return ent.reset_index(drop=True), ex.sort_values(["pair", "exit_signal"]).reset_index(drop=True)


def filled_by(tr: pd.DataFrame, d) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, exits) filled on or before d (a fill after d is at a changed price)."""
    ent = tr[tr["entry"] <= d].sort_values(["pair", "entry_signal"])[ENTRY_COLS]
    ex = tr[tr["exit"] <= d]
    return ent.reset_index(drop=True), ex.sort_values(["pair", "exit_signal"]).reset_index(drop=True)


def decided_by(tr: pd.DataFrame, d) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, exits) decided on or before d, in their decision columns
    (R55): an entry's pair, window, decision and fill dates, side, size and
    β; an exit's pair, entry decision, decision and fill dates and reason —
    also when the fill is after d (at a changed price)."""
    ent = tr[tr["entry_signal"] <= d].sort_values(["pair", "entry_signal"])[DECISION_ENTRY_COLS]
    ex = tr[tr["exit_signal"] <= d].sort_values(["pair", "exit_signal"])[DECISION_EXIT_COLS]
    return ent.reset_index(drop=True), ex.reset_index(drop=True)


def assert_same(a: tuple, b: tuple, what: str = "trades") -> None:
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y, check_exact=True, obj=what)


def same(a: tuple, b: tuple) -> bool:
    try:
        assert_same(a, b)
    except AssertionError:
        return False
    return True


# ── T9.4: every optional flag, truncation invariance ─────────────────────────

# Flag of the plan's list (+ the T9.2 / T9.3 paths, + the six options a
# headline run may use, final review R50) → the golden case that runs it on a
# panel where it binds (make_golden: every case changes its panel's default
# run; here: before the cut).
FLAG_CASES = {
    "--ivol": "ivol", "--volvol": "volvol", "--pairmom": "pairmom", "--rotation": "rotation",
    "--voltarget": "voltarget", "--cusum": "cusum_30", "--aussie-filter": "aussie_filter",
    "--kalman": "kalman", "--hmm block-panic": "hmm_block_panic",
    "--maxconc": "maxconc", "--ticker-overlap": "ticker_overlap",
    "--per-pair-quarter-cap": "per_pair_quarter_cap", "--vol-tight-stop": "vol_tight_stop",
    "--vwz": "vwz", "--exec vwap": "vwap_exec",
}


@pytest.mark.parametrize("flag", list(FLAG_CASES))
def test_flag_decisions_before_a_cut_are_those_of_the_full_data(flag):
    # The plan's truncation test (test_placebo's pattern), at every D of
    # CUT_DATES: the run on the panel cut at D and on the full panel agree on
    # every entry and exit decided and filled before C = min(D, the cut run's
    # last OOS end).
    r = flag_runs(FLAG_CASES[flag])
    for k in r["cuts"]:
        assert_same(filled_before(k["cut"], k["c"]), filled_before(r["full"], k["c"]),
                    f"{flag}, cut at {k['d'].date()}")
    c = r["cuts"][-1]["c"]                                 # the largest compared region
    full = filled_before(r["full"], c)
    assert len(full[0]) > 20 and len(full[1]) > 20
    assert not same(full, filled_before(r["base"], c)), f"{flag} does not bind before the cut"
    if flag == "--cusum":
        # Ruling R29 (T3): a CUSUM break closes an open trade only inside its
        # window's OOS; a break on the training rows of a later window would
        # close it in the full run but not in the cut run (no such window).
        assert (full[1]["reason"] == "cusum_break").any()


@pytest.mark.parametrize("flag", list(FLAG_CASES))
def test_flag_decisions_before_a_date_do_not_depend_on_a_different_future(flag):
    # Stronger than the cut: the windows stay, so a look-ahead inside a window
    # (a fit, a decoding or a normalisation over the whole window or run) is
    # seen.  At every D of CUT_DATES, whatever the data after D: every entry
    # and exit filled on or before D is identical, and every decision made on
    # or before D (R55: also one filled after D) is the same decision.
    r = flag_runs(FLAG_CASES[flag])
    for k in r["cuts"]:
        d, what = k["d"], f"{flag}, future changed after {k['d'].date()}"
        assert_same(filled_by(k["future"], d), filled_by(r["full"], d), what)
        assert_same(decided_by(k["future"], d), decided_by(r["full"], d), what)
    d = r["cuts"][-1]["d"]
    full = filled_by(r["full"], d)
    assert len(full[0]) > 20 and len(full[1]) > 20
    assert not same(full, filled_by(r["base"], d)), f"{flag} does not bind before D"
    if flag == "--cusum":
        # R29 again: a later window's break on its training rows (before D)
        # comes from a z whose β is fitted on rows after D.
        assert (full[1]["reason"] == "cusum_break").any()


def test_legacy_volvol_normalisation_looks_ahead():
    # The pre-T9 --volvol divides by the mean multiplier of ALL trades, later
    # ones included: the cut changes earlier sizes (legacy_sizing_lookahead).
    r = flag_runs("volvol", (("legacy_sizing_lookahead", True),))
    assert not all(same(filled_before(k["cut"], k["c"]), filled_before(r["full"], k["c"]))
                   for k in r["cuts"])


def wfo_dir(tmp_path, monkeypatch) -> list[str]:
    """A CLI run dir: data/closes.csv (3 years of 2 OU pairs from 2005-01-03),
    a frozen point-in-time universe universe_pit_2004-12-31.csv and the same
    pairs as a manual list pairs.csv; the trials registry in tmp_path."""
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=0)
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    closes.to_csv(tmp_path / "data" / "closes.csv")
    legs = [p.split("-") for p in pairs]
    frozen = pd.DataFrame({"pair": pairs, "leg1": [a for a, _ in legs], "leg2": [b for _, b in legs],
                           "sector": "S", "corr": 0.5, "beta": 1.0, "hl_days": 5.0,
                           "johansen_stat": 50.0, "asof": "2004-12-31"}, columns=PIT_COLUMNS)
    frozen.to_csv(tmp_path / "data" / "universe_pit_2004-12-31.csv", index=False)
    frozen[["pair"]].to_csv(tmp_path / "data" / "pairs.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TRIALS_FILE", tmp_path / "TRIALS.csv")
    return pairs


def cli_args(monkeypatch, pairs_file: str, *extra) -> None:
    monkeypatch.setattr(sys, "argv", ["step3j_wfo_daily.py", "--pairs", pairs_file, "--closes",
                                      "closes.csv", "--execution-model", "flat", "--tag", "t", *extra])


@pytest.mark.parametrize("flag", ["--metagate", "--hrp"])
def test_flags_without_a_truncation_test_are_refused_in_a_headline_run(flag, tmp_path, monkeypatch):
    # --metagate / --hrp crash before their first window (Ruling R7), so no
    # truncation test can run them: a run that would be HEADLINE stops before
    # anything runs or is registered; a non-headline run names them in its
    # banner (and then hits R7).
    wfo_dir(tmp_path, monkeypatch)
    cli_args(monkeypatch, "universe_pit_2004-12-31.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main().stats["headline"] is True               # the same run without the flag
    rows = len(pd.read_csv(tmp_path / "TRIALS.csv"))
    cli_args(monkeypatch, "universe_pit_2004-12-31.csv", flag)
    with pytest.raises(SystemExit, match=f"{flag}.*headline"):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main()
    assert len(pd.read_csv(tmp_path / "TRIALS.csv")) == rows
    # Refused before a --final-holdout use is recorded (R41).
    monkeypatch.setattr(cli, "HOLDOUT_USES_FILE", tmp_path / "HOLDOUT_USES.md")
    cli_args(monkeypatch, "universe_pit_2004-12-31.csv", flag, "--final-holdout")
    with pytest.raises(SystemExit, match=f"{flag}.*headline"):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main()
    assert not (tmp_path / "HOLDOUT_USES.md").exists()
    assert len(pd.read_csv(tmp_path / "TRIALS.csv")) == rows
    cli_args(monkeypatch, "pairs.csv", flag)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(UnboundLocalError):
        cli.main()
    banner = [line for line in out.getvalue().splitlines() if line.startswith("NON-HEADLINE RUN")]
    assert any(flag in line and "truncation" in line for line in banner)


@pytest.mark.parametrize("flags", [{"kalman": True}, {"vwz": True}], ids=["kalman", "vwz"])
@pytest.mark.parametrize("legacy", [False, True], ids=["default", "legacy"])
def test_vwap_fills_are_refused_on_the_kalman_and_vwz_paths(flags, legacy):
    # Only the static-β path builds VWAP fills; elsewhere they would be next
    # closes under a "next-day VWAP" header.
    inp = make_golden.case_inputs("vwap_exec")
    cfg = replace(case_cfg("vwap_exec"), **flags, legacy_window_reset=legacy)
    with pytest.raises(ValueError, match="vwap"):
        run_inputs(inp, cfg)


def test_vwap_entry_decisions_do_not_read_whether_the_next_row_has_a_vwap():
    # Final review R51: with --exec vwap the entry decided on row i used to
    # require a VWAP on row i+1 (a next-row read).  Now no decision depends on
    # the VWAPs: removing VWAP cells — one entry's fill row, or 15% at random —
    # leaves every decision and fill date unchanged, and a leg without a VWAP
    # on its fill row fills at its close (an end-of-data exit at the close).
    inp = make_golden.case_inputs("vwap_exec")
    cfg = case_cfg("vwap_exec")
    closes, vwap = as_cli_reads(inp["closes"]), as_cli_reads(inp["vwap"])
    decisions = ["pair", "window_entered", "entry_signal", "entry", "side", "size", "beta",
                 "exit_signal", "exit", "reason"]

    def run(vw: pd.DataFrame) -> pd.DataFrame:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_wfo_daily(closes, inp["pairs"], cfg, volumes=as_cli_reads(inp["volumes"]),
                                 vwap=vw).trades

    full = run(vwap)
    first = full.iloc[0]
    legs = first["pair"].split("-")
    one_row = vwap.copy()
    one_row.loc[first["entry"], legs] = np.nan            # only the fill row of one entry
    random_holes = vwap.mask(np.random.default_rng(3).random(vwap.shape) < 0.15)
    for vw in (one_row, random_holes):
        tr = run(vw)
        pd.testing.assert_frame_equal(tr[decisions], full[decisions], check_exact=True)
        no_vwap = 0                                       # fills at the close for lack of a VWAP
        for _, t in tr.iterrows():
            for leg, col in zip(t["pair"].split("-"), ("p1", "p2")):
                for when in ("entry", "exit"):
                    day, px = t[when], vw.at[t[when], leg]
                    no_vwap += bool(np.isnan(px))
                    if np.isnan(px) or (when == "exit" and t["reason"] == "end_of_data"):
                        px = closes.at[day, leg]
                    assert t[f"{col}_{when}"] == px
        assert no_vwap >= (2 if vw is one_row else 50)
    t = run(one_row).iloc[0]
    assert (t["entry_signal"], t["p1_entry"], t["p2_entry"]) == (
        first["entry_signal"], closes.at[first["entry"], legs[0]], closes.at[first["entry"], legs[1]])


def test_kalman_with_the_hmm_gate_is_refused_in_the_continuous_engine():
    # R41: the Kalman path carries no macro-HMM gate (neither did the pre-audit
    # one): the combination would ignore --hmm, so the default engine refuses
    # it; the legacy per-window engine keeps its pre-audit behaviour.
    inp = make_golden.case_inputs("hmm_block_panic")
    for mode in ("block-panic", "size-panic"):
        with pytest.raises(ValueError, match="Kalman path has no macro-HMM gate"):
            run_inputs(inp, case_cfg("hmm_block_panic", hmm=mode, kalman=True))
    assert not run_inputs(inp, case_cfg("hmm_block_panic", kalman=True,
                                        legacy_window_reset=True)).trades.empty


# ── T9.1: per-window training rows end strictly before oos_s ─────────────────

def training_outputs(monkeypatch, inp: dict, cfg: WfoConfig, closes: pd.DataFrame) -> dict:
    """What the window loop learns from each window's training rows: per pair
    (in call order) the PairWindow of each window (β, eligibility, cost rates),
    the WindowContext of each window (ivol weights), the HMM parameters fitted
    for each window."""
    out = {"pw": [], "ctx": None, "hmm": []}
    real_engine, real_ct, real_fit = core.backtest_pair_continuous, core.continuous_trades, core.fit_macro_hmm

    def engine(idx, leg_vals, n_decide, windows, pair_windows, **kw):
        out["pw"].append(pair_windows)
        return real_engine(idx, leg_vals, n_decide, windows, pair_windows, **kw)

    def ct(closes, pairs, cfg, windows, window_ctx, *a, **kw):
        out["ctx"] = window_ctx
        return real_ct(closes, pairs, cfg, windows, window_ctx, *a, **kw)

    def fit(x, n_components=2, random_state=0):
        m, panic = real_fit(x, n_components=n_components, random_state=random_state)
        out["hmm"].append(None if m is None else
                          (m.startprob_.copy(), m.transmat_.copy(), m.means_.copy(), m.covars_.copy()))
        return m, panic

    with monkeypatch.context() as mp:
        mp.setattr(core, "backtest_pair_continuous", engine)
        mp.setattr(core, "continuous_trades", ct)
        mp.setattr(core, "fit_macro_hmm", fit)
        run_inputs(inp, cfg, closes)
    return out


def test_training_computations_read_only_rows_before_the_oos_start(monkeypatch):
    # Window 6 (OOS from Tuesday 2006-10-03, a trading day of the panel): every
    # price from oos_s on is changed; window 6's β, eligibility, cost rates
    # (almgren: σ of the training rows, ADV at the training end), ivol weight
    # and HMM fit stay exactly the same.
    inp = make_golden.case_inputs("hmm_block_panic")
    cfg = case_cfg("hmm_block_panic", ivol=True)
    closes = as_cli_reads(inp["closes"])
    w = 6
    oos_s = make_windows(closes, cfg.train, cfg.oos, cfg.step)[w - 1][2]
    assert oos_s == pd.Timestamp("2006-10-03") and oos_s in closes.index
    rng = np.random.default_rng(5)
    changed = closes.copy()
    changed.loc[oos_s:] *= np.exp(rng.normal(0.0, 0.02, changed.loc[oos_s:].shape))
    a = training_outputs(monkeypatch, inp, cfg, closes)
    b = training_outputs(monkeypatch, inp, cfg, changed)
    assert cfg.execution_model == "almgren" and inp["volumes"] is not None
    pws = [(p.get(w), q.get(w)) for p, q in zip(a["pw"], b["pw"], strict=True)]
    assert sum(p is not None for p, _ in pws) >= 5
    for p, q in pws:
        assert (p is None) == (q is None)                          # eligibility
        if p is not None:
            assert (p.beta, p.leg_costs) == (q.beta, q.leg_costs)
    assert a["ctx"][w - 1].weights and a["ctx"][w - 1].weights == b["ctx"][w - 1].weights
    for x, y in zip(a["hmm"][w - 1], b["hmm"][w - 1], strict=True):
        np.testing.assert_array_equal(x, y)
    # The change reaches window 6's decisions (the test is not vacuous).
    assert not a["ctx"][w - 1].panic.loc[oos_s:].equals(b["ctx"][w - 1].panic.loc[oos_s:])
    # The legacy per-window engine keeps its training slice [tr_s, oos_s] (T3):
    # its HMM fit sees the changed oos_s row.
    legacy = replace(cfg, legacy_window_reset=True)
    a = training_outputs(monkeypatch, inp, legacy, closes)
    b = training_outputs(monkeypatch, inp, legacy, changed)
    assert not all(np.array_equal(x, y) for x, y in zip(a["hmm"][w - 1], b["hmm"][w - 1]))


def test_window_eligibility_reads_no_print_after_the_oos_start(monkeypatch):
    # R40: window 1 (OOS from 2005-07-03) of a pair whose leg OUB0 starts
    # trading 65 rows before oos_s, and of one with 75 rows (OUB1).  Whether
    # the window is eligible must not depend on prints after oos_s: with the
    # legs delisted 2 rows into the OOS, eligibility is the same.  The rule:
    # ≥ max(60, Z_WIN + 10) = 70 training rows before oos_s — 65 rows: no,
    # 75 rows: yes.  (Before R40, Z_WIN + 10 rows up to oos_e: 65 + 2 rows
    # ineligible, 65 + the OOS rows eligible.)
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=0)
    cfg = WfoConfig(execution_model="flat")
    oos_s = make_windows(closes, cfg.train, cfg.oos, cfg.step)[0][2]
    before = closes.index[closes.index < oos_s]
    listed = closes.copy()
    listed.loc[:before[-65] - pd.Timedelta(days=1), "OUB0"] = np.nan
    listed.loc[:before[-75] - pd.Timedelta(days=1), "OUB1"] = np.nan
    assert [listed.loc[:oos_s, c].notna().sum() for c in ("OUB0", "OUB1")] == [65, 75]
    delisted = listed.copy()
    delisted.loc[closes.index[closes.index > oos_s][2]:, ["OUB0", "OUB1"]] = np.nan
    inp = {"closes": None, "pairs": pairs, "volumes": None, "vwap": None}
    eligible = {}
    for name, px in (("listed", listed), ("delisted", delisted)):
        pws = training_outputs(monkeypatch, inp, cfg, px)["pw"]
        eligible[name] = [1 in pw for pw in pws]
    assert eligible["listed"] == eligible["delisted"] == [False, True]


# ── T9.2: Kalman, the one-step prior ─────────────────────────────────────────

def log_pair(n: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log2 = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    return 0.4 + 1.3 * log2 + rng.normal(0.0, 0.01, n), log2


def test_kalman_prior_is_the_state_before_the_update_with_y_t():
    p1, p2 = log_pair()
    old = kalman.kalman_hedge(p1, p2, delta=1e-6)
    *new, prior = kalman.kalman_hedge(p1, p2, delta=1e-6, return_prior=True)
    assert len(old) == 4                                     # other callers' outputs unchanged
    for x, y in zip(old, new, strict=True):
        np.testing.assert_array_equal(x, y)
    beta = old[1]
    assert prior[0] == 1.0 and np.array_equal(prior[1:], beta[:-1])   # β_{t|t−1} = β_{t−1|t−1}
    t = 250
    q1 = p1.copy()
    q1[t] += 0.05                                            # a different y_t
    _, beta_q, _, _, prior_q = kalman.kalman_hedge(q1, p2, delta=1e-6, return_prior=True)
    assert np.array_equal(prior_q[:t + 1], prior[:t + 1])    # β used at t: unchanged
    assert prior_q[t + 1] != prior[t + 1] and beta_q[t] != beta[t]   # the posterior sees y_t


def test_daily_kalman_spread_uses_the_prior_beta():
    p1, p2 = log_pair()
    df = pd.DataFrame({"A": np.exp(p1), "B": np.exp(p2)}, index=pd.bdate_range("2010-01-01", periods=len(p1)))
    spread, z, beta = core.kalman_spread_z(df, "A", "B", prior=True)
    *_, prior = kalman.kalman_hedge(np.log(df["A"].values), np.log(df["B"].values), delta=1e-6,
                                     beta_init=1.0, return_prior=True)
    np.testing.assert_array_equal(beta.to_numpy(), prior)
    np.testing.assert_array_equal(spread.to_numpy(), np.log(df["A"].values) - prior * np.log(df["B"].values))
    # y_t enters the spread at t only through its direct term.
    t = 250
    bumped = df.copy()
    bumped.iloc[t, 0] *= 1.05
    spread_b, _, beta_b = core.kalman_spread_z(bumped, "A", "B", prior=True)
    assert beta_b.iloc[t] == beta.iloc[t] and spread_b.iloc[:t].equals(spread.iloc[:t])
    assert spread_b.iloc[t] - spread.iloc[t] == pytest.approx(np.log(1.05), abs=1e-12)
    # legacy: the posterior (updated with y_t), as before T9.
    _, _, posterior = core.kalman_spread_z(df, "A", "B")
    _, beta_post, _, _ = kalman.kalman_hedge(np.log(df["A"].values), np.log(df["B"].values), delta=1e-6)
    np.testing.assert_array_equal(posterior.to_numpy(), beta_post)
    _, _, posterior_b = core.kalman_spread_z(bumped, "A", "B")
    assert posterior_b.iloc[t] != posterior.iloc[t]


def test_window_z_applies_one_beta_to_the_whole_trailing_window():
    p1, p2 = log_pair(200)
    beta = np.linspace(1.0, 1.6, 200)
    z = core.window_z(p1, p2, beta)
    assert np.isnan(z[:core.Z_WIN - 1]).all()
    for t in (59, 100, 199):
        s = p1[t - 59:t + 1] - beta[t] * p2[t - 59:t + 1]
        assert z[t] == pytest.approx((s[-1] - s.mean()) / s.std(ddof=1), abs=1e-12)
        # the z rolling_z gives on row t for the spread a trade entered with β_t carries
        ref = core.rolling_z(pd.Series(p1 - beta[t] * p2)).iloc[t]
        assert z[t] == pytest.approx(ref, abs=1e-9)
        assert core.window_z(p1, p2, beta[t])[t] == z[t]          # bit for bit


def pair_rows(closes: pd.DataFrame, volumes: pd.DataFrame, pair: str, first, last) -> pd.DataFrame:
    """The continuous engine's decision rows of a pair: both legs print (after
    the stale-print mask) from the first training start to the last OOS end."""
    masked = mask_stale_prints(closes, volumes)
    return masked.loc[first:last, pair.split("-")].dropna()


@pytest.mark.parametrize("legacy", [False, True], ids=["prior", "legacy_posterior"])
def test_kalman_trade_uses_the_prior_beta_and_the_z_of_its_own_spread(legacy):
    # β_e = β_{t|t−1} of the entry decision row t; the entry z on row t is the
    # rolling z of log p1 − β_e·log p2 over the last Z_WIN rows — the z the
    # trade then exits on, so entry and exit z agree on the entry row (R28).
    # legacy_kalman_posterior: β_e = the posterior β_t, the entry z on the
    # dynamic-β spread (as before T9).
    inp = make_golden.case_inputs("kalman")
    cfg = case_cfg("kalman", legacy_kalman_posterior=legacy)
    tr = run_inputs(inp, cfg).trades
    closes, volumes = as_cli_reads(inp["closes"]), as_cli_reads(inp["volumes"])
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step)
    agree = 0
    for pair, g in tr.groupby("pair"):
        px = pair_rows(closes, volumes, pair, windows[0][0], windows[-1][3])
        l1, l2 = np.log(px.iloc[:, 0].to_numpy()), np.log(px.iloc[:, 1].to_numpy())
        _, post, _, _, prior = kalman.kalman_hedge(l1, l2, delta=1e-6, return_prior=True)
        rows = px.index.get_indexer(g["entry_signal"])
        assert (rows >= 0).all()
        np.testing.assert_array_equal(g["beta"].to_numpy(), (post if legacy else prior)[rows])
        for row, b in zip(rows, g["beta"]):
            entry_z, exit_z = core.window_z(l1, l2, prior)[row], core.window_z(l1, l2, b)[row]
            if not legacy:
                assert entry_z == exit_z and ENTRY_Z < abs(entry_z) < STOP_Z
            agree += entry_z == exit_z
    assert len(tr) > 100 and (agree == len(tr)) != legacy


# ── T9.3: macro HMM, a causal forward filter ─────────────────────────────────

def regime_returns(n: int = 900, seed: int = 2) -> np.ndarray:
    """Daily returns with a volatility of 2% on every third 40-day block, else 1%."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 40) % 3 == 2, 0.02, 0.01)
    return rng.normal(0.0, vol)


def test_forward_filter_is_the_filtered_posterior():
    x = regime_returns()
    m, panic = core.fit_macro_hmm(x[:600])
    alpha = core.hmm_filter(m, x)
    assert alpha.shape == (len(x), 2) and np.allclose(alpha.sum(axis=1), 1.0)
    # P(s_t | x_1..t): hmmlearn's smoothed posterior of the prefix x_1..t at its
    # last day (used here as an oracle only).
    for t in (0, 1, 50, 300, 599, 600, 750, 899):
        np.testing.assert_allclose(alpha[t], m.predict_proba(x[:t + 1].reshape(-1, 1))[-1], atol=1e-9)
    # Truncation invariance: the filter on x cut at k is the full filter's first k rows.
    for k in (1, 100, 450, 899):
        np.testing.assert_array_equal(core.hmm_filter(m, x[:k]), alpha[:k])


def test_panic_states_before_a_cut_do_not_depend_on_later_returns():
    # The state of day t (argmax of the filtered probabilities; for two
    # states P(panic) > 0.5) from returns up to t only.  The legacy Viterbi
    # path of the whole window does depend on later returns.
    x = regime_returns()
    for n_comp in (2, 3):
        m, _ = core.fit_macro_hmm(x[:600], n_components=n_comp)
        full, full_v = core.macro_states(m, x), core.macro_states(m, x, viterbi=True)
        np.testing.assert_array_equal(full, core.hmm_filter(m, x).argmax(axis=1))
        moved = []
        for k in range(250, 900, 10):
            np.testing.assert_array_equal(core.macro_states(m, x[:k]), full[:k])
            moved.append(not np.array_equal(core.macro_states(m, x[:k], viterbi=True), full_v[:k]))
        assert any(moved)


def window_panic(monkeypatch, inp: dict, cfg: WfoConfig, closes: pd.DataFrame) -> list:
    """The macro-HMM panic flags of each window (WindowContext.panic)."""
    return [c.panic for c in training_outputs(monkeypatch, inp, cfg, closes)["ctx"]]


@pytest.mark.parametrize("legacy", [False, True], ids=["filter", "legacy_viterbi"])
@pytest.mark.parametrize("mode", ["block-panic", "extreme-3state", "block-calm", "size-panic"])
def test_window_panic_flags_before_a_date_do_not_depend_on_a_different_future(monkeypatch, mode,
                                                                              legacy):
    # Run level, every --hmm mode on the golden SPY panel: the panic series of
    # every window whose OOS has begun by D (its fit ends before oos_s ≤ D),
    # computed with a different future after D, equals the full-data series on
    # every date up to D (the default); the legacy Viterbi decoding of the
    # whole window does not.
    inp = make_golden.case_inputs("hmm_block_panic")
    cfg = case_cfg("hmm_block_panic", hmm=mode, legacy_hmm_viterbi=legacy)
    closes = as_cli_reads(inp["closes"])
    d = cut_date(closes)
    full = window_panic(monkeypatch, inp, cfg, closes)
    other = window_panic(monkeypatch, inp, cfg, future_changed(closes, d))
    windows = make_windows(closes, cfg.train, cfg.oos, cfg.step)
    begun = [k for k, w in enumerate(windows) if w[2] <= d]
    assert windows[begun[-1]][3] > d                               # D inside the last one's OOS
    fitted = [k for k in begun if full[k] is not None]
    assert len(fitted) >= 5 and any(full[k].loc[windows[k][2]:d].any() for k in fitted)
    same_before_d = [(full[k] is None and other[k] is None) or full[k].loc[:d].equals(other[k].loc[:d])
                     for k in begun]
    assert all(same_before_d) != legacy


def test_hmmlearn_decoding_is_not_called_on_the_default_path(monkeypatch):
    # predict / predict_proba / decode / score_samples decode or smooth over
    # the whole sequence.  The default gate uses none of them, and it binds.
    from hmmlearn.hmm import GaussianHMM
    inp = make_golden.case_inputs("hmm_block_panic")
    calls = []

    def forbidden(name):
        def call(self, *a, **k):
            calls.append(name)
            raise AssertionError(f"GaussianHMM.{name} called")
        return call

    for name in ("predict", "predict_proba", "decode", "score_samples"):
        monkeypatch.setattr(GaussianHMM, name, forbidden(name))
    gated = run_inputs(inp, case_cfg("hmm_block_panic")).trades
    assert calls == []
    assert len(gated) < len(run_inputs(inp, case_cfg("hmm_block_panic", hmm="off")).trades)
    with pytest.raises(AssertionError, match="predict"):
        run_inputs(inp, case_cfg("hmm_block_panic", legacy_hmm_viterbi=True))


def test_macro_proxy_is_the_ew_mean_of_us_closes():
    # Without SPY: the EW mean of the log returns of the US listings only —
    # no *_USD leg (converted with a daily FX close, fx_unaligned), no other
    # exchange, no unknown suffix; their closes are all printed at 16:00 ET,
    # so the day-t value may gate a decision at the day-t close.
    idx = pd.bdate_range("2015-01-01", periods=6)
    rng = np.random.default_rng(4)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (6, 6)), axis=0)), index=idx,
                      columns=["AAA", "BBB", "BHP.AX_USD", "RY.TO", "SHEL.L", "XL0.SY"])
    px.iloc[2, 1] = np.nan
    ret, label = core.macro_proxy(px)
    np.testing.assert_array_equal(ret.to_numpy(), np.log(px[["AAA", "BBB"]]).diff().mean(axis=1).to_numpy())
    assert "2" in label and "US" in label
    ret, _ = core.macro_proxy(px, legacy=True)                    # pre-T9: every column
    np.testing.assert_array_equal(ret.to_numpy(), np.log(px).diff().mean(axis=1).to_numpy())
    ret, label = core.macro_proxy(px.assign(SPY=px["AAA"] * 1.1))
    np.testing.assert_array_equal(ret.to_numpy(), np.log(px["AAA"] * 1.1).diff().to_numpy())
    assert "SPY" in label


# ── T9.4: sizing from data at the entry decision ─────────────────────────────

def test_pairmom_uses_only_trades_closed_by_the_entry_decision():
    # Momentum of trade k: the PnL of the pair's earlier trades whose exit
    # filled on or before k's entry decision (last 10; < 3 → 1; sh = mean/std
    # → clip(1 + sh/4, 0.5, 1.5)).  Pre-T9 every earlier-entered trade
    # counted, also one still open at k's decision (the per-window engine: a
    # window's trade can fill its exit after the next window's first entry
    # decision).
    days = pd.bdate_range("2020-01-01", periods=40)
    # Trades 0–2 closed early; trade 3 is entered before trade 4 is decided
    # (day 20) but exits after it (day 22) with a large loss.
    sig, ent, ex = [0, 4, 8, 12, 20], [1, 5, 9, 13, 21], [3, 7, 11, 22, 30]
    tr = pd.DataFrame({"pair": "A-B", "entry_signal": days[sig], "entry": days[ent],
                       "exit": days[ex], "pnl": [0.01, 0.02, 0.015, -0.5, 0.01]})
    closed = tr["pnl"].to_numpy()[:3]
    expected = float(np.clip(1 + 0.25 * closed.mean() / closed.std(), 0.5, 1.5))
    assert core.pair_momentum(tr).tolist() == [1.0, 1.0, 1.0, expected, expected]
    with_open = tr["pnl"].to_numpy()[:4]
    assert core.pair_momentum(tr, legacy_open_trades=True).iloc[4] == float(
        np.clip(1 + 0.25 * with_open.mean() / with_open.std(), 0.5, 1.5))
    # The default engine: a pair's trades never overlap (a prior exit fills by
    # the next entry decision), so the rule changes nothing there.
    inp = make_golden.case_inputs("pairmom")
    on = run_inputs(inp, case_cfg("pairmom")).trades
    pd.testing.assert_frame_equal(on, run_inputs(inp, case_cfg("pairmom", legacy_sizing_lookahead=True)).trades,
                                  check_exact=True)
    assert (on["size"] != 1.0).sum() > 50


def test_voltarget_leverage_is_the_one_available_at_the_entry_decision():
    # A trade decided on day 30 fills on day 33 (its legs' next common print);
    # another trade exits on day 31 with a large PnL.  The leverage of the
    # first comes from exits filled up to its decision (day 30): the day-31
    # exit cannot change it.  Pre-T9 it was looked up at the fill date.
    days = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(6)
    filler = pd.DataFrame({"entry_signal": days[:28], "entry": days[1:29], "exit": days[2:30],
                           "pnl": rng.normal(0, 0.01, 28), "size": 1.0})
    late = pd.DataFrame({"entry_signal": [days[30]], "entry": [days[33]], "exit": [days[40]],
                         "pnl": [0.01], "size": [1.0]})

    def lev(big: float, legacy: bool) -> float:
        other = pd.DataFrame({"entry_signal": [days[29]], "entry": [days[30]], "exit": [days[31]],
                              "pnl": [big], "size": [1.0]})
        tr = pd.concat([filler, other, late], ignore_index=True)
        out = core.apply_vol_target(tr, 0.10, days, legacy_fill_date=legacy)
        return float(out.loc[out["entry"] == days[33], "size"].iloc[0])

    assert lev(0.5, legacy=False) == lev(-0.5, legacy=False) == lev(0.0, legacy=False)
    assert lev(0.5, legacy=True) != lev(0.0, legacy=True)
    # A fill on the day after the decision: the same leverage either way.
    nxt = pd.concat([filler, late.assign(entry=days[31])], ignore_index=True)
    assert core.apply_vol_target(nxt, 0.10, days)["size"].equals(
        core.apply_vol_target(nxt, 0.10, days, legacy_fill_date=True)["size"])
