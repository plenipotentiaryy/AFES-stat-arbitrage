"""Point-in-time universe (audit task T5): `build_universe_pit.py` forms the
pairs from the index members as of the as-of date by a mechanical rule (same
industry, else same sector), screens them with `bias_test`'s gates on prices up
to the as-of date only, and reports the members without prices (survivorship
gap); the daily WFO CLI marks a run HEADLINE only on such a frozen file; the
OOS-PnL curation of `step7_curate_universe.py` is refused outside a legacy run.

All data are synthetic; nothing reads data/."""
import contextlib
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bias_test  # noqa: E402
import build_universe_pit as pit  # noqa: E402
import step7_curate_universe  # noqa: E402
import synthetic  # noqa: E402
from step3j_wfo_daily import NON_HEADLINE_BANNER, headline_exclusions, main as wfo_main  # noqa: E402
from wfo_daily_core import WfoConfig, run_wfo_daily  # noqa: E402

ASOF = "2014-12-31"
EXPECTED = ["EA-EB", "UA-UB", "UA-UD", "UB-UD"]


# ── Synthetic constituents and prices ───────────────────────────────────────

def pit_closes(seed: int = 0) -> pd.DataFrame:
    """Business days 2010-2019.  E* and UC share the factor fe, U* (not UC)
    the factor fu; each name = its factor + an OU spread (half-life 5 d), so
    any two names on one factor are cointegrated; EC is an independent random
    walk.  UD stops printing on 2017-06-30 (delisted in the OOS period), UN
    prints only from 2016 (no price before the as-of date), ZZ has no column.
    XA / XA.TO_USD and XB / XB.L_USD are cross-listings (one price each)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", "2019-12-31", name="Date")
    n = len(idx)

    def rw():
        return np.cumsum(rng.normal(0.0, 0.015, n))

    def ou(sd=0.02):
        phi = 0.5 ** (1 / 5)
        e = rng.normal(0.0, sd * np.sqrt(1 - phi ** 2), n)
        e[0] = rng.normal(0.0, sd)
        return lfilter([1.0], [1.0, -phi], e)

    fe, fu, fa, fb = rw(), rw(), rw(), rw()
    logs = {"EA": fe + ou(), "EB": fe + ou(), "EC": rw(), "EJ": fe + ou(), "EX": fe + ou(),
            "UA": fu + ou(), "UB": fu + ou(), "UD": fu + ou(), "UN": fu + ou(), "UC": fe + ou(),
            "XA": fa + ou(0.005), "XA.TO_USD": fa + ou(0.005),
            "XB": fb + ou(0.005), "XB.L_USD": fb + ou(0.005)}
    closes = pd.DataFrame({k: 100 * np.exp(v) for k, v in logs.items()}, index=idx)
    closes.loc["2017-07-01":, "UD"] = np.nan
    closes.loc[:"2015-12-31", "UN"] = np.nan
    return closes


def pit_constituents() -> pd.DataFrame:
    """Membership spells as read from a CSV (strings, "" = still a member).
    As of 2014-12-31: EA EB EC ZZ (Energy), UA UB UC UD UN (Utilities); EJ
    joins in 2016, EX left in 2013 (both cointegrated with EA and EB)."""
    rows = [("EA", "2005-01-03", "", "Energy"), ("EB", "2008-03-24", "", "Energy"),
            ("EC", "2005-01-03", "", "Energy"), ("ZZ", "2005-01-03", "2016-05-02", "Energy"),
            ("EJ", "2016-01-04", "", "Energy"), ("EX", "2005-01-03", "2013-06-28", "Energy"),
            ("UA", "2005-01-03", "", "Utilities"), ("UB", "2005-01-03", "", "Utilities"),
            ("UC", "2011-09-19", "", "Utilities"), ("UD", "2005-01-03", "2017-06-30", "Utilities"),
            ("UN", "2014-12-31", "", "Utilities")]
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "sector"])


@pytest.fixture(scope="module")
def built():
    closes = pit_closes()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        universe, missing = pit.build_universe(pit_constituents(), closes, ASOF)
    return closes, universe, missing, out.getvalue()


def sha_of(df: pd.DataFrame, path) -> str:
    """The SHA-256 of the file `pit.write_csv` writes, checked against the bytes."""
    digest = pit.write_csv(df, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# ── Mechanical membership and pairing ───────────────────────────────────────

def test_asof_members_are_the_spells_covering_the_asof_date():
    cons = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E", "F", "F", "G", "H"],
        "start": ["2000-01-03", "2014-12-31", "2015-01-02", "2000-01-03", "2000-01-03",
                  "2000-01-03", "2016-01-04", "2010-01-04", "2000-01-03"],
        "end": ["", "", "", "2014-12-31", "2015-01-02", "2012-01-03", "", "2020-01-02",
                "2010-01-04"],
        "sector": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2", "S2"]})
    # start ≤ asof < end (end empty = still a member): A (no end), B (joins on
    # the as-of date), E (leaves the day after), G; not C (joins later), D
    # (leaves on the as-of date), F (out between 2012 and 2016), H.
    m = pit.asof_members(cons, ASOF)
    assert list(m.columns) == ["ticker", "sector"]
    assert list(m["ticker"]) == ["A", "B", "E", "G"]
    assert list(m["sector"]) == ["S1", "S1", "S2", "S2"]


def test_a_member_with_two_classifications_at_the_asof_date_is_refused():
    cons = pd.DataFrame({"ticker": ["A", "A", "B"], "start": ["2000-01-03"] * 3,
                         "end": ["", "", ""], "sector": ["S1", "S1", "S1"]})
    assert list(pit.asof_members(cons, ASOF)["ticker"]) == ["A", "B"]     # duplicate spell
    cons.loc[1, "sector"] = "S2"
    with pytest.raises(ValueError, match="A"):
        pit.asof_members(cons, ASOF)


def test_a_spell_without_a_start_date_is_refused():
    cons = pd.DataFrame({"ticker": ["A", "B"], "start": ["2000-01-03", ""], "end": ["", ""],
                         "sector": ["S1", "S1"]})
    with pytest.raises(ValueError, match=r"no start date.*'B'"):
        pit.asof_members(cons, ASOF)
    rules = pd.DataFrame({"leg1": ["XA"], "leg2": ["XA.TO_USD"], "start": [""], "end": [""]})
    with pytest.raises(ValueError, match="no start date"):
        pit.cross_listing_pairs(rules, ASOF)


def test_tickers_with_a_hyphen_are_refused(built):
    # The CLI splits pair names on "-": "BRK-B" cannot be a leg.
    closes = built[0].assign(**{"BRK-B": 100.0, "BF-B": 50.0})
    cons = pd.concat([pit_constituents(), pd.DataFrame(
        {"ticker": ["BRK-B", "BF-B"], "start": ["2005-01-03"] * 2, "end": [""] * 2,
         "sector": ["Energy", "Utilities"]})], ignore_index=True)
    with pytest.raises(ValueError, match=r"\['BF-B', 'BRK-B'\].*splits pair names on '-'"):
        pit.build_universe(cons, closes, ASOF)
    rules = pd.DataFrame({"leg1": ["XA"], "leg2": ["XA-T"], "start": ["2010-01-04"], "end": [""]})
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"\['XA-T'\]"):
            pit.build_universe(pit_constituents(), built[0], ASOF, cross_listings=rules)
    # A hyphenated ticker that is not a member at the as-of date does not matter.
    left = cons.assign(end=lambda c: c["end"].where(~c["ticker"].str.contains("-"), "2010-01-04"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert list(pit.build_universe(left, closes, ASOF)[0]["pair"]) == EXPECTED


def test_pairs_are_all_same_sector_pairs_or_same_industry_if_present():
    cons = pd.DataFrame({"ticker": ["D", "B", "A", "C", "E"], "start": ["2000-01-03"] * 5,
                         "end": [""] * 5, "sector": ["S1", "S1", "S1", "S1", "S2"]})
    pairs = pit.same_group_pairs(pit.asof_members(cons, ASOF))
    assert list(pairs.columns) == ["pair", "leg1", "leg2", "sector"]
    assert list(pairs["pair"]) == ["A-B", "A-C", "A-D", "B-C", "B-D", "C-D"]   # E is alone
    assert (pairs["leg1"] < pairs["leg2"]).all() and set(pairs["sector"]) == {"S1"}
    # With an industry column the pairs share the industry; a member without
    # one forms no pair.
    cons["industry"] = ["I2", "I1", "I1", "I2", ""]
    pairs = pit.same_group_pairs(pit.asof_members(cons, ASOF))
    assert list(pairs["pair"]) == ["A-B", "C-D"]
    assert list(pairs["sector"]) == ["I1", "I2"]
    cons["industry"] = ["I2", "I1", "I1", "I2", "I3"]
    cons.loc[4, "sector"] = "S1"
    assert list(pit.same_group_pairs(pit.asof_members(cons, ASOF))["pair"]) == ["A-B", "C-D"]


# ── Screens on prices up to the as-of date ──────────────────────────────────

def test_universe_is_the_screened_same_sector_pairs_of_the_asof_members(built, tmp_path):
    closes, universe, _, _ = built
    assert list(universe.columns) == pit.COLUMNS
    # The file: a header and rows with floats written as %.12g.
    sha_of(universe, tmp_path / "u.csv")
    lines = (tmp_path / "u.csv").read_text().splitlines()
    r = universe.iloc[0]
    assert lines[0] == ",".join(pit.COLUMNS) and len(lines) == len(universe) + 1
    assert lines[1] == ",".join([r["pair"], r["leg1"], r["leg2"], r["sector"]]
                                + [f"{r[c]:.12g}" for c in ("corr", "beta", "hl_days", "johansen_stat")]
                                + [ASOF])
    assert list(universe["pair"]) == EXPECTED                 # sorted by pair
    assert (universe["pair"] == universe["leg1"] + "-" + universe["leg2"]).all()
    assert list(universe["sector"]) == ["Energy", "Utilities", "Utilities", "Utilities"]
    assert (universe["asof"] == ASOF).all()
    # UD, delisted in the OOS period, is in; EJ (joins after the as-of date)
    # and EX (left before it) are not, although they cointegrate with EA and
    # EB; UC cointegrates with EA and EB but is in another sector.
    legs = set(universe["leg1"]) | set(universe["leg2"])
    assert "UD" in legs and not {"EJ", "EX", "UC"} & legs
    # The screens are bias_test's: the same candidates through
    # select_pairs_on_train on the slice up to the as-of date give the same
    # pairs and the same corr / beta / hl_days, bit for bit.
    members = pit.asof_members(pit_constituents(), ASOF)
    candidates = pit.same_group_pairs(members)["pair"].tolist()
    ref = bias_test.select_pairs_on_train(closes.loc[:ASOF], candidates).set_index("pair")
    assert sorted(ref.index) == EXPECTED
    for col in ("corr", "beta", "hl_days"):
        assert universe[col].tolist() == ref.loc[EXPECTED, col].tolist()
    # johansen_stat: the trace statistic of bias_test.johansen_coint's test.
    for _, row in universe.iterrows():
        df = np.log(closes.loc[:ASOF, [row["leg1"], row["leg2"]]]).dropna()
        assert row["johansen_stat"] == float(coint_johansen(df, det_order=0, k_ar_diff=1).lr1[0])


def test_survivorship_gap_is_reported(built, tmp_path):
    _, _, missing, out = built
    assert list(missing.columns) == ["ticker", "sector"]
    assert list(missing["ticker"]) == ["UN", "ZZ"]           # no close up to the as-of date
    assert "survivorship gap: 2 of 9 as-of members (22.2%)" in out


def test_screens_read_no_data_after_the_asof_date(built, tmp_path):
    closes, universe, missing, _ = built
    ref = (sha_of(universe, tmp_path / "u0.csv"), sha_of(missing, tmp_path / "m0.csv"))
    # Post-as-of prices perturbed wildly: other values, holes, new prints for
    # the names without pre-as-of prices, a new column → identical output.
    rng = np.random.default_rng(1)
    wild = closes.copy()
    post = wild.index > pd.Timestamp(ASOF)
    wild.loc[post] = wild.loc[post] * np.exp(rng.normal(0.0, 0.5, size=(post.sum(), wild.shape[1])))
    wild.loc[post & (rng.random(len(wild)) < 0.3), ["EA", "UB"]] = np.nan
    wild.loc[post, "ZZ"] = 50.0
    wild.loc[post, "NEW"] = 10.0
    with contextlib.redirect_stdout(io.StringIO()):
        u, m = pit.build_universe(pit_constituents(), wild, ASOF)
    assert (sha_of(u, tmp_path / "u1.csv"), sha_of(m, tmp_path / "m1.csv")) == ref
    # Power: one pre-as-of price changed → a different universe file.
    tamper = closes.copy()
    tamper.loc["2013-06-03", "EA"] *= 1.5
    with contextlib.redirect_stdout(io.StringIO()):
        u, _ = pit.build_universe(pit_constituents(), tamper, ASOF)
    assert sha_of(u, tmp_path / "u2.csv") != ref[0]


def test_cross_listings_enter_only_through_the_asof_rule(built):
    closes = built[0]
    rules = pd.DataFrame({"leg1": ["XA", "XB"], "leg2": ["XA.TO_USD", "XB.L_USD"],
                          "start": ["2010-01-04", "2016-01-04"], "end": ["", ""]})
    pairs = pit.cross_listing_pairs(rules, ASOF)
    assert list(pairs.columns) == ["pair", "leg1", "leg2", "sector"]
    assert pairs.to_dict("records") == [{"pair": "XA-XA.TO_USD", "leg1": "XA",
                                         "leg2": "XA.TO_USD", "sector": "cross-listing"}]
    with contextlib.redirect_stdout(io.StringIO()):
        universe, _ = pit.build_universe(pit_constituents(), closes, ASOF, cross_listings=rules)
    assert list(universe["pair"]) == sorted(EXPECTED + ["XA-XA.TO_USD"])
    assert universe.set_index("pair").at["XA-XA.TO_USD", "sector"] == "cross-listing"
    # ... and are then subject to T2: the CLI's headline policy excludes it.
    assert list(headline_exclusions(universe["pair"].tolist()).index) == ["XA-XA.TO_USD"]


# ── Acceptance: truncating the input after the as-of date ──────────────────

def run_builder(monkeypatch, capsys, *args) -> str:
    monkeypatch.setattr(sys, "argv", ["build_universe_pit.py", *map(str, args)])
    pit.main()
    return capsys.readouterr().out


def test_acceptance_truncating_the_input_after_the_asof_date_gives_identical_files(
        tmp_path, monkeypatch, capsys):
    closes = pit_closes()
    pit_constituents().to_csv(tmp_path / "constituents.csv", index=False)
    closes.to_csv(tmp_path / "closes_full.csv")
    closes.loc[:ASOF].to_csv(tmp_path / "closes_cut.csv")
    assert closes.index.max() > pd.Timestamp("2019-12-01")      # well past the as-of date
    outs = {}
    for name in ("full", "cut"):
        (tmp_path / name).mkdir()
        outs[name] = run_builder(monkeypatch, capsys, "--asof", ASOF,
                                 "--constituents", tmp_path / "constituents.csv",
                                 "--closes", tmp_path / f"closes_{name}.csv",
                                 "--out", tmp_path / name / f"universe_pit_{ASOF}.csv")
    files = [f"universe_pit_{ASOF}.csv", f"universe_pit_{ASOF}.missing.csv"]
    for f in files:
        assert (tmp_path / "full" / f).read_bytes() == (tmp_path / "cut" / f).read_bytes()
    universe = pd.read_csv(tmp_path / "full" / files[0])
    assert list(universe["pair"]) == EXPECTED
    assert list(pd.read_csv(tmp_path / "full" / files[1])["ticker"]) == ["UN", "ZZ"]
    digest = hashlib.sha256((tmp_path / "full" / files[0]).read_bytes()).hexdigest()
    assert f"SHA-256 {digest}" in outs["full"] and f"SHA-256 {digest}" in outs["cut"]
    assert "survivorship gap: 2 of 9 as-of members (22.2%)" in outs["full"]


def test_default_output_path_is_in_the_data_dir():
    assert str(pit.default_out(ASOF)) == os.path.join("data", f"universe_pit_{ASOF}.csv")


# ── Headline marker of the daily WFO CLI ────────────────────────────────────

def pit_frame(pairs: list[str], asof: str) -> pd.DataFrame:
    """A file shaped like the builder's (its columns, every value present)."""
    legs = [p.split("-") for p in pairs]
    return pd.DataFrame({"pair": pairs, "leg1": [a for a, _ in legs], "leg2": [b for _, b in legs],
                         "sector": "S", "corr": 0.5, "beta": 1.0, "hl_days": 5.0,
                         "johansen_stat": 50.0, "asof": asof}, columns=pit.COLUMNS)


def run_wfo(monkeypatch, pairs_file: str, *extra):
    """The CLI in the current directory (data/ holds the closes and the pairs file)."""
    monkeypatch.setattr(sys, "argv", ["step3j_wfo_daily.py", "--pairs", pairs_file,
                                      "--closes", "closes.csv", "--execution-model", "flat",
                                      "--tag", "t", *extra])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = wfo_main()
    return result, out.getvalue()


def assert_non_headline(result, out, why, legacy: bool = False):
    assert result.stats["headline"] is False, why
    assert "headline=False" in out
    assert not any(line.startswith("HEADLINE RUN") for line in out.splitlines())
    banners = [line for line in out.splitlines() if line.startswith("NON-HEADLINE RUN")]
    assert len(banners) >= 2, why                                     # header and summary
    if legacy:
        assert any("legacy" in b for b in banners)
    else:
        assert out.count(NON_HEADLINE_BANNER) >= 2
        # in the header (before the run header) and in the summary
        assert out.index(NON_HEADLINE_BANNER) < out.index("DAILY WFO  (")
        assert out.rindex(NON_HEADLINE_BANNER) > out.index("DAILY WFO SUMMARY")


@pytest.fixture
def wfo_dir(tmp_path, monkeypatch):
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=0)  # from 2005-01-03
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    closes.to_csv(tmp_path / "data" / "closes.csv")
    monkeypatch.chdir(tmp_path)

    def run(pairs_file: str, frame: pd.DataFrame, *extra):
        frame.to_csv(tmp_path / "data" / pairs_file, index=False)
        return run_wfo(monkeypatch, pairs_file, *extra)
    return run, pairs


def test_cli_marks_a_run_headline_only_on_a_frozen_pit_universe(wfo_dir):
    run, pairs = wfo_dir
    # First OOS date = 2005-01-03 + 6 months = 2005-07-03.
    frozen = pit_frame(pairs, "2004-12-31")
    result, out = run("universe_pit_2004-12-31.csv", frozen)
    assert result.stats["headline"] is True
    assert "HEADLINE RUN" in out and "headline=True" in out
    digest = hashlib.sha256(open("data/universe_pit_2004-12-31.csv", "rb").read()).hexdigest()
    assert f"SHA-256 {digest}" in out                       # the universe hash, for the record
    assert NON_HEADLINE_BANNER not in out and "NON-HEADLINE" not in out
    assert "HEADLINE UNIVERSE POLICY: excludes" in out      # T2's policy line, renamed
    # A hand-appended pair: the asof (or another builder column) left blank.
    appended = pd.concat([frozen, pd.DataFrame({"pair": ["OUA0-OUB1"], "asof": [None]})])
    blank_corr = pd.concat([frozen, pit_frame(["OUA0-OUB1"], "2004-12-31").assign(corr=None)])
    for pairs_file, frame, extra, why in [
            ("pairs.csv", frozen, (), "a manual pair list"),
            ("universe_pit_2004-12-31.csv", frozen.drop(columns="asof"), (), "no asof column"),
            ("universe_pit_2004-12-31.csv", appended, (), "an appended row without asof"),
            ("universe_pit_2004-12-31.csv", blank_corr, (), "an appended row without corr"),
            ("universe_pit_2004-12-31.csv", frozen[["pair", "asof"]], (),
             "a renamed hindsight file: pair and asof only"),
            ("universe_pit_2004-12-31.csv", pit_frame(pairs, "2003-12-31"), (), "name ≠ asof column"),
            ("universe_pit_2005-07-05.csv", pit_frame(pairs, "2005-07-05"), (),
             "asof after the first OOS date"),
            ("universe_pit_2005-07-03.csv", pit_frame(pairs, "2005-07-03"), (),
             "asof = the first OOS date")]:
        result, out = run(pairs_file, frame, *extra)
        assert_non_headline(result, out, why)
    for extra in [("--legacy-cross-listed",), ("--legacy-exit-date-pnl",)]:
        result, out = run("universe_pit_2004-12-31.csv", frozen, *extra)
        assert_non_headline(result, out, extra, legacy=True)


def test_builder_output_certifies_a_headline_run_and_a_tampered_copy_does_not(
        tmp_path, monkeypatch, capsys):
    # The chain: build_universe_pit.py → data/universe_pit_<asof>.csv → the
    # WFO CLI from the as-of date (first OOS date 2015-06-30).
    closes = pit_closes()
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    pit_constituents().to_csv(tmp_path / "constituents.csv", index=False)
    closes.to_csv(tmp_path / "data" / "closes.csv")
    monkeypatch.chdir(tmp_path)
    run_builder(monkeypatch, capsys, "--asof", ASOF, "--constituents", "constituents.csv",
                "--closes", "data/closes.csv")                   # default --out: data/
    frozen = pd.read_csv(f"data/universe_pit_{ASOF}.csv")
    assert list(frozen["pair"]) == EXPECTED
    result, out = run_wfo(monkeypatch, f"universe_pit_{ASOF}.csv", "--start", ASOF)
    assert result.stats["headline"] is True and "headline=True" in out
    assert set(result.trades["pair"]) <= set(EXPECTED) and len(result.trades) > 0
    # A tampered copy: a hand-picked pair (UC is cointegrated with EA, but in
    # another sector) appended with only its name and the as-of date.
    tampered = pd.concat([frozen, pd.DataFrame({"pair": ["EA-UC"], "asof": [ASOF]})])
    tampered.to_csv(f"data/universe_pit_{ASOF}.csv", index=False)
    result, out = run_wfo(monkeypatch, f"universe_pit_{ASOF}.csv", "--start", ASOF)
    assert_non_headline(result, out, "tampered")


def test_run_wfo_daily_alone_never_certifies_a_headline_run():
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        for cfg in (WfoConfig(execution_model="flat"),
                    WfoConfig(execution_model="flat", legacy_exit_date_pnl=True)):
            assert run_wfo_daily(closes, pairs, cfg).stats["headline"] is False
        flat = closes.assign(OUA1=100.0, OUB1=50.0)      # constant prices: no print, no trade
        no_trades = run_wfo_daily(flat, ["OUA1-OUB1"], WfoConfig(execution_model="flat"))
    assert no_trades.trades.empty and no_trades.stats["headline"] is False


# ── step7: OOS-PnL curation never feeds a headline run ──────────────────────

def test_step7_curation_is_refused_outside_a_legacy_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    pd.DataFrame({"pair": ["A-B", "A-B", "C-D", "E-F"], "oos_pnl": [0.5, 0.1, -0.2, 0.3],
                  "oos_trades": [3, 3, 9, 2], "oos_sharpe": [1.0, 0.5, -0.4, 2.0]}) \
        .to_csv(tmp_path / "data" / "wfo_params.csv", index=False)
    pd.DataFrame({"pair": ["A-B", "C-D", "E-F"], "beta": [1.0, 2.0, 3.0]}) \
        .to_csv(tmp_path / "data" / "pairs_selected.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["step7_curate_universe.py"])
    with pytest.raises(SystemExit, match="--legacy-oos-curation"):
        step7_curate_universe.main()
    assert "never be used in a headline run" in capsys.readouterr().out
    assert not (tmp_path / "data" / "pairs_live.csv").exists()
    monkeypatch.setattr(sys, "argv", ["step7_curate_universe.py", "--legacy-oos-curation"])
    step7_curate_universe.main()
    assert "never be used in a headline run" in capsys.readouterr().out
    live = pd.read_csv(tmp_path / "data" / "pairs_live.csv")
    assert live.to_dict("records") == [{"pair": "A-B", "beta": 1.0}]    # pnl > 0 and ≥ 5 trades
