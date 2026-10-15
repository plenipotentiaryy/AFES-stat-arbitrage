"""Costs that scale with the position (audit task T8).

Every fill of every leg costs rate_leg × |leg notional at that fill| (per $1
of leg-1 notional the entry notionals are 1 and |β|, the exit notionals
|q1·P1_exit| and |q2·P2_exit|), charged on the fill date; a round trip at
unchanged prices costs 2·(c + s)·(1 + |β|).  An FX conversion cost is added
to each fill of a non-USD leg (`*_USD`).  Borrow accrues on each held row on
the notional of the short leg(s) at that row's close: leg 1 is short when
side = −1, leg 2 when side·β > 0 (so with β < 0 a short spread borrows both
legs and a long spread none), at the US or the non-US rate (flat model) or
the leg's tier rate (almgren).  `legacy_flat_cost` restores the pre-T8 flat
round trip per trade, whatever |β|, with borrow per unit size per held day.

Engine level: hand-made z and leg prices in both engines (`backtest_pair`,
`backtest_pair_continuous`).  Run level (`run_wfo_daily`): an OU panel with
`*_USD` legs.  CLI: the cost model in the run header and in the
wfo_daily_meta<tag>.json sidecar.
"""
import contextlib
import dataclasses
import hashlib
import io
import json
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

import mtm  # noqa: E402
import synthetic  # noqa: E402
import wfo_daily_core as core  # noqa: E402
from almgren_chriss_repricing import borrow_log_per_day, classify_ticker, leg_cost_log  # noqa: E402
from cross_listing import mask_stale_prints  # noqa: E402
from wfo_daily_core import LegCosts, WfoConfig, run_wfo_daily  # noqa: E402

# ── Engine level ────────────────────────────────────────────────────────────

DAYS = pd.bdate_range("2021-03-01", periods=12)
WINDOWS = [(DAYS[0], DAYS[0], DAYS[0], DAYS[-1])]     # every row but the last is OOS
FLAT = 10e-4                                           # (commission 5 + slippage 5) bp per leg per fill
ENGINES = ["per_window", "continuous"]


def z_trade(side: int) -> np.ndarray:
    """Entry decided on row 4 (|z| = 3; z > 0: short spread), filled on row 5;
    target (z = 0) decided on row 7, filled on row 8: held rows 6, 7, 8."""
    z = np.full(len(DAYS), np.nan)
    z[:4] = 0.0
    z[4] = -3.0 * side
    z[5:7] = -2.5 * side
    z[7:] = 0.0
    return z


def legs(moving: bool) -> np.ndarray:
    """[P1, P2] per row: constant (50, 20), or moving every row."""
    r = np.arange(len(DAYS), dtype=float)
    if not moving:
        return np.column_stack([np.full(len(DAYS), 50.0), np.full(len(DAYS), 20.0)])
    return np.column_stack([50.0 + r, 20.0 + 0.5 * r])


def one_trade(engine: str, side: int, beta: float, lc: LegCosts | None, px: np.ndarray,
              cost_per_trade: float = 0.0, borrow_per_day: float = 0.0) -> pd.Series:
    z = z_trade(side)
    if engine == "per_window":
        tr = core.backtest_pair(pd.Series(0.0, index=DAYS), DAYS[0], DAYS[-1] + pd.Timedelta(days=1),
                                cost_per_trade, z_series=pd.Series(z, index=DAYS),
                                borrow_per_day=borrow_per_day,
                                leg_prices=pd.DataFrame(px, index=DAYS, columns=["P1", "P2"]),
                                beta=beta, leg_costs=lc)
    else:
        pw = {1: core.PairWindow(beta=beta, z=z, cost=cost_per_trade, borrow=borrow_per_day,
                                 leg_costs=lc)}
        tr = core.backtest_pair_continuous(DAYS, px, len(DAYS) - 1, WINDOWS, pw)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert (t["entry"], t["exit"], t["held"], t["side"]) == (DAYS[5], DAYS[8], 3, side)
    assert t["pnl"] == pytest.approx(t["size"] * (t["gross"] - t["cost"] - t["borrow"]), abs=1e-15)
    return t


@pytest.mark.parametrize("engine", ENGINES)
def test_acceptance_round_trip_cost_for_beta_2_is_three_times_beta_0(engine):
    # Flat model, zero borrow, unchanged prices: round trip = 2·(c + s)·(1 + |β|).
    lc = LegCosts(fill1=FLAT, fill2=FLAT, borrow1=0.0, borrow2=0.0)
    c0 = one_trade(engine, -1, 0.0, lc, legs(moving=False))["cost"]
    c2 = one_trade(engine, -1, 2.0, lc, legs(moving=False))["cost"]
    assert c0 == pytest.approx(2 * FLAT * (1 + 0.0), rel=1e-12)        # 20 bp
    assert c2 == pytest.approx(2 * FLAT * (1 + 2.0), rel=1e-12)        # 60 bp
    assert c2 == pytest.approx(3 * c0, rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0, -2.0, 7.5])
def test_round_trip_is_two_fills_per_leg_on_the_leg_notionals(engine, beta):
    lc = LegCosts(fill1=FLAT, fill2=3 * FLAT, borrow1=0.0, borrow2=0.0)
    t = one_trade(engine, +1, beta, lc, legs(moving=False))
    assert t["cost_entry"] == pytest.approx(FLAT * 1.0 + 3 * FLAT * abs(beta), rel=1e-12)
    assert t["cost_exit"] == pytest.approx(t["cost_entry"], rel=1e-12)
    assert t["cost"] == t["cost_entry"] + t["cost_exit"]
    # Moving prices: each fill on the leg notionals at that fill's prices.
    t = one_trade(engine, +1, beta, lc, legs(moving=True))
    q1, q2 = t["q1"], t["q2"]
    assert t["cost_entry"] == pytest.approx(FLAT * abs(q1 * t["p1_entry"]) + 3 * FLAT * abs(q2 * t["p2_entry"]),
                                            rel=1e-12)
    assert t["cost_exit"] == pytest.approx(FLAT * abs(q1 * t["p1_exit"]) + 3 * FLAT * abs(q2 * t["p2_exit"]),
                                           rel=1e-12)
    assert t["cost_exit"] > t["cost_entry"]          # the legs' prices rose: larger exit notionals


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("side,beta,short1,short2", [
    (-1, 1.5, True, False),     # short spread, β > 0: short leg 1, long leg 2
    (+1, 1.5, False, True),     # long spread, β > 0: long leg 1, short leg 2
    (-1, -1.5, True, True),     # short spread, β < 0: both legs short
    (+1, -1.5, False, False),   # long spread, β < 0: both legs long, no borrow
    (+1, 0.0, False, False),    # β = 0: leg 2 holds nothing
], ids=["short_pos_beta", "long_pos_beta", "short_neg_beta", "long_neg_beta", "long_zero_beta"])
def test_borrow_on_the_short_legs_marked_notional(engine, side, beta, short1, short2):
    b1, b2 = 1e-4, 3e-4
    lc = LegCosts(fill1=0.0, fill2=0.0, borrow1=b1, borrow2=b2)
    px = legs(moving=True)
    t = one_trade(engine, side, beta, lc, px)
    r1, r2 = (b1 if short1 else 0.0), (b2 if short2 else 0.0)
    assert (t["borrow_rate1"], t["borrow_rate2"]) == (r1, r2)
    q1, q2 = 1 / px[5, 0], beta / px[5, 1]
    held = px[6:9]                                   # the closes of the held rows 6, 7, 8
    expected = sum(r1 * abs(q1 * p1) + r2 * abs(q2 * p2) for p1, p2 in held)
    assert t["borrow"] == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert (t["borrow"] > 0) == (short1 or short2)
    assert t["cost"] == 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_almgren_leg2_impact_grows_with_the_square_root_of_its_participation(engine):
    # Ruling R38: leg 2 trades |β|·notional, so its square-root impact per $1
    # is impact2·√|β| on top of its half-spread; the rate applies to both fills.
    h, i = 5e-4, 2e-4
    lc = LegCosts(fill1=0.0, fill2=h, borrow1=0.0, borrow2=0.0, impact2=i)
    rate = {}
    for beta in (0.25, 1.0, 4.0, -4.0):
        t = one_trade(engine, -1, beta, lc, legs(moving=False))
        assert t["cost_entry"] == pytest.approx((h + i * np.sqrt(abs(beta))) * abs(beta), rel=1e-12)
        assert t["cost_exit"] == pytest.approx(t["cost_entry"], rel=1e-12)
        rate[beta] = t["cost_entry"] / abs(beta) - h               # leg 2's impact per $1
    assert rate[4.0] == pytest.approx(2 * rate[1.0], rel=1e-9)     # 4× the notional: 2× the impact
    assert rate[0.25] == pytest.approx(rate[1.0] / 2, rel=1e-9)
    assert rate[-4.0] == pytest.approx(rate[4.0], rel=1e-12)
    flat = LegCosts(fill1=0.0, fill2=h, borrow1=0.0, borrow2=0.0)  # impact2 = 0: no β dependence
    assert one_trade(engine, -1, 4.0, flat, legs(moving=False))["cost_entry"] == pytest.approx(4 * h, rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_legacy_flat_cost_is_one_round_trip_per_trade_whatever_beta(engine):
    # leg_costs None (legacy_flat_cost): the pre-T8 scalar per trade, borrow per held day.
    for beta in (0.0, 2.0, -1.5):
        t = one_trade(engine, -1, beta, None, legs(moving=True), cost_per_trade=20e-4, borrow_per_day=3e-6)
        assert (t["cost"], t["borrow"]) == (20e-4, 3 * 3e-6)
        assert (t["cost_entry"], t["cost_exit"]) == (10e-4, 10e-4)
        assert np.isnan(t["borrow_rate1"]) and np.isnan(t["borrow_rate2"])
        assert t["pnl"] == t["size"] * t["gross"] - t["size"] * 20e-4 - t["size"] * (3 * 3e-6)


def test_short_borrow_rates_helper():
    assert core.short_borrow_rates(-1, 2.0, 1.0, 2.0) == (1.0, 0.0)
    assert core.short_borrow_rates(+1, 2.0, 1.0, 2.0) == (0.0, 2.0)
    assert core.short_borrow_rates(-1, -2.0, 1.0, 2.0) == (1.0, 2.0)
    assert core.short_borrow_rates(+1, -2.0, 1.0, 2.0) == (0.0, 0.0)
    assert core.short_borrow_rates(-1, 0.0, 1.0, 2.0) == (1.0, 0.0)


def test_non_us_listing_is_a_usd_converted_local_leg():
    for t in ("RY.TO_USD", "SAP.DE_USD", "BHP.AX_USD", "OUB1.TO_USD"):
        assert core.is_non_us_listing(t)
    for t in ("RY", "BHP", "SAP", "AAPL", "OUA0"):          # US listings (ADRs included)
        assert not core.is_non_us_listing(t)


# ── Trade-level MTM ─────────────────────────────────────────────────────────

MDAYS = pd.bdate_range("2021-01-04", periods=8, name="Date")
PA = [100.0, 100.0, 98.0, 95.0, 99.0, 101.0, 101.0, 101.0]
PB = [100.0, 100.0, 102.0, np.nan, 101.0, 99.0, 99.0, 99.0]


def test_mtm_books_each_fills_cost_on_its_date_and_borrow_on_the_held_rows_closes():
    # Short spread with β = −1: both legs short; PB has no print on day 3 (no
    # borrow that row: the pair's rows are the dates on which both legs print).
    px = pd.DataFrame({"PA": PA, "PB": PB}, index=MDAYS)
    side, beta, size = -1, -1.0, 0.5
    p1e, p2e, p1x, p2x = 100.0, 100.0, 101.0, 99.0                # fills on days 1 and 5
    q1, q2 = core.entry_quantities(p1e, p2e, beta)
    r1, r2 = 1e-4, 2e-4
    ce = mtm.fill_cost(FLAT, 2 * FLAT, q1, q2, p1e, p2e)
    cx = mtm.fill_cost(FLAT, 2 * FLAT, q1, q2, p1x, p2x)
    rows = [2, 4, 5]                                               # held rows with both legs
    per_row = [r1 * abs(q1 * PA[k]) + r2 * abs(q2 * PB[k]) for k in rows]
    gross = core.leg_gross(side, q1, q2, p1e, p2e, p1x, p2x)
    borrow = float(np.sum(per_row))
    t = {"pair": "PA-PB", "entry": MDAYS[1], "exit": MDAYS[5], "held": 3, "side": side,
         "size": size, "beta": beta, "q1": q1, "q2": q2, "p1_entry": p1e, "p2_entry": p2e,
         "p1_exit": p1x, "p2_exit": p2x, "gross": gross, "cost": ce + cx, "borrow": borrow,
         "cost_entry": ce, "cost_exit": cx, "borrow_rate1": r1, "borrow_rate2": r2,
         "pnl": core.net_pnl(size, gross, ce + cx, borrow), "reason": "target"}
    m = mtm.trade_mtm(pd.Series(t), px)
    assert ce == pytest.approx(FLAT * 1.0 + 2 * FLAT * 1.0)
    assert m["cost"].tolist() == [ce, 0.0, 0.0, 0.0, cx]
    np.testing.assert_allclose(m["borrow"], [0.0, per_row[0], 0.0, per_row[1], per_row[2]], rtol=1e-15)
    net = size * (m["gross"] - m["cost"] - m["borrow"])
    assert net.sum() == pytest.approx(t["pnl"], abs=1e-15)
    assert m["gross"].sum() == pytest.approx(gross, abs=1e-15)


# ── Run level ───────────────────────────────────────────────────────────────

USD_LEGS = {"OUB1": "OUB1.TO_USD", "OUA2": "OUA2.TO_USD", "OUA3": "OUA3.TO_USD", "OUB3": "OUB3.TO_USD"}


@pytest.fixture(scope="module")
def usd_panel():
    """OU panel (3 years, 4 pairs): pair 0 US only, pair 1 leg 2 non-USD,
    pair 2 leg 1 non-USD, pair 3 both legs non-USD."""
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=4, seed=5)
    closes = closes.rename(columns=USD_LEGS)
    pairs = ["-".join(USD_LEGS.get(t, t) for t in p.split("-")) for p in pairs]
    return closes, pairs


def quiet(closes, pairs, cfg, **frames):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_wfo_daily(closes, pairs, cfg, **frames)


def usd(t: str) -> float:
    return 1.0 if t.endswith("_USD") else 0.0


@pytest.mark.parametrize("model", ["flat", "almgren"])
def test_fx_cost_only_on_the_fills_of_usd_converted_legs(usd_panel, model):
    closes, pairs = usd_panel
    base = WfoConfig(execution_model=model, cost=0.0, slip=0.0, borrow=0.0, borrow_nonus=0.0,
                     fx_cost=0.0)
    with_fx = quiet(closes, pairs, replace(base, fx_cost=2.0)).trades
    no_fx = quiet(closes, pairs, base).trades
    assert set(with_fx["pair"]) == set(pairs)                       # every kind of pair trades
    # Costs drive no decision: the same trades, only the costs differ.
    same = ["pair", "entry", "exit", "side", "beta", "gross", "borrow"]
    pd.testing.assert_frame_equal(with_fx[same], no_fx[same])
    fx = 2.0 / 1e4
    for (_, t), (_, u) in zip(with_fx.iterrows(), no_fx.iterrows()):
        a, b = t["pair"].split("-")
        exp_e = fx * (usd(a) * abs(t["q1"] * t["p1_entry"]) + usd(b) * abs(t["q2"] * t["p2_entry"]))
        exp_x = fx * (usd(a) * abs(t["q1"] * t["p1_exit"]) + usd(b) * abs(t["q2"] * t["p2_exit"]))
        assert t["cost_entry"] - u["cost_entry"] == pytest.approx(exp_e, rel=1e-9, abs=1e-15)
        assert t["cost_exit"] - u["cost_exit"] == pytest.approx(exp_x, rel=1e-9, abs=1e-15)
        if model == "flat":
            assert u["cost"] == 0.0
            if t["pair"] == pairs[0]:
                assert t["cost"] == 0.0                              # a US-only pair: no FX


@pytest.mark.parametrize("engine", ["continuous", "per_window"])
def test_almgren_rate_of_each_leg_is_its_cost_on_its_own_notional(usd_panel, engine):
    # Ruling R38: leg 1 at --notional, leg 2 at |β_e|·notional (σ of the window's
    # training rows; no volumes file: the tier-default ADV), + FX on *_USD legs.
    closes, pairs = usd_panel
    cfg = WfoConfig(execution_model="almgren", legacy_window_reset=engine == "per_window")
    tr = quiet(closes, pairs, cfg).trades
    masked = mask_stale_prints(closes)
    windows = core.make_windows(masked, cfg.train, cfg.oos, cfg.step)
    adv = {"us_mid_cap": 3.0e8, "canadian_local": 1.0e8}
    assert (tr["beta"].abs() > 1).any() and (tr["beta"].abs() < 1).any()
    for _, t in tr.iterrows():
        a, b = t["pair"].split("-")
        tr_s, tr_e, oos_s, _ = windows[t["window"] - 1]
        rows = masked[[a, b]].dropna()
        end = oos_s if engine == "continuous" else tr_e + pd.Timedelta(days=1)   # T9.1 / legacy train
        train = rows[(rows.index >= tr_s) & (rows.index < end)]
        sig = np.log(train).diff().std()
        r1 = leg_cost_log(a, sig[a], cfg.notional, adv[classify_ticker(a)]) + usd(a) * 2e-4
        r2 = leg_cost_log(b, sig[b], abs(t["beta"]) * cfg.notional, adv[classify_ticker(b)]) + usd(b) * 2e-4
        assert t["cost_entry"] == pytest.approx(r1 * abs(t["q1"] * t["p1_entry"]) + r2 * abs(t["q2"] * t["p2_entry"]),
                                                rel=1e-9)
        assert t["cost_exit"] == pytest.approx(r1 * abs(t["q1"] * t["p1_exit"]) + r2 * abs(t["q2"] * t["p2_exit"]),
                                               rel=1e-9)


def borrow_rate(model: str, cfg: WfoConfig, ticker: str) -> float:
    if model == "almgren":
        return borrow_log_per_day(ticker)
    return (cfg.borrow_nonus if ticker.endswith("_USD") else cfg.borrow) / (252 * 1e4)


@pytest.mark.parametrize("engine", ["continuous", "per_window"])
@pytest.mark.parametrize("model", ["flat", "almgren"])
def test_borrow_rate_of_each_short_leg_us_or_non_us(usd_panel, model, engine):
    closes, pairs = usd_panel
    cfg = WfoConfig(execution_model=model, cost=0.0, slip=0.0, fx_cost=0.0,
                    legacy_window_reset=engine == "per_window")
    tr = quiet(closes, pairs, cfg).trades
    masked = mask_stale_prints(closes)
    seen = set()
    for _, t in tr.iterrows():
        a, b = t["pair"].split("-")
        r1, r2 = core.short_borrow_rates(t["side"], t["beta"], borrow_rate(model, cfg, a),
                                         borrow_rate(model, cfg, b))
        assert (t["borrow_rate1"], t["borrow_rate2"]) == (r1, r2)
        rows = masked[[a, b]].dropna()
        held = rows[(rows.index > t["entry"]) & (rows.index <= t["exit"])]
        assert len(held) == t["held"]
        expected = (r1 * (t["q1"] * held[a]).abs() + r2 * (t["q2"] * held[b]).abs()).sum()
        assert t["borrow"] == pytest.approx(expected, rel=1e-12, abs=1e-18)
        seen |= {(a.endswith("_USD"), r1 > 0), (b.endswith("_USD"), r2 > 0)}
    assert {(True, True), (False, True)} <= seen                    # US and non-US short legs borrowed


@pytest.mark.parametrize("engine", ["continuous", "per_window"])
def test_daily_mtm_invariants_hold_with_the_position_costs(usd_panel, engine):
    # Default T8 costs (flat 5 + 5 bp, FX 2 bp, borrow 80 / 300 bp/yr): each
    # trade's daily net sums to its pnl, its fill costs sit on its fill dates,
    # and the run's daily_pnl is the MTM of the final trades.
    closes, pairs = usd_panel
    cfg = WfoConfig(execution_model="flat", legacy_window_reset=engine == "per_window")
    res = quiet(closes, pairs, cfg)
    tr = res.trades
    masked = mask_stale_prints(closes)
    assert len(tr) > 30 and (tr["borrow"] > 0).any() and (tr["cost_entry"] != tr["cost_exit"]).any()
    np.testing.assert_allclose(tr["pnl"], tr["size"] * (tr["gross"] - tr["cost"] - tr["borrow"]),
                               rtol=1e-12, atol=1e-15)
    assert (tr["cost"] == tr["cost_entry"] + tr["cost_exit"]).all()
    for _, t in tr.iterrows():
        a, b = t["pair"].split("-")
        f1, f2 = (cfg.cost + cfg.slip + cfg.fx_cost * usd(a)) / 1e4, (cfg.cost + cfg.slip + cfg.fx_cost * usd(b)) / 1e4
        assert t["cost_entry"] == pytest.approx(f1 * 1.0 + f2 * abs(t["beta"]), rel=1e-12)
        assert t["cost_exit"] == pytest.approx(f1 * abs(t["q1"] * t["p1_exit"]) + f2 * abs(t["q2"] * t["p2_exit"]),
                                               rel=1e-12)
        m = mtm.trade_mtm(t, masked)
        if len(m) == 1:                              # entry and exit fill on one date
            assert m["cost"].iloc[0] == t["cost"]
        else:
            assert (m["cost"].iloc[0], m["cost"].iloc[-1]) == (t["cost_entry"], t["cost_exit"])
            assert (m["cost"].iloc[1:-1] == 0).all()
        assert m["borrow"].sum() == pytest.approx(t["borrow"], rel=1e-12, abs=1e-18)
        assert m["gross"].sum() == pytest.approx(t["gross"], abs=1e-12)
        assert (t["size"] * (m["gross"] - m["cost"] - m["borrow"])).sum() == pytest.approx(t["pnl"], abs=1e-12)
    span = res.daily_pnl.index
    pd.testing.assert_series_equal(res.daily_pnl, mtm.daily_mtm(tr, masked, span))
    assert res.daily_pnl.sum() == pytest.approx(tr["pnl"].sum(), abs=1e-9)


def test_legacy_flat_cost_restores_the_pre_t8_cost_per_trade(usd_panel):
    closes, pairs = usd_panel
    res = quiet(closes, pairs, WfoConfig(execution_model="flat", legacy_flat_cost=True))
    tr = res.trades
    rt, bpd = 2 * (5.0 + 5.0) / 1e4, 80.0 / (252 * 1e4)
    assert (tr["cost"] == rt).all() and (tr["cost_entry"] == rt / 2).all() and (tr["cost_exit"] == rt / 2).all()
    assert (tr["borrow"] == tr["held"] * bpd).all()
    assert tr["borrow_rate1"].isna().all() and tr["borrow_rate2"].isna().all()
    assert (tr["pnl"] == tr["size"] * tr["gross"] - tr["size"] * rt - tr["size"] * (tr["held"] * bpd)).all()
    # The same trades with the T8 model cost more on |β| > 0 (both fills, both legs).
    t8 = quiet(closes, pairs, WfoConfig(execution_model="flat", fx_cost=0.0)).trades
    pd.testing.assert_frame_equal(t8[["pair", "entry", "exit", "gross"]], tr[["pair", "entry", "exit", "gross"]])
    assert (t8["cost"] > tr["cost"]).all()


def test_legacy_spread_pnl_needs_the_legacy_flat_cost(usd_panel):
    # The spread PnL has no leg notionals to charge per fill.
    closes, pairs = usd_panel
    cfg = WfoConfig(execution_model="flat", legacy_window_reset=True, legacy_spread_pnl=True,
                    legacy_exit_date_pnl=True)
    with pytest.raises(ValueError, match="legacy_flat_cost"):
        quiet(closes, pairs, cfg)
    quiet(closes, pairs, replace(cfg, legacy_flat_cost=True))


def test_default_execution_model_is_the_conservative_one():
    # Ruling (AUDIT_FIX_LOG.md T8): the default is the model with the LOWER
    # development-period Sharpe on the real default run — flat (+0.278) below
    # almgren (+0.369) under the T8 cost model.
    import step3j_wfo_daily
    assert WfoConfig().execution_model == "flat"
    assert step3j_wfo_daily.build_parser().parse_args([]).execution_model == "flat"


def test_cost_model_describes_the_run():
    flat = core.cost_model(WfoConfig(execution_model="flat", cost=4.0, slip=6.0, fx_cost=3.0,
                                     borrow=90.0, borrow_nonus=310.0, notional=50_000))
    assert flat["execution_model"] == "flat" and flat["legacy_flat_cost"] is False
    assert (flat["fill_bp_per_leg"], flat["commission_bp"], flat["slippage_bp"]) == (10.0, 4.0, 6.0)
    assert flat["fx_cost_bp"] == 3.0
    assert (flat["borrow_bp_per_year_us"], flat["borrow_bp_per_year_non_us"]) == (90.0, 310.0)
    assert flat["notional_usd_per_leg"] == 50_000
    alm = core.cost_model(WfoConfig(execution_model="almgren"))
    assert alm["execution_model"] == "almgren" and "tiers" in alm and alm["fx_cost_bp"] == 2.0
    assert alm["tiers"]["us_large_cap"]["borrow_bp_per_year"] == 50
    legacy = core.cost_model(WfoConfig(execution_model="flat", legacy_flat_cost=True))
    assert legacy["legacy_flat_cost"] is True and legacy["round_trip_bp_per_trade"] == 20.0
    json.dumps(flat), json.dumps(alm), json.dumps(legacy)                 # serialisable


# ── CLI: run header and the meta sidecar ────────────────────────────────────

@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory):
    """The CLI on a small OU panel (default and legacy cost) in a temp run dir."""
    import step3j_wfo_daily
    root = tmp_path_factory.mktemp("t8run")
    closes, pairs = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=1)
    (root / "data").mkdir()
    (root / "output").mkdir()
    closes.to_csv(root / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(root / "data" / "pairs.csv", index=False)
    argv, cwd = sys.argv, os.getcwd()
    out = {}
    try:
        os.chdir(root)
        for tag, extra in (("t8", []), ("t8_legacy", ["--legacy-flat-cost"])):
            sys.argv = ["step3j_wfo_daily.py", "--pairs", "pairs.csv", "--closes", "closes.csv",
                        "--execution-model", "flat", "--tag", tag, *extra]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = step3j_wfo_daily.main()
            out[tag] = (res, buf.getvalue(), sys.argv[1:])
    finally:
        os.chdir(cwd)
        sys.argv = argv
    return root, out


def test_run_header_prints_the_full_cost_model(cli_runs):
    _, out = cli_runs
    text = out["t8"][1]
    assert "costs:      FLAT" in text
    for needle in ("per fill per leg", "10.0 bp", "2·(c + s)·(1 + |β|)", "FX conversion 2.0 bp",
                   "*_USD", "borrow", "80.0 bp/yr", "300.0 bp/yr", "short leg", "notional"):
        assert needle in text, needle
    assert "DAILY WFO SUMMARY — NET flat 10.0 bp/leg/fill + FX 2.0 bp" in text
    legacy = out["t8_legacy"][1]
    assert "LEGACY" in legacy and "20.0 bp round trip per trade" in legacy
    assert "DAILY WFO SUMMARY — NET 5.0 bps/leg" in legacy               # the pre-T8 label


def test_meta_sidecar_records_cost_model_config_hash_git_universe_and_headline(cli_runs):
    import step3j_wfo_daily as cli
    root, out = cli_runs
    for tag in ("t8", "t8_legacy"):
        res, _, args = out[tag]
        meta = json.loads((root / "data" / f"wfo_daily_meta_{tag}.json").read_text())
        ns = cli.build_parser().parse_args(args)
        cfg = WfoConfig(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(WfoConfig)})
        assert meta["config"] == json.loads(json.dumps(dataclasses.asdict(cfg)))
        assert meta["cost_model"] == json.loads(json.dumps(core.cost_model(cfg)))
        pairs_file = root / "data" / "pairs.csv"
        assert meta["pairs_file"] == "pairs.csv"
        assert meta["pairs_sha256"] == hashlib.sha256(pairs_file.read_bytes()).hexdigest()
        run_pairs = pd.read_csv(pairs_file)["pair"].tolist()          # US-only: every pair runs
        assert meta["config_hash"] == cli.config_hash(cfg, pairs_file, "closes.csv", run_pairs, False)
        assert meta["headline"] is False and meta["headline"] == res.stats["headline"]
        assert isinstance(meta["git_revision"], str) and meta["git_revision"]
        assert meta["legacy_options"] == ([] if tag == "t8" else ["--legacy-flat-cost"])
        assert meta["files"] == {"trades": f"wfo_daily_results_{tag}.csv",
                                 "mtm": f"wfo_daily_mtm_{tag}.csv",
                                 "mtm_pairs": f"wfo_daily_mtm_pairs_{tag}.csv"}
        assert all((root / "data" / f).exists() for f in meta["files"].values())
