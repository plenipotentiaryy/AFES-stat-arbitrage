"""Data layer of cross-listed legs (audit task T2): exchange close times,
stale-print masking, FX at the local close, union-calendar panels."""
import os
import sys
from datetime import time

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cross_listing import (  # noqa: E402
    Exchange, assemble_panel, build_cross_listed_usd_at_close, close_time,
    close_timestamps, mask_stale_prints, repeats_previous_print,
)

ET = "America/New_York"


@pytest.mark.parametrize("ticker,exchange", [
    ("SLB",         Exchange("NYSE/Nasdaq", "America/New_York", time(16, 0), True, "North America")),
    ("RY.TO_USD",   Exchange("TSX", "America/Toronto", time(16, 0), True, "North America")),
    ("RY.TO",       Exchange("TSX", "America/Toronto", time(16, 0), True, "North America")),
    ("BHP.AX_USD",  Exchange("ASX", "Australia/Sydney", time(16, 0), False, "Asia-Pacific")),
    ("SAP.DE_USD",  Exchange("Xetra", "Europe/Berlin", time(17, 30), True, "Europe")),
    ("SAN.PA_USD",  Exchange("Euronext Paris", "Europe/Paris", time(17, 30), True, "Europe")),
    ("NOVN.SW_USD", Exchange("SIX", "Europe/Zurich", time(17, 30), True, "Europe")),
    ("SHEL.L_USD",  Exchange("LSE", "Europe/London", time(16, 30), True, "Europe")),
    ("7203.T",      Exchange("TSE", "Asia/Tokyo", time(15, 0), False, "Asia-Pacific")),
])
def test_close_time_by_suffix(ticker, exchange):
    assert close_time(ticker) == exchange


@pytest.mark.parametrize("ticker", ["XL0.SY", "XL0.SY_USD", "BRK.B"])
def test_unknown_suffix_raises_key_error(ticker):
    with pytest.raises(KeyError, match=r"cross_listing\.EXCHANGES") as err:
        close_time(ticker)
    assert ticker in str(err.value)


def test_close_timestamps_are_dst_correct_new_york_times():
    def et(*stamps):
        return [pd.Timestamp(s, tz=ET) for s in stamps]

    # ASX 16:00 Sydney: AEST/EDT in July → 02:00 ET; AEDT/EDT before Sydney's
    # April switch → 01:00 ET; AEDT/EST in December → 00:00 ET (same date).
    asx = close_timestamps("BHP.AX_USD", pd.DatetimeIndex(["2021-07-14", "2021-03-31", "2021-12-15"]))
    assert str(asx.tz) == ET
    assert list(asx) == et("2021-07-14 02:00", "2021-03-31 01:00", "2021-12-15 00:00")
    # Xetra 17:30 Berlin: 11:30 ET in July and December; 12:30 ET between the US
    # (14 Mar 2021) and the EU (28 Mar 2021) switch to summer time.
    xetra = close_timestamps("SAP.DE_USD", pd.DatetimeIndex(["2021-07-14", "2021-12-15", "2021-03-17"]))
    assert list(xetra) == et("2021-07-14 11:30", "2021-12-15 11:30", "2021-03-17 12:30")
    assert list(close_timestamps("SLB", pd.DatetimeIndex(["2021-03-17"]))) == et("2021-03-17 16:00")
    # An explicit exchange (the synthetic one: ET, close 16:00 − 15 h).
    sy = Exchange("SY", ET, time(1, 0), False, "synthetic")
    assert list(close_timestamps("XL0.SY", pd.DatetimeIndex(["2021-07-14"]), exchange=sy)) == \
        et("2021-07-14 01:00")


def test_mask_stale_prints_drops_unchanged_closes_without_volume():
    days = pd.bdate_range("2021-01-04", periods=6)
    closes = pd.DataFrame({
        "A": [10.0, 10.0, 10.0, 11.0, np.nan, 11.0],
        "B": [5.0, 5.0, 5.0, 5.5, 5.5, 5.5],
        "C": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0],          # no volume series
    }, index=days)
    volumes = pd.DataFrame({                             # no row for the last day
        "A": [100.0, 0.0, 50.0, 10.0, 10.0],
        "B": [100.0, np.nan, 20.0, 0.0, 30.0],
    }, index=days[:-1])
    before = closes.copy()
    out = mask_stale_prints(closes, volumes)
    expected = pd.DataFrame({
        # unchanged + volume 0 → no print; unchanged + volume 50 → a print; the
        # last day equals the previous non-NaN close (11.0) and has no volume row.
        "A": [10.0, np.nan, 10.0, 11.0, np.nan, np.nan],
        # unchanged + volume NaN → no print; unchanged + volume 30 → a print.
        "B": [5.0, np.nan, 5.0, 5.5, 5.5, np.nan],
        # no volume series: every unchanged close is no print.
        "C": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
    }, index=days)
    pd.testing.assert_frame_equal(out, expected)
    pd.testing.assert_frame_equal(closes, before)        # input untouched
    # Without volumes every exactly unchanged close is no print.
    pd.testing.assert_frame_equal(mask_stale_prints(closes), pd.DataFrame({
        "A": [10.0, np.nan, np.nan, 11.0, np.nan, np.nan],
        "B": [5.0, np.nan, np.nan, 5.5, np.nan, np.nan],
        "C": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
    }, index=days))


def test_repeats_previous_print_is_an_exact_zero_return_against_the_last_print():
    s = pd.Series([1.0, 1.0, np.nan, 1.0, 1.5, 1.5 + 1e-12, 1.5 + 1e-12])
    assert repeats_previous_print(s).tolist() == [False, True, False, True, False, False, True]
    frame = pd.DataFrame({"A": s, "B": s[::-1].to_numpy()})
    pd.testing.assert_series_equal(repeats_previous_print(frame)["A"], repeats_previous_print(s),
                                   check_names=False)


# ── Corrected builder: FX at the local close time ───────────────────────────

FX_GRID = pd.date_range(pd.Timestamp("2021-07-12", tz=ET), pd.Timestamp("2021-07-17", tz=ET),
                        freq="30min", inclusive="left")
# Every quote is distinct: the rate tells when it was observed.
FX = pd.DataFrame({"EURUSD=X": 1.10 + np.arange(len(FX_GRID)) * 1e-4}, index=FX_GRID)
LOCAL_DAYS = pd.DatetimeIndex(["2021-07-12", "2021-07-13", "2021-07-15", "2021-07-16"])  # 14th: holiday


def fx_at(stamp: str) -> float:
    return FX.at[pd.Timestamp(stamp, tz=ET), "EURUSD=X"]


def test_builder_converts_each_print_with_the_fx_quote_at_its_own_close():
    local = pd.DataFrame({"SAP.DE": [100.0, 101.0, 102.0, np.nan]}, index=LOCAL_DAYS)
    usd, unaligned = build_cross_listed_usd_at_close(
        local, FX, {"SAP": ("SAP.DE", "EURUSD=X")}, "usd_per_local")
    assert unaligned == {"SAP.DE": False}
    assert usd.index.equals(LOCAL_DAYS)                 # the local leg's own dates
    # Xetra closes at 17:30 Berlin = 11:30 ET in July.
    expected = [100.0 * fx_at("2021-07-12 11:30"), 101.0 * fx_at("2021-07-13 11:30"),
                102.0 * fx_at("2021-07-15 11:30"), np.nan]
    np.testing.assert_array_equal(usd["SAP.DE_USD"].to_numpy(), expected)
    # Not at the (later) daily FX close.
    assert usd.at[LOCAL_DAYS[0], "SAP.DE_USD"] != 100.0 * fx_at("2021-07-12 17:00")
    # On the union calendar the holiday stays NaN; nothing is filled.
    us = pd.DataFrame({"SAP": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=pd.bdate_range("2021-07-12", periods=5))
    panel = assemble_panel(us, usd)
    assert panel.index.equals(us.index)
    assert np.isnan(panel.at[pd.Timestamp("2021-07-14"), "SAP.DE_USD"])
    assert np.isnan(panel.at[pd.Timestamp("2021-07-16"), "SAP.DE_USD"])


def test_builder_local_per_usd_quotes_and_explicit_exchange():
    days = pd.DatetimeIndex(["2021-07-12", "2021-07-13"])
    local = pd.DataFrame({"RY.TO": [130.0, 131.0], "XL0.SY": [50.0, 51.0]}, index=days)
    fx = FX.rename(columns={"EURUSD=X": "USDCAD=X"}).assign(**{"SYUSD=X": FX["EURUSD=X"] / 2})
    adrs = {"RY": ("RY.TO", "USDCAD=X"), "XU0": ("XL0.SY", "SYUSD=X")}
    sy = Exchange("SY", ET, time(1, 0), False, "synthetic")
    usd, unaligned = build_cross_listed_usd_at_close(local, fx[["USDCAD=X"]], {"RY": adrs["RY"]},
                                                     "local_per_usd")
    assert unaligned == {"RY.TO": False}
    # TSX closes at 16:00 Toronto = 16:00 ET; USDCAD is CAD per USD → divide.
    np.testing.assert_array_equal(usd["RY.TO_USD"].to_numpy(),
                                  [130.0 / fx_at("2021-07-12 16:00"), 131.0 / fx_at("2021-07-13 16:00")])
    usd, _ = build_cross_listed_usd_at_close(local, fx, {"XU0": adrs["XU0"]}, "usd_per_local",
                                             exchanges={"XL0.SY": sy})
    np.testing.assert_array_equal(usd["XL0.SY_USD"].to_numpy(),
                                  [50.0 * (fx_at("2021-07-12 01:00") / 2), 51.0 * (fx_at("2021-07-13 01:00") / 2)])


def test_builder_leaves_a_print_unconverted_when_the_fx_quote_is_too_old():
    stale = FX[~((FX.index >= pd.Timestamp("2021-07-13 10:30", tz=ET))
                 & (FX.index < pd.Timestamp("2021-07-13 14:00", tz=ET)))]   # quotes stop at 10:00
    local = pd.DataFrame({"SAP.DE": [100.0, 101.0]}, index=LOCAL_DAYS[:2])
    adrs = {"SAP": ("SAP.DE", "EURUSD=X")}
    usd, _ = build_cross_listed_usd_at_close(local, stale, adrs, "usd_per_local")   # max age 1 h
    assert np.isnan(usd.at[LOCAL_DAYS[1], "SAP.DE_USD"])
    assert usd.at[LOCAL_DAYS[0], "SAP.DE_USD"] == 100.0 * fx_at("2021-07-12 11:30")
    usd, _ = build_cross_listed_usd_at_close(local, stale, adrs, "usd_per_local", fx_max_age="2h")
    assert usd.at[LOCAL_DAYS[1], "SAP.DE_USD"] == 101.0 * fx_at("2021-07-13 10:00")


@pytest.mark.parametrize("fx", [None, FX.rename(columns={"EURUSD=X": "GBPUSD=X"})],
                         ids=["no_intraday_fx", "fx_column_missing"])
def test_builder_without_intraday_fx_writes_no_column_and_flags_the_leg(fx):
    # Never a fallback to a daily FX close.
    local = pd.DataFrame({"SAP.DE": [100.0, 101.0]}, index=LOCAL_DAYS[:2])
    usd, unaligned = build_cross_listed_usd_at_close(
        local, fx, {"SAP": ("SAP.DE", "EURUSD=X")}, "usd_per_local")
    assert unaligned == {"SAP.DE": True}
    assert list(usd.columns) == [] and usd.index.equals(LOCAL_DAYS[:2])


def test_assemble_panel_is_the_union_calendar_without_fill():
    a = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.DatetimeIndex(["2021-01-05", "2021-01-04"]))
    b = pd.DataFrame({"B": [3.0, 4.0]}, index=pd.DatetimeIndex(["2021-01-06", "2021-01-04"]))
    panel = assemble_panel(a, b)
    expected = pd.DataFrame({"A": [2.0, 1.0, np.nan], "B": [4.0, np.nan, 3.0]},
                            index=pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06"]))
    pd.testing.assert_frame_equal(panel, expected, check_freq=False)


@pytest.mark.parametrize("fx", [
    FX.set_axis(FX.index.tz_localize(None)),            # quote times without a time zone
    FX.iloc[::-1],                                       # not increasing
], ids=["tz_naive", "unsorted"])
def test_builder_rejects_fx_quotes_it_cannot_place_in_time(fx):
    local = pd.DataFrame({"SAP.DE": [100.0, 101.0]}, index=LOCAL_DAYS[:2])
    with pytest.raises(ValueError):
        build_cross_listed_usd_at_close(local, fx, {"SAP": ("SAP.DE", "EURUSD=X")}, "usd_per_local")
