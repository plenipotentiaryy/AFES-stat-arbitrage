"""Headline universe of the daily WFO CLI (audit task T2): pairs with an
fx_unaligned leg (any "*_USD" leg), a leg without NYSE session overlap (ASX,
TSE) or a European leg are excluded and listed in a separate, labelled table."""
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic  # noqa: E402
from step3j_wfo_daily import cross_listed_table, headline_exclusions, main as cli_main  # noqa: E402

PAIRS = ["SLB-HAL", "BHP-BHP.AX_USD", "RY-TD", "SAP-SAP.DE_USD", "RY-RY.TO_USD",
         "TM-7203.T", "SHEL-SHEL.L"]


def test_headline_excludes_fx_unaligned_no_overlap_and_european_legs():
    ex = headline_exclusions(PAIRS)
    assert list(ex.index) == ["BHP-BHP.AX_USD", "SAP-SAP.DE_USD", "RY-RY.TO_USD",
                              "TM-7203.T", "SHEL-SHEL.L"]
    assert [p for p in PAIRS if p not in ex.index] == ["SLB-HAL", "RY-TD"]
    reasons, exchanges = ex["reasons"], ex["exchanges"]
    assert "fx_unaligned" in reasons["BHP-BHP.AX_USD"] and "ASX" in reasons["BHP-BHP.AX_USD"]
    assert "fx_unaligned" in reasons["SAP-SAP.DE_USD"] and "Xetra" in reasons["SAP-SAP.DE_USD"]
    assert reasons["RY-RY.TO_USD"] == "fx_unaligned"          # TSX itself overlaps NYSE
    assert "TSE" in reasons["TM-7203.T"] and "fx_unaligned" not in reasons["TM-7203.T"]
    assert "LSE" in reasons["SHEL-SHEL.L"] and "fx_unaligned" not in reasons["SHEL-SHEL.L"]
    assert exchanges["BHP-BHP.AX_USD"] == "NYSE/Nasdaq + ASX"
    assert exchanges["RY-RY.TO_USD"] == "NYSE/Nasdaq + TSX"


def test_fx_unaligned_is_decided_by_the_leg_name():
    # Any "*_USD" leg, whatever columns a panel has; a US-suffix "_USD" leg too.
    ex = headline_exclusions(["SLB-HAL", "SLB-HAL_USD"])
    assert list(ex.index) == ["SLB-HAL_USD"]
    assert (ex.at["SLB-HAL_USD", "reasons"], ex.at["SLB-HAL_USD", "exchanges"]) == \
        ("fx_unaligned", "NYSE/Nasdaq + NYSE/Nasdaq")


def test_unknown_exchange_suffix_stops_the_headline_but_not_the_legacy_run():
    with pytest.raises(KeyError, match=r"cross_listing\.EXCHANGES") as err:
        headline_exclusions(["SLB-HAL", "SLB-BRK.B"])
    assert "BRK.B" in str(err.value)
    ex = headline_exclusions(["SLB-HAL", "SLB-BRK.B", "SLB-XL0.SY_USD"], unknown_ok=True)
    assert list(ex.index) == ["SLB-BRK.B", "SLB-XL0.SY_USD"]
    assert ex.at["SLB-BRK.B", "reasons"] == "unknown exchange (BRK.B)"
    assert ex.at["SLB-BRK.B", "exchanges"] == "NYSE/Nasdaq + unknown"
    assert ex.at["SLB-XL0.SY_USD", "reasons"] == "fx_unaligned; unknown exchange (XL0.SY_USD)"


def test_cross_listed_table_lists_the_pairs_and_labels_legacy_numbers():
    ex = headline_exclusions(PAIRS)
    text = cross_listed_table(ex)
    assert "CROSS-LISTED PAIRS (not in headline)" in text
    assert all(p in text for p in ex.index) and "SLB-HAL" not in text
    assert "NOT VALID" not in text
    trades = pd.DataFrame({"pair": ["BHP-BHP.AX_USD"] * 3 + ["SLB-HAL"],
                           "pnl": [0.01, 0.02, -0.01, 0.5]})
    text = cross_listed_table(ex, trades)
    assert "LEGACY — NOT VALID" in text and "SLB-HAL" not in text

    def numbers(pair: str) -> tuple[int, str, float]:
        """(trades, win rate, total PnL): the row's last three fields."""
        row = next(line for line in text.splitlines() if line.startswith(pair + " "))
        n, win, pnl = row.split()[-3:]
        return int(n), win, float(pnl)

    assert numbers("BHP-BHP.AX_USD") == (3, "66.7%", pytest.approx(0.02))
    assert numbers("SAP-SAP.DE_USD") == (0, "—", 0.0)          # no trades


def test_cli_runs_the_headline_universe_and_prints_the_cross_listed_table(tmp_path, monkeypatch, capsys):
    closes, _ = synthetic.ou_positive_control(n_years=3, n_pairs=2, seed=0)
    closes = closes.rename(columns={"OUB1": "OUB1.AX_USD"})
    pairs = ["OUA0-OUB0", "OUA1-OUB1.AX_USD"]
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    closes.to_csv(tmp_path / "data" / "closes.csv")
    pd.DataFrame({"pair": pairs}).to_csv(tmp_path / "data" / "pairs.csv", index=False)
    monkeypatch.chdir(tmp_path)

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["step3j_wfo_daily.py", "--pairs", "pairs.csv",
                                          "--closes", "closes.csv", "--execution-model", "flat",
                                          "--tag", "t", *extra])
        cli_main()
        return pd.read_csv(tmp_path / "data" / "wfo_daily_results_t.csv"), capsys.readouterr().out

    trades, out = run()
    assert set(trades["pair"]) == {"OUA0-OUB0"}
    assert "CROSS-LISTED PAIRS (not in headline)" in out and "OUA1-OUB1.AX_USD" in out
    assert "NOT VALID" not in out
    trades, out = run("--legacy-cross-listed")
    assert set(trades["pair"]) == set(pairs)
    assert "LEGACY — NOT VALID" in out
    # A leg with an unknown exchange suffix: the headline run stops, the
    # legacy run lists it.
    pd.DataFrame({"pair": pairs + ["OUA0-OUB0.XX"]}).to_csv(tmp_path / "data" / "pairs.csv", index=False)
    with pytest.raises(KeyError, match="OUB0.XX"):
        run()
    trades, out = run("--legacy-cross-listed")
    assert set(trades["pair"]) == set(pairs)                   # the pair has no prices
    assert "unknown exchange (OUB0.XX)" in out
