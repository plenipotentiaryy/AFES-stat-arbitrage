"""Regenerate the T0 golden fixtures: trades of the PRE-refactor CLI on
small synthetic panels.

    python tests/fixtures/t0_golden/make_golden.py

`step3j_wfo_daily.py` and its helper modules are extracted as of commit
46ffdfb (`git show`) into a temp dir, which is the subprocess's only
PYTHONPATH entry, so later changes to the repo do not reach the fixtures.
For every case in CASES the case's synthetic inputs (closes, pairs and, if
the case uses them, volumes / VWAP) are written as CSVs into a temp run dir,
that CLI runs there, and its `wfo_daily_results_<case>.csv` is copied next
to this script.  A case without trades has no fixture file.  Every case
must change the trades of its panel's default run (the option binds),
except those listed in SAME_AS_BASELINE.

`tests/test_wfo_core_identity.py` rebuilds the same inputs with
`case_inputs` and checks that `run_wfo_daily` reproduces each fixture byte
for byte.  The fixtures are the permanent "legacy reproduces the pre-audit
code" anchor.  They also depend on the library versions recorded in
tests/README.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import synthetic  # noqa: E402

PRE_REFACTOR = "46ffdfb"
# Repo modules the 46ffdfb CLI imports; extracted from that commit with it.
HELPERS = ["config.py", "kalman.py", "metagate_daily.py", "hrp_weights.py",
           "almgren_chriss_repricing.py"]
N_YEARS = 6


def base_panel() -> tuple[pd.DataFrame, list[str]]:
    """4 OU pairs, 1 random-walk pair and 1 momentum (losing) pair over
    6 years, plus a pair sharing tickers with two others and a pair with a
    ticker absent from the panel; OUB3 is listed late and misses 15% of its
    prints, OUA1 has two volatility bursts."""
    ou, ou_pairs = synthetic.ou_positive_control(n_years=N_YEARS, n_pairs=4, seed=11)
    rw, rw_pairs = synthetic.independent_rw(n_years=N_YEARS, n_pairs=1, seed=12)
    mom, _ = synthetic.independent_rw(n_years=N_YEARS, n_pairs=1, seed=15)
    rng = np.random.default_rng(13)
    # Log-spread with AR(1) increments (φ = 0.8): moves keep going → the
    # mean-reversion rules lose on this pair.
    spread = np.cumsum(lfilter([1.0], [1.0, -0.8], rng.normal(0.0, 0.004, len(mom))))
    mom = pd.DataFrame({"MOA0": mom["RWB0"] * np.exp(spread), "MOB0": mom["RWB0"]})
    closes = pd.concat([ou, rw, mom], axis=1)
    closes.loc[:"2006-12-31", "OUB3"] = np.nan
    closes.loc[rng.random(len(closes)) < 0.15, "OUB3"] = np.nan
    for start, end in [("2007-03-01", "2007-04-15"), ("2009-02-01", "2009-03-15")]:
        burst = (closes.index >= start) & (closes.index <= end)
        closes.loc[burst, "OUA1"] *= np.exp(rng.normal(0.0, 0.05, burst.sum()))
    return closes, ou_pairs + rw_pairs + ["MOA0-MOB0", "OUA0-RWB0", "OUA2-GONE"]


def spy_panel() -> tuple[pd.DataFrame, list[str]]:
    """Base panel plus a SPY column (the HMM macro proxy when present)."""
    closes, pairs = base_panel()
    spy, _ = synthetic.independent_rw(n_years=N_YEARS, n_pairs=1, seed=14)
    return closes.assign(SPY=spy["RWA0"]), pairs


def aussie_panel() -> tuple[pd.DataFrame, list[str]]:
    """Base panel with three legs renamed to .AX_USD (Aussie-filter pairs)."""
    closes, pairs = base_panel()
    ren = {"OUB0": "OUB0.AX_USD", "OUB1": "OUB1.AX_USD", "RWB0": "RWB0.AX_USD"}
    pairs = ["-".join(ren.get(t, t) for t in p.split("-")) for p in pairs]
    return closes.rename(columns=ren), pairs


PANELS = {"base": base_panel, "spy": spy_panel, "aussie": aussie_panel}


def volumes_for(closes: pd.DataFrame) -> pd.DataFrame:
    """Synthetic daily share volumes (lognormal around 1M); the last ticker
    has none, as foreign legs lack volumes in the real data."""
    rng = np.random.default_rng(21)
    return pd.DataFrame(rng.lognormal(np.log(1e6), 0.5, size=closes.shape),
                        index=closes.index, columns=closes.columns).iloc[:, :-1]


def vwap_for(closes: pd.DataFrame) -> pd.DataFrame:
    """Synthetic daily VWAPs: the close times a 30 bp lognormal error."""
    rng = np.random.default_rng(22)
    return closes * np.exp(rng.normal(0.0, 0.003, size=closes.shape))


# case → CLI args (+ panel, and whether volumes_daily.csv / vwap_daily.csv exist).
CASES: dict[str, dict] = {
    "flat":                 {"args": ["--execution-model", "flat"]},
    "flat_zero_cost":       {"args": ["--execution-model", "flat", "--cost", "0",
                                      "--slip", "0", "--borrow", "0"]},
    "almgren":              {"args": []},
    "almgren_no_volumes":   {"args": [], "volumes": False},
    "kalman":               {"args": ["--kalman"]},
    "vwz":                  {"args": ["--vwz"]},
    "hmm_block_panic":      {"args": ["--hmm", "block-panic"], "panel": "spy"},
    "hmm_size_panic":       {"args": ["--hmm", "size-panic"], "panel": "spy"},
    "hmm_block_calm":       {"args": ["--hmm", "block-calm"]},
    "hmm_extreme_3state":   {"args": ["--hmm", "extreme-3state"], "panel": "spy"},
    "cusum":                {"args": ["--cusum", "5"]},
    "cusum_10":             {"args": ["--cusum", "10"]},
    "cusum_30":             {"args": ["--cusum", "30"]},
    "vol_tight_stop":       {"args": ["--vol-tight-stop"]},
    "vwap_exec":            {"args": ["--vwap-exec"], "vwap": True},
    "aussie_filter":        {"args": ["--aussie-filter"], "panel": "aussie"},
    "ivol":                 {"args": ["--ivol"]},
    "volvol":               {"args": ["--volvol"]},
    "pairmom":              {"args": ["--pairmom"]},
    "rotation":             {"args": ["--rotation"]},
    "voltarget":            {"args": ["--voltarget", "0.10"]},
    "maxconc":              {"args": ["--maxconc", "3"]},
    "ticker_overlap":       {"args": ["--ticker-overlap"]},
    "per_pair_quarter_cap": {"args": ["--per-pair-quarter-cap", "2"]},
    "start_end":            {"args": ["--start", "2006-03-15", "--end", "2010-06-30"]},
    "train12_oos6_step6":   {"args": ["--train", "12", "--oos", "6", "--step", "6"]},
}

# Cases allowed to reproduce their panel's default run (no options), with the reason.
SAME_AS_BASELINE = {
    "almgren": "it is the base panel's default run",
}


# The 46ffdfb CLI's default --execution-model.  Audit T8 made `flat` the
# default, so a spec without the option names this one: the same trades from
# the pre-refactor CLI, and the same config from the current one.
PRE_AUDIT_MODEL = ["--execution-model", "almgren"]


def inputs(spec: dict) -> dict:
    """closes, pairs, volumes (or None), vwap (or None) and CLI args of a spec
    (with PRE_AUDIT_MODEL unless the spec picks a model)."""
    closes, pairs = PANELS[spec.get("panel", "base")]()
    args = spec["args"] if "--execution-model" in spec["args"] else [*PRE_AUDIT_MODEL, *spec["args"]]
    return {
        "closes": closes,
        "pairs": pairs,
        "volumes": volumes_for(closes) if spec.get("volumes", True) else None,
        "vwap": vwap_for(closes) if spec.get("vwap", False) else None,
        "args": args,
    }


def case_inputs(case: str) -> dict:
    """closes, pairs, volumes (or None), vwap (or None) and CLI args of a case."""
    return inputs(CASES[case])


def fixture_path(case: str) -> Path:
    return HERE / f"wfo_daily_results_{case}.csv"


def run_old_cli(src: Path, run: Path, inp: dict, tag: str) -> bytes:
    """Run the 46ffdfb CLI in `src` on `inp` in run dir `run`; return its
    trades CSV (b"" when it found no trades)."""
    (run / "data").mkdir(parents=True)
    (run / "output").mkdir()
    inp["closes"].to_csv(run / "data" / "closes.csv")
    pd.DataFrame({"pair": inp["pairs"]}).to_csv(run / "data" / "pairs.csv", index=False)
    if inp["volumes"] is not None:
        inp["volumes"].to_csv(run / "data" / "volumes_daily.csv")
    if inp["vwap"] is not None:
        inp["vwap"].to_csv(run / "data" / "vwap_daily.csv")
    env = {**os.environ, "PYTHONPATH": str(src)}
    env.setdefault("MPLCONFIGDIR", str(run / "mpl"))
    proc = subprocess.run(
        [sys.executable, str(src / "step3j_wfo_daily.py"), "--pairs", "pairs.csv",
         "--closes", "closes.csv", "--tag", tag, *inp["args"]],
        cwd=run, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{tag}: pre-refactor CLI failed\n{proc.stdout}\n{proc.stderr}")
    produced = run / "data" / f"wfo_daily_results_{tag}.csv"
    return produced.read_bytes() if produced.exists() else b""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir()
        for name in ["step3j_wfo_daily.py", *HELPERS]:
            (src / name).write_bytes(subprocess.run(
                ["git", "show", f"{PRE_REFACTOR}:{name}"],
                cwd=ROOT, check=True, capture_output=True).stdout)
        trades = {case: run_old_cli(src, tmp / case, case_inputs(case), case) for case in CASES}
        baseline = {panel: run_old_cli(src, tmp / f"baseline_{panel}",
                                       inputs({"args": [], "panel": panel}), f"baseline_{panel}")
                    for panel in PANELS}
    for case, spec in CASES.items():
        if case not in SAME_AS_BASELINE and trades[case] == baseline[spec.get("panel", "base")]:
            raise SystemExit(f"{case}: same trades as the default run on its panel, so the "
                             "case does not exercise its option — fixtures not updated")
    for case, data in trades.items():
        n_trades = data.count(b"\n") - 1 if data else 0
        if data:
            fixture_path(case).write_bytes(data)
        else:
            fixture_path(case).unlink(missing_ok=True)
        print(f"{case:22s} {n_trades:5d} trades")


if __name__ == "__main__":
    main()
