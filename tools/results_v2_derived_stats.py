"""T11 derived statistics of the two existing runs (no strategy evaluation, no
registry row): PSR(0), DSR at the final registry, sensitivity, turnover,
MaxDD, US block — the source of RESULTS_v2.md section 1 (section 6).

Run with cwd = output/audit_runs/t11/ (the T11 run dir; its data/ holds the
runs' trades / MTM / per-pair MTM files):

    cd output/audit_runs/t11 && python ../../../tools/results_v2_derived_stats.py

It prints the statistics (the T11 output is derived_stats.log in that dir).
Reads only: the run files and the repo-root TRIALS.csv (via
step3j_wfo_daily.registry_trials); writes nothing, appends no registry row.
(Committed in the final review, R53/M4, from output/audit_runs/t11/derived_stats.py
with the logic unchanged; only the paths now resolve from that run dir.)"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repo root
import perf_stats as ps, mtm
import step3j_wfo_daily as cli

RUN = "data"
CAP = 26 * 2.0
trials = cli.registry_trials()
srs = np.array(list(trials.values()), dtype=float)
known = srs[np.isfinite(srs)]
N, V = len(srs), float(np.var(known, ddof=1))
reg = pd.read_csv(cli.TRIALS_FILE, float_precision="round_trip", dtype={"config_hash": str})
print(f"registry: {len(reg)} rows, N = {N} distinct hashes, {len(known)} recorded Sharpe ratios, V = {V:.6g} (sd {math.sqrt(V):.5f}/day = {math.sqrt(V)*math.sqrt(252):.3f} ann.)")
k = (1 - ps.EULER_GAMMA) * stats.norm.ppf(1 - 1 / N) + ps.EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * math.e))
sr0 = ps.sr0_threshold(N, V)
print(f"SR0 = sqrt(V)*k(N) = {math.sqrt(V):.6f} * {k:.4f} = {sr0:.5f}/day = {sr0*math.sqrt(252):.3f} ann.")
trades = {}
for tag in ("default", "almgren"):
    m = mtm.read_mtm(f"{RUN}/wfo_daily_mtm_t11_{tag}.csv")
    ret = m["ret"].astype(float)
    T = len(ret); sr = ps.sharpe_per_period(ret)
    g3 = float(stats.skew(ret)); g4 = float(stats.kurtosis(ret, fisher=False))
    st = mtm.mtm_stats(m["pnl"], CAP)
    psr0 = ps.psr(ret, 0.0); dsr = ps.deflated_sharpe(ret, srs)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr)
    # sensitivity: V for DSR = 0.5 (SR0 = SR); DSR(V=0) = PSR(0)
    v_half = (sr / k) ** 2
    dsr_v0 = ps.psr(ret, ps.sr0_threshold(N, 0.0))
    # per-day Sharpe needed for DSR = 0.95 at this N, V, with T, g3, g4 fixed
    f = lambda s: stats.norm.cdf((s - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * s + (g4 - 1) / 4 * s * s)) - 0.95
    s95 = optimize.brentq(f, sr0, 1.0)
    print(f"\n[{tag}] T={T} span {m.index[0].date()} -> {m.index[-1].date()}  Sharpe(ann)={ps.portfolio_sharpe(ret):+.4f} (mtm_stats {st['sharpe']:+.4f}) per day {sr:+.6f}")
    print(f"  ann_return {st['ann_return']*100:+.3f}%  ann_vol {st['ann_vol']*100:.3f}%  MaxDD {st['max_dd']*100:.3f}% peak {st['max_dd_peak'].date()} trough {st['max_dd_trough'].date()}")
    print(f"  skew {g3:+.3f}  raw kurt {g4:.2f}  PSR(0) = {psr0:.4f}  DSR(N={N}, V={V:.5g}) = {dsr:.3g}  (z = {z:+.2f}; SR0 {sr0:.5f}/day)")
    print(f"  sensitivity: DSR at V = 0 (SR0 = 0) = {dsr_v0:.4f} = PSR(0); DSR > 0.5 needs V < {v_half:.3g} ({v_half/V:.4f} x registry V; sd {math.sqrt(v_half)*math.sqrt(252):.3f} ann.)")
    print(f"  per-day Sharpe needed for DSR = 0.95 at N={N}, V: {s95:.4f} ({s95*math.sqrt(252):.2f} ann.)")
    tr = pd.read_csv(f"{RUN}/wfo_daily_results_t11_{tag}.csv", float_precision="round_trip")
    trades[tag] = tr
    legs = (tr["size"] * ((tr["q1"] * tr["p1_entry"]).abs() + (tr["q2"] * tr["p2_entry"]).abs()
                          + (tr["q1"] * tr["p1_exit"]).abs() + (tr["q2"] * tr["p2_exit"]).abs()))
    years = T / 252
    print(f"  trades {len(tr)}  sizes {sorted(tr['size'].unique())}  max|q1*p1_entry - 1| {np.abs(tr['q1']*tr['p1_entry']-1).max():.2e}  mean|beta| {tr['beta'].abs().mean():.4f}")
    print(f"  turnover: sum |leg notional| over {4*len(tr)} fills = {legs.sum():.4f} USD per $1 of leg-1 notional; / capital {CAP} = {legs.sum()/CAP:.3f}; / years {years:.4f} = {legs.sum()/CAP/years:.3f} x capital per year")
    print(f"  trades: win {(tr['pnl']>0).mean()*100:.1f}%  total pnl {tr['pnl'].sum():+.4f}  gross {(tr['size']*tr['gross']).sum():.4f}  cost {(tr['size']*tr['cost']).sum():.4f}  borrow {(tr['size']*tr['borrow']).sum():.4f}")
    pp = mtm.read_mtm(f"{RUN}/wfo_daily_mtm_pairs_t11_{tag}.csv")
    blk = pp.sum(axis=1)
    bs = mtm.mtm_stats(blk, pp.shape[1] * 2.0)
    print(f"  US block ({pp.shape[1]} pairs, capital {pp.shape[1]*2.0}): Sharpe {bs['sharpe']:+.4f}  MaxDD {bs['max_dd']*100:.3f}%  max|block - portfolio| {np.abs(blk.values - m['pnl'].values).max():.2e}")
a, b = trades["default"], trades["almgren"]
cost_cols = {"pnl", "cost", "borrow", "cost_entry", "cost_exit", "borrow_rate1", "borrow_rate2"}
same = [c for c in a.columns if c not in cost_cols]
print("\nsame non-cost columns in both runs:", a[same].equals(b[same]), len(same), "columns")

# PnL concentration: trades by entry window; daily MTM PnL by calendar year (% of capital)
for tag, tr in trades.items():
    w = tr.groupby("window_entered")["pnl"].agg(["count", "sum"])
    print(f"[{tag}] window 19 (oos_start {tr.loc[tr.window_entered == 19, 'oos_start'].iloc[0]}): {int(w.loc[19, 'count'])} trades, net {w.loc[19, 'sum']:+.4f}; other windows {w.drop(19)['sum'].sum():+.4f}; total {tr['pnl'].sum():+.4f}")
    m = mtm.read_mtm(f"{RUN}/wfo_daily_mtm_t11_{tag}.csv")
    y = m["pnl"].groupby(m.index.year).sum() / CAP * 100
    print(f"  MTM PnL by year, % of capital: " + ", ".join(f"{k} {v:+.2f}" for k, v in y.items()))
    r = m["ret"].astype(float)
    print(f"  skew {stats.skew(r):+.5f}  raw kurtosis {stats.kurtosis(r, fisher=False):.4f}")
