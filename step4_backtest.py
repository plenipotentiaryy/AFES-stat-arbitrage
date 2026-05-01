import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    ENTRY_Z, EXIT_Z, STOP_Z, ENTRY_Z_VOLATILE,
    COST_PER_SIDE, BORROW_RATE_ANNUAL, PAIR_MAX_LOSS,
    RTH_START, RTH_END, SIGNAL_START, RECENT_BARS,
    DATA_DIR, OUTPUT_DIR,
)

BARS_PER_TRADING_DAY = 26


def load_closes() -> pd.DataFrame:
    closes = pd.read_csv(DATA_DIR / "closes_15min.csv", index_col=0, parse_dates=True)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        closes.index = closes.index.tz_convert("US/Eastern")
    return closes.between_time(RTH_START, RTH_END).dropna().tail(RECENT_BARS)


def build_signals(closes, t1, t2, beta, half_life) -> pd.DataFrame:
    spread = closes[t1] - beta * closes[t2]
    window = max(20, min(int(half_life), 200))
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return pd.DataFrame({
        f"{t1}_close": closes[t1],
        f"{t2}_close": closes[t2],
        "spread": spread,
        "zscore": zscore,
    }).dropna().between_time(SIGNAL_START, RTH_END)


def backtest_pair(df, t1, t2, beta, regime_dict: dict | None = None) -> pd.DataFrame:
    t1_col, t2_col = f"{t1}_close", f"{t2}_close"
    position = 0
    entry_spread = entry_t1 = entry_t2 = 0.0
    entry_bar = 0
    cumulative_pnl = 0.0
    trades = []

    for i in range(len(df)):
        z          = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        t1_price   = df[t1_col].iloc[i]
        t2_price   = df[t2_col].iloc[i]

        if position != 0:
            exit_signal = (position == 1 and z > -EXIT_Z) or (position == -1 and z < EXIT_Z)
            stop_signal = (position == 1 and z < -STOP_Z) or (position == -1 and z > STOP_Z)

            if exit_signal or stop_signal:
                gross_pnl      = position * (spread_now - entry_spread)
                notional       = entry_t1 + beta * entry_t2
                tx_cost        = 2 * notional * COST_PER_SIDE
                holding_days   = (i - entry_bar) / BARS_PER_TRADING_DAY
                short_notional = (beta * entry_t2) if position == 1 else entry_t1
                borrow_cost    = short_notional * BORROW_RATE_ANNUAL * holding_days / 252
                net_pnl        = gross_pnl - tx_cost - borrow_cost
                cumulative_pnl += net_pnl

                trades.append({
                    "pair":         f"{t1}-{t2}",
                    "entry_time":   df.index[entry_bar],
                    "exit_time":    df.index[i],
                    "direction":    "LONG" if position == 1 else "SHORT",
                    "holding_bars": i - entry_bar,
                    "gross_pnl":    round(gross_pnl, 4),
                    "tx_cost":      round(tx_cost, 4),
                    "borrow_cost":  round(borrow_cost, 4),
                    "net_pnl":      round(net_pnl, 4),
                    "cum_pnl":      round(cumulative_pnl, 4),
                    "exit_reason":  "STOP" if stop_signal else "SIGNAL",
                    "entry_z":      round(df["zscore"].iloc[entry_bar], 2),
                    "exit_z":       round(z, 2),
                })
                position = 0

                # ── Auto-disable if pair is consistently losing ───────────
                if cumulative_pnl < PAIR_MAX_LOSS:
                    break

        if position == 0:
            is_volatile = regime_dict.get(df.index[i], 0) == 1 if regime_dict else False
            threshold   = ENTRY_Z_VOLATILE if is_volatile else ENTRY_Z
            if z < -threshold:
                position = 1
            elif z > threshold:
                position = -1
            if position != 0:
                entry_spread = spread_now
                entry_t1     = t1_price
                entry_t2     = t2_price
                entry_bar    = i

    return pd.DataFrame(trades)


# ── Load ─────────────────────────────────────────────────────────────────────
closes = load_closes()
pairs  = pd.read_csv(DATA_DIR / "pairs_selected.csv")

# Load per-pair regimes (from step5_regime.py)
# Format: wide CSV — index=timestamp, columns=pair names, values=0/1
regime_data: dict[str, dict] = {}
regimes_path = DATA_DIR / "regimes.csv"
if regimes_path.exists():
    reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    if reg_df.index.tz is None:
        reg_df.index = reg_df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        reg_df.index = reg_df.index.tz_convert("US/Eastern")
    for col in reg_df.columns:
        regime_data[col] = reg_df[col].to_dict()
    avg_vol = reg_df.mean().mean() * 100
    print(f"Regimes loaded for {len(regime_data)} pairs "
          f"(avg {avg_vol:.0f}% volatile bars) → entry_z={ENTRY_Z_VOLATILE} when volatile")
else:
    print("No regimes.csv — fixed entry_z (run step5_regime.py to enable HMM filter)")

if pairs.empty:
    raise SystemExit("pairs_selected.csv is empty — run step2_pairs.py first")

print(f"Trading {len(pairs)} pairs | {closes.shape[0]} bars per ticker")
print(f"Period: {closes.index[0]} — {closes.index[-1]}")
print(f"Pair max loss cutoff: {PAIR_MAX_LOSS}\n")

# ── Run backtest per pair ─────────────────────────────────────────────────────
pair_results = {}

for _, row in pairs.iterrows():
    t1, t2    = row["pair"].split("-")
    beta      = row["beta"]
    half_life = row["half_life_bars"]

    if t1 not in closes.columns or t2 not in closes.columns:
        print(f"  SKIP {row['pair']}: missing ticker data")
        continue

    df_sig = build_signals(closes, t1, t2, beta, half_life)
    pair_regime = regime_data.get(row["pair"])
    trades = backtest_pair(df_sig, t1, t2, beta, pair_regime)

    if trades.empty:
        print(f"  {row['pair']:12s}  0 trades")
        continue

    pnl      = trades["net_pnl"]
    win_rate = (pnl > 0).mean() * 100
    disabled = trades["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status   = " [DISABLED — max loss hit]" if disabled else ""

    pair_results[row["pair"]] = {"trades": trades, "signals": df_sig}
    print(f"  {row['pair']:12s}  trades={len(trades):3d}  "
          f"WR={win_rate:4.1f}%  net P&L={pnl.sum():+.4f}{status}")

if not pair_results:
    raise SystemExit("No trades generated.")

# ── Combine ───────────────────────────────────────────────────────────────────
df_trades = (pd.concat([v["trades"] for v in pair_results.values()])
               .sort_values("exit_time")
               .reset_index(drop=True))

pnl             = df_trades["net_pnl"]
winning         = df_trades[pnl > 0]
losing          = df_trades[pnl <= 0]
stops           = df_trades[df_trades["exit_reason"] == "STOP"]
cumulative      = pnl.cumsum()
max_drawdown    = (cumulative - cumulative.cummax()).min()
days_total      = pd.to_datetime(df_trades["exit_time"].iloc[-1]) - pd.to_datetime(df_trades["exit_time"].iloc[0])
trades_per_year = len(df_trades) / (days_total.days / 365.25)
sharpe          = (pnl.mean() / pnl.std() * np.sqrt(trades_per_year)
                   if pnl.std() > 0 else 0.0)
profit_factor   = (winning["net_pnl"].sum() / abs(losing["net_pnl"].sum())
                   if len(losing) > 0 and losing["net_pnl"].sum() != 0 else float("inf"))

print(f"\n{'='*60}")
print(f"PORTFOLIO  ({len(pair_results)} pairs)")
print(f"{'='*60}")
print(f"Trades:        {len(df_trades)}  ({trades_per_year:.0f}/yr)")
print(f"Win rate:      {len(winning)/len(df_trades)*100:.1f}%")
print(f"Stops:         {len(stops)}")
print(f"Gross P&L:     {df_trades['gross_pnl'].sum():+.4f}")
print(f"Costs:         {(df_trades['tx_cost']+df_trades['borrow_cost']).sum():.4f}")
print(f"Net P&L:       {pnl.sum():+.4f}")
print(f"Avg trade:     {pnl.mean():+.4f}")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown:  {max_drawdown:.4f}")
print(f"Sharpe:        {sharpe:.2f}")
print(f"Avg hold:      {df_trades['holding_bars'].mean():.0f} bars "
      f"({df_trades['holding_bars'].mean()/BARS_PER_TRADING_DAY:.1f} days)")

print(f"\n{'─'*65}")
print(f"{'Pair':<12} {'Trades':>6} {'WR':>6} {'Net P&L':>10} {'Sharpe':>7} {'AvgHold':>8} {'Status':>10}")
print(f"{'─'*65}")
for pair_name, data in pair_results.items():
    t  = data["trades"]
    p  = t["net_pnl"]
    wr = (p > 0).mean() * 100
    tpy = len(t) / (days_total.days / 365.25)
    sh  = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0.0
    ah  = t["holding_bars"].mean() / BARS_PER_TRADING_DAY
    disabled = t["cum_pnl"].iloc[-1] < PAIR_MAX_LOSS
    status = "DISABLED" if disabled else "active"
    print(f"{pair_name:<12} {len(t):>6} {wr:>5.1f}% {p.sum():>+10.4f} "
          f"{sh:>7.2f} {ah:>6.1f}d {status:>10}")

df_trades.to_csv(DATA_DIR / "trades.csv", index=False)
print(f"\nSaved {len(df_trades)} trades to {DATA_DIR / 'trades.csv'}")

# ── Charts ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
exit_times = pd.to_datetime(df_trades["exit_time"])

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(exit_times, cumulative.values, color="blue", lw=2, label="Portfolio net P&L")
ax.plot(exit_times, df_trades["gross_pnl"].cumsum().values,
        color="blue", lw=1, linestyle="--", alpha=0.35, label="Gross P&L")
ax.axhline(PAIR_MAX_LOSS, color="red", linestyle=":", lw=1, alpha=0.5, label=f"Max loss cutoff ({PAIR_MAX_LOSS})")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Portfolio Equity Curve")
ax.set_ylabel("Cumulative net P&L")
ax.legend()

ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 1, len(pair_results)))
for (pair_name, data), color in zip(pair_results.items(), colors):
    t  = data["trades"]
    et = pd.to_datetime(t["exit_time"])
    ax.plot(et, t["net_pnl"].cumsum().values, label=pair_name, color=color, lw=1.5)
ax.axhline(PAIR_MAX_LOSS, color="red", linestyle=":", lw=1, alpha=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Per-pair Equity Curves")
ax.set_ylabel("Cumulative net P&L")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_results.png", dpi=150)
print(f"Chart saved to {OUTPUT_DIR / 'backtest_results.png'}")
# plt.show()
