"""Print the placebo report table (the table each audit task pastes into
AUDIT_FIX_LOG.md).

    python tests/placebo_report.py

Per variant × dataset: mean gross Sharpe ± SE over seeds, pooled per-trade
t-stat, trade count and win rate; then the OU positive control.
"""
from __future__ import annotations

import placebo_harness as ph


def main():
    res = ph.results(ph.all_jobs())
    print(f"Gross (flat, cost=slip=borrow=0). Null seeds {ph.NULL_SEEDS.start}..{ph.NULL_SEEDS.stop - 1}, "
          f"control seeds {ph.CONTROL_SEEDS.start}..{ph.CONTROL_SEEDS.stop - 1}.")
    print("Pass: null mean Sharpe in [-0.2, +0.2] and |t| < 3; control mean Sharpe > 1.0.\n")
    print("| variant | dataset | seeds | mean Sharpe ± SE | per-trade t | trades | win rate |")
    print("|---|---|---|---|---|---|---|")
    for (variant, dataset), s in res.items():
        print(f"| {variant} | {dataset} | {len(s.sharpes)} | {s.mean_sharpe:+.3f} ± {s.se_sharpe:.3f} "
              f"| {s.tstat:+.1f} | {s.n_trades} | {s.win_rate:.1%} |")


if __name__ == "__main__":
    main()
