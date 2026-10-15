# HOLDOUT_USES

Every run that may use the final holdout (dates from 2024-01-01,
`wfo_daily_core.HOLDOUT_START`) needs `--final-holdout`; `step3j_wfo_daily.py`
appends one line here BEFORE such a run starts: UTC time | git revision
(`+dirty` = uncommitted changes to tracked files other than TRIALS.csv and this
file) | command line | config hash (as in TRIALS.csv).  Only the CLI records:
a direct run_wfo_daily(final_holdout=True) call is not recorded here.

