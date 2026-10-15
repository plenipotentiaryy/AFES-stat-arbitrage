"""Session-wide test setup.

Audit T6: every CLI run in the suite (step3j_wfo_daily.main() in a temp run
dir) would append to the committed repo-root TRIALS.csv / HOLDOUT_USES.md.
Both paths point to a session temp dir before any test or fixture runs (a
session-scoped autouse fixture is set up before module-scoped ones); tests
that count rows redirect them again to their own tmp_path."""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True, scope="session")
def registry_files_in_tmp(tmp_path_factory):
    import step3j_wfo_daily
    reg = tmp_path_factory.mktemp("registry")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(step3j_wfo_daily, "TRIALS_FILE", reg / "TRIALS.csv")
        mp.setattr(step3j_wfo_daily, "HOLDOUT_USES_FILE", reg / "HOLDOUT_USES.md")
        yield reg
