"""Shared fixtures: an isolated runs/ and reports/ directory per test."""

import os

import pytest

from eval import constants as c

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    """Point RUNS_DIR and REPORTS_DIR at a temporary directory.

    Parameters
    ----------
    tmp_path: Path
        pytest temporary directory.
    monkeypatch: MonkeyPatch
        pytest monkeypatch.

    Returns
    -------
    Path
        Temporary runs directory.
    """
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    runs.mkdir()
    monkeypatch.setattr(c, "RUNS_DIR", runs)
    monkeypatch.setattr(c, "REPORTS_DIR", reports)
    return runs
