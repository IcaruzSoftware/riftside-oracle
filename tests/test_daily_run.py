"""daily_run: every part is attempted, any failure fails the run."""

from pathlib import Path

import pytest


@pytest.fixture
def daily_run(monkeypatch):
    sys_path_scripts = Path(__file__).parent.parent / "src" / "scripts"
    monkeypatch.syspath_prepend(str(sys_path_scripts))
    import daily_run
    return daily_run


def test_daily_run_attempts_every_part(daily_run, monkeypatch):
    calls = []

    def boom(*a):
        calls.append("evaluate")
        raise RuntimeError("supabase down")
    monkeypatch.setattr(daily_run, "run_evaluate", boom)
    monkeypatch.setattr(daily_run, "run_state_update", lambda d, dr: calls.append("state"))
    monkeypatch.setattr(daily_run, "run_predict", lambda d, dr: calls.append("predict"))
    assert daily_run.main([]) == 1
    assert calls == ["evaluate", "state", "predict"]


def test_daily_run_state_failure_still_predicts(daily_run, monkeypatch):
    calls = []
    monkeypatch.setattr(daily_run, "run_evaluate", lambda dr: calls.append("evaluate"))

    def blocked(*a):
        raise RuntimeError("OE download failed (Quota exceeded)")
    monkeypatch.setattr(daily_run, "run_state_update", blocked)
    monkeypatch.setattr(daily_run, "run_predict", lambda d, dr: calls.append("predict"))
    assert daily_run.main([]) == 1
    assert calls == ["evaluate", "predict"]
    calls.clear()
    assert daily_run.main(["--predict"]) == 0 and calls == ["predict"]
