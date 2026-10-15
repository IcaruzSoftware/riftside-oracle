"""evaluate with a FakeSupabase: winner/score/correct, unresolved warnings, running_record."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import pipeline.evaluate as evaluate
from pipeline import lolesports

FIXTURES = Path(__file__).parent / "fixtures" / "lolesports"
NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def _details(t1_wins, t2_wins, bo=5, outcomes=None):
    event = json.loads((FIXTURES / "event_details.json").read_text())["data"]["event"]
    event["match"]["strategy"]["count"] = bo
    for t, w, o in zip(event["match"]["teams"], (t1_wins, t2_wins), outcomes or (None, None)):
        t["result"] = {"gameWins": w, **({"outcome": o} if o else {})}
    return event                                      # teams: T1, KT


def _pred(match_id, start, predicted="T1", match_date=None, **kw):
    return {"id": match_id, "match_id": match_id, "start_time_utc": start, "team1_code": "T1",
            "team2_code": "KT", "predicted_team": predicted,
            "match_date": match_date or start[:10], "actual_winner": None, "correct": None, **kw}


@pytest.fixture
def db(fake_supabase, monkeypatch):
    details = {"won": _details(3, 2), "lost": _details(1, 3), "open": _details(1, 0),
               "stale": _details(0, 0)}
    monkeypatch.setattr(lolesports, "get_event_details", lambda match_id: details[match_id])
    return fake_supabase({"predictions": [
        _pred("won", "2026-10-19T10:00:00+00:00"),
        _pred("lost", "2026-10-19T13:00:00+00:00"),
        _pred("open", "2026-10-20T11:00:00+00:00"),
        _pred("stale", "2026-10-10T11:00:00+00:00"),
        _pred("future", "2026-10-21T11:00:00+00:00"),        # not started: never queried
        _pred("old", "2026-10-01T10:00:00+00:00", actual_winner="T1", correct=True),  # before RECORD_START
    ]})


def _row(db, match_id):
    return db.table("predictions").select("*").eq("match_id", match_id).execute().data[0]


def test_resolves_winner_score_and_correct(db):
    evaluate.evaluate(db=db, now=NOW)
    assert {k: _row(db, "won")[k] for k in ("actual_winner", "score", "correct")} == \
        {"actual_winner": "T1", "score": "3-2", "correct": True}
    assert {k: _row(db, "lost")[k] for k in ("actual_winner", "score", "correct")} == \
        {"actual_winner": "KT", "score": "1-3", "correct": False}
    for m in ("open", "stale", "future"):
        assert _row(db, m)["actual_winner"] is None


def test_unresolved_after_7_days_warns(db, caplog):
    evaluate.evaluate(db=db, now=NOW)
    assert "T1 vs KT (stale) still unresolved" in caplog.text
    assert "(open) still unresolved" not in caplog.text


def test_running_record_from_record_start(db):
    evaluate.evaluate(db=db, now=NOW)
    rec = db.table("running_record").select("*").execute().data[0]
    assert (rec["total_correct"], rec["total_incorrect"], rec["accuracy"]) == (1, 1, 0.5)


def test_running_record_recomputed_without_new_results(fake_supabase):
    db = fake_supabase({"predictions": [_pred("a", "2026-10-16T10:00:00+00:00", correct=True,
                                              actual_winner="T1")]})
    evaluate.evaluate(db=db, now=NOW)
    assert db.table("running_record").select("*").execute().data[0]["total_correct"] == 1


def test_dry_run_writes_nothing(db):
    evaluate.evaluate(db=db, now=NOW, dry_run=True)
    assert db.writes == [] and _row(db, "won")["actual_winner"] is None


def test_match_result_prefers_outcome():
    assert evaluate.match_result(_details(0, 0, outcomes=("loss", "win"))) == ("KT", "0-0")
    assert evaluate.match_result(_details(2, 1, bo=3)) == ("T1", "2-1")
    assert evaluate.match_result(_details(1, 1, bo=3)) is None


def _with_events(monkeypatch, overrides):
    """get_event_details as in the db fixture, but overrides[match_id] (None or an exception) wins."""
    real = lolesports.get_event_details

    def get(match_id):
        if match_id not in overrides:
            return real(match_id)
        if isinstance(overrides[match_id], Exception):
            raise overrides[match_id]
        return overrides[match_id]
    monkeypatch.setattr(lolesports, "get_event_details", get)


def test_unknown_match_id_counts_as_unresolved(db, monkeypatch, caplog):
    # getEventDetails answers an unknown/removed id with data.event = null.
    _with_events(monkeypatch, {"stale": None})
    evaluate.evaluate(db=db, now=NOW)
    assert _row(db, "stale")["actual_winner"] is None
    assert "(stale) still unresolved" in caplog.text
    assert _row(db, "won")["actual_winner"] == "T1"


def test_one_failing_match_does_not_block_the_others(db, monkeypatch, caplog):
    _with_events(monkeypatch, {"won": RuntimeError("HTTP 500")})
    with pytest.raises(RuntimeError, match="evaluation failed for 1 match"):
        evaluate.evaluate(db=db, now=NOW)
    assert "won: evaluation failed: HTTP 500" in caplog.text
    assert _row(db, "won")["actual_winner"] is None and _row(db, "lost")["actual_winner"] == "KT"
    rec = db.table("running_record").select("*").execute().data[0]
    assert (rec["total_correct"], rec["total_incorrect"]) == (0, 1)
