"""Evaluate: the running record counts only predictions from RECORD_START onward."""

from datetime import datetime

import pipeline.evaluate as ev


def test_running_record_excludes_pre_record_start(monkeypatch, fake_supabase):
    seed = {
        "predictions": [
            # pending current-season game (will be resolved this run)
            {"id": "a", "game_id": "0022600001", "game_date": "2026-11-01",
             "predicted_team": "BOS", "home_team": "BOS", "away_team": "DET",
             "actual_winner": None, "correct": None},
            # already-resolved current-season game
            {"id": "b", "game_id": "0022600002", "game_date": "2026-11-02",
             "predicted_team": "LAL", "actual_winner": "LAL", "correct": True},
            # pre-RECORD_START seeded Finals rows — must be excluded from the record
            {"id": "c", "game_id": "0042500401", "game_date": "2026-06-03",
             "predicted_team": "NYK", "actual_winner": "NYK", "correct": True},
            {"id": "d", "game_id": "0042500403", "game_date": "2026-06-08",
             "predicted_team": "NYK", "actual_winner": "SAS", "correct": False},
        ],
        "running_record": [],
    }
    db = fake_supabase(seed)
    monkeypatch.setattr(ev, "_fetch_results_cdn", lambda ids: {"0022600001": "BOS"})

    ev.evaluate(dry_run=False, db=db)

    rec = [w[2] for w in db.writes if w[0] == "running_record" and w[1] == "upsert"][-1]
    # Counted: a (BOS==BOS, correct) + b (correct) = 2/2. Finals c/d excluded.
    assert rec["total_correct"] == 2
    assert rec["total_incorrect"] == 0
    assert rec["accuracy"] == 1.0


def test_dry_run_writes_nothing(monkeypatch, fake_supabase):
    seed = {"predictions": [
        {"id": "a", "game_id": "0022600001", "game_date": "2026-11-01",
         "predicted_team": "BOS", "home_team": "BOS", "away_team": "DET",
         "actual_winner": None, "correct": None},
    ], "running_record": []}
    db = fake_supabase(seed)
    monkeypatch.setattr(ev, "_fetch_results_cdn", lambda ids: {"0022600001": "BOS"})

    ev.evaluate(dry_run=True, db=db)
    assert db.writes == []


def test_running_record_recomputed_when_no_pending(fake_supabase):
    # Everything already graded (no pending), but running_record is stale (seeded 4-1).
    seed = {
        "predictions": [
            {"id": "b", "game_id": "0022600002", "game_date": "2026-11-02",
             "predicted_team": "LAL", "actual_winner": "LAL", "correct": True},
            {"id": "e", "game_id": "0022600003", "game_date": "2026-11-03",
             "predicted_team": "NYK", "actual_winner": "BOS", "correct": False},
            {"id": "c", "game_id": "0042500401", "game_date": "2026-06-03",
             "predicted_team": "NYK", "actual_winner": "NYK", "correct": True},
        ],
        "running_record": [{"id": 1, "total_correct": 4, "total_incorrect": 1, "accuracy": 0.8}],
    }
    db = fake_supabase(seed)
    ev.evaluate(dry_run=False, db=db)

    rec = [w[2] for w in db.writes if w[0] == "running_record" and w[1] == "upsert"][-1]
    # Current-season only: b (correct) + e (wrong) = 1/1/0.5; Finals row c excluded.
    assert rec["total_correct"] == 1
    assert rec["total_incorrect"] == 1
    assert rec["accuracy"] == 0.5
    assert rec["last_updated"]
    datetime.fromisoformat(rec["last_updated"])
