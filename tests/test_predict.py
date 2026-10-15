"""predict_upcoming with a FakeSupabase and a stubbed schedule/model: skips, side-neutral p, upserts."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.elo as elo
import pipeline.features as features
import pipeline.predict as predict
from pipeline import lolesports, oe
from pipeline.features import series_win_prob

FIXTURES = Path(__file__).parent / "fixtures" / "oe"
TEAM_GAMES = pd.DataFrame({
    "date": pd.to_datetime(["2026-09-01"] * 4),
    "teamid": ["oe:team:t1", "oe:team:kt", "oe:team:geng", "oe:team:hle"],
    "teamname": ["T1", "KT Rolster", "Gen.G", "Hanwha Life Esports"],
})
# The real functions, before the autouse stubs replace them.
REAL = {n: getattr(predict, n) for n in ("build_feature_matrix", "_predict_proba", "_shap_top", "_load_model")}
P_BLUE = {"oe:team:t1": 0.7, "oe:team:kt": 0.4, "oe:team:geng": 0.55, "oe:team:hle": 0.5}


def _event(match_id, start, teams, state="unstarted", bo=3, block="Playoffs", slug="lck"):
    return {"startTime": start, "state": state, "type": "match", "blockName": block,
            "league": {"name": slug.upper(), "slug": slug},
            "match": {"id": match_id, "strategy": {"type": "bestOf", "count": bo},
                      "teams": [{"name": n, "code": c, "image": f"http://img/{c}.png"} for n, c in teams]}}


EVENTS = [
    _event("ok", "2026-10-15T08:00:00Z", [("T1", "T1"), ("kt Rolster", "KT")]),
    _event("started", "2026-10-15T09:00:00Z", [("T1", "T1"), ("Gen.G Esports", "GEN")], state="inProgress"),
    _event("tbd", "2026-10-15T10:00:00Z", [("TBD", "TBD"), ("TBD", "TBD")]),
    _event("unmapped", "2026-10-15T11:00:00Z", [("T1", "T1"), ("Nobody Gaming", "NOB")]),
    _event("resolved", "2026-10-15T12:00:00Z", [("Gen.G Esports", "GEN"), ("Hanwha Life Esports", "HLE")]),
    _event("refresh", "2026-10-15T13:00:00Z", [("Hanwha Life Esports", "HLE"), ("T1", "T1")], bo=5),
    _event("later", "2026-10-17T13:00:00Z", [("T1", "T1"), ("kt Rolster", "KT")]),
]


@pytest.fixture(autouse=True)
def stubs(monkeypatch, tmp_path):
    aliases = tmp_path / "team_aliases.csv"
    aliases.write_text("lolesports_code,lolesports_name,oe_teamid\nKT,kt Rolster,oe:team:kt\n")
    monkeypatch.setattr(lolesports, "ALIASES_PATH", aliases)
    monkeypatch.setattr(lolesports, "league_ids", lambda slugs: ["id"])
    monkeypatch.setattr(lolesports, "get_schedule", lambda ids, token=None: {
        "pages": {"older": None, "newer": None}, "events": copy.deepcopy(EVENTS)})
    monkeypatch.setattr(predict, "load_team_games", lambda: TEAM_GAMES)
    monkeypatch.setattr(predict, "_load_model", lambda: {})
    # Each orientation's "features" just name the blue team; P(blue wins) is P_BLUE[blue].
    monkeypatch.setattr(predict, "build_feature_matrix",
                        lambda blue, red, day, ctx, tg, live: {"blue": blue, "ctx": ctx})
    monkeypatch.setattr(predict, "_predict_proba",
                        lambda bundle, feats: np.array([P_BLUE[f["blue"]] for f in feats]))
    monkeypatch.setattr(predict, "_shap_top", lambda bundle, feats: [("diff_outcome_elo", 0.25)])


def _db(fake_supabase):
    return fake_supabase({"predictions": [
        {"id": "r1", "match_id": "resolved", "predicted_team": "HLE", "team1_game_prob": 0.1,
         "actual_winner": "GEN", "score": "2-0", "correct": False},
        {"id": "r2", "match_id": "refresh", "predicted_team": "HLE", "team1_game_prob": 0.9,
         "actual_winner": None},
    ]})


def test_predicts_only_eligible_matches(fake_supabase):
    db = _db(fake_supabase)
    rows = predict.predict_upcoming(date="2026-10-15", window_hours=36, db=db)
    assert [r["match_id"] for r in rows] == ["ok", "refresh"]
    assert [w[2]["match_id"] for w in db.writes if w[:2] == ("predictions", "upsert")] == ["ok", "refresh"]


def test_never_overwrites_a_resolved_row(fake_supabase):
    db = _db(fake_supabase)
    predict.predict_upcoming(date="2026-10-15", db=db)
    resolved = db.table("predictions").select("*").eq("match_id", "resolved").execute().data[0]
    assert resolved == {"id": "r1", "match_id": "resolved", "predicted_team": "HLE",
                        "team1_game_prob": 0.1, "actual_winner": "GEN", "score": "2-0",
                        "correct": False}


def test_side_neutral_probability_and_series_wiring(fake_supabase):
    db = _db(fake_supabase)
    rows = {r["match_id"]: r for r in predict.predict_upcoming(date="2026-10-15", db=db)}
    p = (0.7 + 1 - 0.4) / 2                                # T1 blue 0.7, KT blue 0.4
    ok = rows["ok"]
    assert ok["team1_game_prob"] == round(p, 4)
    assert ok["team1_series_prob"] == round(series_win_prob(p, 3), 4)
    assert ok["team2_series_prob"] == round(1 - series_win_prob(p, 3), 4)
    assert ok["predicted_team"] == "T1" and ok["confidence"] == ok["team1_series_prob"]
    assert (ok["match_date"], ok["best_of"], ok["block_name"]) == ("2026-10-15", 3, "Playoffs")
    q = (0.5 + 1 - 0.7) / 2                                # HLE vs T1, Bo5
    ref = rows["refresh"]
    assert ref["team1_series_prob"] == round(series_win_prob(q, 5), 4) and ref["predicted_team"] == "T1"
    assert ref["confidence"] == ref["team2_series_prob"]
    # The refreshed row was updated in place (same id) with SHAP rows attached.
    stored = db.table("predictions").select("*").eq("match_id", "refresh").execute().data[0]
    assert stored["id"] == "r2" and stored["team1_game_prob"] == round(q, 4)
    shap = db.table("shap_values").select("*").eq("prediction_id", "r2").execute().data
    assert [(s["feature_name"], s["shap_value"]) for s in shap] == [("diff_outcome_elo", 0.25)]


def test_dry_run_writes_nothing():
    rows = predict.predict_upcoming(date="2026-10-15", dry_run=True)
    assert [r["match_id"] for r in rows] == ["ok", "resolved", "refresh"]   # no db: no resolved check


def test_unmapped_team_is_logged(fake_supabase, caplog):
    predict.predict_upcoming(date="2026-10-15", db=_db(fake_supabase))
    assert "unmapped team NOB/Nobody Gaming" in caplog.text


@pytest.mark.parametrize("slug, block, expected", [
    ("lck", "Playoffs", {"league": "LCK", "playoffs": 1}),
    ("lpl", "Play In Knockouts", {"league": "LPL", "playoffs": 1}),
    ("lec", "Finals", {"league": "LEC", "playoffs": 1}),
    ("lck", "Week 3", {"league": "LCK", "playoffs": 0}),
    ("lcs", "Swiss", {"league": "LCS", "playoffs": 0}),
    ("worlds", "Quarterfinals", {"league": "WLDs", "playoffs": 0}),
    ("first_stand", None, {"league": "FST", "playoffs": 0}),
])
def test_match_context(slug, block, expected):
    assert predict.match_context(slug, block) == expected


def test_predict_match_with_the_committed_model(monkeypatch, tmp_path):
    """Real model + features on a tiny fixture state: valid probabilities, 10 SHAP rows."""
    for name, fn in REAL.items():
        monkeypatch.setattr(predict, name, fn)
    for name in ("HISTORY_PATH", "CURRENT_PATH", "RECENT_PATH", "ROLE_NORMS_PATH"):
        monkeypatch.setattr(elo, name, tmp_path / getattr(elo, name).name)
    monkeypatch.setattr(features, "_cache", {})
    base = oe.load_year(2026, FIXTURES)
    elo.build_elo(force=True, games=base, norms=elo.compute_role_norms(oe.player_rows(base), years=None))
    tg = oe.team_rows(base)
    hle, blg = tg.loc[tg["gameid"] == "LOLTMNT01_423456", "teamid"]
    pred = predict.predict_match(hle, blg, 5, "2026-10-15", {"league": "WLDs", "playoffs": 0},
                                 tg, predict._load_model())
    assert 0 < pred["game_prob"] < 1
    assert pred["series_prob"] == pytest.approx(series_win_prob(pred["game_prob"], 5))
    assert len(pred["shap"]) == predict.SHAP_TOP_N
    assert [abs(v) for _, v in pred["shap"]] == sorted((abs(v) for _, v in pred["shap"]), reverse=True)


def test_one_failing_match_does_not_block_the_others(fake_supabase, monkeypatch, caplog):
    events = copy.deepcopy(EVENTS)
    del events[0]["match"]["strategy"]                      # "ok" is malformed now
    monkeypatch.setattr(lolesports, "get_schedule", lambda ids, token=None: {
        "pages": {"older": None, "newer": None}, "events": events})
    db = _db(fake_supabase)
    with pytest.raises(RuntimeError, match="prediction failed for 1 match"):
        predict.predict_upcoming(date="2026-10-15", db=db)
    assert "ok: prediction failed" in caplog.text
    assert [w[2]["match_id"] for w in db.writes if w[:2] == ("predictions", "upsert")] == ["refresh"]
