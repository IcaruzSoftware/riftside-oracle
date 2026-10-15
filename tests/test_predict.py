"""predict_todays_games: fixture schedule + stub DB, live features, all columns."""

import pandas as pd

import pipeline.predict as pred
from pipeline.predict import PREDICTION_COLS


def test_predict_todays_games_rows_and_writes(tiny_state, fake_supabase):
    db = fake_supabase({"predictions": [], "shap_values": []})

    rows = pred.predict_todays_games(
        date=tiny_state["date"], dry_run=False,
        schedule=tiny_state["schedule"], db=db,
    )

    assert len(rows) == 1
    row = rows[0]
    for col in PREDICTION_COLS:
        assert col in row
    assert row["game_id"] == "0042500405"
    assert row["game_id"] != "0000000000"
    assert {row["home_team"], row["away_team"]} == {"NYK", "SAS"}
    assert row["game_time_utc"] == "2026-06-14T00:30:00Z"
    assert 0.0 <= row["home_win_prob"] <= 1.0

    upserts = [w[2] for w in db.writes if w[0] == "predictions" and w[1] == "upsert"]
    assert len(upserts) == 1
    for col in PREDICTION_COLS:
        assert col in upserts[0]
    # SHAP rows are replaced (delete then insert)
    assert any(w[0] == "shap_values" and w[1] == "delete" for w in db.writes)
    assert any(w[0] == "shap_values" and w[1] == "insert" for w in db.writes)


def test_no_games_returns_empty(tiny_state, fake_supabase):
    empty_schedule = {"leagueSchedule": {"gameDates": []}}
    rows = pred.predict_todays_games(
        date="2026-06-13", dry_run=True, schedule=empty_schedule,
    )
    assert rows == []


def test_skips_started_games(tiny_state, fake_supabase):
    import copy
    started = copy.deepcopy(tiny_state["schedule"])
    started["leagueSchedule"]["gameDates"][0]["games"][0]["gameStatus"] = 3  # final
    db = fake_supabase({"predictions": [], "shap_values": []})
    rows = pred.predict_todays_games(
        date=tiny_state["date"], dry_run=False, schedule=started, db=db,
    )
    assert rows == []
    assert not any(w[0] == "predictions" and w[1] == "upsert" for w in db.writes)


def test_never_overwrites_graded_prediction(tiny_state, fake_supabase):
    db = fake_supabase({
        "predictions": [{"id": "x", "game_id": "0042500405",
                         "actual_winner": "NYK", "correct": True}],
        "shap_values": [],
    })
    rows = pred.predict_todays_games(
        date=tiny_state["date"], dry_run=False, schedule=tiny_state["schedule"], db=db,
    )
    assert rows == []
    assert not any(w[0] == "predictions" and w[1] == "upsert" for w in db.writes)


def test_live_features_have_nonzero_elo_diff(tiny_state):
    from pipeline.features import build_feature_matrix
    from pipeline import nba_cdn

    game_log = pd.read_csv(tiny_state["raw"] / "game_log_playoffs_2025-26.csv")
    stats = pd.read_csv(tiny_state["raw"] / "player_season_stats_2025-26.csv")
    home_id = str(nba_cdn.tricode_to_id("SAS"))
    away_id = str(nba_cdn.tricode_to_id("NYK"))

    feats = build_feature_matrix(
        game_id="0042500405", home_team_id=home_id, away_team_id=away_id,
        game_date="2026-06-13", season="2025-26", game_log_df=game_log,
        player_stats_df=stats, use_current_elo=True, prev_player_stats_df=None,
    )
    # Lineups came from the ELO state (not the default 1000-per-skill fallback)...
    assert feats["home_general_elo"] > 5000.0
    assert feats["away_general_elo"] > 5000.0
    # ...and the two teams differ.
    assert feats["elo_general_diff"] != 0.0
