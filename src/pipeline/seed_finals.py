"""
One-time seed: insert 2026 NBA Finals predictions into Supabase.

Runs all 5 games through the model, stores predictions + SHAP values,
marks actual results, and sets the running_record to 4/5 (80%).

Run once after Supabase schema is set up:
  SUPABASE_URL=... SUPABASE_KEY=... python src/pipeline/seed_finals.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd
from supabase import create_client
from pipeline.predict import (
    _load_game_logs, _load_player_stats, _team_id,
    _predict_proba, _shap_top_features, _load_model, _date_to_season,
)
from pipeline.features import build_feature_matrix

FINALS = [
    # (home, away, date, game_id,        actual_winner)
    ("NYK", "SAS", "2026-06-03", "0042500401", "NYK"),  # NYK wins 105-95
    ("NYK", "SAS", "2026-06-05", "0042500402", "NYK"),  # NYK wins 105-104
    ("SAS", "NYK", "2026-06-08", "0042500403", "SAS"),  # SAS wins 115-111
    ("SAS", "NYK", "2026-06-10", "0042500404", "NYK"),  # NYK wins 107-106
    ("NYK", "SAS", "2026-06-13", "0042500405", "NYK"),  # NYK wins 94-90
]


def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("Set SUPABASE_URL and SUPABASE_KEY env vars.")

    db = create_client(url, key)

    logger.info("Loading data...")
    game_log_df     = _load_game_logs()
    finals_season   = _date_to_season(pd.to_datetime(FINALS[0][2]))
    player_stats_df = _load_player_stats(finals_season)
    model_bundle    = _load_model()

    correct_count = 0

    for home_abbrev, away_abbrev, date, game_id, actual in FINALS:
        home_id = _team_id(home_abbrev, game_log_df)
        away_id = _team_id(away_abbrev, game_log_df)

        logger.info("Predicting %s @ %s on %s ...", away_abbrev, home_abbrev, date)

        feature_dict = build_feature_matrix(
            game_id         = game_id,
            home_team_id    = home_id,
            away_team_id    = away_id,
            game_date       = date,
            season          = _date_to_season(pd.to_datetime(date)),
            game_log_df     = game_log_df,
            player_stats_df = player_stats_df,
            use_current_elo = False,
        )

        home_win_prob  = _predict_proba(model_bundle, feature_dict)
        away_win_prob  = round(1 - home_win_prob, 4)
        home_win_prob  = round(home_win_prob, 4)
        predicted      = "home" if home_win_prob >= 0.5 else "away"
        predicted_team = home_abbrev if predicted == "home" else away_abbrev
        confidence     = home_win_prob if predicted == "home" else away_win_prob
        correct        = (predicted_team == actual)
        if correct:
            correct_count += 1

        # Upsert prediction row
        row = {
            "game_id":          game_id,
            "game_date":        date,
            "home_team":        home_abbrev,
            "away_team":        away_abbrev,
            "home_team_id":     home_id,
            "away_team_id":     away_id,
            "predicted_winner": predicted,
            "predicted_team":   predicted_team,
            "home_win_prob":    home_win_prob,
            "away_win_prob":    away_win_prob,
            "confidence":       round(float(confidence), 4),
            "actual_winner":    actual,
            "correct":          correct,
        }

        resp = db.table("predictions").upsert(row, on_conflict="game_id").execute()
        pred_id = resp.data[0]["id"]
        logger.info("  → %s wins (%d%%) | actual: %s | %s | id=%s",
                    predicted_team, confidence * 100, actual,
                    "CORRECT" if correct else "WRONG", pred_id)

        # Insert SHAP values
        shap_pairs = _shap_top_features(model_bundle, feature_dict, n=10)
        if shap_pairs:
            shap_rows = [
                {
                    "prediction_id": pred_id,
                    "feature_name":  feat,
                    "shap_value":    round(float(val), 6),
                    "feature_value": round(float(feature_dict.get(feat, 0)), 4),
                }
                for feat, val in shap_pairs
            ]
            db.table("shap_values").insert(shap_rows).execute()
            logger.info("  → %d SHAP values stored", len(shap_rows))

    # Update running record
    total = len(FINALS)
    incorrect = total - correct_count
    accuracy  = round(correct_count / total, 4)

    db.table("running_record").upsert({
        "id":              1,
        "total_correct":   correct_count,
        "total_incorrect": incorrect,
        "accuracy":        accuracy,
    }, on_conflict="id").execute()

    logger.info("Running record updated: %d/%d (%.0f%%)", correct_count, total, accuracy * 100)
    logger.info("Seed complete.")


if __name__ == "__main__":
    main()
