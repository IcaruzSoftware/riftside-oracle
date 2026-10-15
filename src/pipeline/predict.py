"""
Predict the outcome of an NBA game.

Two modes
---------
1. Historical game (already in our data):
     python src/pipeline/predict.py --game-id 0042500237

2. Any matchup by team abbreviation + date:
     python src/pipeline/predict.py --home NYK --away OKC --date 2026-06-05

Output
------
  - Predicted winner + home win probability
  - Top 10 most influential features (SHAP values)
  - Actual result (if the game is in our data)
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODEL_PATH    = MODELS_DIR / "xgb_model.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_game_logs() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(RAW_DIR.glob("game_log_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["GAME_ID"]   = df["GAME_ID"].astype(str).str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])


def _load_player_stats(season: str) -> pd.DataFrame:
    path = RAW_DIR / f"player_season_stats_{season}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _date_to_season(date: pd.Timestamp) -> str:
    y = date.year
    return f"{y}-{str(y+1)[2:]}" if date.month >= 10 else f"{y-1}-{str(y)[2:]}"


def _team_id(abbrev: str, game_log_df: pd.DataFrame) -> str:
    rows = game_log_df[game_log_df["TEAM_ABBREVIATION"].str.upper() == abbrev.upper()]
    if rows.empty:
        raise ValueError(f"Unknown team abbreviation: {abbrev}")
    return str(rows["TEAM_ID"].iloc[0])


def _team_abbrev(team_id: str, game_log_df: pd.DataFrame) -> str:
    rows = game_log_df[game_log_df["TEAM_ID"] == int(team_id)]
    return str(rows["TEAM_ABBREVIATION"].iloc[0]) if not rows.empty else team_id


def _predict_proba(model_bundle: dict, feature_dict: dict) -> float:
    """Run the Platt-calibrated model and return P(home win)."""
    feature_names = model_bundle["feature_names"]
    base_model    = model_bundle["base_model"]
    platt         = model_bundle["platt"]

    row = np.array([[feature_dict.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    raw = base_model.predict_proba(row)[:, 1].reshape(-1, 1)
    return float(platt.predict_proba(raw)[0, 1])


def _shap_top_features(model_bundle: dict, feature_dict: dict, n: int = 10) -> list[tuple]:
    """Return top-n (feature_name, shap_value) sorted by abs impact."""
    try:
        import shap
    except ImportError:
        return []

    feature_names = model_bundle["feature_names"]
    base_model    = model_bundle["base_model"]
    row = np.array([[feature_dict.get(f, 0.0) for f in feature_names]], dtype=np.float32)

    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(row)[0]

    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_matchup(
    home_team_id: str,
    away_team_id: str,
    game_date: str,
    game_log_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    season: str,
    game_id: str = "0000000000",
) -> dict:
    """
    Build features and predict a single game.

    Args:
        home_team_id:    NBA team ID for the home team
        away_team_id:    NBA team ID for the away team
        game_date:       ISO date string "YYYY-MM-DD"
        game_log_df:     Combined game log (all seasons)
        player_stats_df: Season player stats for the relevant season
        season:          Season string e.g. "2023-24"
        game_id:         10-digit game ID if known (fake ID for hypothetical games)

    Returns:
        dict with prediction results and feature values
    """
    from pipeline.features import build_feature_matrix

    feature_dict = build_feature_matrix(
        game_id         = game_id,
        home_team_id    = home_team_id,
        away_team_id    = away_team_id,
        game_date       = game_date,
        season          = season,
        game_log_df     = game_log_df,
        player_stats_df = player_stats_df,
        use_current_elo = (game_id == "0000000000"),
    )

    model_bundle   = _load_model()
    home_win_prob  = _predict_proba(model_bundle, feature_dict)
    shap_features  = _shap_top_features(model_bundle, feature_dict)

    home_abbrev = _team_abbrev(home_team_id, game_log_df)
    away_abbrev = _team_abbrev(away_team_id, game_log_df)

    predicted_winner = home_abbrev if home_win_prob >= 0.5 else away_abbrev
    winner_prob      = home_win_prob if home_win_prob >= 0.5 else (1 - home_win_prob)

    return {
        "game_id":        game_id,
        "game_date":      game_date,
        "home_team":      home_abbrev,
        "away_team":      away_abbrev,
        "home_win_prob":  round(home_win_prob, 4),
        "away_win_prob":  round(1 - home_win_prob, 4),
        "predicted_winner": predicted_winner,
        "confidence":     round(winner_prob, 4),
        "shap_features":  shap_features,
        "feature_dict":   feature_dict,
    }


# ---------------------------------------------------------------------------
# Today's games (daily pipeline entry point)
# ---------------------------------------------------------------------------

PREDICTION_COLS = [
    "game_id", "game_date", "game_time_utc", "home_team", "away_team",
    "home_team_id", "away_team_id", "predicted_winner", "predicted_team",
    "home_win_prob", "away_win_prob", "confidence",
]


def _today_et() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _prev_season(season: str) -> str:
    start = int(season[:4])
    return f"{start - 1}-{str(start)[2:]}"


def _get_supabase():
    from supabase import create_client
    import os
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def predict_todays_games(
    date: str | None = None,
    dry_run: bool = False,
    schedule: dict | None = None,
    db=None,
) -> list[dict]:
    """
    Predict every stateful game (regular season / playoffs / play-in / Cup final)
    on an ET date and upsert the results to Supabase.

    Preseason and all-star games are skipped. Live features are used (team lineups
    from the ELO state, no future box score). For each game the prediction is
    upserted on game_id and its SHAP rows are replaced (delete-then-insert top 10).

    Args:
        date:     ET date 'YYYY-MM-DD' (defaults to today, Eastern time).
        dry_run:  print instead of writing to Supabase.
        schedule: pre-fetched CDN schedule dict (fetched from the CDN if None).
        db:       Supabase client (created from env when None and not dry_run).

    Returns the list of prediction rows (empty when there are no games).
    """
    from pipeline import nba_cdn
    from pipeline.features import build_feature_matrix

    et_date = date or _today_et()
    # Only predict stateful games that have not tipped off yet (gameStatus 1 =
    # scheduled). Predicting a live/final game would let a manual re-run rewrite a
    # pick after the fact.
    games = [g for g in nba_cdn.games_for_date(et_date, schedule=schedule)
             if nba_cdn.is_stateful(g["game_id"]) and g["game_status"] == 1]

    if not games:
        logger.info("No unstarted games to predict on %s.", et_date)
        return []

    logger.info("Predicting %d game(s) on %s ...", len(games), et_date)

    game_log_df     = _load_game_logs()
    season          = _date_to_season(pd.to_datetime(et_date))
    player_stats_df = _load_player_stats(season)
    prev_stats_df   = _load_player_stats(_prev_season(season))
    model_bundle    = _load_model()

    if not dry_run and db is None:
        db = _get_supabase()

    results: list[dict] = []
    for g in games:
        # Never overwrite a prediction that has already been graded (guards manual
        # re-runs after a game has been played).
        if not dry_run:
            graded = (db.table("predictions").select("actual_winner")
                        .eq("game_id", g["game_id"]).execute().data)
            if graded and graded[0].get("actual_winner") is not None:
                logger.info("  %s already graded — leaving prediction untouched", g["game_id"])
                continue

        home_id = str(g["home_team_id"])
        away_id = str(g["away_team_id"])
        home_ab = g["home_tricode"]
        away_ab = g["away_tricode"]

        feats = build_feature_matrix(
            game_id         = g["game_id"],
            home_team_id    = home_id,
            away_team_id    = away_id,
            game_date       = et_date,
            season          = season,
            game_log_df     = game_log_df,
            player_stats_df = player_stats_df,
            use_current_elo = True,
            prev_player_stats_df = prev_stats_df,
        )

        home_win_prob = round(_predict_proba(model_bundle, feats), 4)
        away_win_prob = round(1 - home_win_prob, 4)
        is_home_pick  = home_win_prob >= 0.5

        row = {
            "game_id":          g["game_id"],
            "game_date":        et_date,
            "game_time_utc":    g["game_time_utc"],
            "home_team":        home_ab,
            "away_team":        away_ab,
            "home_team_id":     home_id,
            "away_team_id":     away_id,
            "predicted_winner": "home" if is_home_pick else "away",
            "predicted_team":   home_ab if is_home_pick else away_ab,
            "home_win_prob":    home_win_prob,
            "away_win_prob":    away_win_prob,
            "confidence":       home_win_prob if is_home_pick else away_win_prob,
        }
        shap_pairs = _shap_top_features(model_bundle, feats, n=10)

        if dry_run:
            logger.info(
                "  %s @ %s  →  %s (%.1f%%)  | elo_general_diff=%.1f",
                away_ab, home_ab, row["predicted_team"], row["confidence"] * 100,
                feats.get("elo_general_diff", 0.0),
            )
        else:
            resp = db.table("predictions").upsert(row, on_conflict="game_id").execute()
            pred_id = resp.data[0]["id"]
            db.table("shap_values").delete().eq("prediction_id", pred_id).execute()
            if shap_pairs:
                db.table("shap_values").insert([
                    {
                        "prediction_id": pred_id,
                        "feature_name":  feat,
                        "shap_value":    round(float(val), 6),
                        "feature_value": round(float(feats.get(feat, 0.0)), 4),
                    }
                    for feat, val in shap_pairs
                ]).execute()

        results.append(row)

    logger.info("Predicted %d game(s) on %s.", len(results), et_date)
    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_prediction(result: dict, actual_winner: str | None = None) -> None:
    home = result["home_team"]
    away = result["away_team"]
    pred = result["predicted_winner"]
    conf = result["confidence"] * 100

    print()
    print("=" * 52)
    print(f"  {away}  @  {home}    ({result['game_date']})")
    print("=" * 52)
    print(f"  Home win prob : {result['home_win_prob']*100:.1f}%")
    print(f"  Away win prob : {result['away_win_prob']*100:.1f}%")
    print(f"  Prediction    : {pred} wins  ({conf:.1f}% confident)")

    if actual_winner is not None:
        correct = "✓ CORRECT" if actual_winner == pred else "✗ WRONG"
        print(f"  Actual winner : {actual_winner}   {correct}")

    if result["shap_features"]:
        print()
        print("  Top factors (SHAP):")
        for feat, val in result["shap_features"]:
            direction = "→ home" if val > 0 else "→ away"
            print(f"    {feat:<38} {val:+.4f}  {direction}")

    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict NBA game outcome")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game-id",  help="10-digit game ID (historical game in our data)")
    group.add_argument("--home",     help="Home team abbreviation (e.g. NYK)")

    parser.add_argument("--away",  help="Away team abbreviation (required with --home)")
    parser.add_argument("--date",  help="Game date YYYY-MM-DD (required with --home)")
    args = parser.parse_args()

    logger.info("Loading game logs...")
    game_log_df = _load_game_logs()

    # ── Mode 1: predict by game ID ───────────────────────────────────────────
    if args.game_id:
        game_id = str(args.game_id).zfill(10)
        row = game_log_df[game_log_df["GAME_ID"] == game_id]
        if row.empty:
            print(f"Game ID {game_id} not found in game logs.")
            sys.exit(1)

        home_row  = row[row["MATCHUP"].str.contains(" vs. ")]
        away_row  = row[row["MATCHUP"].str.contains(" @ ")]
        if home_row.empty:
            print(f"Cannot determine home team for {game_id}.")
            sys.exit(1)

        home_team_id = str(home_row["TEAM_ID"].iloc[0])
        away_team_id = str(away_row["TEAM_ID"].iloc[0]) if not away_row.empty else ""
        game_date    = str(home_row["GAME_DATE"].iloc[0].date())
        game_date_ts = pd.to_datetime(game_date)
        season       = _date_to_season(game_date_ts)

        # Actual result from the log
        actual_home_wl = home_row["WL"].iloc[0]
        home_abbrev    = _team_abbrev(home_team_id, game_log_df)
        away_abbrev    = _team_abbrev(away_team_id, game_log_df) if away_team_id else "?"
        actual_winner  = home_abbrev if actual_home_wl == "W" else away_abbrev

    # ── Mode 2: predict by team abbreviations ────────────────────────────────
    else:
        if not args.away or not args.date:
            parser.error("--away and --date are required when using --home")

        home_team_id = _team_id(args.home, game_log_df)
        away_team_id = _team_id(args.away, game_log_df)
        game_date    = args.date
        game_date_ts = pd.to_datetime(game_date)
        season       = _date_to_season(game_date_ts)
        game_id      = "0000000000"
        actual_winner = None

        logger.info("Predicting %s @ %s on %s (season %s)...",
                    args.away, args.home, game_date, season)

    player_stats_df = _load_player_stats(season)
    if player_stats_df.empty:
        logger.warning("No player stats found for season %s — availability features will be empty", season)

    result = predict_matchup(
        home_team_id    = home_team_id,
        away_team_id    = away_team_id,
        game_date       = game_date,
        game_log_df     = game_log_df,
        player_stats_df = player_stats_df,
        season          = season,
        game_id         = game_id,
    )

    _print_prediction(result, actual_winner=actual_winner)


if __name__ == "__main__":
    main()
