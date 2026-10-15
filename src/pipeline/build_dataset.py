"""
Build the full training feature matrix across all historical games.

Loads all game logs, iterates every game chronologically, calls
build_feature_matrix() for each game, and saves a flat parquet file
ready for train.py.

Output
------
  data/processed/feature_matrix.parquet
    One row per game. Columns:
      game_id, game_date, season, home_team_id, away_team_id,
      home_win  (label: 1 = home won, 0 = away won),
      + ~100 feature columns

Run
---
  python src/pipeline/build_dataset.py           # build from scratch
  python src/pipeline/build_dataset.py --force   # overwrite existing
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_to_season(date: pd.Timestamp) -> str:
    """Map a game date to its NBA season string (e.g. 2024-03-20 → '2023-24')."""
    y = date.year
    if date.month >= 10:
        return f"{y}-{str(y + 1)[2:]}"
    return f"{y - 1}-{str(y)[2:]}"


def load_all_game_logs() -> pd.DataFrame:
    """Combine all regular-season and playoff game log CSVs into one DataFrame."""
    frames = []
    for csv in sorted(RAW_DIR.glob("game_log_*.csv")):
        frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"No game_log_*.csv files found in {RAW_DIR}")
    combined = pd.concat(frames, ignore_index=True)
    combined["GAME_ID"]   = combined["GAME_ID"].astype(str).str.zfill(10)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
    return combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])


def load_player_stats_by_season() -> dict[str, pd.DataFrame]:
    """Return {season_str: player_stats_df} for every available season."""
    stats: dict[str, pd.DataFrame] = {}
    for csv in sorted(RAW_DIR.glob("player_season_stats_*.csv")):
        season = csv.stem.replace("player_season_stats_", "")
        stats[season] = pd.read_csv(csv)
    logger.info("Loaded player stats for seasons: %s", sorted(stats.keys()))
    return stats


def build_game_index(game_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row-per-game index with home team, away team, and label.

    Home team is identified by MATCHUP containing ' vs. ' (e.g. 'OKC vs. SAC').
    Away team is identified by ' @ '    (e.g. 'SAC @ OKC').
    """
    home = (
        game_log_df[game_log_df["MATCHUP"].str.contains(" vs. ", na=False)]
        .drop_duplicates("GAME_ID")
        .copy()
    )
    away = (
        game_log_df[game_log_df["MATCHUP"].str.contains(" @ ", na=False)]
        [["GAME_ID", "TEAM_ID"]]
        .drop_duplicates("GAME_ID")
        .rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})
    )

    home["season"]       = home["GAME_DATE"].apply(_date_to_season)
    home["home_team_id"] = home["TEAM_ID"].astype(str)
    home["home_win"]     = (home["WL"] == "W").astype(int)

    index = home[["GAME_ID", "GAME_DATE", "season", "home_team_id", "home_win"]].merge(
        away, on="GAME_ID", how="left"
    )
    index["away_team_id"] = index["AWAY_TEAM_ID"].astype(str)
    index = (
        index
        .drop(columns=["AWAY_TEAM_ID"])
        .rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    return index


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(force: bool = False) -> pd.DataFrame:
    """
    Iterate every game chronologically and build the feature matrix.

    For each game:
      1. Identify home team, away team, label (home_win)
      2. Pull player stats for the correct season
      3. Call build_feature_matrix() → ~100 features
      4. Append row to results

    Saves to data/processed/feature_matrix.parquet.

    Args:
        force: Overwrite existing output if True.
    """
    out_path = PROCESSED_DIR / "feature_matrix.parquet"
    if out_path.exists() and not force:
        logger.info("Feature matrix already exists — loading from %s", out_path)
        return pd.read_parquet(out_path)

    logger.info("Loading game logs...")
    game_log_df = load_all_game_logs()
    logger.info("  %d team-game rows across all seasons", len(game_log_df))

    logger.info("Loading player stats...")
    player_stats = load_player_stats_by_season()

    logger.info("Building game index...")
    game_index = build_game_index(game_log_df)
    logger.info("  %d unique games to process", len(game_index))

    rows:   list[dict] = []
    errors: int        = 0

    for _, meta in tqdm(game_index.iterrows(), total=len(game_index),
                        desc="Building features", unit="game", ncols=88):
        game_id      = meta["game_id"]
        game_date    = meta["game_date"]
        season       = meta["season"]
        home_team_id = meta["home_team_id"]
        away_team_id = meta["away_team_id"]
        home_win     = int(meta["home_win"])

        ps_df = player_stats.get(season, pd.DataFrame())

        try:
            feats = build_feature_matrix(
                game_id      = game_id,
                home_team_id = home_team_id,
                away_team_id = away_team_id,
                game_date    = str(game_date.date()),
                season       = season,
                game_log_df  = game_log_df,
                player_stats_df = ps_df,
            )
        except Exception as exc:
            logger.warning("FAILED %s: %s", game_id, exc)
            errors += 1
            continue

        rows.append({
            "game_id":      game_id,
            "game_date":    game_date,
            "season":       season,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_win":     home_win,
            **feats,
        })

    logger.info("Processed %d games (%d errors / skipped)", len(rows), errors)

    result = pd.DataFrame(rows)
    result.to_parquet(out_path, index=False)
    logger.info("Saved feature matrix → %s  shape=%s", out_path, result.shape)

    # Quick sanity print
    n_features = len(result.columns) - 6  # subtract metadata cols
    label_dist = result["home_win"].value_counts().to_dict()
    logger.info("Features per game: %d | Label: %s", n_features, label_dist)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build XGBoost training feature matrix")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    df = build_dataset(force=args.force)

    feature_cols = [c for c in df.columns
                    if c not in {"game_id", "game_date", "season",
                                 "home_team_id", "away_team_id", "home_win"}]

    print(f"\nDone.")
    print(f"  Rows     : {len(df):,}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Home wins: {df['home_win'].sum():,} / {len(df):,} "
          f"({df['home_win'].mean():.1%})")
    print(f"\nSample feature names:")
    for col in feature_cols[:15]:
        print(f"  {col}")
