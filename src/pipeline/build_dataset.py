"""
Build the training feature matrix: one row per OE game, blue side vs red side.

Loads the team rows of every OE year into one team_games frame, walks every game
from START_DATE onward chronologically (all leagues), and calls
build_feature_matrix() in training mode (game_id -> the starters who actually
played, ELO from player_elo.parquet). Games where either side has no teamid are
skipped.

Output
------
  data/processed/feature_matrix.parquet   (not committed)
    One row per game: META_COLS (label blue_win: 1 = blue side won)
    + the feature columns.

Run
---
  python src/pipeline/build_dataset.py           # build if missing
  python src/pipeline/build_dataset.py --force   # overwrite existing
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import elo, oe
from pipeline.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROCESSED_DIR / "feature_matrix.parquet"

START_DATE = "2017-01-01"   # ELO warms up on 2014-2016
META_COLS = ["game_id", "game_date", "league", "blue_team_id", "red_team_id", "blue_win"]


def load_team_games() -> pd.DataFrame:
    """The team rows (oe.TEAM_GAMES_COLS) of every OE year in oe.OE_DIR."""
    years = sorted(int(f.name[:4]) for f in oe.OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    if not years:
        raise FileNotFoundError(f"No OE year files in {oe.OE_DIR} - run bootstrap_data.py")
    logger.info("Loading OE team rows %d-%d", years[0], years[-1])
    return pd.concat([oe.team_rows(oe.load_year(y)) for y in years], ignore_index=True)


def build_game_index(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per game from START_DATE: META_COLS + playoffs, sorted by (game_date, game_id)."""
    blue = team_games[team_games["side"] == "Blue"].set_index("gameid")
    red = team_games[team_games["side"] == "Red"].set_index("gameid")["teamid"]
    index = pd.DataFrame({
        "game_id": blue.index, "game_date": blue["date"].to_numpy(),
        "league": blue["league"].to_numpy(), "playoffs": blue["playoffs"].to_numpy(),
        "blue_team_id": blue["teamid"].to_numpy(), "red_team_id": red.reindex(blue.index).to_numpy(),
        "blue_win": blue["result"].astype(int).to_numpy(),
    })
    index = index[index["game_date"] >= START_DATE]
    return index.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def check_elo_history(games: pd.DataFrame) -> None:
    """Raise if elo.HISTORY_PATH is missing or ends before the last game (features would use future ELO)."""
    if not elo.HISTORY_PATH.exists():
        raise FileNotFoundError(f"No ELO history at {elo.HISTORY_PATH} - run python src/pipeline/elo.py --force first")
    # Dates, not gameid sets: incomplete games / missing playerids legitimately have no ELO rows.
    hist_end = pd.read_parquet(elo.HISTORY_PATH, columns=["game_date"])["game_date"].max().normalize()
    games_end = games["game_date"].max().normalize()
    if not hist_end >= games_end:                           # also catches an empty history (NaT)
        raise RuntimeError(f"ELO history is stale: player_elo.parquet ends {hist_end.date()}, "
                           f"games run to {games_end.date()} -run python src/pipeline/elo.py --force first")


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Build (or load) the feature matrix and save it to OUT_PATH."""
    if OUT_PATH.exists() and not force:
        logger.info("Feature matrix already exists — loading from %s", OUT_PATH)
        return pd.read_parquet(OUT_PATH)

    team_games = load_team_games()
    games = build_game_index(team_games)
    no_team = games["blue_team_id"].isna() | games["red_team_id"].isna()
    logger.info("%d games since %s; skipping %d with a missing teamid",
                len(games), START_DATE, int(no_team.sum()))
    games = games[~no_team]
    check_elo_history(games)

    rows = []
    for m in tqdm(games.itertuples(index=False), total=len(games),
                  desc="Building features", unit="game", ncols=88):
        feats = build_feature_matrix(
            m.blue_team_id, m.red_team_id, m.game_date,
            {"league": m.league, "playoffs": m.playoffs},
            team_games, use_current_elo=False, game_id=m.game_id,
        )
        rows.append({c: getattr(m, c) for c in META_COLS} | feats)

    result = pd.DataFrame(rows)
    result.to_parquet(OUT_PATH, index=False)
    logger.info("Saved feature matrix → %s  shape=%s", OUT_PATH, result.shape)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the XGBoost training feature matrix")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    t0 = time.monotonic()
    df = build_dataset(force=args.force)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print("\nDone.")
    print(f"  Rows      : {len(df):,}  ({df['game_date'].min():%Y-%m-%d} .. {df['game_date'].max():%Y-%m-%d})")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Blue wins : {df['blue_win'].sum():,} / {len(df):,} ({df['blue_win'].mean():.1%})")
    print(f"  Runtime   : {time.monotonic() - t0:.0f}s")
