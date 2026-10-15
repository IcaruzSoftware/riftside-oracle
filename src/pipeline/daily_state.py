"""
Daily state update: Oracle's Elixir (OE) -> team_games_{year}.csv + incremental ELO.

  1. Download the OE year file(s) covering (last game in team_games - LOOKBACK_DAYS)
     through the target date (early January: both years).
  2. New games = gameids not in team_games_*.csv, dated strictly BEFORE the target
     day (UTC). Same-day games wait for the next run, so the ELO state never holds
     day d while live features cut team_games at day < d.
  3. Keep only complete games (12 rows, 5 roles per side, results present); an
     incomplete one is logged and skipped, and a later run picks it up.
  4. elo.update_elo on exactly those games, then append their team rows to
     team_games_{year}.csv (year of the game's date). ELO goes first: if the append
     fails, the next run hands the same games to update_elo, which skips them.

Idempotent: a second run finds no new games and writes nothing. --dry-run writes
no state (the OE download still lands in the gitignored src/data/raw/oe/ cache).
A blocked OE download (Drive "Quota exceeded") raises with a clear message; the
committed state stays as it was and the next run catches up via the lookback.

One-time setup: --init writes team_games_{year}.csv from the local OE files (the
same files the committed ELO state was built from).

Run
---
  python src/pipeline/daily_state.py                 # catch up through yesterday (UTC)
  python src/pipeline/daily_state.py --date 2026-11-05
  python src/pipeline/daily_state.py --dry-run
  python src/pipeline/daily_state.py --init          # one-time, from src/data/raw/oe/
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import elo, oe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = oe.RAW_DIR
LOOKBACK_DAYS = 14
_STR_COLS = [c for c in oe._STR_COLS if c in oe.TEAM_GAMES_COLS] + ["starters"]


def team_games_path(year: int) -> Path:
    return RAW_DIR / f"team_games_{year}.csv"


def load_team_games() -> pd.DataFrame:
    """All committed team_games_*.csv rows (oe.TEAM_GAMES_COLS), ids and patch kept as str."""
    paths = sorted(RAW_DIR.glob("team_games_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No team_games_*.csv in {RAW_DIR} - run daily_state.py --init once")
    df = pd.concat([pd.read_csv(p, dtype={c: str for c in _STR_COLS}, parse_dates=["date"])
                    for p in paths], ignore_index=True)
    return df[oe.TEAM_GAMES_COLS]


def complete_gameids(rows: pd.DataFrame) -> set:
    """Gameids with exactly 12 rows, 2 team rows, 5 roles per side and every result present."""
    players = rows[rows["position"] != "team"]
    roles_ok = (players.groupby(["gameid", "side"])["position"].agg(lambda s: "|".join(sorted(s)))
                == "|".join(sorted(oe.ROLES)))
    g = (rows.assign(is_team=rows["position"].eq("team"), no_result=rows["result"].isna())
         .groupby("gameid").agg(n=("is_team", "size"), teams=("is_team", "sum"),
                                no_result=("no_result", "sum")))
    g["sides_ok"] = roles_ok.groupby(level="gameid").sum().reindex(g.index, fill_value=0)
    return set(g.index[(g["n"] == 12) & (g["teams"] == 2) & (g["no_result"] == 0)
                       & (g["sides_ok"] == 2)])


def _append(team: pd.DataFrame, dry_run: bool) -> None:
    """Append team rows to team_games_{year}.csv by the year of each game's date."""
    for year, rows in team.groupby(team["date"].dt.year):
        path = team_games_path(int(year))
        if dry_run:
            logger.info("[dry-run] %s: +%d row(s)", path.name, len(rows))
            continue
        # Fixed date_format: an all-midnight batch would otherwise be written date-only.
        rows.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n",
                    date_format="%Y-%m-%d %H:%M:%S")
        logger.info("Appended %d row(s) to %s", len(rows), path.name)


def _download(years: range) -> pd.DataFrame:
    """Download (or reuse the fresh cache of) the OE files for these years and load them."""
    try:
        files = oe.list_files()
        frames = []
        for y in years:
            if y not in files:
                logger.info("OE %d: no file in the folder yet, skipping", y)
                continue
            oe.download_year(y, file_id=files[y])
            frames.append(oe.load_year(y))
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        raise RuntimeError(
            f"OE download failed ({exc}). State NOT updated; predictions use the committed "
            f"state, and the next run catches up (lookback {LOOKBACK_DAYS} days).") from exc
    if not frames:
        raise RuntimeError(f"OE: no file for any of the years {list(years)}")
    return pd.concat(frames, ignore_index=True)


def run(date: str | None = None, dry_run: bool = False) -> list[str]:
    """
    Catch the committed state up through the day before `date` (UTC 'YYYY-MM-DD',
    default today). Returns the newly applied gameids in (date, gameid) order.
    """
    target = pd.Timestamp(date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    tg = load_team_games()
    start = tg["date"].max().normalize() - pd.Timedelta(days=LOOKBACK_DAYS)
    logger.info("Scanning OE %s..%s (games before %s)", start.date(), target.date(), target.date())

    raw = _download(range(start.year, target.year + 1))
    rows = raw[(raw["date"] >= start) & (raw["date"] < target) & ~raw["gameid"].isin(set(tg["gameid"]))]
    ok = complete_gameids(rows)
    for gid in sorted(set(rows["gameid"]) - ok):
        logger.warning("  %s: incomplete game (rows/roles/result), skipped for now", gid)
    new = rows[rows["gameid"].isin(ok)]
    if new.empty:
        logger.info("State already up to date - nothing to apply.")
        return []

    team = oe.team_rows(new)
    gids = list(dict.fromkeys(team["gameid"]))
    if dry_run:
        logger.info("[dry-run] would apply update_elo to %d game(s) (%s..%s)", len(gids),
                    team["date"].min().date(), team["date"].max().date())
    else:
        elo.update_elo(new)
    _append(team, dry_run)
    logger.info("Applied %d new game(s)%s.", len(gids), " [dry-run]" if dry_run else "")
    return gids


def init_team_games() -> dict[int, int]:
    """
    One-time: write team_games_{year}.csv from every local OE file (complete games
    only, as run() would ingest them). Refuses to overwrite existing files.
    Returns {year: rows written}.
    """
    existing = sorted(RAW_DIR.glob("team_games_*.csv"))
    if existing:
        raise FileExistsError(f"{existing[0].name} exists - delete the team_games files to re-init")
    years = sorted(int(f.name[:4]) for f in oe.OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    if not years:
        raise FileNotFoundError(f"No OE year files in {oe.OE_DIR} - run bootstrap_data.py")
    frames = []
    for y in years:
        df = oe.load_year(y)
        ok = complete_gameids(df)
        logger.info("OE %d: %d games, %d complete", y, df["gameid"].nunique(), len(ok))
        frames.append(oe.team_rows(df[df["gameid"].isin(ok)]))
    team = pd.concat(frames, ignore_index=True).sort_values(["date", "gameid", "side"], kind="stable")
    _append(team, dry_run=False)
    return {int(y): len(r) for y, r in team.groupby(team["date"].dt.year)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily OE-driven state + ELO update")
    parser.add_argument("--date", help="Target UTC date YYYY-MM-DD (default: today); "
                                       "games before this day are ingested")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing")
    parser.add_argument("--init", action="store_true",
                        help="One-time: write team_games_*.csv from the local OE files")
    args = parser.parse_args()
    if args.init:
        print(init_team_games())
    else:
        run(date=args.date, dry_run=args.dry_run)
