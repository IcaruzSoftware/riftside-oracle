"""
Daily state update — replaces update_season.py.

Brings the committed live state up to date from the NBA CDN (no proxies, no full
historical dataset on the runner):

  1. Find completed stateful games (regular season / playoffs / play-in / Cup final)
     on the ET dates from the last logged game through the target date that are not
     already in the committed game logs.
  2. Fetch each CDN box score, append both team rows to that season's game-log CSV,
     and fold the per-player lines into that season's player_season_stats CSV.
  3. Run incremental ELO on those games (elo.update_elo).
  4. Preseason games only refresh each player's team assignment (no ELO change).
  5. All-star games are ignored entirely.

Idempotent: games already in the game logs are skipped, cached files are only ever
appended to after a successful fetch (never deleted first), and update_elo skips
games already reflected in the ELO state. Catches up over multiple missed days.

Run
---
  python src/pipeline/daily_state.py                 # catch up through today (ET)
  python src/pipeline/daily_state.py --date 2026-11-05
  python src/pipeline/daily_state.py --dry-run
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import nba_cdn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Re-scan a few days before the last logged game so a game that had no box score on
# its own day (but whose later neighbours got logged) is still picked up. The
# existing-game filter prevents re-doing games already in the logs.
LOOKBACK_DAYS = 7

GAME_LOG_COLS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
                 "WL", "PTS", "PLUS_MINUS", "AST", "REB", "TOV"]


# ---------------------------------------------------------------------------
# Date / season helpers
# ---------------------------------------------------------------------------

def _today_et() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _date_to_season(date: str) -> str:
    ts = pd.to_datetime(date)
    y = ts.year
    return f"{y}-{str(y + 1)[2:]}" if ts.month >= 10 else f"{y - 1}-{str(y)[2:]}"


def _slug(game_id: str) -> str:
    return "playoffs" if nba_cdn.game_prefix(game_id) == "004" else "regular"


def _date_range(start: str, end: str) -> list[str]:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    if s > e:
        s = e
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(s, e, freq="D")]


# ---------------------------------------------------------------------------
# Existing state
# ---------------------------------------------------------------------------

def _game_log_frames() -> list[pd.DataFrame]:
    return [pd.read_csv(f) for f in sorted(RAW_DIR.glob("game_log_*.csv"))]


def _existing_game_ids() -> set[str]:
    ids: set[str] = set()
    for df in _game_log_frames():
        if "GAME_ID" in df.columns:
            ids.update(df["GAME_ID"].astype(str).str.zfill(10))
    return ids


def _last_logged_date() -> str | None:
    latest = None
    for df in _game_log_frames():
        if "GAME_DATE" in df.columns and not df.empty:
            d = pd.to_datetime(df["GAME_DATE"]).max()
            latest = d if latest is None else max(latest, d)
    return latest.strftime("%Y-%m-%d") if latest is not None else None


# ---------------------------------------------------------------------------
# CDN box score -> rows
# ---------------------------------------------------------------------------

def _game_log_rows(game: dict, game_id: str, game_date: str) -> list[dict]:
    home, away = game["homeTeam"], game["awayTeam"]
    h_ab, a_ab = home["teamTricode"], away["teamTricode"]
    h_pts, a_pts = home["score"], away["score"]

    def _row(team, matchup, pts, opp_pts):
        st = team.get("statistics", {}) or {}
        return {
            "GAME_ID":           game_id,
            "GAME_DATE":         game_date,
            "TEAM_ID":           int(team["teamId"]),
            "TEAM_ABBREVIATION": team["teamTricode"],
            "MATCHUP":           matchup,
            "WL":                "W" if pts > opp_pts else "L",
            "PTS":               pts,
            "PLUS_MINUS":        pts - opp_pts,
            "AST":               st.get("assists"),
            "REB":               st.get("reboundsTotal"),
            "TOV":               st.get("turnovers"),
        }

    return [
        _row(home, f"{h_ab} vs. {a_ab}", h_pts, a_pts),
        _row(away, f"{a_ab} @ {h_ab}",   a_pts, h_pts),
    ]


def _player_contribs(game: dict) -> list[tuple[str, int, float, str]]:
    """(player_id, team_id, points, player_name) for every player who logged minutes."""
    from pipeline.elo import _parse_iso_minutes

    out: list[tuple[str, int, float, str]] = []
    for team in (game["homeTeam"], game["awayTeam"]):
        tid = int(team["teamId"])
        for p in team.get("players", []):
            st = p.get("statistics", {}) or {}
            if _parse_iso_minutes(st.get("minutes")) <= 0:
                continue
            name = p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            out.append((str(p["personId"]), tid, float(st.get("points", 0)), name))
    return out


# ---------------------------------------------------------------------------
# Writers (append-only, idempotent)
# ---------------------------------------------------------------------------

def _append_game_logs(rows_by_file: dict, dry_run: bool) -> None:
    for (slug, season), rows in rows_by_file.items():
        path = RAW_DIR / f"game_log_{slug}_{season}.csv"
        new = pd.DataFrame(rows, columns=GAME_LOG_COLS)
        if path.exists():
            old = pd.read_csv(path)
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new
        combined["GAME_ID"] = combined["GAME_ID"].astype(str).str.zfill(10)
        combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="first")
        if dry_run:
            logger.info("[dry-run] %s: +%d row(s) -> %d total", path.name, len(new), len(combined))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(path, index=False)
            logger.info("Wrote %s (%d rows)", path.name, len(combined))


def _update_player_stats(season: str, contribs: list[tuple[str, int, float, str]], dry_run: bool) -> None:
    if not contribs:
        return
    path = RAW_DIR / f"player_season_stats_{season}.csv"

    agg: dict[int, list] = {}
    for pid, tid, pts, name in contribs:
        a = agg.setdefault(int(pid), [0.0, 0, tid, ""])
        a[0] += pts
        a[1] += 1
        a[2] = tid
        a[3] = name or a[3]

    if path.exists():
        df = pd.read_csv(path)
        df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
        df = df.set_index("PLAYER_ID")
    else:
        df = pd.DataFrame(columns=["PLAYER_NAME", "TEAM_ID", "GP", "PTS"])
        df.index.name = "PLAYER_ID"

    for pid, (pts_sum, gp, tid, name) in agg.items():
        if pid in df.index and "GP" in df.columns and pd.notna(df.at[pid, "GP"]):
            old_gp  = float(df.at[pid, "GP"])
            old_ppg = float(df.at[pid, "PTS"])
            new_gp  = old_gp + gp
            df.at[pid, "GP"]      = new_gp
            df.at[pid, "PTS"]     = round((old_ppg * old_gp + pts_sum) / new_gp, 4)
            df.at[pid, "TEAM_ID"] = tid
            if name:
                df.at[pid, "PLAYER_NAME"] = name
        else:
            df.loc[pid, "PLAYER_NAME"] = name
            df.loc[pid, "TEAM_ID"]     = tid
            df.loc[pid, "GP"]          = gp
            df.loc[pid, "PTS"]         = round(pts_sum / gp, 4)

    df = df.reset_index()
    if dry_run:
        logger.info("[dry-run] %s: %d player line(s) updated", path.name, len(agg))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Wrote %s (%d players)", path.name, len(df))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(date: str | None = None, dry_run: bool = False, schedule: dict | None = None) -> list[str]:
    """Catch the committed state up through ``date`` (ET, defaults to today). Returns
    the list of newly-applied stateful game IDs."""
    et_target = date or _today_et()
    if schedule is None:
        schedule = nba_cdn.fetch_schedule()

    existing = _existing_game_ids()
    last = _last_logged_date()
    if last is not None:
        start = (pd.to_datetime(last) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    else:
        start = et_target
    dates = _date_range(start, et_target)
    logger.info("Scanning %s..%s (%d day(s)) for completed games", dates[0], dates[-1], len(dates))

    stateful_boxes: dict[str, dict] = {}   # game_id -> game dict
    stateful_dates: dict[str, str]  = {}
    preseason_boxes: dict[str, dict] = {}
    preseason_dates: dict[str, str]  = {}

    for d in dates:
        for g in nba_cdn.games_for_date(d, schedule):
            gid = g["game_id"]
            if g["game_status"] != 3 or nba_cdn.is_all_star(gid):
                continue
            if nba_cdn.is_preseason(gid):
                box = nba_cdn.fetch_boxscore(gid)
                if box:
                    preseason_boxes[gid] = box
                    preseason_dates[gid] = d
                continue
            if not nba_cdn.is_stateful(gid) or gid in existing:
                continue
            box = nba_cdn.fetch_boxscore(gid)
            if not box:
                logger.info("  %s: box score not available yet", gid)
                continue
            stateful_boxes[gid] = box
            stateful_dates[gid] = d

    if not stateful_boxes and not preseason_boxes:
        logger.info("State already up to date — nothing to apply.")
        return []

    # 1. game logs + player season stats
    rows_by_file: dict = defaultdict(list)
    contribs_by_season: dict = defaultdict(list)
    for gid, box in stateful_boxes.items():
        d = stateful_dates[gid]
        season = _date_to_season(d)
        rows_by_file[(_slug(gid), season)].extend(_game_log_rows(box, gid, d))
        contribs_by_season[season].extend(_player_contribs(box))
    _append_game_logs(rows_by_file, dry_run)
    for season, contribs in contribs_by_season.items():
        _update_player_stats(season, contribs, dry_run)

    from pipeline.elo import load_cdn_game_players, update_elo, update_team_assignments

    # 2. incremental ELO on new stateful games (reuse already-fetched box scores)
    if stateful_boxes:
        games = [(gid, stateful_dates[gid]) for gid in stateful_boxes]
        if dry_run:
            logger.info("[dry-run] would apply update_elo to %d game(s)", len(games))
        else:
            update_elo(games, loader=lambda g: load_cdn_game_players(stateful_boxes.get(g)))

    # 3. preseason: refresh player -> team assignment only
    if preseason_boxes:
        games = [(gid, preseason_dates[gid]) for gid in preseason_boxes]
        if dry_run:
            logger.info("[dry-run] would refresh team assignment for %d preseason game(s)", len(games))
        else:
            update_team_assignments(games, loader=lambda g: load_cdn_game_players(preseason_boxes.get(g)))

    logger.info("Applied %d stateful game(s), %d preseason game(s)%s.",
                len(stateful_boxes), len(preseason_boxes), " [dry-run]" if dry_run else "")
    return list(stateful_boxes.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily CDN-driven state + ELO update")
    parser.add_argument("--date", help="Target ET date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing")
    args = parser.parse_args()
    run(date=args.date, dry_run=args.dry_run)
