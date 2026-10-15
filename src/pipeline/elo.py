"""
Player ELO calculation from per-game box score data.

Design
------
Each player has 7 skill ELOs + 1 general ELO (sum of all skills).

Every game, for each skill:
  1. Qualify players with >= MIN_MINUTES played.
  2. Compute a composite skill score (z-scored components averaged).
  3. Rank players 1..N by score (1 = best).
  4. Convert rank → raw delta via a dynamic zero-centered scale:
       Odd  N: middle player gets 0, above +1/+2/..., below -1/-2/...
       Even N: no zero; top half +1/+2/..., bottom half -1/-2/...
  5. Scale delta by sqrt(minutes / 36) — shrinks noise for short stints.
  6. Mean-subtract all scaled deltas so the game is exactly zero-sum.
  7. Add delta to each player's skill ELO.

Skills
------
  scoring     — points/36 + true shooting %
  playmaking  — assists/36 + assist-to-turnover ratio
  defense     — defensive rating (inverted) + steals/36 + blocks/36
  rebounding  — rebound percentage
  efficiency  — PIE (NBA's Player Impact Estimate)
  hustle      — speed + distance + touches/36  [skipped if no tracking file]
  three_point — 3PM/36 + 3P%  [only for players who attempted >= 1 three]

Outputs
-------
  data/processed/player_elo.parquet
    One row per player per game — the ELO snapshot BEFORE that game.
    Columns: game_id, game_date, player_id,
             pre_{skill}_elo × 7, pre_general_elo
    Use in features.py: filter by game_id to get all players' incoming ELOs.

  data/processed/player_elo_current.parquet
    Most recent ELO per player — used for today's game predictions.

Run
---
  python src/pipeline/elo.py                # build from scratch
  python src/pipeline/elo.py --force        # reprocess even if output exists
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_ELO = 1000.0
MIN_MINUTES = 10.0      # minimum minutes to qualify for ranking in a game
RECENT_N    = 11        # pre-game snapshots kept per player in the "recent" state file
                        # (the form feature looks back up to 10 games → needs 11 rows)

SKILLS = [
    "scoring",
    "playmaking",
    "defense",
    "rebounding",
    "efficiency",
    "hustle",
    "three_point",
]

HISTORY_PATH = PROCESSED_DIR / "player_elo.parquet"          # full history (not committed)
CURRENT_PATH = PROCESSED_DIR / "player_elo_current.parquet"  # committed live state (post-game)
RECENT_PATH  = PROCESSED_DIR / "player_elo_recent.parquet"   # committed live state (pre-game tail)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _parse_minutes(value) -> float:
    """Parse 'MM:SS' string or bare float to float minutes. Returns 0.0 on failure."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60.0
        return float(s)
    except ValueError:
        return 0.0


def _load_player_dataset(game_id: str, endpoint: str) -> pd.DataFrame | None:
    """
    Load the PlayerStats dataset from one box score JSON file.
    Returns None if the file is absent, a sentinel, or unreadable.
    """
    path = RAW_DIR / f"boxscore_{endpoint}_{game_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get("_absent"):
        return None
    if "PlayerStats" not in data:
        return None
    ds = data["PlayerStats"]
    return pd.DataFrame(ds["data"], columns=ds["headers"])


def _name_column(df: pd.DataFrame) -> pd.Series:
    """Build 'First Last' from a frame's firstName/familyName columns (blank-safe)."""
    def col(name):
        return df[name].fillna("").astype(str) if name in df.columns else pd.Series("", index=df.index)
    return (col("firstName") + " " + col("familyName")).str.strip()


def load_game_players(game_id: str) -> pd.DataFrame | None:
    """
    Merge traditional + advanced + tracking box scores for one game.

    Returns a single DataFrame with one row per player containing all stats
    needed for ELO computation (plus a human-readable player_name). Returns None
    if core data is missing.
    """
    trad = _load_player_dataset(game_id, "traditional")
    adv  = _load_player_dataset(game_id, "advanced")

    if trad is None or adv is None:
        return None

    # Parse minutes to float
    trad = trad.copy()
    trad["minutes_float"] = trad["minutes"].apply(_parse_minutes)

    # Keep only what we need from advanced (avoid duplicate columns)
    adv_cols = [
        "personId",
        "defensiveRating",
        "trueShootingPercentage",
        "usagePercentage",
        "assistToTurnover",
        "reboundPercentage",
        "PIE",
    ]
    adv_cols = [c for c in adv_cols if c in adv.columns]
    df = trad.merge(adv[adv_cols], on="personId", how="left")

    # Optionally add tracking (hustle skill)
    track = _load_player_dataset(game_id, "tracking")
    if track is not None:
        track_cols = [c for c in ["personId", "speed", "distance", "touches"] if c in track.columns]
        df = df.merge(track[track_cols], on="personId", how="left")

    df["personId"] = df["personId"].astype(str)
    df["player_name"] = _name_column(df)
    return df


# ---------------------------------------------------------------------------
# CDN box-score adapter
# ---------------------------------------------------------------------------
#
# The daily pipeline reads box scores from cdn.nba.com's liveData feed, which
# carries only basic counting stats (no advanced or tracking box score). This
# adapter derives the advanced components ELO needs from those basics so the CDN
# feed produces the SAME frame shape as load_game_players():
#
#   trueShootingPercentage = PTS / (2 · (FGA + 0.44 · FTA))
#   assistToTurnover       = AST / TO
#   reboundPercentage      = 100 · TRB · (TeamMin/5) / (Min · (TeamTRB + OppTRB))
#   PIE                    = player game-impact numerator / game total
#
# defensiveRating and the tracking stats (speed/distance/touches) are absent —
# compute_skill_scores already falls back gracefully (_col median-fill for
# defensive rating; hustle skipped when tracking columns are missing).

def _parse_iso_minutes(value) -> float:
    """Parse an ISO-8601 duration like 'PT39M12.00S' to float minutes."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s.startswith("PT"):
        return _parse_minutes(s)
    mins = secs = 0.0
    num = ""
    for ch in s[2:]:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch == "M":
            mins = float(num) if num else 0.0
            num = ""
        elif ch == "S":
            secs = float(num) if num else 0.0
            num = ""
    return mins + secs / 60.0


def _pie_numerator(s: dict) -> float:
    """Player Impact Estimate numerator from basic counting stats."""
    return (
        float(s.get("points", 0))
        + float(s.get("fieldGoalsMade", 0)) + float(s.get("freeThrowsMade", 0))
        - float(s.get("fieldGoalsAttempted", 0)) - float(s.get("freeThrowsAttempted", 0))
        + float(s.get("reboundsDefensive", 0)) + 0.5 * float(s.get("reboundsOffensive", 0))
        + float(s.get("assists", 0)) + float(s.get("steals", 0))
        + 0.5 * float(s.get("blocks", 0))
        - float(s.get("foulsPersonal", 0)) - float(s.get("turnovers", 0))
    )


def load_cdn_game_players(game: dict | None) -> pd.DataFrame | None:
    """
    Build a per-player DataFrame from one cdn.nba.com liveData box score.

    Args:
        game: the ``game`` sub-dict of a boxscore JSON (both teams' ``players``).

    Returns a frame matching load_game_players()'s columns (personId, teamId,
    minutes_float, plus the counting/derived stats compute_skill_scores reads),
    or None if the box score is missing or has no player rows.
    """
    if not game:
        return None

    teams = [game.get("homeTeam", {}), game.get("awayTeam", {})]
    if not any(t.get("players") for t in teams):
        return None

    # Team-level rebound totals + minutes, for reboundPercentage.
    team_reb: dict[int, float] = {}
    team_min: dict[int, float] = {}
    for t in teams:
        tid = int(t.get("teamId", 0))
        reb = mins = 0.0
        for p in t.get("players", []):
            st = p.get("statistics", {}) or {}
            reb += float(st.get("reboundsTotal", 0))
            mins += _parse_iso_minutes(st.get("minutes"))
        team_reb[tid] = reb
        team_min[tid] = mins
    total_reb = sum(team_reb.values())

    rows: list[dict] = []
    for t in teams:
        tid = int(t.get("teamId", 0))
        opp_reb = total_reb - team_reb.get(tid, 0.0)
        for p in t.get("players", []):
            st = p.get("statistics", {}) or {}
            mp  = _parse_iso_minutes(st.get("minutes"))
            fga = float(st.get("fieldGoalsAttempted", 0))
            fta = float(st.get("freeThrowsAttempted", 0))
            pts = float(st.get("points", 0))
            ast = float(st.get("assists", 0))
            tov = float(st.get("turnovers", 0))
            trb = float(st.get("reboundsTotal", 0))

            tsa = fga + 0.44 * fta
            ts  = pts / (2.0 * tsa) if tsa > 0 else 0.0
            a2t = ast / tov if tov > 0 else ast
            denom = mp * (team_reb.get(tid, 0.0) + opp_reb)
            reb_pct = (100.0 * trb * (team_min.get(tid, 0.0) / 5.0) / denom) if denom > 0 else 0.0

            name = p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            rows.append({
                "personId":                str(p.get("personId")),
                "player_name":             name,
                "teamId":                  tid,
                "minutes_float":           mp,
                "points":                  pts,
                "assists":                 ast,
                "steals":                  float(st.get("steals", 0)),
                "blocks":                  float(st.get("blocks", 0)),
                "turnovers":               tov,
                "reboundsTotal":           trb,
                "threePointersMade":       float(st.get("threePointersMade", 0)),
                "threePointersAttempted":  float(st.get("threePointersAttempted", 0)),
                "threePointersPercentage": float(st.get("threePointersPercentage", 0) or 0.0),
                "trueShootingPercentage":  ts,
                "assistToTurnover":        a2t,
                "reboundPercentage":       reb_pct,
                "_pie_num":                _pie_numerator(st),
            })

    df = pd.DataFrame(rows)
    game_num = df["_pie_num"].sum()
    df["PIE"] = df["_pie_num"] / game_num if game_num else 0.0
    df = df.drop(columns=["_pie_num"])
    return df


# ---------------------------------------------------------------------------
# Skill score computation
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series) -> pd.Series:
    """Z-score a series within a game. Returns zeros if std == 0 or all NaN."""
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _per36(stat: pd.Series, minutes: pd.Series) -> pd.Series:
    """Normalize a counting stat to per-36-minutes rate."""
    return stat / minutes.clip(lower=0.01) * 36.0


def _col(df: pd.DataFrame, name: str, fill=0.0) -> pd.Series:
    """Get a column from df, filling with `fill` if absent or all-NaN."""
    if name not in df.columns:
        return pd.Series(fill, index=df.index, dtype=float)
    return df[name].fillna(fill).astype(float)


def compute_skill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a score_{skill} column for each skill to df.

    Each skill score is the sum of z-scored components. Z-scoring within
    the game puts all components on the same scale regardless of units.
    NaN scores mean the player is excluded from that skill's ranking.
    """
    df = df.copy()
    m = df["minutes_float"]

    # ── scoring ──────────────────────────────────────────────────────────────
    pts36 = _per36(_col(df, "points"), m)
    ts    = _col(df, "trueShootingPercentage", fill=df["trueShootingPercentage"].median()
                 if "trueShootingPercentage" in df.columns else 0.0)
    df["score_scoring"] = _zscore(pts36) + _zscore(ts)

    # ── playmaking ───────────────────────────────────────────────────────────
    ast36 = _per36(_col(df, "assists"), m)
    a2t   = _col(df, "assistToTurnover", fill=0.0)
    df["score_playmaking"] = _zscore(ast36) + _zscore(a2t)

    # ── defense ──────────────────────────────────────────────────────────────
    # defensiveRating: lower = better → negate z-score
    drtg  = _col(df, "defensiveRating",
                 fill=df["defensiveRating"].median() if "defensiveRating" in df.columns else 110.0)
    stl36 = _per36(_col(df, "steals"), m)
    blk36 = _per36(_col(df, "blocks"), m)
    df["score_defense"] = -_zscore(drtg) + _zscore(stl36) + _zscore(blk36)

    # ── rebounding ───────────────────────────────────────────────────────────
    reb_pct = _col(df, "reboundPercentage",
                   fill=_per36(_col(df, "reboundsTotal"), m))
    df["score_rebounding"] = _zscore(reb_pct)

    # ── efficiency ───────────────────────────────────────────────────────────
    pie = _col(df, "PIE", fill=0.0)
    df["score_efficiency"] = _zscore(pie)

    # ── hustle (requires tracking data) ──────────────────────────────────────
    has_tracking = all(c in df.columns for c in ["speed", "distance", "touches"])
    if has_tracking:
        speed  = _col(df, "speed",    fill=0.0)
        dist   = _col(df, "distance", fill=0.0)
        tch36  = _per36(_col(df, "touches", fill=0.0), m)
        df["score_hustle"] = _zscore(speed) + _zscore(dist) + _zscore(tch36)
    else:
        df["score_hustle"] = np.nan  # skip hustle this game

    # ── three_point (only for players who attempted >= 1 three) ──────────────
    tpa   = _col(df, "threePointersAttempted", fill=0.0)
    tpm36 = _per36(_col(df, "threePointersMade"), m)
    tp_pct = _col(df, "threePointersPercentage", fill=0.0)
    three_score = _zscore(tpm36) + _zscore(tp_pct)
    df["score_three_point"] = np.where(tpa > 0, three_score, np.nan)

    return df


# ---------------------------------------------------------------------------
# Rank → delta
# ---------------------------------------------------------------------------

def rank_to_delta(rank: int, n: int) -> int:
    """
    Convert 1-indexed rank to ELO delta on a dynamic zero-centered scale.

    Odd  N=15: rank 1 → +7, rank 8 → 0,  rank 15 → -7
    Even N=20: rank 1 → +10, rank 10 → +1, rank 11 → -1, rank 20 → -10

    Sum of all deltas is always 0 (zero-sum per game per skill).
    """
    if n % 2 == 1:          # odd — single middle player gets 0
        return (n + 1) // 2 - rank
    else:                   # even — no zero, gap between the two middle ranks
        half = n // 2
        return half - rank + (1 if rank <= half else 0)


def compute_deltas(scores: pd.Series, minutes: pd.Series) -> pd.Series:
    """
    Given per-player skill scores and minutes, return minutes-weighted
    zero-sum ELO deltas.

    Players with NaN score or < MIN_MINUTES receive NaN (no ELO update).

    Steps:
      1. Filter to qualified players (score not NaN, minutes >= MIN_MINUTES)
      2. Rank by score (1 = best); ties → average rank, rounded to nearest int
      3. raw_delta = rank_to_delta(rank, n)
      4. scaled    = raw_delta × sqrt(minutes / 36)
      5. final     = scaled − mean(scaled)   ← preserves zero-sum

    Returns a Series aligned to the original index, NaN for excluded players.
    """
    valid = scores.notna() & (minutes >= MIN_MINUTES)
    if valid.sum() < 2:
        return pd.Series(np.nan, index=scores.index)

    valid_scores  = scores[valid]
    valid_minutes = minutes[valid]
    n             = len(valid_scores)

    # Rank: 1 = highest score. Average ties, round to integer.
    ranks = valid_scores.rank(ascending=False, method="average").round().astype(int)
    raw   = ranks.map(lambda r: rank_to_delta(r, n)).astype(float)

    # Scale by minutes played
    scaled = raw * np.sqrt(valid_minutes / 36.0)

    # Mean-subtract → zero-sum
    final = scaled - scaled.mean()

    return final.reindex(scores.index)   # NaN for excluded players


# ---------------------------------------------------------------------------
# Game date index
# ---------------------------------------------------------------------------

def build_game_date_index() -> pd.DataFrame:
    """
    Return a DataFrame of (GAME_ID, GAME_DATE) sorted chronologically,
    built from all game log CSVs collected in Phase 1.
    """
    frames = []
    for csv in sorted(RAW_DIR.glob("game_log_*.csv")):
        df = pd.read_csv(csv, usecols=["GAME_ID", "GAME_DATE"])
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No game_log_*.csv files found in {RAW_DIR}")

    index = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("GAME_ID")
        .assign(
            GAME_ID   = lambda d: d["GAME_ID"].astype(str).str.zfill(10),
            GAME_DATE = lambda d: pd.to_datetime(d["GAME_DATE"]),
        )
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )
    return index


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

HISTORY_COLS = (
    ["game_id", "game_date", "player_id", "player_name", "pre_general_elo"]
    + [f"pre_{s}_elo" for s in SKILLS]
)


def _process_game(elo, meta, game_id, game_date, players_df, records) -> bool:
    """
    Apply one game to the live ELO state, in place.

    Snapshots each qualifying player's PRE-game ELO into ``records`` (identical to
    the historical layout), records each player's post-game team + date into
    ``meta``, then updates ``elo`` with the game's zero-sum skill deltas.

    Shared by both build_elo (stats box scores) and update_elo (CDN box scores).
    Returns True if the game contributed (had qualifying players), else False.
    """
    if players_df is None or players_df.empty:
        return False

    df = players_df[players_df["minutes_float"] >= MIN_MINUTES].copy()
    if df.empty:
        return False

    name_by_pid = (
        dict(zip(df["personId"], df["player_name"].fillna("").astype(str)))
        if "player_name" in df.columns else {}
    )

    # ── Snapshot PRE-game ELO for every qualifying player ────────────────────
    for pid in df["personId"].unique():
        player_elo = elo[pid]
        rec = {
            "game_id":         game_id,
            "game_date":       game_date,
            "player_id":       pid,
            "player_name":     name_by_pid.get(pid, ""),
            "pre_general_elo": sum(player_elo[s] for s in SKILLS),
        }
        for skill in SKILLS:
            rec[f"pre_{skill}_elo"] = player_elo[skill]
        records.append(rec)

    # ── Compute skill scores ─────────────────────────────────────────────────
    # Deduplicate personId — a player traded same-day can appear for both teams
    # in one game file. Keep the entry with the most minutes.
    df = compute_skill_scores(df)
    df = (df.sort_values("minutes_float", ascending=False)
            .drop_duplicates("personId", keep="first")
            .set_index("personId"))

    # ── Post-game team assignment (for live-lineup building) ──────────────────
    if "teamId" in df.columns:
        for pid, tid in df["teamId"].items():
            name = name_by_pid.get(pid, "") or meta.get(pid, {}).get("player_name", "")
            meta[pid] = {"team_id": int(tid), "last_game_date": game_date, "player_name": name}

    # ── For each skill: rank → delta → update ELO ────────────────────────────
    for skill in SKILLS:
        score_col = f"score_{skill}"
        if score_col not in df.columns:
            continue
        deltas = compute_deltas(df[score_col], df["minutes_float"])
        for pid, delta in deltas.items():
            if pd.notna(delta):
                elo[pid][skill] = elo[pid][skill] + float(delta)

    return True


def _tail_recent(records: list[dict]) -> pd.DataFrame:
    """Return the last RECENT_N pre-game snapshots per player, chronologically."""
    if not records:
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.DataFrame(records)[HISTORY_COLS]
    df = (df.sort_values("game_date")
            .groupby("player_id", group_keys=False)
            .tail(RECENT_N)
            .reset_index(drop=True))
    return df


def _write_state(elo, meta, records) -> pd.DataFrame:
    """Write the two committed live-state files (post-game current + recent tail)."""
    cur_rows = []
    for pid, skills in elo.items():
        m = meta.get(pid, {})
        row = {
            "player_id":      pid,
            "player_name":    m.get("player_name", ""),
            "team_id":        m.get("team_id"),
            "last_game_date": m.get("last_game_date"),
            "general_elo":    sum(skills[s] for s in SKILLS),
        }
        for s in SKILLS:
            row[f"{s}_elo"] = skills[s]
        cur_rows.append(row)

    current = pd.DataFrame(cur_rows)
    current["last_game_date"] = pd.to_datetime(current["last_game_date"])
    current = current.sort_values("player_id").reset_index(drop=True)
    current.to_parquet(CURRENT_PATH, index=False)
    logger.info("Saved current ELO (post-game): %d players → %s", len(current), CURRENT_PATH)

    recent = _tail_recent(records)
    recent.to_parquet(RECENT_PATH, index=False)
    logger.info("Saved recent ELO snapshots: %d rows → %s", len(recent), RECENT_PATH)
    return current


def _new_elo_state():
    return defaultdict(lambda: {s: INITIAL_ELO for s in SKILLS}), {}


def build_elo(force: bool = False, loader=load_game_players, game_index=None) -> pd.DataFrame:
    """
    Process all games chronologically and compute the full player ELO history.

    Also writes the two committed live-state files (player_elo_current.parquet with
    post-game ELO + team_id + last_game_date, and player_elo_recent.parquet).

        python src/pipeline/elo.py --force

    rebuilds everything from the box scores in data/raw/.

    Args:
        force:      Reprocess and overwrite even if the history file exists.
        loader:     game_id -> players DataFrame (defaults to the stats.nba.com
                    box-score loader; the CDN adapter can be injected for tests).
        game_index: optional (GAME_ID, GAME_DATE) frame; built from the raw game
                    logs when omitted.

    Returns:
        DataFrame with pre-game ELO snapshots (one row per player per game).
    """
    if game_index is None:
        if HISTORY_PATH.exists() and not force:
            logger.info("ELO already built — loading from %s", HISTORY_PATH)
            return pd.read_parquet(HISTORY_PATH)
        game_index = build_game_date_index()

    logger.info("Building ELO across %d games", len(game_index))

    elo, meta = _new_elo_state()
    records: list[dict] = []

    for _, row in tqdm(game_index.iterrows(), total=len(game_index), desc="ELO  games"):
        game_id   = str(row["GAME_ID"]).zfill(10)
        game_date = row["GAME_DATE"]
        _process_game(elo, meta, game_id, game_date, loader(game_id), records)

    if not records:
        raise RuntimeError("No ELO records generated — verify box score files exist")

    result = pd.DataFrame(records)[HISTORY_COLS]
    result.to_parquet(HISTORY_PATH, index=False)
    logger.info("Saved ELO history: %d records → %s", len(result), HISTORY_PATH)

    _write_state(elo, meta, records)
    return result


def _load_state():
    """Load the ELO dict + team/date meta + recent pre-game records from the state files."""
    elo, meta = _new_elo_state()
    if CURRENT_PATH.exists():
        cur = pd.read_parquet(CURRENT_PATH)
        cur["player_id"] = cur["player_id"].astype(str)
        has_name = "player_name" in cur.columns  # backward-compat with name-less state
        for _, r in cur.iterrows():
            pid = r["player_id"]
            elo[pid] = {s: float(r[f"{s}_elo"]) for s in SKILLS}
            tid = r["team_id"]
            meta[pid] = {
                "team_id":        int(tid) if pd.notna(tid) else None,
                "last_game_date": r["last_game_date"],
                "player_name":    (str(r["player_name"]) if has_name and pd.notna(r["player_name"]) else ""),
            }
    records: list[dict] = []
    if RECENT_PATH.exists():
        rec = pd.read_parquet(RECENT_PATH)
        rec["player_id"] = rec["player_id"].astype(str)
        records = rec.to_dict("records")
    return elo, meta, records


def _default_cdn_loader(game_id: str):
    from pipeline.nba_cdn import fetch_boxscore
    return load_cdn_game_players(fetch_boxscore(game_id))


def update_elo(games, loader=None) -> pd.DataFrame:
    """
    Incrementally continue the saved ELO state with new games (idempotent).

    Uses the exact same algorithm as build_elo, resuming from the committed state
    files. Games already reflected in player_elo_recent.parquet are skipped, so
    re-running with the same games is a no-op.

    Args:
        games:  iterable of (game_id, game_date) — applied in chronological order.
        loader: game_id -> players DataFrame (defaults to the CDN adapter).

    Returns the refreshed post-game current-state DataFrame.
    """
    if loader is None:
        loader = _default_cdn_loader

    elo, meta, records = _load_state()
    applied = {str(r["game_id"]).zfill(10) for r in records}

    ordered = sorted(
        ((str(g).zfill(10), pd.to_datetime(d)) for g, d in games),
        key=lambda x: (x[1], x[0]),
    )

    n_new = 0
    for game_id, game_date in ordered:
        if game_id in applied:
            logger.info("update_elo: %s already applied — skipping", game_id)
            continue
        if _process_game(elo, meta, game_id, game_date, loader(game_id), records):
            applied.add(game_id)
            n_new += 1
        else:
            logger.info("update_elo: %s had no box score / qualifying players", game_id)

    current = _write_state(elo, meta, records)
    logger.info("update_elo: applied %d new game(s); state now %d players", n_new, len(current))
    return current


def update_team_assignments(games, loader=None) -> pd.DataFrame:
    """
    Refresh only player -> team assignment (team_id + last_game_date) from games,
    without changing any ELO values or the recent snapshots. Used for preseason
    games, which set the current-season roster but must not affect skill ELO.
    """
    if loader is None:
        loader = _default_cdn_loader

    elo, meta, records = _load_state()
    for game_id, game_date in games:
        frame = loader(game_id)
        if frame is None or frame.empty or "teamId" not in frame.columns:
            continue
        gd = pd.to_datetime(game_date)
        df = frame[frame["minutes_float"] >= MIN_MINUTES]
        has_name = "player_name" in df.columns
        for _, r in df.iterrows():
            pid = str(r["personId"])
            existing = meta.get(pid)
            # Only move a player's team forward in time — never let an older game
            # (e.g. a preseason game in the same scan window) overwrite a newer one.
            if existing and existing.get("last_game_date") is not None \
                    and gd < pd.to_datetime(existing["last_game_date"]):
                continue
            _ = elo[pid]  # ensure the player exists in the state (defaults to 1000)
            name = (str(r["player_name"]) if has_name else "") or (existing or {}).get("player_name", "")
            meta[pid] = {"team_id": int(r["teamId"]), "last_game_date": gd, "player_name": name}

    return _write_state(elo, meta, records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build player ELO from box score data")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    args = parser.parse_args()

    df = build_elo(force=args.force)
    print(f"\nDone. {len(df):,} records, {df['player_id'].nunique():,} unique players.")
    print(df.head(3).to_string())
