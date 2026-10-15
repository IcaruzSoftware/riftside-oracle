"""
Feature engineering for NBA game outcome prediction.

Entry point: build_feature_matrix(...) → dict[str, float]
Returns a flat dict of named features ready for XGBoost.

Feature groups
--------------
  elo          — usage-weighted team ELO per skill + differentials
  rolling      — last-5 / last-10 team stats (pts, opp_pts, net, win%)
  rest         — days rest, back-to-back, games in last 7 days
  splits       — home vs away win% / scoring for each team this season
  h2h          — head-to-head record between the two teams this season
  availability — star player out, missing PPG weight
  efficiency   — rolling offensive vs defensive net rating proxy
  form         — ELO trajectory over last 10 games (rising / falling)
  streak       — current win/loss streak + last-5 win%

Pre-loading pattern (for the matrix builder script)
----------------------------------------------------
Load game_log_df and player_stats_df once across all seasons, then pass
them into build_feature_matrix() per game. The ELO dataframes are cached
at module level (loaded on first access).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

ELO_SKILLS = [
    "scoring", "playmaking", "defense",
    "rebounding", "efficiency", "hustle", "three_point",
]
_ELO_INITIAL = 1000.0

# ---------------------------------------------------------------------------
# ELO cache — loaded once, reused across all feature calls
# ---------------------------------------------------------------------------

_elo_history:   pd.DataFrame | None = None
_elo_current:   pd.DataFrame | None = None
_elo_recent:    pd.DataFrame | None = None
_elo_by_game:   dict | None = None   # game_id  → sub-DataFrame (O(1) lookup)
_elo_by_player: dict | None = None   # player_id → sorted sub-DataFrame (O(1) lookup)
_elo_recent_by_player: dict | None = None   # live-mode form: player_id → recent snapshots


def _get_elo_history() -> pd.DataFrame:
    global _elo_history
    if _elo_history is None:
        p = PROCESSED_DIR / "player_elo.parquet"
        if not p.exists():
            raise FileNotFoundError("player_elo.parquet not found — run elo.py first")
        _elo_history = pd.read_parquet(p)
        _elo_history["player_id"] = _elo_history["player_id"].astype(str)
    return _elo_history


def _get_elo_current() -> pd.DataFrame:
    global _elo_current
    if _elo_current is None:
        p = PROCESSED_DIR / "player_elo_current.parquet"
        if not p.exists():
            raise FileNotFoundError("player_elo_current.parquet not found — run elo.py first")
        _elo_current = pd.read_parquet(p)
        _elo_current["player_id"] = _elo_current["player_id"].astype(str)
    return _elo_current


def _get_elo_by_game() -> dict:
    """Dict of game_id → ELO sub-DataFrame. Built once, O(1) per lookup."""
    global _elo_by_game
    if _elo_by_game is None:
        df = _get_elo_history()
        _elo_by_game = {gid: sub for gid, sub in df.groupby("game_id")}
    return _elo_by_game


def _get_elo_by_player() -> dict:
    """Dict of player_id → ELO history sorted by date. Built once, O(1) per lookup."""
    global _elo_by_player
    if _elo_by_player is None:
        df = _get_elo_history()
        _elo_by_player = {
            pid: sub.sort_values("game_date").reset_index(drop=True)
            for pid, sub in df.groupby("player_id")
        }
    return _elo_by_player


def _get_elo_recent() -> pd.DataFrame:
    """Live-mode source for the form feature: last-N pre-game snapshots per player."""
    global _elo_recent
    if _elo_recent is None:
        p = PROCESSED_DIR / "player_elo_recent.parquet"
        if not p.exists():
            raise FileNotFoundError("player_elo_recent.parquet not found — run elo.py first")
        _elo_recent = pd.read_parquet(p)
        _elo_recent["player_id"] = _elo_recent["player_id"].astype(str)
    return _elo_recent


def _get_elo_recent_by_player() -> dict:
    """Dict of player_id → recent pre-game snapshots sorted by date (live form feature)."""
    global _elo_recent_by_player
    if _elo_recent_by_player is None:
        df = _get_elo_recent()
        _elo_recent_by_player = {
            pid: sub.sort_values("game_date").reset_index(drop=True)
            for pid, sub in df.groupby("player_id")
        }
    return _elo_recent_by_player


# ---------------------------------------------------------------------------
# Live-mode helpers (season derivation, PPG weights, roster from ELO state)
# ---------------------------------------------------------------------------

def _season_of(date) -> str:
    """NBA season string for a date (Oct–Sep season boundary), e.g. Mar 2024 → '2023-24'."""
    ts = pd.to_datetime(date)
    y = ts.year
    return f"{y}-{str(y + 1)[2:]}" if ts.month >= 10 else f"{y - 1}-{str(y)[2:]}"


def _prev_season(season: str) -> str:
    start = int(season[:4])
    return f"{start - 1}-{str(start)[2:]}"


def _build_ppg_lookup(*stats_frames) -> dict[str, float]:
    """player_id → PPG. Later frames override earlier ones (pass previous season first)."""
    ppg: dict[str, float] = {}
    for df in stats_frames:
        if df is not None and not df.empty and "PLAYER_ID" in df.columns:
            for _, row in df.iterrows():
                ppg[str(int(row["PLAYER_ID"]))] = max(float(row.get("PTS", 5.0)), 1.0)
    return ppg


def _live_team_lineup(current_elo: pd.DataFrame, team_id: str, season: str) -> pd.DataFrame:
    """
    Players making up a team's live lineup, taken from the ELO state.

    A player belongs to the team when their latest appearance (the team_id stored
    in the post-game state) is that team. Current-season appearances are preferred;
    if the team has fewer than 8 of those (e.g. early in the season, before rosters
    have played enough), we add previous-season appearances too. Players whose latest
    game is older than the previous season (i.e. retired / long gone) are excluded.
    """
    same_team = current_elo[
        pd.to_numeric(current_elo["team_id"], errors="coerce") == int(team_id)
    ]
    if same_team.empty:
        return same_team
    seasons = same_team["last_game_date"].apply(_season_of)
    in_season = same_team[seasons == season]
    if len(in_season) >= 8:
        return in_season
    return same_team[seasons.isin({season, _prev_season(season)})]


# ---------------------------------------------------------------------------
# Box score helpers
# ---------------------------------------------------------------------------

def _load_inactive_ids(game_id: str, team_id: str) -> set[str]:
    """Return personIds of inactive players for a team in a game."""
    path = RAW_DIR / f"boxscore_summary_{game_id}.json"
    if not path.exists():
        return set()
    try:
        d = json.loads(path.read_text())
        if d.get("_absent") or "InactivePlayers" not in d:
            return set()
        ds = d["InactivePlayers"]
        df = pd.DataFrame(ds["data"], columns=ds["headers"])
        team_rows = df[df["teamId"].astype(str) == str(team_id)]
        return set(team_rows["personId"].astype(str))
    except Exception:
        return set()


def _load_team_player_ids(game_id: str, team_id: str) -> set[str]:
    """Return personIds of all players who appeared for a team in a game."""
    path = RAW_DIR / f"boxscore_traditional_{game_id}.json"
    if not path.exists():
        return set()
    try:
        d = json.loads(path.read_text())
        if "PlayerStats" not in d:
            return set()
        ds = d["PlayerStats"]
        df = pd.DataFrame(ds["data"], columns=ds["headers"])
        team_rows = df[df["teamId"].astype(str) == str(team_id)]
        return set(team_rows["personId"].astype(str))
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# 1. ELO features
# ---------------------------------------------------------------------------

def _get_elo_features_live(
    home_team_id: str,
    away_team_id: str,
    season: str,
    player_stats_df: pd.DataFrame,
    prev_player_stats_df: pd.DataFrame | None,
) -> dict:
    """
    Live-prediction ELO features (no future box score exists yet).

    Each team's lineup is read from the ELO state (players whose latest appearance
    was for that team) instead of from the game's box score, and each team's
    composite ELO is the PPG-weighted average of those players' post-game ELOs.
    PPG weights come from the current season, falling back to the previous season
    for players without a current-season line yet.
    """
    current = _get_elo_current()
    ppg_lookup = _build_ppg_lookup(prev_player_stats_df, player_stats_df)
    all_skills = ELO_SKILLS + ["general"]

    def _weighted_team_elo(team_id: str) -> dict[str, float]:
        lineup = _live_team_lineup(current, team_id, season)
        if lineup.empty:
            return {s: _ELO_INITIAL for s in all_skills}

        weights: list[float] = []
        vals: dict[str, list] = {s: [] for s in all_skills}
        for _, row in lineup.iterrows():
            pid = str(row["player_id"])
            weights.append(ppg_lookup.get(pid, 5.0))
            for s in ELO_SKILLS:
                vals[s].append(float(row[f"{s}_elo"]))
            vals["general"].append(float(row["general_elo"]))

        total_w = sum(weights)
        return {
            s: sum(v * w for v, w in zip(vals[s], weights)) / total_w
            for s in all_skills
        }

    home_elo = _weighted_team_elo(home_team_id)
    away_elo = _weighted_team_elo(away_team_id)

    features: dict = {}
    for s in all_skills:
        features[f"home_{s}_elo"] = home_elo[s]
        features[f"away_{s}_elo"] = away_elo[s]
        features[f"elo_{s}_diff"] = home_elo[s] - away_elo[s]
    return features


def get_elo_features(
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    player_stats_df: pd.DataFrame,
    use_current: bool = False,
    season: str | None = None,
    prev_player_stats_df: pd.DataFrame | None = None,
) -> dict:
    """
    Compute usage-weighted team ELO per skill + home/away differentials.

    Each team's composite ELO is the PPG-weighted average of its active
    players' ELOs. Inactive players (from the summary file) are excluded
    before weighting so injuries are automatically reflected.

    Args:
        game_id:         NBA game ID (10-digit string)
        home_team_id:    NBA team ID for home team
        away_team_id:    NBA team ID for away team
        player_stats_df: Season player stats (for PPG weights)
        use_current:     True for live predictions — builds each team's lineup from
                         the ELO state (post-game player_elo_current) instead of
                         from this game's box score, which does not exist yet.
        season:          current season string (live mode only)
        prev_player_stats_df: previous-season player stats (live PPG fallback)

    Returns features:
        home_{skill}_elo, away_{skill}_elo, elo_{skill}_diff  × 8 skills
    """
    if use_current:
        return _get_elo_features_live(
            home_team_id, away_team_id, season, player_stats_df, prev_player_stats_df
        )

    game_elos = _get_elo_by_game().get(game_id, pd.DataFrame())

    # player_id → {skill: elo}
    elo_lookup: dict[str, dict] = {}
    for _, row in game_elos.iterrows():
        pid = str(row["player_id"])
        elo_lookup[pid] = {s: float(row[f"pre_{s}_elo"]) for s in ELO_SKILLS}
        elo_lookup[pid]["general"] = float(row["pre_general_elo"])

    # player_id → PPG weight (default 5.0 for unknown players)
    ppg_lookup: dict[str, float] = {}
    if player_stats_df is not None and not player_stats_df.empty:
        for _, row in player_stats_df.iterrows():
            pid = str(int(row["PLAYER_ID"]))
            ppg_lookup[pid] = max(float(row.get("PTS", 5.0)), 1.0)

    all_skills = ELO_SKILLS + ["general"]

    def _weighted_team_elo(team_id: str) -> dict[str, float]:
        player_ids = _load_team_player_ids(game_id, team_id)
        inactive   = _load_inactive_ids(game_id, team_id)
        active     = player_ids - inactive

        vals:    dict[str, list] = {s: [] for s in all_skills}
        weights: list[float] = []

        for pid in active:
            if pid not in elo_lookup:
                continue
            w = ppg_lookup.get(pid, 5.0)
            weights.append(w)
            for s in all_skills:
                vals[s].append(elo_lookup[pid][s])

        if not weights:
            return {s: _ELO_INITIAL for s in all_skills}

        total_w = sum(weights)
        return {
            s: sum(v * w for v, w in zip(vals[s], weights)) / total_w
            for s in all_skills
        }

    home_elo = _weighted_team_elo(home_team_id)
    away_elo = _weighted_team_elo(away_team_id)

    features: dict = {}
    for s in all_skills:
        features[f"home_{s}_elo"] = home_elo[s]
        features[f"away_{s}_elo"] = away_elo[s]
        features[f"elo_{s}_diff"] = home_elo[s] - away_elo[s]

    return features


# ---------------------------------------------------------------------------
# 2. Team rolling stats
# ---------------------------------------------------------------------------

def get_team_rolling_stats(
    team_id: str,
    game_date: str,
    game_log_df: pd.DataFrame,
) -> dict:
    """
    Rolling team stats (last-5 and last-10 games) entering a game.

    Returns keys: pts_last5, pts_last10, opp_pts_last5, opp_pts_last10,
                  net_last5, net_last10, win_pct_last5, win_pct_last10,
                  ast_last5, reb_last5, tov_last5
    """
    team  = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
    team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"])
    target = pd.to_datetime(game_date)
    prior  = team[team["GAME_DATE"] < target].sort_values("GAME_DATE")

    features: dict = {}

    for window, suffix in [(5, "last5"), (10, "last10")]:
        recent = prior.tail(window)
        if recent.empty:
            for key in ["pts", "opp_pts", "net", "win_pct"]:
                features[f"{key}_{suffix}"] = 0.0
            continue

        pts     = recent["PTS"].mean()
        net     = recent["PLUS_MINUS"].mean()
        opp_pts = pts - net                          # opp_pts = PTS - PLUS_MINUS
        wins    = (recent["WL"] == "W").mean()

        features[f"pts_{suffix}"]     = float(pts)
        features[f"opp_pts_{suffix}"] = float(opp_pts)
        features[f"net_{suffix}"]     = float(net)
        features[f"win_pct_{suffix}"] = float(wins)

    # Additional stats (last-5 only — enough signal, keeps feature count down)
    recent5 = prior.tail(5)
    for col, key in [("AST", "ast_last5"), ("REB", "reb_last5"), ("TOV", "tov_last5")]:
        features[key] = float(recent5[col].mean()) if not recent5.empty and col in recent5.columns else 0.0

    return features


# ---------------------------------------------------------------------------
# 3. Home / away splits
# ---------------------------------------------------------------------------

def get_home_away_splits(
    team_id: str,
    season: str,
    game_log_df: pd.DataFrame,
    game_date: str | None = None,
) -> dict:
    """
    Season-to-date home vs away performance splits (only prior games).

    Returns: win_pct_home, pts_avg_home, opp_pts_avg_home,
             win_pct_away, pts_avg_away, opp_pts_avg_away,
             home_away_win_pct_diff
    """
    home_games, away_games = split_home_away(team_id, game_log_df, game_date=game_date)

    def _split_stats(games: pd.DataFrame, label: str) -> dict:
        if games.empty:
            return {f"win_pct_{label}": 0.5, f"pts_avg_{label}": 110.0,
                    f"opp_pts_avg_{label}": 110.0}
        net  = games["PLUS_MINUS"].mean()
        pts  = games["PTS"].mean()
        return {
            f"win_pct_{label}":     float((games["WL"] == "W").mean()),
            f"pts_avg_{label}":     float(pts),
            f"opp_pts_avg_{label}": float(pts - net),
        }

    features = {}
    features.update(_split_stats(home_games, "home"))
    features.update(_split_stats(away_games, "away"))
    features["home_away_win_pct_diff"] = (
        features["win_pct_home"] - features["win_pct_away"]
    )
    return features


# ---------------------------------------------------------------------------
# 4. Rest and schedule
# ---------------------------------------------------------------------------

def get_rest_features(
    team_id: str,
    game_date: str,
    game_log_df: pd.DataFrame,
) -> dict:
    """
    Rest and schedule density features.

    Returns: days_rest, is_back_to_back, games_last_7
    """
    rest   = days_rest_calculator(team_id, game_date, game_log_df)
    b2b    = is_back_to_back(team_id, game_date, game_log_df)

    team   = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
    team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"])
    target = pd.to_datetime(game_date)
    window_start = target - pd.Timedelta(days=7)
    games_last_7 = int(
        ((team["GAME_DATE"] >= window_start) & (team["GAME_DATE"] < target)).sum()
    )

    return {
        "days_rest":      rest,
        "is_back_to_back": b2b,
        "games_last_7":   games_last_7,
    }


# ---------------------------------------------------------------------------
# 5. Head-to-head
# ---------------------------------------------------------------------------

def get_head_to_head_features(
    home_team_id: str,
    away_team_id: str,
    season: str,
    game_log_df: pd.DataFrame,
    game_date: str | None = None,
) -> dict:
    """
    Head-to-head record between the two teams this season (only prior games).

    Returns: h2h_home_wins, h2h_away_wins, h2h_home_win_pct
    """
    home_id = int(home_team_id)
    away_id = int(away_team_id)

    log = game_log_df.copy()
    if game_date is not None:
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
        log = log[log["GAME_DATE"] < pd.to_datetime(game_date)]

    # Find game_ids where both teams appear
    home_games = set(log[log["TEAM_ID"] == home_id]["GAME_ID"])
    away_games = set(log[log["TEAM_ID"] == away_id]["GAME_ID"])
    shared_ids = home_games & away_games

    if not shared_ids:
        return {"h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_home_win_pct": 0.5}

    h2h = game_log_df[
        (game_log_df["GAME_ID"].isin(shared_ids)) &
        (game_log_df["TEAM_ID"] == home_id)
    ]

    home_wins = int((h2h["WL"] == "W").sum())
    away_wins = int((h2h["WL"] == "L").sum())
    total     = home_wins + away_wins

    return {
        "h2h_home_wins":    home_wins,
        "h2h_away_wins":    away_wins,
        "h2h_home_win_pct": home_wins / total if total > 0 else 0.5,
    }


# ---------------------------------------------------------------------------
# 6. Player availability
# ---------------------------------------------------------------------------

def get_player_availability_features(
    team_id: str,
    game_id: str,
    player_stats_df: pd.DataFrame,
) -> dict:
    """
    Estimate the impact of missing players on team strength.

    Returns: star_player_out (1 if 20+ PPG player is inactive),
             missing_ppg (total PPG of inactive players),
             availability_score (active PPG / total roster PPG, 0–1)

    Live-mode note: there is no injury/inactives source for a game that has not been
    played (the box-score summary file does not exist yet), so these features stay
    neutral (star_player_out=0, missing_ppg=0, availability_score=1.0). Wire up an
    injury feed here if per-game availability is wanted for live predictions.
    """
    inactive_ids = _load_inactive_ids(game_id, team_id)

    if player_stats_df is None or player_stats_df.empty:
        return {"star_player_out": 0, "missing_ppg": 0.0, "availability_score": 1.0}

    # Filter to players on this team
    team_stats = player_stats_df[
        player_stats_df["TEAM_ID"].astype(str) == str(team_id)
    ].copy()

    if team_stats.empty:
        return {"star_player_out": 0, "missing_ppg": 0.0, "availability_score": 1.0}

    total_ppg   = float(team_stats["PTS"].sum())
    team_stats["pid_str"] = team_stats["PLAYER_ID"].astype(int).astype(str)
    inactive_rows = team_stats[team_stats["pid_str"].isin(inactive_ids)]

    missing_ppg  = float(inactive_rows["PTS"].sum())
    star_out     = int((inactive_rows["PTS"] >= 20.0).any())
    avail_score  = (total_ppg - missing_ppg) / total_ppg if total_ppg > 0 else 1.0

    return {
        "star_player_out":    star_out,
        "missing_ppg":        missing_ppg,
        "availability_score": float(np.clip(avail_score, 0.0, 1.0)),
    }


# ---------------------------------------------------------------------------
# 7. Efficiency differential
# ---------------------------------------------------------------------------

def get_efficiency_differential_features(
    home_team_id: str,
    away_team_id: str,
    game_log_df: pd.DataFrame,
    game_date: str,
) -> dict:
    """
    Offensive vs defensive net rating matchup (last-10 proxy).

    home_off_vs_away_def_diff: home rolling pts/game − away rolling opp_pts/game
    away_off_vs_home_def_diff: away rolling pts/game − home rolling opp_pts/game
    net_efficiency_advantage:  home rolling net − away rolling net
    """
    def _rolling(team_id: str, window: int = 10) -> dict:
        team   = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
        team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"])
        target = pd.to_datetime(game_date)
        recent = (
            team[team["GAME_DATE"] < target]
            .sort_values("GAME_DATE")
            .tail(window)
        )
        if recent.empty:
            return {"pts": 110.0, "opp_pts": 110.0, "net": 0.0}
        pts = float(recent["PTS"].mean())
        net = float(recent["PLUS_MINUS"].mean())
        return {"pts": pts, "opp_pts": pts - net, "net": net}

    home = _rolling(home_team_id)
    away = _rolling(away_team_id)

    return {
        "home_off_vs_away_def_diff": home["pts"]     - away["opp_pts"],
        "away_off_vs_home_def_diff": away["pts"]     - home["opp_pts"],
        "net_efficiency_advantage":  home["net"]      - away["net"],
    }


# ---------------------------------------------------------------------------
# 8. Player form (ELO trajectory)
# ---------------------------------------------------------------------------

def get_player_form_features(
    team_id: str,
    game_id: str,
    game_date: str,
    player_stats_df: pd.DataFrame,
    n_games: int = 10,
    use_current: bool = False,
) -> dict:
    """
    Measure recent form of top players via ELO trajectory.

    Finds the top 3 players on the team by PPG, then measures how much
    their general ELO changed over the last n_games. A rising ELO means
    they've been outperforming peers recently.

    In live mode (use_current) the trajectory is read from the small recent-snapshot
    state file instead of the full ELO history (which is not shipped to the runner).

    Returns: top{1,2,3}_elo_trend, avg_top3_elo_trend
    """
    elo_by_player = _get_elo_recent_by_player() if use_current else _get_elo_by_player()

    # Get top 3 players by PPG for this team
    if player_stats_df is None or player_stats_df.empty:
        return {"top1_elo_trend": 0.0, "top2_elo_trend": 0.0,
                "top3_elo_trend": 0.0, "avg_top3_elo_trend": 0.0}

    team_stats = player_stats_df[
        player_stats_df["TEAM_ID"].astype(str) == str(team_id)
    ].nlargest(3, "PTS")

    trends: list[float] = []

    for _, row in team_stats.iterrows():
        pid = str(int(row["PLAYER_ID"]))
        player_history = elo_by_player.get(pid)
        if player_history is None:
            trends.append(0.0)
            continue
        # Games strictly before this game
        before = player_history[
            player_history["game_date"] < pd.to_datetime(game_date)
        ]

        if len(before) < 2:
            trends.append(0.0)
            continue

        recent    = float(before["pre_general_elo"].iloc[-1])
        past      = float(before["pre_general_elo"].iloc[max(-n_games - 1, -len(before))])
        trends.append(recent - past)

    # Pad to 3 if fewer than 3 players found
    while len(trends) < 3:
        trends.append(0.0)

    return {
        "top1_elo_trend":     trends[0],
        "top2_elo_trend":     trends[1],
        "top3_elo_trend":     trends[2],
        "avg_top3_elo_trend": float(np.mean(trends)),
    }


# ---------------------------------------------------------------------------
# 9. Streak features
# ---------------------------------------------------------------------------

def get_streak_features(
    team_id: str,
    game_date: str,
    game_log_df: pd.DataFrame,
) -> dict:
    """
    Win/loss streak and recent win percentage entering a game.

    Returns: win_streak, loss_streak, last5_win_pct
    """
    team   = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
    team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"])
    target = pd.to_datetime(game_date)
    prior  = (
        team[team["GAME_DATE"] < target]
        .sort_values("GAME_DATE")["WL"]
        .tolist()
    )

    win_streak = loss_streak = 0
    if prior:
        last = prior[-1]
        for result in reversed(prior):
            if result == last:
                if last == "W":
                    win_streak += 1
                else:
                    loss_streak += 1
            else:
                break

    last5       = prior[-5:] if len(prior) >= 5 else prior
    last5_win_pct = sum(r == "W" for r in last5) / len(last5) if last5 else 0.5

    return {
        "win_streak":      win_streak,
        "loss_streak":     loss_streak,
        "last5_win_pct":   last5_win_pct,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    game_date: str,
    season: str,
    game_log_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    use_current_elo: bool = False,
    prev_player_stats_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Build a single flat feature dict for one game, ready for XGBoost.

    All features are prefixed with "home_" or "away_" where applicable.
    Differential features (home − away) are included for key metrics.

    Args:
        game_id:          NBA 10-digit game ID
        home_team_id:     NBA team ID for home team
        away_team_id:     NBA team ID for away team
        game_date:        ISO date string "YYYY-MM-DD"
        season:           Season string e.g. "2023-24"
        game_log_df:      Combined game log DataFrame (all seasons / types)
        player_stats_df:  Season player stats DataFrame for this season
        use_current_elo:  True when predicting today's games (no future games yet)

    Returns:
        dict[str, float] — flat feature vector, human-readable keys
    """
    features: dict = {}

    # ELO (most important feature group)
    try:
        elo = get_elo_features(
            game_id, home_team_id, away_team_id,
            player_stats_df, use_current=use_current_elo,
            season=season, prev_player_stats_df=prev_player_stats_df,
        )
        features.update(elo)
    except Exception as exc:
        logger.warning("elo_features %s: %s", game_id, exc)

    # Rolling team stats
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            rolling = get_team_rolling_stats(team_id, game_date, game_log_df)
            features.update({f"{prefix}_{k}": v for k, v in rolling.items()})
        except Exception as exc:
            logger.warning("rolling_stats %s %s: %s", prefix, game_id, exc)

    # Rolling stat differentials (home − away)
    for key in ["pts_last5", "pts_last10", "net_last5", "net_last10",
                "win_pct_last5", "win_pct_last10"]:
        h = features.get(f"home_{key}", 0.0)
        a = features.get(f"away_{key}", 0.0)
        features[f"diff_{key}"] = h - a

    # Home/away splits
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            splits = get_home_away_splits(team_id, season, game_log_df, game_date=game_date)
            features.update({f"{prefix}_{k}": v for k, v in splits.items()})
        except Exception as exc:
            logger.warning("splits %s %s: %s", prefix, game_id, exc)

    # Rest / schedule
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            rest = get_rest_features(team_id, game_date, game_log_df)
            features.update({f"{prefix}_{k}": v for k, v in rest.items()})
        except Exception as exc:
            logger.warning("rest %s %s: %s", prefix, game_id, exc)

    features["rest_diff"] = (
        features.get("home_days_rest", 0) - features.get("away_days_rest", 0)
    )
    features["b2b_diff"] = (
        features.get("home_is_back_to_back", 0) - features.get("away_is_back_to_back", 0)
    )

    # Head-to-head
    try:
        h2h = get_head_to_head_features(
            home_team_id, away_team_id, season, game_log_df, game_date=game_date
        )
        features.update(h2h)
    except Exception as exc:
        logger.warning("h2h %s: %s", game_id, exc)

    # Player availability
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            avail = get_player_availability_features(team_id, game_id, player_stats_df)
            features.update({f"{prefix}_{k}": v for k, v in avail.items()})
        except Exception as exc:
            logger.warning("availability %s %s: %s", prefix, game_id, exc)

    # Efficiency differential
    try:
        eff = get_efficiency_differential_features(
            home_team_id, away_team_id, game_log_df, game_date
        )
        features.update(eff)
    except Exception as exc:
        logger.warning("efficiency %s: %s", game_id, exc)

    # Player form (ELO trajectory)
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            form = get_player_form_features(
                team_id, game_id, game_date, player_stats_df,
                use_current=use_current_elo,
            )
            features.update({f"{prefix}_{k}": v for k, v in form.items()})
        except Exception as exc:
            logger.warning("form %s %s: %s", prefix, game_id, exc)

    # Streaks
    for prefix, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            streak = get_streak_features(team_id, game_date, game_log_df)
            features.update({f"{prefix}_{k}": v for k, v in streak.items()})
        except Exception as exc:
            logger.warning("streak %s %s: %s", prefix, game_id, exc)

    return {k: float(v) for k, v in features.items()}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def rolling_window(series: pd.Series, window: int) -> float:
    valid = series.dropna()
    if len(valid) < window:
        return float("nan")
    return float(valid.iloc[-window:].mean())


def split_home_away(
    team_id: str, game_log_df: pd.DataFrame, game_date: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a team's game log into home and away games, optionally capped before game_date."""
    team_games = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
    if game_date is not None:
        team_games["GAME_DATE"] = pd.to_datetime(team_games["GAME_DATE"])
        team_games = team_games[team_games["GAME_DATE"] < pd.to_datetime(game_date)]
    home_mask = team_games["MATCHUP"].str.contains(" vs. ", na=False)
    return team_games[home_mask], team_games[~home_mask]


def days_rest_calculator(
    team_id: str, game_date: str, game_log_df: pd.DataFrame
) -> int:
    """Days between the team's last game and game_date. Returns 7 if no prior game."""
    team = game_log_df[game_log_df["TEAM_ID"] == int(team_id)].copy()
    team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"])
    target = pd.to_datetime(game_date)
    prior  = team[team["GAME_DATE"] < target].sort_values("GAME_DATE")
    if prior.empty:
        return 7
    return int((target - prior.iloc[-1]["GAME_DATE"]).days - 1)


def is_back_to_back(
    team_id: str, game_date: str, game_log_df: pd.DataFrame
) -> int:
    """Return 1 if team played yesterday, else 0."""
    return int(days_rest_calculator(team_id, game_date, game_log_df) == 0)
