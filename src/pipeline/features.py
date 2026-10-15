"""
Feature engineering for pro LoL: P(blue side wins one game).

Entry point: build_feature_matrix(...) -> dict[str, float]
Returns a flat dict of named features ready for XGBoost. Per-team features come
as blue_*, red_* and diff_* (blue - red).

Feature groups (one get_*_features() each, each in its own try/except)
-----------------------------------------------------------------------
  elo      starter means of outcome/general/5 skill ELOs + per-role lane diffs
  rolling  last-5 / last-10 games: win%, gd@15, game length, kills/deaths per min,
           first blood/dragon/herald/tower/baron rates, dragons per game
  side     win% on blue / on red in the last 365 days, plus the side-matching one
  rest     days since the last game, games in the last 7 days
  h2h      head-to-head games / blue-team wins / win% in the last 365 days
  roster   starters shared by the team's two latest lineups, rookies (< 20 games)
  form     mean starter change in general/outcome ELO over their last 10 games
  streak   signed current win/loss streak (last-5 win% is rolling's win_pct_last5)
  context  is_international, playoffs, league tier, each team's home-league mean
           outcome ELO

Point-in-time rule (day level)
------------------------------
A game on calendar day d sees only team_games rows with day < d and player ELO
"as of the start of day d":
  training  pre-game snapshot of the player's first game on/after d
            (player_elo.parquet), else the player's current value
  live      the player's row in player_elo_current.parquet
Live predictions are made before a series with data up to the previous day, so in
training game 2 of a series must not see game 1 either.

Training vs live (the one difference: rosters)
----------------------------------------------
  starters  training: the 5 who played (team_games row of game_id); live: the
            team's latest lineup before d (or the caller's starters)
  missing   training: an empty slot / unknown id is left out of team means (its
            lane diffs are NaN); live: §5.2 init value (outcome = mean of current
            players last seen in the team's home league if >= 10, else the league
            prior; skills 1000 each)
  form      training: history snapshots; live: player_elo_recent.parquet

team_games_df is the oe.TEAM_GAMES_COLS frame (training: all OE years in memory,
live: the committed team_games_*.csv). It is indexed once per DataFrame object, so
pass the same object for every game and don't mutate it in between.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline import elo, oe

logger = logging.getLogger(__name__)

VALS = ["outcome", "general"] + elo.SKILLS      # column order of every ELO vector
_SNAP_COLS = [f"pre_{v}_elo" for v in VALS]
_CUR_COLS = [f"{v}_elo" for v in VALS]

ROLL_WINDOWS = (5, 10)
YEAR_DAYS = 365
REST_WINDOW = 7
FORM_N = 10
ROOKIE_GAMES = 20
STAT_NAMES = ["win_pct", "gd15", "gamelength", "kills_pm", "deaths_pm", "firstblood_rate",
              "firstdragon_rate", "firstherald_rate", "firsttower_rate", "firstbaron_rate",
              "dragons"]
_NO_STARTERS = (None,) * len(oe.ROLES)

# Lazy caches: ELO state per mode ("train"/"live") and the team_games index ("tg").
# Paths are elo.HISTORY_PATH / CURRENT_PATH / RECENT_PATH, read at load time.
_cache: dict = {}


def _days(dates) -> np.ndarray:
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)


def _day(date) -> int:
    return int(np.datetime64(pd.Timestamp(date).date(), "D").astype(np.int64))


def _nanmean(x: np.ndarray) -> np.ndarray:
    """Column-wise NaN-skipping mean of a 2D array; all-NaN or empty column -> NaN."""
    ok = ~np.isnan(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, x, 0.0).sum(0) / ok.sum(0)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else np.nan


# ---------------------------------------------------------------------------
# ELO state
# ---------------------------------------------------------------------------

def _by_player(df: pd.DataFrame) -> dict:
    """player_id -> (day array, value matrix in VALS order), chronological per player."""
    pid = df["player_id"].to_numpy()
    order = np.argsort(pid, kind="stable")          # files are chronological: keep that order
    pid, days, vals = pid[order], _days(df["game_date"])[order], df[_SNAP_COLS].to_numpy(float)[order]
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]]) if len(pid) else np.array([], int)
    ends = np.r_[starts[1:], len(pid)]
    return {pid[a]: (days[a:b], vals[a:b]) for a, b in zip(starts, ends)}


def _elo_state(live: bool) -> dict:
    """
    cur:    player_id -> (post-game ELO vector, games_played) from the current file
    snaps:  player_id -> pre-game snapshots (training: full history, live: recent tail)
    league: OE league -> (outcome sum, player count) by last league (live init only)
    """
    key = "live" if live else "train"
    if key not in _cache:
        cur = pd.read_parquet(elo.CURRENT_PATH)
        st = {"cur": dict(zip(cur["player_id"], zip(cur[_CUR_COLS].to_numpy(float),
                                                     cur["games_played"].tolist()))),
              "snaps": _by_player(pd.read_parquet(elo.RECENT_PATH if live else elo.HISTORY_PATH))}
        agg = cur.groupby("league")["outcome_elo"].agg(["sum", "count"])
        st["league"] = dict(zip(agg.index, zip(agg["sum"], agg["count"])))
        _cache[key] = st
    return _cache[key]


def _player(g: "_Game", pid) -> tuple[np.ndarray | None, int]:
    """(ELO vector as of the start of g.day, games played before it); (None, 0) if unknown."""
    if g.live:
        return g.st["cur"].get(pid, (None, 0))
    s = g.st["snaps"].get(pid)
    if s is None:
        return None, 0
    i = int(np.searchsorted(s[0], g.day))
    return (s[1][i] if i < len(s[0]) else g.st["cur"][pid][0]), i


def _init(g: "_Game", league) -> np.ndarray:
    """Live §5.2 init vector for a player missing from the ELO state."""
    s, c = g.st["league"].get(league, (0.0, 0))
    o = s / c if c >= elo.MIN_LEAGUE_PLAYERS else elo.LEAGUE_PRIOR.get(league, elo.DEFAULT_PRIOR)
    return np.array([o, elo.INITIAL_ELO * len(elo.SKILLS)] + [elo.INITIAL_ELO] * len(elo.SKILLS))


def _lineup(g: "_Game", starters, team_id) -> np.ndarray:
    """(5, len(VALS)) ELO matrix of a lineup in role order; missing -> NaN (training) / init (live)."""
    rows = []
    for pid in starters:
        v = _player(g, pid)[0] if pid else None
        if v is None:
            v = _init(g, _home_league(g, team_id) or g.league) if g.live else np.full(len(VALS), np.nan)
        rows.append(v)
    return np.array(rows)


# ---------------------------------------------------------------------------
# team_games index
# ---------------------------------------------------------------------------

def _parse_starters(s) -> tuple:
    return tuple(p if isinstance(p, str) and p else None for p in s) if s is not None else _NO_STARTERS


def _team_index(tg: pd.DataFrame) -> dict:
    """
    Built once per team_games DataFrame object:
      teams:    teamid -> per-row arrays sorted by (date, gameid) (NaN teamids dropped)
      games:    (gameid, side) -> starters tuple
      leagues:  non-international league -> (day array, teamid array) of its team rows
      lmean:    (league, day, live) -> league mean outcome ELO (filled lazily)
    """
    hit = _cache.get("tg")
    if hit is not None and hit[0] is tg:
        return hit[1]
    t = tg.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    starters = [_parse_starters(s.split("|") if isinstance(s, str) else None) for s in t["starters"]]
    keys = list(zip(t["gameid"], t["side"]))
    side_team = dict(zip(keys, t["teamid"]))
    opp = np.array([side_team.get((gid, "Red" if side == "Blue" else "Blue")) for gid, side in keys],
                   dtype=object)
    minutes = t["gamelength"].astype(float) / 60.0
    stats = np.column_stack([
        t["result"].astype(float), t["golddiffat15"], minutes,
        t["teamkills"] / minutes, t["teamdeaths"] / minutes,
        t["firstblood"], t["firstdragon"], t["firstherald"], t["firsttower"], t["firstbaron"],
        t["dragons"],
    ]).astype(float)
    day = _days(t["date"])
    blue = (t["side"] == "Blue").to_numpy()
    league = t["league"].to_numpy(dtype=object)
    domestic = ~t["league"].isin(elo.INTERNATIONAL).to_numpy()
    teamid = t["teamid"].to_numpy(dtype=object)
    known = t["teamid"].notna().to_numpy()

    teams, pos = {}, np.flatnonzero(known)
    for tid, rows in t[known].groupby("teamid").indices.items():
        rows = pos[rows]
        last_dom = np.where(domestic[rows], np.arange(len(rows)), -1)
        teams[tid] = {"day": day[rows], "stats": stats[rows], "blue": blue[rows], "opp": opp[rows],
                      "league": league[rows], "starters": [starters[i] for i in rows],
                      "home": np.maximum.accumulate(last_dom)}
    dom = np.flatnonzero(known & domestic)
    leagues = {lg: (day[dom[rows]], teamid[dom[rows]])
               for lg, rows in pd.Series(league[dom]).groupby(league[dom]).indices.items()}
    ix = {"teams": teams, "games": dict(zip(keys, starters)), "leagues": leagues, "lmean": {}}
    _cache["tg"] = (tg, ix)
    return ix


_EMPTY_TEAM = {"day": np.empty(0, np.int64), "stats": np.empty((0, len(STAT_NAMES))),
               "blue": np.empty(0, bool), "opp": np.empty(0, object), "league": np.empty(0, object),
               "starters": [], "home": np.empty(0, np.int64)}


def _prior(g: "_Game", team_id) -> tuple[dict, int]:
    """(the team's index arrays, number of its rows before g.day)."""
    t = g.ix["teams"].get(team_id, _EMPTY_TEAM)
    return t, int(np.searchsorted(t["day"], g.day))


def _home_league(g: "_Game", team_id):
    """The team's most recent non-international league before g.day (within 365 days), or None."""
    t, n = _prior(g, team_id)
    k = t["home"][n - 1] if n else -1
    return t["league"][k] if k >= 0 and t["day"][k] >= g.day - YEAR_DAYS else None


@dataclass
class _Game:
    ix: dict
    st: dict
    day: int
    live: bool
    blue: str
    red: str
    blue_starters: tuple
    red_starters: tuple
    league: str
    playoffs: int


def _pair(b: dict, r: dict) -> dict:
    return ({f"blue_{k}": v for k, v in b.items()} | {f"red_{k}": v for k, v in r.items()}
            | {f"diff_{k}": b[k] - r[k] for k in b})


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def get_elo_features(g: _Game) -> dict:
    """Starter means of the VALS ELOs per team + diffs, and per-role general/outcome lane diffs."""
    b, r = _lineup(g, g.blue_starters, g.blue), _lineup(g, g.red_starters, g.red)
    out = _pair(dict(zip([f"{v}_elo" for v in VALS], _nanmean(b))),
                dict(zip([f"{v}_elo" for v in VALS], _nanmean(r))))
    for i, role in enumerate(oe.ROLES):
        out[f"diff_{role}_general_elo"] = b[i, 1] - r[i, 1]
        out[f"diff_{role}_outcome_elo"] = b[i, 0] - r[i, 0]
    return out


def get_team_rolling_stats(g: _Game) -> dict:
    def one(team_id):
        t, n = _prior(g, team_id)
        out = {}
        for w in ROLL_WINDOWS:
            out.update({f"{s}_last{w}": v
                        for s, v in zip(STAT_NAMES, _nanmean(t["stats"][max(n - w, 0):n]))})
        return out
    return _pair(one(g.blue), one(g.red))


def get_side_splits(g: _Game) -> dict:
    def one(team_id):
        t, n = _prior(g, team_id)
        lo = int(np.searchsorted(t["day"], g.day - YEAR_DAYS))
        res, blue = t["stats"][lo:n, 0], t["blue"][lo:n]
        return {"win_pct_on_blue": _mean(res[blue]), "win_pct_on_red": _mean(res[~blue])}
    b, r = one(g.blue), one(g.red)
    out = _pair(b, r)
    out["blue_win_pct_own_side"] = b["win_pct_on_blue"]
    out["red_win_pct_own_side"] = r["win_pct_on_red"]
    out["diff_win_pct_own_side"] = b["win_pct_on_blue"] - r["win_pct_on_red"]
    return out


def get_rest_features(g: _Game) -> dict:
    def one(team_id):
        t, n = _prior(g, team_id)
        return {"days_since_last": g.day - t["day"][n - 1] if n else np.nan,
                "games_last_7": n - int(np.searchsorted(t["day"], g.day - REST_WINDOW))}
    return _pair(one(g.blue), one(g.red))


def get_head_to_head_features(g: _Game) -> dict:
    t, n = _prior(g, g.blue)
    lo = int(np.searchsorted(t["day"], g.day - YEAR_DAYS))
    met = t["opp"][lo:n] == g.red
    games, wins = int(met.sum()), float(t["stats"][lo:n, 0][met].sum())
    return {"h2h_games": games, "h2h_blue_wins": wins,
            "h2h_blue_win_pct": wins / games if games else np.nan}


def get_roster_features(g: _Game) -> dict:
    def one(team_id, starters):
        t, n = _prior(g, team_id)
        churn = (len({p for p in t["starters"][n - 1] if p} & {p for p in t["starters"][n - 2] if p})
                 if n >= 2 else np.nan)
        rookies = sum(_player(g, p)[1] < ROOKIE_GAMES for p in starters if p)
        return {"roster_churn": churn, "rookie_count": rookies}
    return _pair(one(g.blue, g.blue_starters), one(g.red, g.red_starters))


def get_player_form_features(g: _Game) -> dict:
    """Mean over starters of (ELO as of today - pre-game ELO FORM_N games ago, or first game)."""
    def one(starters):
        deltas = []
        for pid in starters:
            s = g.st["snaps"].get(pid) if pid else None
            n = int(np.searchsorted(s[0], g.day)) if s is not None else 0
            if n:
                deltas.append(_player(g, pid)[0][:2] - s[1][max(n - FORM_N, 0), :2])
        m = np.mean(deltas, axis=0) if deltas else (np.nan, np.nan)
        return {"form_outcome_elo": m[0], "form_general_elo": m[1]}
    return _pair(one(g.blue_starters), one(g.red_starters))


def get_streak_features(g: _Game) -> dict:
    def one(team_id):
        t, n = _prior(g, team_id)
        res = t["stats"][:n, 0]
        k = 0
        while k < n and res[n - 1 - k] == res[n - 1]:
            k += 1
        return {"streak": k if n and res[n - 1] == 1 else -k}
    return _pair(one(g.blue), one(g.red))


def _league_mean(g: _Game, league) -> float:
    """
    Mean team strength of the league as of g.day: over every team whose home league
    is `league`, strength = mean as-of outcome ELO of its last lineup. Cached per day.
    """
    if league is None:
        return np.nan
    key = (league, g.day, g.live)
    if key not in g.ix["lmean"]:
        days, teams = g.ix["leagues"].get(league, (np.empty(0, np.int64), np.empty(0, object)))
        lo, hi = np.searchsorted(days, [g.day - YEAR_DAYS, g.day])
        strengths = []
        for tid in dict.fromkeys(teams[lo:hi]):        # ordered unique
            if _home_league(g, tid) == league:
                t, n = _prior(g, tid)
                s = _nanmean(_lineup(g, t["starters"][n - 1], tid)[:, :1])[0]
                if not np.isnan(s):
                    strengths.append(s)
        g.ix["lmean"][key] = float(np.mean(strengths)) if strengths else np.nan
    return g.ix["lmean"][key]


def get_context_features(g: _Game) -> dict:
    out = {"is_international": g.league in elo.INTERNATIONAL, "playoffs": g.playoffs,
           "league_tier": elo.LEAGUE_PRIOR.get(g.league, elo.DEFAULT_PRIOR)}
    out.update(_pair({"league_mean_outcome_elo": _league_mean(g, _home_league(g, g.blue))},
                     {"league_mean_outcome_elo": _league_mean(g, _home_league(g, g.red))}))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _starters(g: _Game, team_id, side: str, game_id) -> tuple:
    """Training: who played game_id on that side. Live: the team's latest lineup before g.day."""
    if not g.live and game_id is not None:
        return g.ix["games"].get((game_id, side), _NO_STARTERS)
    t, n = _prior(g, team_id)
    return t["starters"][n - 1] if n else _NO_STARTERS


def build_feature_matrix(
    blue_team_id: str,
    red_team_id: str,
    game_date,
    context: dict,
    team_games_df: pd.DataFrame,
    use_current_elo: bool,
    game_id: str | None = None,
    blue_starters=None,
    red_starters=None,
) -> dict[str, float]:
    """
    One flat feature dict for one game, blue side vs red side.

    Args:
        blue_team_id / red_team_id: OE teamids.
        game_date:       the game's date (anything pd.Timestamp parses); only the day counts.
        context:         {"league": OE league code of the game, "playoffs": 0/1}.
        team_games_df:   oe.TEAM_GAMES_COLS rows (see module docstring).
        use_current_elo: False = training (history snapshots), True = live (current state).
        game_id:         training: the OE gameid whose actual starters are used.
        blue_starters / red_starters: 5 playerids in role order (None/"" = empty slot);
                         overrides the default lineup.

    Every group runs in its own try/except; a failing group's keys are just missing.
    """
    g = _Game(ix=_team_index(team_games_df), st=_elo_state(use_current_elo), day=_day(game_date),
              live=use_current_elo, blue=blue_team_id, red=red_team_id, blue_starters=_NO_STARTERS,
              red_starters=_NO_STARTERS, league=context["league"], playoffs=int(context["playoffs"]))
    g.blue_starters = (_parse_starters(blue_starters) if blue_starters is not None
                       else _starters(g, blue_team_id, "Blue", game_id))
    g.red_starters = (_parse_starters(red_starters) if red_starters is not None
                      else _starters(g, red_team_id, "Red", game_id))

    features: dict = {}
    for group in (get_elo_features, get_team_rolling_stats, get_side_splits, get_rest_features,
                  get_head_to_head_features, get_roster_features, get_player_form_features,
                  get_streak_features, get_context_features):
        try:
            features.update(group(g))
        except Exception as exc:
            logger.warning("%s %s %s-%s: %s", getattr(group, "__name__", group), game_id,
                           blue_team_id, red_team_id, exc)
    return {k: float(v) for k, v in features.items()}


def series_win_prob(p: float, best_of: int) -> float:
    """
    P(winning a best-of-n series) = P(first to ceil(n/2) wins) with per-game win prob p.
    Bo1 p, Bo3 p^2(3-2p), Bo5 p^3(10-15p+6p^2).
    """
    # ponytail: iid games, ignores momentum and side alternation
    need = math.ceil(best_of / 2)
    return sum(math.comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))
