"""
Data collection from nba_api — full historical pull (2015-16 onward).

Pulls everything needed for the player-centric ELO + play% prediction system.
Supports proxy rotation for parallel collection — each proxy gets its own rate-limit
bucket, so N proxies = N× throughput with full safety.

Configuration via environment variables:
  PROXIES   Comma-separated proxy URLs, e.g.:
              PROXIES=http://user:pass@host1:port,http://user:pass@host2:port
            Leave unset to run single-threaded without proxies.
  WORKERS   Number of concurrent workers (defaults to number of proxies, min 1).

What gets pulled:
  Per season  — game logs (regular + playoffs), player matchups (off + def),
                player tracking stats (6 types), tracking defense, 5-man lineups
  Per game    — traditional, advanced, summary (inactives), four factors, misc,
                player tracking box scores
  Per player  — bio info (position, height, weight, age, draft year)

Run collect_all() once to bootstrap. Already-cached files are skipped at the
individual file level — safe to kill and restart at any point.

Expected runtime:
  No proxies  : ~12–18 hours
  3 proxies   : ~4–6 hours
  5 proxies   : ~2.5–4 hours
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscorefourfactorsv3,
    boxscorehustlev2,
    boxscorematchupsv3,
    boxscoremiscv3,
    boxscoreplayertrackv3,
    boxscoresummaryv3,
    boxscoretraditionalv3,
    commonplayerinfo,
    leaguedashlineups,
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptstats,
    leaguegamelog,
    leaguehustlestatsplayer,
    leagueseasonmatchups,
    scoreboardv2,
    synergyplaytypes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2026)]

REQUEST_DELAY   = 0.6    # minimum seconds between calls on the same proxy slot
RETRY_BASE_WAIT = 2.0    # seconds before first retry (doubles each attempt)
MAX_RETRIES     = 3

PT_STAT_TYPES = [
    "Drives",
    "CatchShoot",
    "PullUpShot",
    "Possessions",
    "Passing",
    "SpeedDistance",
]

# Core endpoints — full retry logic, data should always be present
BOXSCORE_ENDPOINTS_CORE = [
    ("traditional", boxscoretraditionalv3.BoxScoreTraditionalV3),
    ("advanced",    boxscoreadvancedv3.BoxScoreAdvancedV3),
    ("summary",     boxscoresummaryv3.BoxScoreSummaryV3),
    ("fourfactors", boxscorefourfactorsv3.BoxScoreFourFactorsV3),
    ("misc",        boxscoremiscv3.BoxScoreMiscV3),
    ("tracking",    boxscoreplayertrackv3.BoxScorePlayerTrackV3),
]

# Optional endpoints — try once, fail silently. Data is absent for older games
# and retrying wastes 14+ seconds per game across 14,000 games.
BOXSCORE_ENDPOINTS_OPTIONAL = [
    ("matchups", boxscorematchupsv3.BoxScoreMatchupsV3),
    ("hustle",   boxscorehustlev2.BoxScoreHustleV2),
]


# ---------------------------------------------------------------------------
# Proxy pool — round-robin with per-slot rate limiting
# ---------------------------------------------------------------------------

class ProxyPool:
    """
    Manages a list of proxy slots with independent rate-limit tracking.

    Each slot gets its own lock and last-call timestamp, so concurrent workers
    on different proxies never interfere with each other's delays.

    Usage:
        result = pool.call(SomeEndpoint, param=value)
    """

    def __init__(self, proxies: list[str | None]) -> None:
        self._slots   = proxies
        self._n       = len(proxies)
        self._locks   = [threading.Lock() for _ in proxies]
        self._last    = [0.0] * len(proxies)
        self._rr_idx  = 0
        self._rr_lock = threading.Lock()

    def _next(self) -> int:
        """Return the next slot index in round-robin order."""
        with self._rr_lock:
            idx = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % self._n
        return idx

    def call(self, endpoint_cls, **kwargs):
        """
        Make one API call through the next available proxy slot.
        Respects per-slot rate limiting and retries with exponential backoff.
        """
        idx   = self._next()
        proxy = self._slots[idx]

        for attempt in range(MAX_RETRIES):
            # Enforce rate limit on this specific slot
            with self._locks[idx]:
                wait = REQUEST_DELAY - (time.time() - self._last[idx])
                if wait > 0:
                    time.sleep(wait)
                self._last[idx] = time.time()

            try:
                if proxy:
                    return endpoint_cls(**kwargs, proxy=proxy)
                return endpoint_cls(**kwargs)

            except Exception as exc:
                retry_wait = RETRY_BASE_WAIT * (2 ** attempt)
                logger.warning(
                    "%s failed (slot %d, attempt %d/%d, retry in %.0fs): %s",
                    endpoint_cls.__name__, idx, attempt + 1, MAX_RETRIES, retry_wait, exc,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_wait)
                else:
                    raise


# Module-level pool — initialised in collect_all() before any threads start
_pool: ProxyPool | None = None


def _api_call(endpoint_cls, **kwargs):
    """Thin wrapper so all callers go through the global pool."""
    assert _pool is not None, "collect_all() must be called before _api_call()"
    return _pool.call(endpoint_cls, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(result) -> dict:
    """
    Serialise all result sets from an nba_api response to a plain dict.

    Uses the library's own parser system — works for both V2 (resultSets)
    and V3 (nested JSON with custom parsers) endpoints without any guessing.

    Output format:
        {"DataSetName": {"headers": [...], "data": [[...], ...]}, ...}

    Reconstruct any frame later with:
        pd.DataFrame(d["DataSetName"]["data"], columns=d["DataSetName"]["headers"])

    V3 endpoints (all boxscore types, hustle) have registered parsers that
    flatten the nested homeTeam/awayTeam structure into tabular rows.
    V2 endpoints (commonplayerinfo, etc.) fall back to the resultSets path.
    """
    from nba_api.stats.endpoints._parsers import _PARSER_REGISTRY

    endpoint_name = getattr(result, "endpoint", None)
    # Endpoints not in the registry use the V2 resultSets path (pass None)
    if endpoint_name not in _PARSER_REGISTRY:
        endpoint_name = None
    return result.nba_response.get_data_sets(endpoint_name)


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _player_ids_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c == "PLAYER_ID")
    return set(df["PLAYER_ID"].astype(str).unique()) if "PLAYER_ID" in df.columns else set()


# ---------------------------------------------------------------------------
# Season-level collectors
# ---------------------------------------------------------------------------

def collect_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Pull all games for a season (one row per team per game).

    Files: data/raw/game_log_regular_{season}.csv
           data/raw/game_log_playoffs_{season}.csv
    """
    slug     = "regular" if "Regular" in season_type else "playoffs"
    out_path = RAW_DIR / f"game_log_{slug}_{season}.csv"

    if out_path.exists():
        return pd.read_csv(out_path)

    result = _api_call(
        leaguegamelog.LeagueGameLog,
        season=season,
        season_type_all_star=season_type,
    )
    df = result.get_data_frames()[0]
    df.to_csv(out_path, index=False)
    return df


def collect_player_matchups(season: str) -> None:
    """
    Pull season-level player matchup rollup data.

    Who matched up against whom across the season, possessions faced,
    opponent FG% in that matchup. Supplements the per-game boxscorematchupsv3.

    File: data/raw/player_matchups_{season}.csv
    """
    out_path = RAW_DIR / f"player_matchups_{season}.csv"
    if out_path.exists():
        return
    try:
        result = _api_call(
            leagueseasonmatchups.LeagueSeasonMatchups,
            season=season,
            season_type_playoffs="Regular Season",
            per_mode_simple="PerGame",
        )
        result.get_data_frames()[0].to_csv(out_path, index=False)
    except Exception as exc:
        logger.error("player_matchups %s: %s", season, exc)


def collect_player_season_stats(season: str) -> None:
    """
    Pull per-game season averages for every player.

    Provides PPG, usage rate, efficiency — needed to weight the ELO
    by how much a player actually contributes (usage-weighted ELO).

    File: data/raw/player_season_stats_{season}.csv
    """
    out_path = RAW_DIR / f"player_season_stats_{season}.csv"
    if out_path.exists():
        return
    try:
        result = _api_call(
            leaguedashplayerstats.LeagueDashPlayerStats,
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        result.get_data_frames()[0].to_csv(out_path, index=False)
    except Exception as exc:
        logger.error("player_season_stats %s: %s", season, exc)


def collect_hustle_stats(season: str) -> None:
    """
    Pull season-level hustle stats per player.

    Deflections, charges drawn, contested shots, loose balls —
    effort metrics that often predict outperformance vs raw ELO.

    File: data/raw/hustle_stats_{season}.csv
    """
    out_path = RAW_DIR / f"hustle_stats_{season}.csv"
    if out_path.exists():
        return
    try:
        result = _api_call(
            leaguehustlestatsplayer.LeagueHustleStatsPlayer,
            season=season,
            season_type_all_star="Regular Season",
            per_mode_time="PerGame",
        )
        result.get_data_frames()[0].to_csv(out_path, index=False)
    except Exception as exc:
        logger.error("hustle_stats %s: %s", season, exc)


def collect_synergy_playtypes(season: str) -> None:
    """
    Pull Synergy play type data per player.

    Isolation, pick-and-roll (ball handler + screener), post-up,
    spot-up, transition, cut, handoff — describes HOW each player
    scores and what situations they thrive in. Strong signal for
    matchup-quality features.

    Files: data/raw/synergy_{playtype}_{season}.csv
    """
    play_types = [
        "Isolation", "Transition", "PRBallHandler", "PRRollman",
        "Postup", "Spotup", "Handoff", "Cut", "OffScreen", "Misc",
    ]
    for play_type in play_types:
        out_path = RAW_DIR / f"synergy_{play_type.lower()}_{season}.csv"
        if out_path.exists():
            continue
        try:
            result = _api_call(
                synergyplaytypes.SynergyPlayTypes,
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                play_type_nullable=play_type,
                type_grouping_nullable="offensive",
            )
            result.get_data_frames()[0].to_csv(out_path, index=False)
        except Exception as exc:
            logger.error("synergy %s %s: %s", play_type, season, exc)


def collect_pt_stats(season: str) -> None:
    """
    Pull player tracking stats for each measure type.

    Covers drives, catch-shoot, pull-up shots, possessions/touches,
    passing, and speed/distance — describes HOW players play.

    Files: data/raw/pt_stats_{measure}_{season}.csv
    """
    for measure in PT_STAT_TYPES:
        slug     = measure.lower().replace(" ", "_")
        out_path = RAW_DIR / f"pt_stats_{slug}_{season}.csv"
        if out_path.exists():
            continue
        try:
            result = _api_call(
                leaguedashptstats.LeagueDashPtStats,
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                pt_measure_type=measure,
            )
            result.get_data_frames()[0].to_csv(out_path, index=False)
        except Exception as exc:
            logger.error("pt_stats %s %s: %s", measure, season, exc)


def collect_pt_defend(season: str) -> None:
    """
    Pull defensive matchup tracking stats.

    Contested shots, frequency defending per player, FG% allowed.

    File: data/raw/pt_defend_{season}.csv
    """
    out_path = RAW_DIR / f"pt_defend_{season}.csv"
    if out_path.exists():
        return
    try:
        result = _api_call(
            leaguedashptdefend.LeagueDashPtDefend,
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            defense_category="Overall",
        )
        result.get_data_frames()[0].to_csv(out_path, index=False)
    except Exception as exc:
        logger.error("pt_defend %s: %s", season, exc)


def collect_lineups(season: str) -> None:
    """
    Pull 5-man lineup combination stats for a season.

    Which players share the floor together and how effective those combinations
    are — used to estimate true lineup strength beyond individual ELO sums.

    File: data/raw/lineups_5man_{season}.csv
    """
    out_path = RAW_DIR / f"lineups_5man_{season}.csv"
    if out_path.exists():
        return
    try:
        result = _api_call(
            leaguedashlineups.LeagueDashLineups,
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            group_quantity=5,
        )
        result.get_data_frames()[0].to_csv(out_path, index=False)
    except Exception as exc:
        logger.error("lineups %s: %s", season, exc)


def _collect_season(season: str) -> tuple[set[str], set[str]]:
    """
    Collect all season-level data for one season.
    Returns (game_ids, player_ids) harvested from this season.
    Used as the unit of work for the phase-1 thread pool.
    """
    game_ids: set[str] = set()

    for season_type in ["Regular Season", "Playoffs"]:
        try:
            df = collect_game_log(season, season_type)
            if "GAME_ID" in df.columns:
                game_ids.update(df["GAME_ID"].astype(str).str.zfill(10).unique())
        except Exception as exc:
            logger.error("game_log %s %s: %s", season, season_type, exc)

    collect_player_matchups(season)
    collect_player_season_stats(season)
    collect_hustle_stats(season)
    collect_synergy_playtypes(season)
    collect_pt_stats(season)
    collect_pt_defend(season)
    collect_lineups(season)

    # Harvest player IDs from season stats (all active players appear here)
    player_ids = _player_ids_from_csv(RAW_DIR / f"player_season_stats_{season}.csv")
    return game_ids, player_ids


# ---------------------------------------------------------------------------
# Per-game box score collection
# ---------------------------------------------------------------------------

def collect_game_boxscores(game_id: str) -> None:
    """
    Collect all box score types for one game.

    Core endpoints use full retry logic. Optional endpoints (matchups, hustle)
    are tried once with no retries — they are legitimately absent for older games
    and retrying wastes ~16 seconds per game across 14,000 games.

    Files: data/raw/boxscore_{type}_{game_id}.json
    """
    # Core box scores — full retry via _api_call
    for name, endpoint_cls in BOXSCORE_ENDPOINTS_CORE:
        out_path = RAW_DIR / f"boxscore_{name}_{game_id}.json"
        if out_path.exists() and out_path.stat().st_size > 50:
            continue
        try:
            result = _api_call(endpoint_cls, game_id=game_id)
            _save_json(_to_dict(result), out_path)
        except Exception as exc:
            logger.warning("boxscore_%s %s: %s", name, game_id, exc)

    # Optional box scores — single attempt, silent on failure.
    # On failure we write a sentinel so future runs skip without retrying.
    for name, endpoint_cls in BOXSCORE_ENDPOINTS_OPTIONAL:
        out_path = RAW_DIR / f"boxscore_{name}_{game_id}.json"
        if out_path.exists():  # good data OR known-absent sentinel — either way skip
            continue
        try:
            time.sleep(REQUEST_DELAY)
            result = endpoint_cls(game_id=game_id)
            _save_json(_to_dict(result), out_path)
        except Exception:
            _save_json({"_absent": True}, out_path)  # mark tried, not available


# ---------------------------------------------------------------------------
# Player info collection
# ---------------------------------------------------------------------------

def _collect_player_info_single(player_id: str) -> None:
    """
    Pull bio info for one player (position, height, weight, age, draft year).
    Used as the unit of work for the phase-2 thread pool.

    File: data/raw/player_info_{player_id}.json
    """
    out_path = RAW_DIR / f"player_info_{player_id}.json"
    if out_path.exists() and out_path.stat().st_size > 50:
        return
    try:
        result = _api_call(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
        _save_json(_to_dict(result), out_path)
    except Exception as exc:
        logger.warning("player_info %s: %s", player_id, exc)


# ---------------------------------------------------------------------------
# Today's games (used by daily predict pipeline)
# ---------------------------------------------------------------------------

def collect_todays_games() -> pd.DataFrame:
    """
    Fetch today's scheduled games from the NBA scoreboard.

    Returns a DataFrame with GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, GAME_DATE_EST.
    Also saves to data/raw/todays_games.csv for debugging.
    """
    result = _api_call(scoreboardv2.ScoreboardV2)
    games  = result.game_header.get_data_frame()
    games.to_csv(RAW_DIR / "todays_games.csv", index=False)
    logger.info("Found %d game(s) today", len(games))
    return games


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect_all() -> None:
    """
    Full historical bootstrap. Run once. Resumes safely after interruption.

    Proxy setup:
        Set PROXIES env var to a comma-separated list of proxy URLs before running.
        Without proxies the script runs single-threaded (1 worker).
        Each proxy = 1 additional parallel worker with its own rate-limit bucket.

    Phase 1 — Season-level data (parallelised across seasons):
        Game logs, player matchups, tracking stats, lineups for every season.
        Fast even single-threaded (~1–2 hrs).

    Phase 2 — Player bio info (parallelised per player):
        CommonPlayerInfo for every unique player found in phase 1.
        ~20–40 min.

    Phase 3 — Per-game box scores (parallelised per game):
        6 box score types per game. The long part.
        ~12–18 hrs single-threaded, ~2.5–6 hrs with proxies.
    """
    global _pool

    # Read proxy + worker config from environment
    proxy_env = os.environ.get("PROXIES", "")
    proxies   = [p.strip() for p in proxy_env.split(",") if p.strip()]
    slots     = proxies if proxies else [None]  # None = no proxy
    workers   = int(os.environ.get("WORKERS", len(slots)))

    _pool = ProxyPool(slots)

    # Inject a requests Session with a connection pool sized to match our
    # worker count. nba_api exposes set_session() exactly for this purpose.
    import requests as _requests
    from requests.adapters import HTTPAdapter
    from nba_api.stats.library.http import NBAStatsHTTP
    _session = _requests.Session()
    _adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    _session.mount("https://", _adapter)
    _session.mount("http://", _adapter)
    NBAStatsHTTP.set_session(_session)

    if proxies:
        logger.info("Proxy pool: %d slot(s) | %d worker(s)", len(slots), workers)
    else:
        logger.info("No proxies configured — running single-threaded")

    # -----------------------------------------------------------------------
    # Phase 1: Season-level data
    # -----------------------------------------------------------------------
    all_game_ids:   set[str] = set()
    all_player_ids: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_collect_season, s): s for s in SEASONS}
        with tqdm(total=len(SEASONS), desc="Phase 1  Seasons ",
                  unit="season", ncols=88, colour="cyan") as pbar:
            for f in as_completed(futures):
                try:
                    gids, pids = f.result()
                    all_game_ids.update(gids)
                    all_player_ids.update(pids)
                except Exception as exc:
                    logger.error("Season failed: %s", exc)
                pbar.update(1)

    logger.info(
        "Phase 1 complete — %d unique games | %d unique players",
        len(all_game_ids), len(all_player_ids),
    )

    # -----------------------------------------------------------------------
    # Phase 2: Player bio info
    # -----------------------------------------------------------------------
    to_fetch = sorted(
        pid for pid in all_player_ids
        if not (RAW_DIR / f"player_info_{pid}.json").exists()
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_collect_player_info_single, pid) for pid in to_fetch]
        with tqdm(total=len(to_fetch), desc="Phase 2  Players ",
                  unit="player", ncols=88, colour="yellow") as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    logger.error("Player info failed: %s", exc)
                pbar.update(1)

    logger.info("Phase 2 complete — %d player profiles fetched", len(to_fetch))

    # -----------------------------------------------------------------------
    # Phase 3: Per-game box scores
    # -----------------------------------------------------------------------
    retry_file = os.environ.get("RETRY_FILE", "")
    if retry_file and Path(retry_file).exists():
        sorted_games = [g.strip() for g in Path(retry_file).read_text().splitlines() if g.strip()]
        logger.info("Phase 3  Retry mode: %d games from %s", len(sorted_games), retry_file)
    else:
        sorted_games = sorted(all_game_ids)
        skip = int(os.environ.get("SKIP_GAMES", 0))
        if skip:
            logger.info("Phase 3  Skipping first %d already-done games (SKIP_GAMES=%d)", skip, skip)
            sorted_games = sorted_games[skip:]
    total_games  = len(sorted_games)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(collect_game_boxscores, gid): gid for gid in sorted_games}
        with tqdm(total=total_games, desc="Phase 3  Box scores",
                  unit="game", ncols=88, colour="green") as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    logger.error("Box score failed: %s", exc)
                pbar.update(1)

    logger.info("=== Collection complete. ===")


if __name__ == "__main__":
    collect_all()
