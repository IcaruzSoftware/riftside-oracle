"""
NBA CDN client — schedule + live box scores from cdn.nba.com.

No proxies, no stats.nba.com. This is the read path the daily GitHub Actions
pipeline uses: it needs only the public NBA CDN, which is reachable from a plain
runner as long as the request carries a current browser-like header set (see
CDN_HEADERS). nba_api's own live header set is outdated and now gets a 403 from the
cdn.nba.com Akamai edge, so we send our own verified header constant instead.

Endpoints
---------
  schedule   https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
  box score  https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json

CDN box scores exist from the 2019-20 season onward; older game IDs return 403.

Game-ID prefixes (first 3 chars of the 10-digit id):
  001 preseason   002 regular season   003 all-star
  004 playoffs    005 play-in          006 NBA Cup final
"""

import json
import logging
import os
import urllib.parse

import requests

logger = logging.getLogger(__name__)

# Offline/testing seam: if set, the schedule is read from this local JSON file
# instead of the live CDN (same shape as scheduleLeagueV2.json).
SCHEDULE_ENV = "COURTSIDE_SCHEDULE_JSON"

# Optional residential proxies for the CDN calls. NBA blacklists cloud IP ranges,
# so cdn.nba.com returns 403 from GitHub-hosted runners even with CDN_HEADERS.
# Comma-separated list of http(s)://user:pass@host:port; unset/empty -> direct.
PROXY_ENV = "NBA_PROXIES"

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

_TIMEOUT = 30

# Verified header set for cdn.nba.com's Akamai edge (returns 200). nba_api's live
# header set is outdated (Chrome/87 UA, Accept: text/html, Cache-Control: max-age=0)
# and now gets a 403 here; this browser-like set works. Host is set by requests.
CDN_HEADERS = {
    "Accept":           "application/json, text/plain, */*",
    "Accept-Encoding":  "gzip, deflate, br",
    "Accept-Language":  "en-US,en;q=0.9",
    "Cache-Control":    "no-cache",
    "Connection":       "keep-alive",
    "Origin":           "https://www.nba.com",
    "Pragma":           "no-cache",
    "Referer":          "https://www.nba.com/",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest":   "empty",
    "Sec-Fetch-Mode":   "cors",
    "Sec-Fetch-Site":   "same-site",
    "User-Agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
}

# Game-type prefixes ---------------------------------------------------------
PRESEASON_PREFIX = "001"
ALLSTAR_PREFIX   = "003"
# Games that carry a real, ELO-relevant box score and can be predicted:
# regular season, playoffs, play-in, NBA Cup final. Preseason and all-star excluded.
STATEFUL_PREFIXES = {"002", "004", "005", "006"}

_session: requests.Session | None = None
_channels: list | None = None   # ordered list of proxy URLs, or [None] for direct
_sticky_idx: int = 0            # index of the last channel that worked

# Errors that mean "this channel is bad, try the next one".
_FAILOVER_ERRORS = (
    requests.exceptions.ProxyError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def reset_session() -> None:
    """Drop cached session/channel state (so a changed NBA_PROXIES takes effect)."""
    global _session, _channels, _sticky_idx
    _session = None
    _channels = None
    _sticky_idx = 0


def _validate_proxy(entry: str, idx: int) -> None:
    """Raise ValueError (never echoing the value) if a proxy entry is malformed."""
    bad = False
    try:
        parsed = urllib.parse.urlparse(entry)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.port:
            bad = True
    except ValueError:
        bad = True
    if bad:
        raise ValueError(
            f"{PROXY_ENV} entry #{idx} is malformed. Expected "
            "'http(s)://user:pass@host:port' with a scheme, host and port. "
            "If the password contains special characters (@ : / etc.) they must be "
            "URL-encoded (e.g. '@' -> '%40'), or urllib3 fails to parse the proxy. "
            "(The value is not shown for safety.)"
        )


def _ensure_channels() -> list:
    """Parse + validate NBA_PROXIES once. Returns proxy URLs, or [None] for direct."""
    global _channels
    if _channels is None:
        entries = [p.strip() for p in os.environ.get(PROXY_ENV, "").split(",") if p.strip()]
        for idx, entry in enumerate(entries):
            _validate_proxy(entry, idx)
        _channels = entries or [None]
    return _channels


def _get_session() -> requests.Session:
    """A cached Session carrying the verified cdn.nba.com browser header set.
    Creating it validates any configured proxies."""
    global _session
    _ensure_channels()  # validate proxies at session-creation time
    if _session is None:
        s = requests.Session()
        s.headers.update(CDN_HEADERS)
        _session = s
    return _session


def _proxies_arg(channel):
    return None if channel is None else {"http": channel, "https": channel}


def _host_port(channel) -> str:
    """'host:port' for a proxy (never its credentials), or 'direct'."""
    if channel is None:
        return "direct"
    p = urllib.parse.urlparse(channel)
    return f"{p.hostname}:{p.port}"


def _get_with_failover(url: str) -> requests.Response:
    """
    GET ``url`` through the configured channels with sticky failover.

    Tries the last-working channel first, then the rest in order. A ProxyError /
    ConnectionError / Timeout, or an HTTP 403 (Akamai block of that IP), moves on to
    the next channel. A 200 or 404 is returned (the caller handles 404). If every
    channel fails, raises a RuntimeError summarising each as
    ``proxy #i (host:port): <error type or HTTP status>`` — host:port only, never
    credentials.
    """
    global _sticky_idx
    channels = _ensure_channels()
    session = _get_session()
    n = len(channels)
    failures: list[str] = []

    for k in range(n):
        i = (_sticky_idx + k) % n
        channel = channels[i]
        try:
            resp = session.get(url, timeout=_TIMEOUT, proxies=_proxies_arg(channel))
        except requests.exceptions.RequestException as exc:
            failures.append(f"proxy #{i} ({_host_port(channel)}): {type(exc).__name__}")
            continue
        if resp.status_code in (200, 404):
            _sticky_idx = i
            return resp
        # 403 (IP blocked) or any other status -> try the next channel.
        failures.append(f"proxy #{i} ({_host_port(channel)}): HTTP {resp.status_code}")

    raise RuntimeError("All NBA CDN channels failed: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# Game-type helpers
# ---------------------------------------------------------------------------

def game_prefix(game_id: str) -> str:
    """Return the 3-digit game-type prefix (first 3 chars of the 10-digit ID),
    e.g. 0042500405 -> '004' (playoffs), 0022300001 -> '002' (regular season)."""
    gid = str(game_id).zfill(10)
    return gid[:3]


def is_preseason(game_id: str) -> bool:
    return game_prefix(game_id) == PRESEASON_PREFIX


def is_all_star(game_id: str) -> bool:
    return game_prefix(game_id) == ALLSTAR_PREFIX


def is_stateful(game_id: str) -> bool:
    """True for regular season / playoffs / play-in / Cup final — games that update ELO."""
    return game_prefix(game_id) in STATEFUL_PREFIXES


# ---------------------------------------------------------------------------
# Team tricode <-> team id
# ---------------------------------------------------------------------------

_tricode_to_id: dict[str, int] | None = None
_id_to_tricode: dict[int, str] | None = None


def _load_team_maps() -> None:
    global _tricode_to_id, _id_to_tricode
    if _tricode_to_id is not None:
        return
    from nba_api.stats.static import teams as _teams

    _tricode_to_id = {}
    _id_to_tricode = {}
    for t in _teams.get_teams():
        _tricode_to_id[t["abbreviation"].upper()] = int(t["id"])
        _id_to_tricode[int(t["id"])] = t["abbreviation"].upper()


def tricode_to_id(tricode: str) -> int:
    _load_team_maps()
    return _tricode_to_id[str(tricode).upper()]


def id_to_tricode(team_id) -> str:
    _load_team_maps()
    return _id_to_tricode[int(team_id)]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def fetch_schedule() -> dict:
    """Fetch the full league schedule JSON. Raises on HTTP error.

    Honours the COURTSIDE_SCHEDULE_JSON env var (offline/testing) — when set, the
    schedule is read from that local file instead of the live CDN.
    """
    override = os.environ.get(SCHEDULE_ENV)
    if override:
        logger.info("Reading schedule from %s (%s)", override, SCHEDULE_ENV)
        with open(override, encoding="utf-8") as f:
            return json.load(f)
    resp = _get_with_failover(SCHEDULE_URL)
    resp.raise_for_status()
    return resp.json()


def _parse_schedule_date(raw: str) -> str:
    """Parse a scheduleLeagueV2 gameDate ('MM/DD/YYYY HH:MM:SS') to ISO 'YYYY-MM-DD'."""
    mmddyyyy = raw.split(" ")[0]
    m, d, y = mmddyyyy.split("/")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def games_for_date(et_date: str, schedule: dict | None = None) -> list[dict]:
    """
    Return the games scheduled on a given ET calendar date.

    Args:
        et_date:   ISO date string 'YYYY-MM-DD' (Eastern-time game date).
        schedule:  Pre-fetched schedule dict (fetched if None).

    Returns a list of dicts, one per game:
        game_id, game_time_utc, game_status (1=scheduled, 2=live, 3=final),
        home_team_id, home_tricode, away_team_id, away_tricode,
        home_score, away_score
    """
    if schedule is None:
        schedule = fetch_schedule()

    out: list[dict] = []
    for day in schedule.get("leagueSchedule", {}).get("gameDates", []):
        if _parse_schedule_date(day.get("gameDate", "")) != et_date:
            continue
        for g in day.get("games", []):
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            out.append({
                "game_id":       str(g["gameId"]).zfill(10),
                "game_time_utc": g.get("gameDateTimeUTC"),
                "game_status":   g.get("gameStatus"),
                "home_team_id":  int(home.get("teamId", 0)),
                "home_tricode":  home.get("teamTricode"),
                "away_team_id":  int(away.get("teamId", 0)),
                "away_tricode":  away.get("teamTricode"),
                "home_score":    home.get("score"),
                "away_score":    away.get("score"),
            })
    return out


# ---------------------------------------------------------------------------
# Box score
# ---------------------------------------------------------------------------

def fetch_boxscore(game_id: str) -> dict | None:
    """
    Fetch one game's live box score. Returns the ``game`` sub-dict, or None when the
    box score does not exist yet (HTTP 404 — a scheduled/not-yet-played game).

    A 403 is Akamai "Access Denied" (blocked IP): it triggers proxy failover inside
    _get_with_failover, and if every channel is blocked the whole fetch raises. The
    daily pipeline only fetches games from 2019-20 onward (which all have box scores),
    so a 403 never means "not available".
    """
    gid = str(game_id).zfill(10)
    resp = _get_with_failover(BOXSCORE_URL.format(game_id=gid))
    if resp.status_code == 404:
        logger.info("boxscore %s not available yet (HTTP 404)", gid)
        return None
    try:
        return resp.json().get("game")
    except ValueError:
        raise RuntimeError(f"CDN box score {gid}: invalid JSON response")


# ---------------------------------------------------------------------------
# Connectivity check (CLI)
# ---------------------------------------------------------------------------

def check() -> int:
    """
    Fetch the schedule + box score 0042500405 through each configured proxy (or direct
    if none) and print status / size / latency. Returns 0 if at least one channel
    fully works, else 1.
    """
    import time

    channels = _ensure_channels()
    session = _get_session()
    box_url = BOXSCORE_URL.format(game_id="0042500405")
    any_ok = False

    for i, channel in enumerate(channels):
        label = _host_port(channel)
        ok = True
        for name, url in (("schedule", SCHEDULE_URL), ("boxscore", box_url)):
            t0 = time.monotonic()
            try:
                resp = session.get(url, timeout=_TIMEOUT, proxies=_proxies_arg(channel))
                dt = time.monotonic() - t0
                print(f"proxy #{i} ({label}) {name:9} HTTP {resp.status_code}  "
                      f"{len(resp.content):>9,} B  {dt:5.2f}s")
                if resp.status_code != 200:
                    ok = False
            except requests.exceptions.RequestException as exc:
                dt = time.monotonic() - t0
                print(f"proxy #{i} ({label}) {name:9} {type(exc).__name__}  {dt:5.2f}s")
                ok = False
        any_ok = any_ok or ok

    print("OK - at least one channel works." if any_ok else "FAILED - no working channel.")
    return 0 if any_ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="NBA CDN client")
    parser.add_argument("--check", action="store_true",
                        help="Test each configured proxy (or direct) against the CDN")
    args = parser.parse_args()
    if args.check:
        sys.exit(check())
    parser.print_help()
