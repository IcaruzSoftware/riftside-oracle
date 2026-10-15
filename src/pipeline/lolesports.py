"""
lolesports API client: leagues, schedule and results from the API behind
lolesports.com.

Unofficial and undocumented, so every call fails loudly (HTTP errors raise,
missing keys raise) instead of returning a silent empty result.

Endpoints
---------
  https://esports-api.lolesports.com/persisted/gw/getLeagues?hl=en-US
  .../getSchedule?hl=en-US&leagueId=<id>[,<id>...][&pageToken=...]
  .../getEventDetails?hl=en-US&id=<matchId>

Team resolution: lolesports and Oracle's Elixir share no team id, so
resolve_team() maps a lolesports (name, code) to an OE teamid via
team_aliases.csv first, then an exact normalised-name match (never fuzzy).
"""

import csv
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://esports-api.lolesports.com/persisted/gw/"
# Public key that lolesports.com itself sends with every request; not a secret.
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}

_TIMEOUT = 30
MAX_SCHEDULE_PAGES = 10

ALIASES_PATH = Path(__file__).parent / "team_aliases.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
_SUFFIXES = ("esports", "gaming", "team", "club", "academy")


def _get(endpoint: str, **params) -> dict:
    """GET an API endpoint (hl=en-US always set) and return its `data` object."""
    resp = requests.get(API_BASE + endpoint, params={"hl": "en-US", **params},
                        headers=HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["data"]


def get_leagues() -> list[dict]:
    return _get("getLeagues")["leagues"]


def get_schedule(league_ids: list[str], page_token: str | None = None) -> dict:
    """One schedule page: {"pages": {"older", "newer"}, "events": [...]}."""
    params = {"leagueId": ",".join(league_ids)}
    if page_token is not None:
        params["pageToken"] = page_token
    return _get("getSchedule", **params)["schedule"]


def get_event_details(match_id: str) -> dict:
    return _get("getEventDetails", id=match_id)["event"]


def league_ids(slugs: list[str]) -> list[str]:
    """League slugs -> ids. Raises on an unknown slug."""
    ids = {lg["slug"]: lg["id"] for lg in get_leagues()}
    unknown = [s for s in slugs if s not in ids]
    if unknown:
        raise KeyError(f"unknown lolesports league slugs: {', '.join(unknown)}")
    return [ids[s] for s in slugs]


def _start(event: dict) -> datetime:
    return datetime.fromisoformat(event["startTime"].replace("Z", "+00:00"))


def upcoming_matches(slugs: list[str], start: datetime, end: datetime) -> list[dict]:
    """
    Unstarted match events in [start, end] (aware UTC) for these leagues, without
    TBD teams, de-duplicated by match id, sorted by startTime. Follows `newer`
    pages while the last event still starts before `end`.
    """
    ids = league_ids(slugs)
    events, token = [], None
    for _ in range(MAX_SCHEDULE_PAGES):
        page = get_schedule(ids, token)
        events += page["events"]
        token = page["pages"]["newer"]
        if not token or not events or _start(events[-1]) >= end:
            break
    else:
        raise RuntimeError(f"getSchedule: still before {end} after {MAX_SCHEDULE_PAGES} pages")

    matches = {}
    for e in events:
        if (e["type"] == "match" and e["state"] == "unstarted" and start <= _start(e) <= end
                and not any(t["name"].casefold() == "tbd" for t in e["match"]["teams"])):
            matches[e["match"]["id"]] = e
    return sorted(matches.values(), key=_start)


# ---------------------------------------------------------------------------
# Team resolution: lolesports (name, code) -> OE teamid
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    """Casefold, drop non-alphanumerics and trailing _SUFFIXES words: "Gen.G Esports" -> "geng"."""
    words = [w for w in re.split(r"[^0-9a-z]+", name.casefold()) if w]
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return "".join(words)


def resolve_team(name: str, code: str, team_games: pd.DataFrame | None = None,
                 as_of: datetime | None = None) -> str | None:
    """
    OE teamid for a lolesports team, or None (logged) if unmapped or ambiguous.
    1) team_aliases.csv on the exact (code, name) pair; 2) exact normalised-name
    match against OE teams seen in the 365 days before as_of (default: now UTC).
    team_games defaults to the committed RAW_DIR/team_games_*.csv.
    """
    with open(ALIASES_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["lolesports_code"], row["lolesports_name"]) == (code, name):
                return row["oe_teamid"]

    if team_games is None:
        paths = sorted(RAW_DIR.glob("team_games_*.csv"))
        team_games = (pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in paths])
                      if paths else pd.DataFrame(columns=["date", "teamid", "teamname"]))
    # OE dates are naive (taken as UTC); as_of must be aware.
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc)).tz_convert(None) - timedelta(days=365)
    recent = team_games[pd.to_datetime(team_games["date"]) >= cutoff]
    key = normalize_team_name(name)
    hits = recent.loc[recent["teamname"].fillna("").map(normalize_team_name) == key,
                      "teamid"].dropna().unique() if key else []
    if len(hits) == 1:
        return hits[0]
    logger.warning("unmapped team %s/%s%s", code, name,
                   f" (ambiguous: {len(hits)} OE teamids)" if len(hits) > 1 else "")
    return None


# ---------------------------------------------------------------------------
# Connectivity check (CLI)
# ---------------------------------------------------------------------------

def check() -> int:
    """Call getLeagues and print status / latency. Returns 0 if the core leagues are listed, else 1."""
    required = ("lck", "lpl", "lec", "worlds")
    t0 = time.monotonic()
    try:
        leagues = get_leagues()
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError,
            AttributeError) as exc:
        print(f"getLeagues  {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        print("FAILED - getLeagues did not work.")
        return 1
    slugs = {lg.get("slug") for lg in leagues}
    print(f"getLeagues  HTTP 200  {len(leagues)} leagues  {time.monotonic() - t0:5.2f}s")
    for slug in required:
        print(f"  {slug:7} {'present' if slug in slugs else 'MISSING'}")

    missing = [s for s in required if s not in slugs]
    print("OK - lolesports API works." if not missing
          else f"FAILED - missing league slugs: {', '.join(missing)}.")
    return 0 if not missing else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="lolesports API client")
    parser.add_argument("--check", action="store_true",
                        help="Call getLeagues and verify the core league slugs")
    args = parser.parse_args()
    if args.check:
        sys.exit(check())
    parser.print_help()
