"""
lolesports API client: leagues, schedule and results from the API behind
lolesports.com.

Unofficial and undocumented, so every call fails loudly (HTTP errors raise,
missing keys raise) instead of returning a silent empty result.

Endpoints
---------
  https://esports-api.lolesports.com/persisted/gw/getLeagues?hl=en-US
"""

import time

import requests

API_BASE = "https://esports-api.lolesports.com/persisted/gw/"
# Public key that lolesports.com itself sends with every request; not a secret.
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}

_TIMEOUT = 30


def _get(endpoint: str, **params) -> dict:
    """GET an API endpoint (hl=en-US always set) and return its `data` object."""
    resp = requests.get(API_BASE + endpoint, params={"hl": "en-US", **params},
                        headers=HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["data"]


def get_leagues() -> list[dict]:
    return _get("getLeagues")["leagues"]


# ---------------------------------------------------------------------------
# Connectivity check (CLI)
# ---------------------------------------------------------------------------

def check() -> int:
    """Call getLeagues and print status / latency. Returns 0 if the core leagues are listed, else 1."""
    required = ("lck", "lpl", "lec", "worlds")
    t0 = time.monotonic()
    try:
        leagues = get_leagues()
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
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
