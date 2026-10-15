"""
Oracle's Elixir client: the yearly LoL esports match-data CSVs (2014 onward),
one file per year in a public Google Drive folder, refreshed once a day.

Endpoints
---------
  folder view  https://drive.google.com/embeddedfolderview?id={FOLDER_ID}
  download     https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t

File IDs are resolved from the folder view on every run (a new year means a new
ID). The download endpoint needs no auth. Drive can refuse full downloads with
a "Quota exceeded" page for up to 24 h; --check detects that.

Files land in src/data/raw/oe/ (gitignored). load_year() reads one year into the
NEEDED_COLS subset with normalised dtypes; player_rows()/team_rows() split it.
"""

import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OE_DIR = RAW_DIR / "oe"

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/embeddedfolderview?id={FOLDER_ID}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

_TIMEOUT = 30
MAX_AGE_HOURS = 20          # OE refreshes once a day; younger local files are reused

ROLES = ("top", "jng", "mid", "bot", "sup")

_ID_COLS = ["gameid", "datacompleteness", "league", "year", "split", "playoffs", "date",
            "game", "patch", "participantid", "side", "position", "playername", "playerid",
            "teamname", "teamid", "gamelength", "result"]
_ELO_COLS = ["kills", "deaths", "assists", "dpm", "damageshare", "earned gpm",
             "earnedgoldshare", "cspm", "golddiffat15", "xpdiffat15", "csdiffat15", "vspm",
             "wcpm", "controlwardsbought"]
_TEAM_COLS = ["teamkills", "teamdeaths", "firstblood", "firstdragon", "firstherald",
              "firsttower", "firstbaron", "dragons", "barons", "towers", "opp_towers"]
NEEDED_COLS = _ID_COLS + _ELO_COLS + _TEAM_COLS

_INT_COLS = ["participantid", "result", "playoffs", "game", "year"]
_STR_COLS = ["gameid", "datacompleteness", "league", "split", "patch", "side", "position",
             "playername", "playerid", "teamname", "teamid"]
_FLOAT_COLS = [c for c in NEEDED_COLS if c not in _INT_COLS + _STR_COLS + ["date"]]

# The committed team game log (one row per team per game), see team_rows().
TEAM_GAMES_COLS = ["gameid", "date", "league", "year", "split", "playoffs", "game", "patch",
                   "side", "teamid", "teamname", "result", "gamelength", "teamkills",
                   "teamdeaths", "golddiffat15", "xpdiffat15", "csdiffat15", "firstblood",
                   "firstdragon", "firstherald", "firsttower", "firstbaron", "dragons",
                   "barons", "towers", "opp_towers", "starters"]

# One folder entry: the file id, then (without crossing into the next entry) its title.
_ENTRY_RE = re.compile(
    r'/file/d/([\w-]+)/(?:(?!/file/d/).)*?'
    r'flip-entry-title">(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv<',
    re.DOTALL,
)


def parse_folder(html: str) -> dict[int, str]:
    """Year -> Drive file id from the embedded folder view HTML."""
    return {int(year): file_id for file_id, year in _ENTRY_RE.findall(html)}


def list_files() -> dict[int, str]:
    """Fetch the folder view and return year -> file id. Raises if nothing matched."""
    resp = requests.get(FOLDER_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    files = parse_folder(resp.text)
    if not files:
        raise RuntimeError(f"OE folder {FOLDER_ID}: no year files found (page shape changed?)")
    return files


def year_path(year: int, oe_dir: Path = OE_DIR) -> Path:
    return oe_dir / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def download_year(year: int, dest: Path = OE_DIR, file_id: str | None = None,
                  force: bool = False) -> Path:
    """
    Download one year's CSV into dest (atomic: temp file + os.replace) and return
    its path. Skips the download if the local file is younger than MAX_AGE_HOURS,
    unless force. Raises if Drive returns anything but the OE CSV.
    """
    path = year_path(year, dest)
    if not force and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_HOURS * 3600:
        logger.info("OE %d: %s is fresh, skipping download", year, path.name)
        return path
    if file_id is None:
        file_id = list_files()[year]

    dest.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    t0 = time.monotonic()
    try:
        with requests.get(DOWNLOAD_URL.format(file_id=file_id), stream=True,
                          timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        with open(tmp, "rb") as f:
            head = f.read(4096)
        if not head.startswith(b"gameid,"):
            # Drive answers with an HTML page instead, e.g. "Google Drive - Quota exceeded".
            title = re.search(rb"<title>(.*?)</title>", head)
            raise RuntimeError(f"OE {year}: download {file_id} is not the OE CSV: "
                               f"{title[1].decode(errors='replace') if title else head[:80]!r}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("OE %d: downloaded %s B in %.1fs", year, f"{path.stat().st_size:,}",
                time.monotonic() - t0)
    return path


def load_year(year: int, oe_dir: Path = OE_DIR) -> pd.DataFrame:
    """
    Read one year's CSV, reindexed to NEEDED_COLS (columns absent that year are
    NaN). date -> datetime64, _INT_COLS -> Int64, ids/names/patch -> str (NaN kept),
    stats -> float. No rows are filtered.
    """
    df = pd.read_csv(year_path(year, oe_dir), usecols=lambda c: c in NEEDED_COLS,
                     dtype={c: str for c in _STR_COLS}, low_memory=False)
    df = df.reindex(columns=NEEDED_COLS)
    df["date"] = pd.to_datetime(df["date"])
    for c in _INT_COLS:
        df[c] = pd.to_numeric(df[c]).astype("Int64")
    for c in _FLOAT_COLS:
        df[c] = pd.to_numeric(df[c]).astype(float)
    for c in _STR_COLS:
        df[c] = df[c].astype(object)
    return df


def player_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["position"] != "team"]


def team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    The team rows in TEAM_GAMES_COLS shape, sorted by (date, gameid, side).
    `starters` = the side's playerids in ROLES order, |-joined; always 5 slots
    (empty for a missing role/id), so a slot's position is its role.
    """
    teams = df[df["position"] == "team"]
    by_role = (player_rows(df).groupby(["gameid", "side", "position"])["playerid"].first()
               .unstack().reindex(columns=list(ROLES)))
    starters = (by_role.reindex(pd.MultiIndex.from_frame(teams[["gameid", "side"]]))
                .fillna("").agg("|".join, axis=1))
    teams = teams.assign(starters=starters.to_numpy())
    return (teams.sort_values(["date", "gameid", "side"])[TEAM_GAMES_COLS]
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Connectivity check (CLI)
# ---------------------------------------------------------------------------

def check() -> int:
    """
    List the folder, then start a full (non-Range) download of the latest year's CSV,
    read only its first 20 KB and print status / size / latency. Returns 0 if both
    work and the body is the OE CSV, else 1. A Range request would hide Drive's
    "Quota exceeded" page, which only blocks full downloads.
    """
    t0 = time.monotonic()
    try:
        files = list_files()
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        print(f"folder    {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        print("FAILED - could not list the OE folder.")
        return 1
    print(f"folder    HTTP 200  {len(files)} files ({min(files)}-{max(files)})  "
          f"{time.monotonic() - t0:5.2f}s")

    year = max(files)
    t0 = time.monotonic()
    try:
        # stream + bounded read: only the first 20 KB of the full download is pulled.
        with requests.get(DOWNLOAD_URL.format(file_id=files[year]), stream=True,
                          timeout=_TIMEOUT) as resp:
            body = next(resp.iter_content(chunk_size=20480), b"")
    except requests.exceptions.RequestException as exc:
        print(f"download  {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        print(f"FAILED - could not download the {year} file.")
        return 1
    title = re.search(rb"<title>(.*?)</title>", body)
    print(f"download  {year} HTTP {resp.status_code}  {len(body):,} B  "
          f"{time.monotonic() - t0:5.2f}s  total {resp.headers.get('Content-Length', '?')} B  "
          f"last-modified {resp.headers.get('Last-Modified', '?')}")

    ok = resp.status_code in (200, 206) and body.startswith(b"gameid,")
    print(f"OK - OE folder and {year} download work." if ok
          else f"FAILED - {year} download did not return the OE CSV"
               f"{': ' + title[1].decode(errors='replace') if title else ''}.")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Oracle's Elixir client")
    parser.add_argument("--check", action="store_true",
                        help="List the OE folder and test a download of the latest year's CSV")
    args = parser.parse_args()
    if args.check:
        sys.exit(check())
    parser.print_help()
