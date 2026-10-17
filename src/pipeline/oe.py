"""
Oracle's Elixir client: the yearly LoL esports match-data CSVs (2014 onward),
one file per year in a public Google Drive folder, refreshed once a day.

Endpoints
---------
  With GOOGLE_DRIVE_API_KEY set (Drive API v3, key sent in the X-Goog-Api-Key header):
  folder list  https://www.googleapis.com/drive/v3/files?q='{FOLDER_ID}' in parents and trashed=false
  download     https://www.googleapis.com/drive/v3/files/{file_id}?alt=media
  Without it (public links, no auth):
  folder view  https://drive.google.com/embeddedfolderview?id={FOLDER_ID}
  download     https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t

File IDs are resolved from the folder on every run (a new year means a new ID).
Drive can refuse public full downloads with a "Quota exceeded" page for up to
24 h. An API quota error (listing or download) falls back to the public link
once. --check detects both.

Files land in src/data/raw/oe/ (gitignored). load_year() reads one year into the
NEEDED_COLS subset with normalised dtypes; player_rows()/team_rows() split it.
"""

import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OE_DIR = RAW_DIR / "oe"

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/embeddedfolderview?id={FOLDER_ID}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
API_FILES_URL = "https://www.googleapis.com/drive/v3/files"
API_KEY_ENV = "GOOGLE_DRIVE_API_KEY"
_QUOTA_REASONS = {"downloadQuotaExceeded", "quotaExceeded", "rateLimitExceeded",
                  "userRateLimitExceeded", "dailyLimitExceeded", "RATE_LIMIT_EXCEEDED"}
_KEY_REASONS = {"keyInvalid", "API_KEY_INVALID"}
_REDIRECT_HOSTS = (".googleusercontent.com", ".googleapis.com")   # https only, followed once

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
_NAME_RE = re.compile(r"(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv")


class DriveQuotaError(RuntimeError):
    """The Drive API refused the request for quota / rate-limit reasons."""


def api_key() -> str | None:
    """The Drive API key from GOOGLE_DRIVE_API_KEY; unset, empty or blank means no key."""
    return (os.environ.get(API_KEY_ENV) or "").strip() or None


def _is_redirect(resp: requests.Response) -> bool:
    return 300 <= resp.status_code < 400


def _api_get(url: str, key: str, **kwargs) -> requests.Response:
    """
    GET a Drive API URL. The key goes only in a header, never in a URL (so not in
    exception texts or logs). One redirect to an https *.googleusercontent.com /
    *.googleapis.com host is followed with a plain request without the key; any
    other redirect is left for _raise_for_api to reject.
    """
    resp = requests.get(url, headers={"X-Goog-Api-Key": key}, timeout=_TIMEOUT,
                        allow_redirects=False, **kwargs)
    location = resp.headers.get("Location", "") if _is_redirect(resp) else ""
    target = urlsplit(location)
    if target.scheme == "https" and (target.hostname or "").endswith(_REDIRECT_HOSTS):
        resp.close()
        resp = requests.get(location, timeout=_TIMEOUT, allow_redirects=False,
                            stream=kwargs.get("stream", False))
    return resp


def _raise_for_api(resp: requests.Response) -> None:
    """Raise a clear error for a failed Drive API response (quota, bad key, not found)."""
    if _is_redirect(resp):
        host = urlsplit(resp.headers.get("Location", "")).hostname or "?"
        raise RuntimeError(f"Drive API: unexpected HTTP {resp.status_code} redirect to {host} "
                           f"(not followed)")
    if resp.ok:
        return
    try:
        body = resp.json()
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    err = err if isinstance(err, dict) else {}
    items = [i for k in ("errors", "details") if isinstance(err.get(k), list) for i in err[k]]
    reasons = {i.get("reason") for i in items if isinstance(i, dict)}
    message = err["message"] if isinstance(err.get("message"), str) else str(resp.reason)
    detail = f"HTTP {resp.status_code}: {message}"
    if (resp.status_code == 429 or err.get("status") == "RESOURCE_EXHAUSTED"
            or reasons & _QUOTA_REASONS):
        raise DriveQuotaError(f"Drive API quota exceeded ({detail})")
    if reasons & _KEY_REASONS or "API key not valid" in message:
        raise RuntimeError(f"Drive API key in {API_KEY_ENV} is invalid ({detail})")
    if resp.status_code == 404:
        raise RuntimeError(f"Drive API: file not found or not public ({detail})")
    raise RuntimeError(f"Drive API error ({detail})")


def parse_folder(html: str) -> dict[int, str]:
    """Year -> Drive file id from the embedded folder view HTML."""
    return {int(year): file_id for file_id, year in _ENTRY_RE.findall(html)}


def parse_listing(files: list[dict]) -> dict[int, str]:
    """Year -> Drive file id from a Drive API files.list `files` array."""
    return {int(m[1]): f["id"] for f in files if (m := _NAME_RE.fullmatch(f["name"]))}


def _list_files_api(key: str) -> dict[int, str]:
    resp = _api_get(API_FILES_URL, key, params={
        "q": f"'{FOLDER_ID}' in parents and trashed=false", "fields": "files(id,name)",
        "pageSize": 1000, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"})
    _raise_for_api(resp)
    files = parse_listing(resp.json().get("files", []))
    if not files:
        raise RuntimeError(f"OE folder {FOLDER_ID}: Drive API listing returned no OE files")
    return files


def list_files() -> dict[int, str]:
    """
    Return year -> file id: via the Drive API files.list with a key (one retry via
    the folder view on an API quota error), else from the embedded folder view.
    Raises if nothing matched.
    """
    if key := api_key():
        try:
            return _list_files_api(key)
        except DriveQuotaError as exc:
            logger.warning("OE folder: %s - retrying via the public folder view", exc)
    resp = requests.get(FOLDER_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    files = parse_folder(resp.text)
    if not files:
        raise RuntimeError(f"OE folder {FOLDER_ID}: no year files found (page shape changed?)")
    return files


def year_path(year: int, oe_dir: Path = OE_DIR) -> Path:
    return oe_dir / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _open_download(file_id: str, key: str | None) -> requests.Response:
    """Streaming full GET of a file: Drive API with a key, else the public link."""
    if key:
        return _api_get(f"{API_FILES_URL}/{file_id}", key, params={"alt": "media"}, stream=True)
    return requests.get(DOWNLOAD_URL.format(file_id=file_id), stream=True, timeout=_TIMEOUT)


def download_year(year: int, dest: Path = OE_DIR, file_id: str | None = None,
                  force: bool = False) -> Path:
    """
    Download one year's CSV into dest (atomic: temp file + os.replace) and return
    its path. Skips the download if the local file is younger than MAX_AGE_HOURS,
    unless force. Uses the Drive API when a key is set and falls back to the
    public link once on an API quota error. Raises if Drive returns anything but
    the OE CSV.
    """
    path = year_path(year, dest)
    if not force and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_HOURS * 3600:
        logger.info("OE %d: %s is fresh, skipping download", year, path.name)
        return path
    if file_id is None:
        file_id = list_files()[year]

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    key = api_key()
    try:
        _download(year, file_id, path, key)
    except DriveQuotaError as exc:
        logger.warning("OE %d: %s - retrying via the public link", year, exc)
        key = None
        _download(year, file_id, path, key)
    logger.info("OE %d: downloaded %s B in %.1fs via the %s", year, f"{path.stat().st_size:,}",
                time.monotonic() - t0, "Drive API" if key else "public link")
    return path


def _download(year: int, file_id: str, path: Path, key: str | None) -> None:
    tmp = path.with_name(path.name + ".part")
    try:
        with _open_download(file_id, key) as resp:
            if key:
                _raise_for_api(resp)
            else:
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

def _check_download(year: int, file_id: str, key: str | None) -> str | None:
    """
    Start a full download, read only its first 20 KB and print one status line.
    Returns None if the body is the OE CSV, else why not. Raises DriveQuotaError
    on a Drive API quota error.
    """
    t0 = time.monotonic()
    try:
        # stream + bounded read: only the first 20 KB of the full download is pulled.
        with _open_download(file_id, key) as resp:
            if key:
                _raise_for_api(resp)
            body = next(resp.iter_content(chunk_size=20480), b"")
    except DriveQuotaError:
        raise
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        print(f"download  {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        return f"could not download the {year} file"
    print(f"download  {year} HTTP {resp.status_code}  {len(body):,} B  "
          f"{time.monotonic() - t0:5.2f}s  total {resp.headers.get('Content-Length', '?')} B  "
          f"last-modified {resp.headers.get('Last-Modified', '?')}")
    if resp.status_code in (200, 206) and body.startswith(b"gameid,"):
        return None
    title = re.search(rb"<title>(.*?)</title>", body)
    return (f"{year} download did not return the OE CSV"
            f"{': ' + title[1].decode(errors='replace') if title else ''}")


def check() -> int:
    """
    Say which path is used (Drive API with GOOGLE_DRIVE_API_KEY, else the public
    link), list the folder, then start a full (non-Range) download of the latest
    year's CSV, read only its first 20 KB and print status / size / latency. An API
    quota error retries via the public link, as download_year does. Returns 0 if
    both work and the body is the OE CSV, else 1. A Range request would hide
    Drive's "Quota exceeded" page, which only blocks full downloads.
    """
    key = api_key()
    via = "Drive API" if key else "public link"
    print(f"source    {via} ({API_KEY_ENV} {'set' if key else 'not set'})")
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
    try:
        err = _check_download(year, files[year], key)
    except DriveQuotaError as exc:
        print(f"download  {exc} - retrying via the public link")
        via = "public link"
        err = _check_download(year, files[year], None)
    print(f"OK - OE folder and {year} download work (via the {via})." if err is None
          else f"FAILED - {err}.")
    return 0 if err is None else 1


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
