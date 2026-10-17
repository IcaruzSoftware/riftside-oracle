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

  With GOOGLE_OAUTH_CLIENT_ID / _CLIENT_SECRET / _REFRESH_TOKEN all set (the owner's
  OAuth client, Bearer token in the Authorization header):
  token        POST https://oauth2.googleapis.com/token (grant_type=refresh_token)
  copy         POST https://www.googleapis.com/drive/v3/files/{file_id}/copy
  download     https://www.googleapis.com/drive/v3/files/{copy_id}?alt=media
  delete       DELETE https://www.googleapis.com/drive/v3/files/{copy_id}

File IDs are resolved from the folder on every run (a new year means a new ID).
Drive can refuse full downloads of a file with a "Quota exceeded" page (public
link) or a downloadQuotaExceeded error (API) for up to 24 h; the quota is per
file. Download routes (route_order): with a key, the Drive API, then on an API
quota error a temporary copy in the owner's own Drive (OAuth creds set), then
the public link; without a key, the public link, then the copy. The copy is
expected (not yet verified) to have its own quota, since it is a new file and
copying is not a download; it is deleted afterwards. The copy needs the full
`drive` scope: `drive.file` only reaches files the app created or was given, so
it cannot read the public OE file to copy it. An API quota error on the listing
falls back to the public folder view. --check detects all of it.

Files land in src/data/raw/oe/ (gitignored). load_year() reads one year into the
NEEDED_COLS subset with normalised dtypes; player_rows()/team_rows() split it.
"""

import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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
TOKEN_URL = "https://oauth2.googleapis.com/token"
# OAuth token-request field -> env var; all three set = the copy-to-own-Drive route is on.
OAUTH_ENV = {"client_id": "GOOGLE_OAUTH_CLIENT_ID", "client_secret": "GOOGLE_OAUTH_CLIENT_SECRET",
             "refresh_token": "GOOGLE_OAUTH_REFRESH_TOKEN"}
COPY_NAME = "riftside-oracle tmp OE"       # our temporary copies are "<COPY_NAME> <year>"
_COPY_NAME_RE = re.compile(rf"{re.escape(COPY_NAME)} \d{{4}}")
COPY_TAG = ("riftsideOracleTmp", "1")      # appProperties key/value set on every copy
STALE_COPY_HOURS = 2                        # younger tagged copies may be a concurrent run's
ROUTE_API, ROUTE_COPY, ROUTE_PUBLIC = "the Drive API", "a copy in your Drive", "the public link"

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


def oauth_credentials() -> dict[str, str] | None:
    """
    The copy route's OAuth credentials (token-request field -> stripped value), or
    None unless all three OAUTH_ENV vars are non-blank. A partial set logs a
    warning naming the missing vars (never a value).
    """
    creds = {field: (os.environ.get(env) or "").strip() for field, env in OAUTH_ENV.items()}
    missing = [OAUTH_ENV[field] for field, value in creds.items() if not value]
    if not missing:
        return creds
    if len(missing) < len(OAUTH_ENV):
        logger.warning("OE copy route off: %s not set (the other GOOGLE_OAUTH_* vars are)",
                       ", ".join(missing))
    return None


def _is_redirect(resp: requests.Response) -> bool:
    return 300 <= resp.status_code < 400


def _api_get(url: str, auth: dict[str, str], **kwargs) -> requests.Response:
    """
    GET a Drive API URL. The credential (API key or Bearer token) goes only in the
    `auth` headers, never in a URL (so not in exception texts or logs). One
    redirect to an https *.googleusercontent.com / *.googleapis.com host is
    followed with a plain request without them; any other redirect is left for
    _raise_for_api to reject.
    """
    resp = requests.get(url, headers=auth, timeout=_TIMEOUT, allow_redirects=False, **kwargs)
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
    resp = _api_get(API_FILES_URL, {"X-Goog-Api-Key": key}, params={
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


def _open_download(file_id: str, auth: dict[str, str] | None) -> requests.Response:
    """Streaming full GET of a file: Drive API with auth headers, else the public link."""
    if auth:
        return _api_get(f"{API_FILES_URL}/{file_id}", auth, params={"alt": "media"}, stream=True)
    return requests.get(DOWNLOAD_URL.format(file_id=file_id), stream=True, timeout=_TIMEOUT)


def _access_token(creds: dict[str, str]) -> str:
    """
    A fresh access token from the refresh token. The secrets go only in the POST
    body; errors carry Google's error code, never a credential.
    """
    resp = requests.post(TOKEN_URL, data={**creds, "grant_type": "refresh_token"},
                         timeout=_TIMEOUT, allow_redirects=False)
    try:
        body = resp.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    if resp.ok and isinstance(body.get("access_token"), str):
        return body["access_token"]
    error = body.get("error") if isinstance(body.get("error"), str) else resp.reason
    if error == "invalid_grant":
        raise RuntimeError(f"Google OAuth: the refresh token in {OAUTH_ENV['refresh_token']} is "
                           f"expired or revoked (invalid_grant); re-run "
                           f"src/scripts/google_oauth_token.py and update the secret")
    desc = body.get("error_description")
    raise RuntimeError(f"Google OAuth token request failed (HTTP {resp.status_code}: {error}"
                       f"{': ' + desc if isinstance(desc, str) else ''})")


def _json_dict(resp: requests.Response, what: str) -> dict:
    """The response's JSON object; a RuntimeError for anything else."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise RuntimeError(f"Drive API: unexpected {what} response (not a JSON object)")
    return body


def _delete_copy(copy_id: str, auth: dict[str, str]) -> None:
    """Permanently delete one of our temporary copies; a failure only warns."""
    try:
        resp = requests.delete(f"{API_FILES_URL}/{copy_id}", headers=auth, timeout=_TIMEOUT,
                               allow_redirects=False)
        if resp.status_code == 404:
            logger.warning("OE: temporary Drive copy %s was not found on delete (HTTP 404)",
                           copy_id)
            return
        _raise_for_api(resp)
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        logger.warning("OE: could not delete the temporary Drive copy %s (%s); delete it by "
                       "hand in Drive", copy_id, exc)


def _is_stale_copy(f: object, now: datetime) -> bool:
    """A leftover copy of ours: tagged, exactly named, older than STALE_COPY_HOURS."""
    if not isinstance(f, dict) or not isinstance(f.get("id"), str):
        return False
    props, name = f.get("appProperties"), f.get("name")
    if not (isinstance(props, dict) and props.get(COPY_TAG[0]) == COPY_TAG[1]
            and isinstance(name, str) and _COPY_NAME_RE.fullmatch(name)):
        return False
    try:
        return now - datetime.fromisoformat(f["createdTime"]) > timedelta(hours=STALE_COPY_HOURS)
    except (KeyError, TypeError, ValueError):
        return False


@contextmanager
def _own_copy(year: int, file_id: str, creds: dict[str, str]):
    """
    Copy the OE file into the owner's My Drive (a new file, expected to have its
    own download quota) and yield (copy id, Bearer auth headers); the copy is
    deleted on exit. Leftover copies of crashed runs are deleted first.
    """
    auth = {"Authorization": f"Bearer {_access_token(creds)}"}
    resp = _api_get(API_FILES_URL, auth, params={
        "q": (f"appProperties has {{ key='{COPY_TAG[0]}' and value='{COPY_TAG[1]}' }} "
              f"and 'me' in owners and trashed=false"),
        "fields": "files(id,name,createdTime,appProperties)"})
    _raise_for_api(resp)
    files = _json_dict(resp, "files.list").get("files", [])
    now = datetime.now(timezone.utc)
    for stale in files if isinstance(files, list) else []:
        if _is_stale_copy(stale, now):
            _delete_copy(stale["id"], auth)
    resp = requests.post(f"{API_FILES_URL}/{file_id}/copy", headers=auth,
                         params={"supportsAllDrives": "true", "fields": "id"},
                         json={"name": f"{COPY_NAME} {year}", "parents": ["root"],
                               "appProperties": dict([COPY_TAG])},
                         timeout=_TIMEOUT, allow_redirects=False)
    _raise_for_api(resp)
    copy_id = _json_dict(resp, "files.copy").get("id")
    if not isinstance(copy_id, str):
        raise RuntimeError("Drive API: files.copy returned no file id")
    try:
        yield copy_id, auth
    finally:
        _delete_copy(copy_id, auth)


def route_order(key: str | None, creds: dict[str, str] | None) -> list[str]:
    """
    The download routes in order. With a key: the Drive API, then (OAuth creds)
    a copy in your Drive, then the public link. Without: the public link, then
    (OAuth creds) a copy in your Drive.
    """
    if key:
        return [ROUTE_API] + [ROUTE_COPY] * bool(creds) + [ROUTE_PUBLIC]
    return [ROUTE_PUBLIC] + [ROUTE_COPY] * bool(creds)


def _fetch(year: int, file_id: str, fetch, warn, key: str | None,
           creds: dict[str, str] | None) -> str:
    """
    Run fetch(file_id, auth) along route_order and return the route that worked.
    The Drive API moves on only on a quota error (other API errors raise as they
    are); a copy or public-link failure moves on whatever it is. After a
    fallback, the last route's error names the routes tried. warn(msg) reports
    each fallback.
    """
    def copy(fid):
        with _own_copy(year, fid, creds) as (copy_id, auth):
            fetch(copy_id, auth)

    attempt = {ROUTE_API: lambda fid: fetch(fid, {"X-Goog-Api-Key": key}),
               ROUTE_COPY: copy,
               ROUTE_PUBLIC: lambda fid: fetch(fid, None)}
    label = {ROUTE_API: "", ROUTE_COPY: "copy route failed: ", ROUTE_PUBLIC: "public link failed: "}
    routes = route_order(key, creds)
    for i, route in enumerate(routes):
        try:
            attempt[route](file_id)
            return route
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            if i == len(routes) - 1 or (route == ROUTE_API
                                        and not isinstance(exc, DriveQuotaError)):
                if i == 0:
                    raise
                raise RuntimeError(f"{exc} (tried {', '.join(routes[:i + 1])})") from exc
            warn(f"{label[route]}{exc} - retrying via {routes[i + 1]}")


def download_year(year: int, dest: Path = OE_DIR, file_id: str | None = None,
                  force: bool = False) -> Path:
    """
    Download one year's CSV into dest (atomic: temp file + os.replace) and return
    its path. Skips the download if the local file is younger than MAX_AGE_HOURS,
    unless force. Routes as in route_order / _fetch: with a key the Drive API, on
    an API quota error a temporary copy in the owner's Drive (OAuth creds set),
    then the public link; without a key the public link, then the copy. Raises
    if Drive returns anything but the OE CSV.
    """
    path = year_path(year, dest)
    if not force and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_HOURS * 3600:
        logger.info("OE %d: %s is fresh, skipping download", year, path.name)
        return path
    if file_id is None:
        file_id = list_files()[year]

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    via = _fetch(year, file_id, lambda fid, auth: _download(year, fid, path, auth),
                 lambda msg: logger.warning("OE %d: %s", year, msg), api_key(), oauth_credentials())
    logger.info("OE %d: downloaded %s B in %.1fs via %s", year, f"{path.stat().st_size:,}",
                time.monotonic() - t0, via)
    return path


def _download(year: int, file_id: str, path: Path, auth: dict[str, str] | None) -> None:
    tmp = path.with_name(path.name + ".part")
    try:
        with _open_download(file_id, auth) as resp:
            if auth:
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

def _check_download(year: int, file_id: str, auth: dict[str, str] | None) -> None:
    """
    Start a full download, read only its first 20 KB and print one status line.
    Raises unless the body is the OE CSV (DriveQuotaError on a Drive API quota
    error, else RuntimeError saying why).
    """
    t0 = time.monotonic()
    try:
        # stream + bounded read: only the first 20 KB of the full download is pulled.
        with _open_download(file_id, auth) as resp:
            if auth:
                _raise_for_api(resp)
            body = next(resp.iter_content(chunk_size=20480), b"")
    except DriveQuotaError:
        raise
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        print(f"download  {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        raise RuntimeError(f"could not download the {year} file") from exc
    print(f"download  {year} HTTP {resp.status_code}  {len(body):,} B  "
          f"{time.monotonic() - t0:5.2f}s  total {resp.headers.get('Content-Length', '?')} B  "
          f"last-modified {resp.headers.get('Last-Modified', '?')}")
    if resp.status_code in (200, 206) and body.startswith(b"gameid,"):
        return
    title = re.search(rb"<title>(.*?)</title>", body)
    raise RuntimeError(f"{year} download did not return the OE CSV"
                       f"{': ' + title[1].decode(errors='replace') if title else ''}")


def check() -> int:
    """
    Say which path is used (Drive API with GOOGLE_DRIVE_API_KEY, else the public
    link) and whether the copy route is configured (then test its token; a
    failing token drops it), print the effective route order, list the folder,
    then start a full (non-Range) download of the latest year's CSV
    along the same routes as download_year, read only its first 20 KB and print
    status / size / latency and the route that delivered it. Returns 0 if all of
    that works and the body is the OE CSV, else 1. A Range request would hide
    Drive's "Quota exceeded" page, which only blocks full downloads.
    """
    key = api_key()
    print(f"source    {'Drive API' if key else 'public link'} ({API_KEY_ENV} "
          f"{'set' if key else 'not set'})")
    creds = oauth_credentials()
    copy_err = None
    if creds:
        try:
            _access_token(creds)
            print("copy      configured (GOOGLE_OAUTH_* set), access token OK")
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            copy_err = str(exc)
            print(f"copy      configured (GOOGLE_OAUTH_* set), but {exc}")
    else:
        print("copy      not configured (GOOGLE_OAUTH_* not all set)")
    if copy_err:
        creds = None
    print(f"routes    {' -> '.join(route_order(key, creds))}")
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
        via = _fetch(year, files[year], lambda fid, auth: _check_download(year, fid, auth),
                     lambda msg: print(f"download  {msg}"), key, creds)
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        print(f"FAILED - {exc}.")
        return 1
    if copy_err:
        print(f"FAILED - OE {year} download works (via {via}), but the copy route does not: "
              f"{copy_err}.")
        return 1
    print(f"OK - OE folder and {year} download work (via {via}).")
    return 0


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
