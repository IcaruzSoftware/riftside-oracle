import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from pipeline import oe
from pipeline.oe import parse_folder


def _entry(file_id, title):
    return (f'<div class="flip-entry" id="entry-{file_id}"><a href="https://drive.google.com/'
            f'file/d/{file_id}/view?usp=drive_web"><div class="flip-entry-title">{title}</div></a></div>')


def test_parse_folder_pairs_ids_with_year_titles():
    html = (_entry("id2025", "2025_LoL_esports_match_data_from_OraclesElixir.csv")
            + _entry("idOther", "readme.txt")
            + _entry("id2026", "2026_LoL_esports_match_data_from_OraclesElixir.csv"))
    assert parse_folder(html) == {2025: "id2025", 2026: "id2026"}


# ---------------------------------------------------------------------------
# load_year / row splitting on the captured 2026 fixture (4 games, 48 rows)
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "oe"
LPL_PARTIAL = "13080-13080_game_1"
LPL_MISSING = "13065-13065_game_1"


@pytest.fixture(scope="module")
def df():
    return oe.load_year(2026, FIXTURES)


def test_load_year_columns_and_dtypes(df):
    assert len(df) == 48 and df["gameid"].nunique() == 4
    assert list(df.columns) == oe.NEEDED_COLS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_integer_dtype(df["result"])
    assert set(df["patch"]) == {"16.01", "16.02", "16.13"}   # strings, not floats
    assert df["playerid"].isna().sum() == 8                  # team rows: NaN, not "nan"


def test_load_year_nan_fills_absent_column(tmp_path):
    src = pd.read_csv(oe.year_path(2026, FIXTURES), low_memory=False)
    src.drop(columns=["wcpm"]).to_csv(oe.year_path(2026, tmp_path), index=False)
    assert oe.load_year(2026, tmp_path)["wcpm"].isna().all()


def test_player_and_team_rows(df):
    players, teams = oe.player_rows(df), oe.team_rows(df)
    assert len(players) == 40 and len(teams) == 8
    assert list(teams.columns) == oe.TEAM_GAMES_COLS
    assert teams["date"].is_monotonic_increasing
    for _, t in teams.iterrows():
        side = players[(players["gameid"] == t["gameid"]) & (players["side"] == t["side"])]
        by_role = side.set_index("position")["playerid"]
        assert t["starters"].split("|") == [by_role[r] for r in oe.ROLES]


def test_team_rows_keeps_empty_slot_for_missing_role(df):
    df = df[~((df["gameid"] == LPL_PARTIAL) & (df["side"] == "Blue") & (df["position"] == "mid"))]
    t = oe.team_rows(df)
    starters = t[(t["gameid"] == LPL_PARTIAL) & (t["side"] == "Blue")]["starters"].iloc[0]
    assert starters.split("|")[2] == "" and len(starters.split("|")) == 5


def test_lpl_partial_lacks_early_game_stats(df):
    g = oe.player_rows(df[df["gameid"] == LPL_PARTIAL])
    assert (g["datacompleteness"] == "partial").all()
    assert g["golddiffat15"].isna().all()


def test_missing_playername_keeps_playerid(df):
    row = df[(df["gameid"] == LPL_MISSING) & (df["playername"] == "MISSING")]
    assert len(row) == 1 and row["playerid"].iloc[0].startswith("oe:player:")


# ---------------------------------------------------------------------------
# download_year
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests default to the public-link path, whatever the local environment has set."""
    for env in ("GOOGLE_DRIVE_API_KEY", *oe.OAUTH_ENV.values()):
        monkeypatch.delenv(env, raising=False)


def test_download_year_skips_fresh_file(tmp_path, monkeypatch):
    path = oe.year_path(2026, tmp_path)
    path.write_text("gameid,x\n")
    monkeypatch.setattr(oe.requests, "get", lambda *a, **k: pytest.fail("downloaded"))
    assert oe.download_year(2026, tmp_path, file_id="x") == path


def test_download_year_replaces_stale_file(tmp_path, monkeypatch):
    path = oe.year_path(2026, tmp_path)
    path.write_text("old")
    os.utime(path, (time.time() - 21 * 3600,) * 2)
    monkeypatch.setattr(oe.requests, "get", lambda *a, **k: _Resp(b"gameid,new\n"))
    oe.download_year(2026, tmp_path, file_id="x")
    assert path.read_bytes() == b"gameid,new\n"


def test_download_year_rejects_html(tmp_path, monkeypatch):
    monkeypatch.setattr(oe.requests, "get",
                        lambda *a, **k: _Resp(b"<!DOCTYPE html><title>Quota exceeded</title>"))
    with pytest.raises(RuntimeError):
        oe.download_year(2026, tmp_path, file_id="x")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Drive API path (GOOGLE_DRIVE_API_KEY)
# ---------------------------------------------------------------------------

KEY = "test-key-123"
CSV = b"gameid,new\n"
PUBLIC = "https://drive.usercontent.google.com"
QUOTA = {"error": {"code": 403, "message": "The download quota for this file has been exceeded.",
                   "errors": [{"domain": "usageLimits", "reason": "downloadQuotaExceeded",
                               "message": "The download quota for this file has been exceeded."}]}}
QUOTA_PAGE = b"<!DOCTYPE html><title>Google Drive - Quota exceeded</title>"
LISTING = {"files": [{"id": "id2026", "name": "2026_LoL_esports_match_data_from_OraclesElixir.csv"}]}


class _ApiResp(_Resp):
    def __init__(self, body=b"", status=200, payload=None, location=None):
        super().__init__(json.dumps(payload).encode() if payload is not None else body)
        self.status_code, self.reason = status, "Error"
        self.ok, self.headers = status < 400, {"Location": location} if location else {}
        self.text = self.body.decode()

    def json(self):
        return json.loads(self.body)

    def close(self):
        pass


def _fake_get(monkeypatch, responses, key=KEY):
    """Set the key (None: unset) and stub requests.get: records (url, kwargs), answers by URL prefix."""
    if key is not None:
        monkeypatch.setenv("GOOGLE_DRIVE_API_KEY", key)
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return next(r for prefix, r in responses.items() if url.startswith(prefix))
    monkeypatch.setattr(oe.requests, "get", get)
    return calls


def test_download_year_uses_api_with_key(tmp_path, monkeypatch):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(CSV)})
    path = oe.download_year(2026, tmp_path, file_id="fid")
    assert path.read_bytes() == CSV and list(tmp_path.iterdir()) == [path]   # no .part left
    (url, kwargs), = calls
    assert url == f"{oe.API_FILES_URL}/fid" and kwargs["params"] == {"alt": "media"}
    assert kwargs["headers"] == {"X-Goog-Api-Key": KEY} and kwargs["allow_redirects"] is False


@pytest.mark.parametrize("resp", [
    _ApiResp(status=403, payload=QUOTA),
    _ApiResp(b"", status=429),
    _ApiResp(status=403, payload={"error": {"status": "RESOURCE_EXHAUSTED",
                                            "details": [{"reason": "RATE_LIMIT_EXCEEDED"}]}}),
])
def test_api_quota_falls_back_to_public_link(tmp_path, monkeypatch, resp):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: resp, PUBLIC: _Resp(CSV)})
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert [u.split("?")[0] for u, _ in calls] == [f"{oe.API_FILES_URL}/fid", f"{PUBLIC}/download"]
    assert "headers" not in calls[1][1]          # the key never goes to the public link


def test_api_quota_and_public_block_raise_clear_error(tmp_path, monkeypatch, caplog):
    _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(status=403, payload=QUOTA),
                            PUBLIC: _Resp(QUOTA_PAGE)})
    with pytest.raises(RuntimeError, match="is not the OE CSV: 'Google Drive - Quota exceeded'") as exc:
        oe.download_year(2026, tmp_path, file_id="fid")
    assert "Drive API quota exceeded" in caplog.text
    assert KEY not in str(exc.value) and KEY not in caplog.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("resp, match", [
    (_ApiResp(status=400, payload={"error": {
        "code": 400, "message": "API key not valid. Please pass a valid API key.",
        "errors": [{"reason": "badRequest"}], "details": [{"reason": "API_KEY_INVALID"}]}}),
     "is invalid"),
    (_ApiResp(status=403, payload={"error": {"message": "Bad key", "errors": [{"reason": "keyInvalid"}]}}),
     "is invalid"),
    (_ApiResp(status=404, payload={"error": {"message": "File not found: fid.",
                                             "errors": [{"reason": "notFound"}]}}),
     "not found or not public"),
    (_ApiResp(QUOTA_PAGE), "is not the OE CSV"),                     # API 200, but not the CSV
    # Error bodies of any shape still give a clear RuntimeError:
    (_ApiResp(b"<html>oops</html>", status=500), r"Drive API error \(HTTP 500"),
    (_ApiResp(b"null", status=500), r"Drive API error \(HTTP 500"),
    (_ApiResp(b"[1, 2]", status=500), r"Drive API error \(HTTP 500"),
    (_ApiResp(status=500, payload={"error": "boom"}), r"Drive API error \(HTTP 500"),
    (_ApiResp(status=500, payload={"error": {"errors": "x", "details": [1], "message": 5}}),
     r"Drive API error \(HTTP 500"),
])
def test_api_errors_are_mapped_without_fallback(tmp_path, monkeypatch, resp, match):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: resp})
    with pytest.raises(RuntimeError, match=match) as exc:
        oe.download_year(2026, tmp_path, file_id="fid")
    assert not isinstance(exc.value, oe.DriveQuotaError) and len(calls) == 1
    assert KEY not in str(exc.value) and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("env", [None, "", "  \n"])
def test_no_empty_or_blank_key_uses_public_link(tmp_path, monkeypatch, env):
    calls = _fake_get(monkeypatch, {PUBLIC: _Resp(CSV)}, key=env)
    oe.download_year(2026, tmp_path, file_id="fid")
    (url, kwargs), = calls
    assert url == oe.DOWNLOAD_URL.format(file_id="fid") and "headers" not in kwargs


MEDIA = "https://doc-0s-media.googleusercontent.com/download/abc"


def test_api_redirect_to_google_host_is_followed_without_key(tmp_path, monkeypatch):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(status=302, location=MEDIA),
                                    MEDIA: _ApiResp(CSV)})
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert [u for u, _ in calls] == [f"{oe.API_FILES_URL}/fid", MEDIA]
    assert "headers" not in calls[1][1] and calls[1][1]["allow_redirects"] is False


@pytest.mark.parametrize("location, hops", [
    ("https://elsewhere.example/x", 1),             # foreign host
    ("https://evilgoogleapis.com/x", 1),            # not a *.googleapis.com subdomain
    ("http://doc.googleusercontent.com/x", 1),      # not https
    (None, 1),                                      # no Location
    (MEDIA, 2),                                     # a second redirect
])
def test_api_unexpected_redirect_raises(tmp_path, monkeypatch, location, hops):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(status=302, location=location),
                                    MEDIA: _ApiResp(status=302, location=MEDIA)})
    with pytest.raises(RuntimeError, match="unexpected HTTP 302 redirect") as exc:
        oe.download_year(2026, tmp_path, file_id="fid")
    assert len(calls) == hops and all("headers" not in kw for _, kw in calls[1:])
    assert KEY not in str(exc.value) and list(tmp_path.iterdir()) == []


def test_parse_listing_pairs_ids_with_year_names():
    files = [{"id": "id2025", "name": "2025_LoL_esports_match_data_from_OraclesElixir.csv"},
             {"id": "idOther", "name": "readme.txt"},
             {"id": "idCopy", "name": "Copy of 2024_LoL_esports_match_data_from_OraclesElixir.csv"},
             {"id": "id2026", "name": "2026_LoL_esports_match_data_from_OraclesElixir.csv"}]
    assert oe.parse_listing(files) == {2025: "id2025", 2026: "id2026"}


def test_list_files_uses_api_with_key(monkeypatch):
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(payload=LISTING)})
    assert oe.list_files() == {2026: "id2026"}
    (url, kwargs), = calls
    assert url == oe.API_FILES_URL and kwargs["headers"] == {"X-Goog-Api-Key": KEY}
    assert kwargs["params"]["q"] == f"'{oe.FOLDER_ID}' in parents and trashed=false"
    assert kwargs["params"]["supportsAllDrives"] == "true" and KEY not in str(kwargs["params"])


def test_list_files_api_quota_falls_back_to_folder_view(monkeypatch):
    html = _entry("id2026", "2026_LoL_esports_match_data_from_OraclesElixir.csv").encode()
    calls = _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(b"", status=429),
                                    oe.FOLDER_URL: _ApiResp(html)})
    assert oe.list_files() == {2026: "id2026"}
    assert [u for u, _ in calls] == [oe.API_FILES_URL, oe.FOLDER_URL]
    assert "headers" not in calls[1][1]


def test_list_files_empty_api_listing_says_so(monkeypatch):
    _fake_get(monkeypatch, {oe.API_FILES_URL: _ApiResp(payload={"files": []})})
    with pytest.raises(RuntimeError, match="Drive API listing returned no OE files"):
        oe.list_files()


# ---------------------------------------------------------------------------
# Copy-to-own-Drive route (GOOGLE_OAUTH_*), after an API quota error
# ---------------------------------------------------------------------------

CLIENT_ID, CLIENT_SECRET, REFRESH = "cid-DUMMY-7f3a", "csecret-DUMMY-91b2", "refresh-DUMMY-c4d8"
ACCESS = "access-DUMMY-55e1"
SECRETS = (KEY, CLIENT_SECRET, REFRESH, ACCESS)
OAUTH = {"GOOGLE_OAUTH_CLIENT_ID": CLIENT_ID, "GOOGLE_OAUTH_CLIENT_SECRET": CLIENT_SECRET,
         "GOOGLE_OAUTH_REFRESH_TOKEN": REFRESH}
BEARER = {"Authorization": f"Bearer {ACCESS}"}
API = oe.API_FILES_URL
SWEEP_Q = ("appProperties has { key='riftsideOracleTmp' and value='1' } and 'me' in owners "
           "and trashed=false")
TAG = {"riftsideOracleTmp": "1"}
OLD = "2026-10-17T01:00:00.000Z"        # > 2 h before NOW
NOW = oe.datetime(2026, 10, 17, 7, 45, tzinfo=oe.timezone.utc)


def _copy_route(over=None, fid="fid"):
    """Responses by (method, url): API quota, then token, empty sweep, copy, copy download, delete."""
    return {("GET", f"{API}/{fid}"): _ApiResp(status=403, payload=QUOTA),
            ("POST", oe.TOKEN_URL): _ApiResp(payload={"access_token": ACCESS, "expires_in": 3599}),
            ("GET", API): _ApiResp(payload={"files": []}),
            ("POST", f"{API}/{fid}/copy"): _ApiResp(payload={"id": "cid"}),
            ("GET", f"{API}/cid"): _ApiResp(CSV),
            ("DELETE", f"{API}/cid"): _ApiResp(status=204),
            ("GET", oe.DOWNLOAD_URL.format(file_id=fid)): _ApiResp(CSV),
            **(over or {})}


def _fake_http(monkeypatch, responses, oauth=OAUTH):
    """Set the key + OAuth env; stub requests.get/post/delete, recording (method, url, kwargs).
    A response may be a callable(kwargs) -> response."""
    monkeypatch.setenv("GOOGLE_DRIVE_API_KEY", KEY)
    for env, value in oauth.items():
        monkeypatch.setenv(env, value)
    calls = []

    def stub(method):
        def call(url, **kwargs):
            calls.append((method, url, kwargs))
            r = responses[(method, url)]
            return r(kwargs) if callable(r) else r
        return call
    for method in ("get", "post", "delete"):
        monkeypatch.setattr(oe.requests, method, stub(method.upper()))
    return calls


def _routes(calls):
    return [(m, u) for m, u, _ in calls]


def _assert_no_secrets(calls, *texts):
    urls = str([(u, kw.get("params"), kw.get("json")) for _, u, kw in calls])
    for secret in SECRETS:
        assert all(secret not in t for t in (urls, *texts))


def test_copy_route_after_api_quota(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="pipeline.oe")
    calls = _fake_http(monkeypatch, _copy_route())
    path = oe.download_year(2026, tmp_path, file_id="fid")
    assert path.read_bytes() == CSV and list(tmp_path.iterdir()) == [path]   # atomic, no .part
    assert _routes(calls) == [("GET", f"{API}/fid"), ("POST", oe.TOKEN_URL), ("GET", API),
                              ("POST", f"{API}/fid/copy"), ("GET", f"{API}/cid"),
                              ("DELETE", f"{API}/cid")]
    token, sweep, copy, media, delete = (kw for _, _, kw in calls[1:])
    assert token["data"] == {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                             "refresh_token": REFRESH, "grant_type": "refresh_token"}
    assert "headers" not in token and token["allow_redirects"] is False
    assert sweep["params"] == {"q": SWEEP_Q, "fields": "files(id,name,createdTime,appProperties)"}
    assert copy["params"] == {"supportsAllDrives": "true", "fields": "id"}
    assert copy["json"] == {"name": "riftside-oracle tmp OE 2026", "parents": ["root"],
                            "appProperties": TAG}
    assert media["params"] == {"alt": "media"}
    assert all(kw["headers"] == BEARER and kw["allow_redirects"] is False
               for kw in (sweep, copy, media, delete))
    assert "retrying via a copy in your Drive" in caplog.text
    assert "via a copy in your Drive" in caplog.records[-1].getMessage()
    _assert_no_secrets(calls, caplog.text)


@pytest.mark.parametrize("oauth, missing", [
    ({}, None),
    ({"GOOGLE_OAUTH_CLIENT_ID": CLIENT_ID, "GOOGLE_OAUTH_REFRESH_TOKEN": " \n"},
     "GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REFRESH_TOKEN"),
    ({**OAUTH, "GOOGLE_OAUTH_CLIENT_SECRET": ""}, "GOOGLE_OAUTH_CLIENT_SECRET"),
])
def test_copy_route_off_without_all_creds(tmp_path, monkeypatch, caplog, oauth, missing):
    calls = _fake_http(monkeypatch, _copy_route(), oauth)
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert _routes(calls) == [("GET", f"{API}/fid"), ("GET", oe.DOWNLOAD_URL.format(file_id="fid"))]
    assert "retrying via the public link" in caplog.text
    if missing:
        assert f"OE copy route off: {missing} not set" in caplog.text
    else:
        assert "copy route" not in caplog.text
    assert CLIENT_ID not in caplog.text
    _assert_no_secrets(calls, caplog.text)


def test_invalid_grant_is_clear_and_falls_back_to_public_link(tmp_path, monkeypatch, caplog):
    calls = _fake_http(monkeypatch, _copy_route({("POST", oe.TOKEN_URL): _ApiResp(
        status=400, payload={"error": "invalid_grant",
                             "error_description": "Token has been expired or revoked."})}))
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert _routes(calls) == [("GET", f"{API}/fid"), ("POST", oe.TOKEN_URL),   # no token retry
                              ("GET", oe.DOWNLOAD_URL.format(file_id="fid"))]
    assert "expired or revoked (invalid_grant)" in caplog.text
    assert "re-run src/scripts/google_oauth_token.py" in caplog.text
    _assert_no_secrets(calls, caplog.text)


@pytest.mark.parametrize("oauth, tried", [
    (OAUTH, "the Drive API, a copy in your Drive, the public link"),
    ({}, "the Drive API, the public link"),
])
def test_all_routes_blocked_raise_error_naming_routes(tmp_path, monkeypatch, caplog, oauth, tried):
    calls = _fake_http(monkeypatch, _copy_route({
        ("POST", oe.TOKEN_URL): _ApiResp(status=401, payload={
            "error": "invalid_client", "error_description": "The OAuth client was not found."}),
        ("GET", oe.DOWNLOAD_URL.format(file_id="fid")): _ApiResp(QUOTA_PAGE)}), oauth)
    with pytest.raises(RuntimeError, match=rf"Quota exceeded' \(tried {tried}\)") as exc:
        oe.download_year(2026, tmp_path, file_id="fid")
    if oauth:
        assert "HTTP 401: invalid_client: The OAuth client was not found." in caplog.text
    assert list(tmp_path.iterdir()) == []
    _assert_no_secrets(calls, str(exc.value), caplog.text)


@pytest.mark.parametrize("blocked, routes", [
    (("POST", f"{API}/fid/copy"), ["copy"]),
    (("GET", f"{API}/cid"), ["copy", "media", "delete"]),        # the copy is still deleted
])
def test_copy_quota_falls_back_to_public_link(tmp_path, monkeypatch, caplog, blocked, routes):
    calls = _fake_http(monkeypatch, _copy_route({blocked: _ApiResp(status=403, payload=QUOTA)}))
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    step = {"copy": ("POST", f"{API}/fid/copy"), "media": ("GET", f"{API}/cid"),
            "delete": ("DELETE", f"{API}/cid")}
    assert _routes(calls) == [("GET", f"{API}/fid"), ("POST", oe.TOKEN_URL), ("GET", API),
                              *(step[r] for r in routes),
                              ("GET", oe.DOWNLOAD_URL.format(file_id="fid"))]
    assert "copy route failed: Drive API quota exceeded" in caplog.text
    assert "headers" not in calls[-1][2]                 # no credential to the public link
    _assert_no_secrets(calls, caplog.text)


def test_copy_delete_failure_only_warns(tmp_path, monkeypatch, caplog):
    calls = _fake_http(monkeypatch, _copy_route({("DELETE", f"{API}/cid"): _ApiResp(
        status=500, payload={"error": {"message": "Backend Error"}})}))
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert "could not delete the temporary Drive copy cid" in caplog.text
    _assert_no_secrets(calls, caplog.text)


def _stale(fid, name="riftside-oracle tmp OE 2025", created=OLD, props=TAG):
    f = {"id": fid, "name": name, "createdTime": created}
    return f if props is None else {**f, "appProperties": props}


def test_stale_copies_are_swept_before_copying(tmp_path, monkeypatch):
    class _Now(oe.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(oe, "datetime", _Now)
    calls = _fake_http(monkeypatch, _copy_route({
        ("GET", API): _ApiResp(payload={"files": [
            _stale("old1"),
            _stale("old2", name="riftside-oracle tmp OE 2026", created="2026-10-17T05:44:59Z"),
            _stale("young", created="2026-10-17T06:00:00.000Z"),     # a concurrent run's copy
            _stale("other", name="riftside-oracle tmp OE 2026 notes"),
            _stale("prefix", name="riftside-oracle tmp OEx 2026"),
            _stale("untagged", props=None),
            _stale("wrongtag", props={"riftsideOracleTmp": "0"}),
            _stale("nodate", created=None),
            {"name": "riftside-oracle tmp OE 2024"}, "junk"]}),
        ("DELETE", f"{API}/old1"): _ApiResp(status=204),
        ("DELETE", f"{API}/old2"): _ApiResp(status=204)}))
    oe.download_year(2026, tmp_path, file_id="fid")
    assert _routes(calls)[2:6] == [("GET", API), ("DELETE", f"{API}/old1"),
                                   ("DELETE", f"{API}/old2"), ("POST", f"{API}/fid/copy")]
    assert [u for m, u, _ in calls if m == "DELETE"] == [f"{API}/old1", f"{API}/old2", f"{API}/cid"]
    assert all(kw["headers"] == BEARER for m, _, kw in calls if m == "DELETE")


@pytest.mark.parametrize("over", [
    {("GET", API): _ApiResp(b"<html>oops</html>")},                  # sweep: not JSON
    {("GET", API): _ApiResp(b"[1, 2]")},                             # sweep: not an object
    {("POST", f"{API}/fid/copy"): _ApiResp(b"null")},                # copy: not an object
    {("POST", f"{API}/fid/copy"): _ApiResp(payload={"kind": "drive#file"})},   # copy: no id
    {("POST", f"{API}/fid/copy"): _ApiResp(payload={"id": 5})},
])
def test_malformed_copy_route_bodies_fall_back_to_public_link(tmp_path, monkeypatch, caplog, over):
    calls = _fake_http(monkeypatch, _copy_route(over))
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert "copy route failed: Drive API: " in caplog.text
    assert _routes(calls)[-1] == ("GET", oe.DOWNLOAD_URL.format(file_id="fid"))
    assert ("DELETE", f"{API}/cid") not in _routes(calls)


def test_sweep_with_malformed_files_list_still_copies(tmp_path, monkeypatch):
    calls = _fake_http(monkeypatch, _copy_route({("GET", API): _ApiResp(payload={"files": "x"})}))
    oe.download_year(2026, tmp_path, file_id="fid")
    assert ("POST", f"{API}/fid/copy") in _routes(calls)


def test_copy_delete_404_has_neutral_message(tmp_path, monkeypatch, caplog):
    _fake_http(monkeypatch, _copy_route({("DELETE", f"{API}/cid"): _ApiResp(
        status=404, payload={"error": {"message": "File not found: cid."}})}))
    oe.download_year(2026, tmp_path, file_id="fid")
    assert "temporary Drive copy cid was not found on delete (HTTP 404)" in caplog.text
    assert "not public" not in caplog.text


@pytest.mark.parametrize("key, oauth, order", [
    (KEY, OAUTH, ["the Drive API", "a copy in your Drive", "the public link"]),
    (KEY, {}, ["the Drive API", "the public link"]),
    (None, OAUTH, ["the public link", "a copy in your Drive"]),
    (None, {}, ["the public link"]),
])
def test_route_order(monkeypatch, key, oauth, order):
    for env, value in oauth.items():
        monkeypatch.setenv(env, value)
    assert oe.route_order(key, oe.oauth_credentials()) == order


def test_oauth_without_key_tries_copy_after_public_link(tmp_path, monkeypatch, caplog):
    calls = _fake_http(monkeypatch, _copy_route({
        ("GET", oe.DOWNLOAD_URL.format(file_id="fid")): _ApiResp(QUOTA_PAGE)}))
    monkeypatch.delenv("GOOGLE_DRIVE_API_KEY")
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    assert _routes(calls) == [("GET", oe.DOWNLOAD_URL.format(file_id="fid")),
                              ("POST", oe.TOKEN_URL), ("GET", API), ("POST", f"{API}/fid/copy"),
                              ("GET", f"{API}/cid"), ("DELETE", f"{API}/cid")]
    assert ("public link failed: OE 2026: download fid is not the OE CSV: 'Google Drive - Quota "
            "exceeded' - retrying via a copy in your Drive") in caplog.text
    _assert_no_secrets(calls, caplog.text)


def test_oauth_without_key_all_blocked_names_routes(tmp_path, monkeypatch, caplog):
    calls = _fake_http(monkeypatch, _copy_route({
        ("GET", oe.DOWNLOAD_URL.format(file_id="fid")): _ApiResp(QUOTA_PAGE),
        ("POST", f"{API}/fid/copy"): _ApiResp(status=403, payload=QUOTA)}))
    monkeypatch.delenv("GOOGLE_DRIVE_API_KEY")
    with pytest.raises(RuntimeError, match=r"Drive API quota exceeded .*\(tried the public link, "
                                           r"a copy in your Drive\)") as exc:
        oe.download_year(2026, tmp_path, file_id="fid")
    assert list(tmp_path.iterdir()) == []
    _assert_no_secrets(calls, str(exc.value), caplog.text)


def test_copy_redirect_is_followed_without_authorization(tmp_path, monkeypatch, caplog):
    calls = _fake_http(monkeypatch, _copy_route({
        ("GET", f"{API}/cid"): _ApiResp(status=302, location=MEDIA),
        ("GET", MEDIA): _ApiResp(CSV)}))
    assert oe.download_year(2026, tmp_path, file_id="fid").read_bytes() == CSV
    (_, _, kwargs), = [c for c in calls if c[1] == MEDIA]
    assert "headers" not in kwargs and kwargs["allow_redirects"] is False
    _assert_no_secrets(calls, caplog.text)


def _check_responses(over=None):
    """check(): the API listing (key) and the sweep (Bearer) share the files URL."""
    def listing(kw):
        return _ApiResp(payload=LISTING if "X-Goog-Api-Key" in kw["headers"] else {"files": []})
    return _copy_route({("GET", API): listing, **(over or {})}, fid="id2026")


def test_check_reports_copy_route(monkeypatch, capsys):
    calls = _fake_http(monkeypatch, _check_responses())
    assert oe.check() == 0
    out = capsys.readouterr()
    assert "copy      configured (GOOGLE_OAUTH_* set), access token OK" in out.out
    assert "routes    the Drive API -> a copy in your Drive -> the public link" in out.out
    assert "OK - OE folder and 2026 download work (via a copy in your Drive)." in out.out
    _assert_no_secrets(calls, out.out, out.err)


def test_check_reports_bad_oauth_without_values(monkeypatch, capsys):
    calls = _fake_http(monkeypatch, _check_responses({
        ("POST", oe.TOKEN_URL): _ApiResp(status=401, payload={"error": "invalid_client"}),
        ("GET", f"{API}/id2026"): _ApiResp(CSV)}))
    assert oe.check() == 1
    out = capsys.readouterr()
    assert "but Google OAuth token request failed (HTTP 401: invalid_client)" in out.out
    assert "download works (via the Drive API), but the copy route does not" in out.out
    assert "routes    the Drive API -> the public link" in out.out
    assert _routes(calls).count(("POST", oe.TOKEN_URL)) == 1
    _assert_no_secrets(calls, out.out, out.err)


def test_check_oauth_without_key(monkeypatch, capsys):
    html = _entry("id2026", "2026_LoL_esports_match_data_from_OraclesElixir.csv").encode()
    calls = _fake_http(monkeypatch, _copy_route({
        ("GET", oe.FOLDER_URL): _ApiResp(html),
        ("GET", oe.DOWNLOAD_URL.format(file_id="id2026")): _ApiResp(QUOTA_PAGE)}, fid="id2026"))
    monkeypatch.delenv("GOOGLE_DRIVE_API_KEY")
    assert oe.check() == 0
    out = capsys.readouterr()
    assert "routes    the public link -> a copy in your Drive" in out.out
    assert "download  public link failed: " in out.out
    assert "OK - OE folder and 2026 download work (via a copy in your Drive)." in out.out
    _assert_no_secrets(calls, out.out, out.err)
