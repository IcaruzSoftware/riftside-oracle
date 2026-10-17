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
    monkeypatch.delenv("GOOGLE_DRIVE_API_KEY", raising=False)


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
