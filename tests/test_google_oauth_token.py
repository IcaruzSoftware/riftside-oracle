"""google_oauth_token: PKCE, consent URL, redirect parsing and code exchange (no browser, no server)."""

import base64
import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

SECRET, VERIFIER, CODE = "csecret-DUMMY-91b2", "verifier-DUMMY-0a1b", "code-DUMMY-2c3d"
REDIRECT = "http://127.0.0.1:53682"


@pytest.fixture
def tok(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "src" / "scripts"))
    import google_oauth_token
    return google_oauth_token


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self.ok, self.reason = status, status < 400, "Bad Request"
        self.payload = payload

    def json(self):
        return self.payload


def test_pkce_pair_is_s256(tok):
    verifier, challenge = tok.pkce_pair()
    assert 43 <= len(verifier) <= 128 and re.fullmatch(r"[A-Za-z0-9_-]+", verifier)
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert tok.pkce_pair()[0] != verifier


def test_auth_url_asks_for_offline_drive_access_with_pkce(tok):
    parts = urlsplit(tok.auth_url("cid", REDIRECT, "chal", "st"))
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == tok.AUTH_URL
    assert {k: v[0] for k, v in parse_qs(parts.query).items()} == {
        "client_id": "cid", "redirect_uri": REDIRECT, "response_type": "code",
        "scope": "https://www.googleapis.com/auth/drive", "access_type": "offline",
        "prompt": "consent", "code_challenge": "chal", "code_challenge_method": "S256",
        "state": "st"}


@pytest.mark.parametrize("path, match", [
    ("/?state=other&code=c", "state mismatch"),
    ("/?code=c", "state mismatch"),
    ("/?state=st&error=access_denied", "authorisation refused: access_denied"),
    ("/?state=st", "no code"),
])
def test_parse_callback_rejects_bad_redirects(tok, path, match):
    with pytest.raises(RuntimeError, match=match):
        tok.parse_callback(path, "st")


@pytest.mark.parametrize("path, expected", [
    ("/?state=st&code=c&scope=x", True),
    ("/?error=access_denied", True),
    ("/favicon.ico", False),
    ("/", False),
    ("/?foo=1", False),
])
def test_is_oauth_redirect_ignores_stray_requests(tok, path, expected):
    assert tok.is_oauth_redirect(path) is expected


def test_parse_callback_returns_code(tok):
    assert tok.parse_callback("/?state=st&code=4%2F0Ab&scope=x", "st") == "4/0Ab"


def test_exchange_code_posts_pkce_verifier_and_returns_refresh_token(tok, monkeypatch):
    calls = []
    monkeypatch.setattr(tok.requests, "post", lambda url, **kw: calls.append((url, kw)) or _Resp(
        200, {"access_token": "a", "refresh_token": "r-1", "scope": tok.SCOPE}))
    assert tok.exchange_code(CODE, "cid", SECRET, VERIFIER, REDIRECT) == "r-1"
    (url, kw), = calls
    assert url == tok.TOKEN_URL and kw["data"] == {
        "code": CODE, "client_id": "cid", "client_secret": SECRET, "code_verifier": VERIFIER,
        "redirect_uri": REDIRECT, "grant_type": "authorization_code"}


@pytest.mark.parametrize("resp, match", [
    (_Resp(400, {"error": "invalid_grant", "error_description": "Bad Request"}),
     r"token exchange failed \(HTTP 400: invalid_grant\)"),
    (_Resp(401, None), r"token exchange failed \(HTTP 401: Bad Request\)"),
    (_Resp(200, {"access_token": "a", "scope": "https://www.googleapis.com/auth/drive"}),
     "no refresh token"),
    (_Resp(200, {"access_token": "a", "refresh_token": "r-1", "scope": "openid email"}),
     "Drive access was not granted; re-run and tick the Google Drive checkbox"),
    (_Resp(200, {"access_token": "a", "refresh_token": "r-1"}), "Drive access was not granted"),
    (_Resp(200, {"refresh_token": "r-1",
                 "scope": "https://www.googleapis.com/auth/drive.file"}), "not granted"),
])
def test_exchange_code_errors_carry_no_secret(tok, monkeypatch, resp, match):
    monkeypatch.setattr(tok.requests, "post", lambda url, **kw: resp)
    with pytest.raises(RuntimeError, match=match) as exc:
        tok.exchange_code(CODE, "cid", SECRET, VERIFIER, REDIRECT)
    assert all(s not in str(exc.value) for s in (SECRET, VERIFIER, CODE))
