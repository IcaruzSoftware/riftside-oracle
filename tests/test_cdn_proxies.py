"""NBA_PROXIES: validation, failover + stickiness, all-fail summary, direct default."""

import pytest
import requests

from pipeline import nba_cdn


# ---------------------------------------------------------------------------
# Fake session keyed by the proxy URL each request is routed through
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.content = b"x" * 10

    def json(self):
        return {"game": {"ok": True}, "leagueSchedule": {"gameDates": []}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


class _FakeSession:
    def __init__(self, behavior):
        self.behavior = behavior            # proxy-url (or None) -> ("ok", code) | ("raise", exc)
        self.calls = []                     # proxy-urls used, in order

    def get(self, url, timeout=None, proxies=None):
        key = proxies["https"] if proxies else None
        self.calls.append(key)
        kind, val = self.behavior[key]
        if kind == "raise":
            raise val
        return _Resp(val)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unset_env_is_direct(monkeypatch):
    monkeypatch.delenv(nba_cdn.PROXY_ENV, raising=False)
    assert nba_cdn._ensure_channels() == [None]


def test_valid_proxies_parse(monkeypatch):
    monkeypatch.setenv(nba_cdn.PROXY_ENV, "http://u:p@h1:8080, https://h2:3128")
    assert nba_cdn._ensure_channels() == ["http://u:p@h1:8080", "https://h2:3128"]


@pytest.mark.parametrize("bad", [
    "http://host",            # no port
    "ftp://host:1",           # bad scheme
    "http://host:abc",        # non-numeric port
    "host:1234",              # no scheme
    "http://user:s3cret@:22",  # no host
])
def test_invalid_proxy_raises_without_leaking_value(monkeypatch, bad):
    monkeypatch.setenv(nba_cdn.PROXY_ENV, bad)
    with pytest.raises(ValueError) as ei:
        nba_cdn._ensure_channels()
    msg = str(ei.value)
    assert "#0" in msg and "%40" in msg          # index + URL-encoding hint
    assert bad not in msg                         # never echoes the entry
    assert "s3cret" not in msg                    # nor any credential within it


# ---------------------------------------------------------------------------
# Failover + stickiness
# ---------------------------------------------------------------------------

def test_failover_order_and_stickiness(monkeypatch):
    monkeypatch.setenv(nba_cdn.PROXY_ENV, "http://p0:1,http://p1:2,http://p2:3")
    fake = _FakeSession({
        "http://p0:1": ("raise", requests.exceptions.ProxyError("nope")),
        "http://p1:2": ("ok", 200),
        "http://p2:3": ("ok", 200),
    })
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: fake)

    nba_cdn.fetch_schedule()
    assert fake.calls == ["http://p0:1", "http://p1:2"]   # failover p0 -> p1

    nba_cdn.fetch_schedule()
    assert fake.calls[-1] == "http://p1:2"                # sticky: starts at p1
    assert "http://p0:1" not in fake.calls[2:]            # p0 not retried first


def test_403_triggers_failover(monkeypatch):
    monkeypatch.setenv(nba_cdn.PROXY_ENV, "http://p0:1,http://p1:2")
    fake = _FakeSession({
        "http://p0:1": ("ok", 403),
        "http://p1:2": ("ok", 200),
    })
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: fake)
    nba_cdn.fetch_schedule()
    assert fake.calls == ["http://p0:1", "http://p1:2"]


# ---------------------------------------------------------------------------
# All-fail summary never leaks credentials
# ---------------------------------------------------------------------------

def test_all_fail_summary_has_hostport_not_credentials(monkeypatch):
    monkeypatch.setenv(nba_cdn.PROXY_ENV, "http://user:p%40ss@127.0.0.1:9")
    fake = _FakeSession({
        # message deliberately contains the credentials to prove they are not echoed
        "http://user:p%40ss@127.0.0.1:9": (
            "raise", requests.exceptions.ProxyError("fail via http://user:p%40ss@127.0.0.1:9")),
    })
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: fake)

    with pytest.raises(RuntimeError) as ei:
        nba_cdn.fetch_schedule()
    msg = str(ei.value)
    assert "127.0.0.1:9" in msg
    assert "ProxyError" in msg
    assert "user" not in msg
    assert "p%40ss" not in msg
