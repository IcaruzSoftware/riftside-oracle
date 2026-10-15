"""fetch_boxscore treats 403 (Akamai block) as an error, 404 as not-yet-available."""

import pytest

from pipeline import nba_cdn


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, status, payload=None):
        self._resp = _Resp(status, payload)

    def get(self, url, timeout=None, proxies=None):
        return self._resp


def test_403_raises(monkeypatch):
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: _Session(403))
    with pytest.raises(RuntimeError):
        nba_cdn.fetch_boxscore("0042500405")


def test_404_returns_none(monkeypatch):
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: _Session(404))
    assert nba_cdn.fetch_boxscore("0042500405") is None


def test_200_returns_game(monkeypatch):
    monkeypatch.setattr(nba_cdn, "_get_session", lambda: _Session(200, {"game": {"gameId": "x"}}))
    assert nba_cdn.fetch_boxscore("0042500405") == {"gameId": "x"}
