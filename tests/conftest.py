"""Shared test fixtures: import path, a fake Supabase client, and a tiny live-state
builder used by the prediction/evaluation tests."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fake Supabase — records writes, answers the query chains evaluate/predict use
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, table):
        self.t = table
        self.kind = "select"
        self.cols = "*"
        self.payload = None
        self.on_conflict = None
        self.filters = []
        self._negate = False

    def select(self, cols="*"):
        self.kind, self.cols = "select", cols
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        self.filters.append(("is", col, val, self._negate)); self._negate = False
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val, self._negate)); self._negate = False
        return self

    def update(self, payload):
        self.kind, self.payload = "update", payload
        return self

    def upsert(self, payload, on_conflict=None):
        self.kind, self.payload, self.on_conflict = "upsert", payload, on_conflict
        return self

    def insert(self, payload):
        self.kind, self.payload = "insert", payload
        return self

    def delete(self):
        self.kind = "delete"
        return self

    def _match(self, row):
        for kind, col, val, neg in self.filters:
            if kind == "is" and val == "null":
                ok = row.get(col) is None
            elif kind == "eq":
                ok = row.get(col) == val
            else:
                ok = True
            if neg:
                ok = not ok
            if not ok:
                return False
        return True

    def execute(self):
        t = self.t
        if self.kind == "select":
            rows = [r for r in t.rows if self._match(r)]
            return _Result([dict(r) for r in rows])
        if self.kind == "update":
            for r in t.rows:
                if self._match(r):
                    r.update(self.payload)
            return _Result([])
        if self.kind == "upsert":
            key = self.on_conflict or "id"
            row = dict(self.payload)
            existing = next((r for r in t.rows if r.get(key) == row.get(key)), None)
            if existing:
                existing.update(row)
                out = existing
            else:
                row.setdefault("id", f"id-{len(t.rows)+1}")
                t.rows.append(row)
                out = row
            t.db.writes.append((t.name, "upsert", dict(out)))
            return _Result([dict(out)])
        if self.kind == "insert":
            payload = self.payload if isinstance(self.payload, list) else [self.payload]
            for row in payload:
                row = dict(row)
                row.setdefault("id", f"id-{len(t.rows)+1}")
                t.rows.append(row)
                t.db.writes.append((t.name, "insert", row))
            return _Result([dict(r) for r in payload])
        if self.kind == "delete":
            kept = [r for r in t.rows if not self._match(r)]
            removed = len(t.rows) - len(kept)
            t.rows[:] = kept
            t.db.writes.append((t.name, "delete", {"removed": removed}))
            return _Result([])
        return _Result([])


class _Table:
    def __init__(self, name, db):
        self.name = name
        self.db = db
        self.rows = db.seed.get(name, [])

    def __getattr__(self, item):
        return getattr(_Query(self), item)


class FakeSupabase:
    def __init__(self, seed=None):
        self.seed = {k: [dict(r) for r in v] for k, v in (seed or {}).items()}
        self._tables = {}
        self.writes = []

    def table(self, name):
        if name not in self._tables:
            self._tables[name] = _Table(name, self)
        return self._tables[name]


@pytest.fixture
def fake_supabase():
    return FakeSupabase


@pytest.fixture(autouse=True)
def _reset_nba_cdn():
    """Clear cached CDN session/proxy state so NBA_PROXIES changes take effect per test."""
    from pipeline import nba_cdn
    nba_cdn.reset_session()
    yield
    nba_cdn.reset_session()


# ---------------------------------------------------------------------------
# Fixtures: real CDN box score + tiny live state built from it
# ---------------------------------------------------------------------------

@pytest.fixture
def finals_game():
    return json.loads((FIXTURES / "boxscore_0042500405.json").read_text(encoding="utf-8"))["game"]


def make_schedule(et_date: str, game: dict, status: int = 1) -> dict:
    """A scheduleLeagueV2-shaped dict with one game on et_date (default status 1 =
    scheduled, i.e. predictable)."""
    home, away = game["homeTeam"], game["awayTeam"]
    y, m, d = et_date[:4], et_date[5:7], et_date[8:10]
    return {"leagueSchedule": {"gameDates": [{
        "gameDate": f"{m}/{d}/{y} 00:00:00",
        "games": [{
            "gameId":          game["gameId"],
            "gameDateTimeUTC": game.get("gameTimeUTC"),
            "gameStatus":      status,
            "homeTeam": {"teamId": home["teamId"], "teamTricode": home["teamTricode"], "score": home.get("score")},
            "awayTeam": {"teamId": away["teamId"], "teamTricode": away["teamTricode"], "score": away.get("score")},
        }],
    }]}}


@pytest.fixture
def tiny_state(tmp_path, monkeypatch, finals_game):
    """Build a minimal committed live state (ELO + game logs + player stats) from the
    finals fixture in a temp dir, and repoint every module's data dirs at it."""
    import pipeline.elo as elo
    import pipeline.features as feat
    import pipeline.predict as pred
    import pipeline.daily_state as ds

    raw = tmp_path / "raw"; raw.mkdir()
    proc = tmp_path / "processed"; proc.mkdir()

    monkeypatch.setattr(elo, "PROCESSED_DIR", proc)
    monkeypatch.setattr(elo, "HISTORY_PATH", proc / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", proc / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", proc / "player_elo_recent.parquet")
    monkeypatch.setattr(feat, "PROCESSED_DIR", proc)
    monkeypatch.setattr(feat, "RAW_DIR", raw)
    monkeypatch.setattr(pred, "PROCESSED_DIR", proc)
    monkeypatch.setattr(pred, "RAW_DIR", raw)
    monkeypatch.setattr(ds, "RAW_DIR", raw)
    for attr in ["_elo_current", "_elo_recent", "_elo_recent_by_player",
                 "_elo_history", "_elo_by_game", "_elo_by_player"]:
        monkeypatch.setattr(feat, attr, None)

    gid, date = "0042500405", "2026-06-13"
    frame = elo.load_cdn_game_players(finals_game)
    game_index = pd.DataFrame({"GAME_ID": [gid], "GAME_DATE": [pd.Timestamp(date)]})
    elo.build_elo(force=True, loader=lambda g: frame.copy(), game_index=game_index)

    # Minimal game logs (both teams) from the same box score.
    rows = ds._game_log_rows(finals_game, gid, date)
    pd.DataFrame(rows).to_csv(raw / "game_log_playoffs_2025-26.csv", index=False)

    # Minimal player season stats (PPG weights) for the players in the game.
    contribs = ds._player_contribs(finals_game)
    stats = pd.DataFrame(
        [{"PLAYER_ID": int(pid), "PLAYER_NAME": name, "TEAM_ID": tid, "GP": 1, "PTS": pts}
         for pid, tid, pts, name in contribs]
    )
    stats.to_csv(raw / "player_season_stats_2025-26.csv", index=False)

    return {"raw": raw, "proc": proc, "date": date, "game_id": gid,
            "schedule": make_schedule(date, finals_game)}
