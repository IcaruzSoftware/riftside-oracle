import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pipeline import lolesports

FIXTURES = Path(__file__).parent / "fixtures" / "lolesports"
UTC = timezone.utc


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_upcoming_matches_filters_tbd_completed_and_shows(monkeypatch):
    monkeypatch.setattr(lolesports, "league_ids", lambda slugs: ["98767975604431411"])
    page = _load("schedule_page.json")["data"]["schedule"]
    page["pages"]["newer"] = None
    monkeypatch.setattr(lolesports, "get_schedule", lambda ids, token=None: page)
    got = lolesports.upcoming_matches(["worlds"], datetime(2025, 1, 1, tzinfo=UTC),
                                      datetime(2026, 10, 17, tzinfo=UTC))
    assert [e["match"]["id"] for e in got] == ["115660588583171786", "115660588583040710"]


def test_upcoming_matches_respects_window(monkeypatch):
    monkeypatch.setattr(lolesports, "league_ids", lambda slugs: ["x"])
    page = _load("schedule_page.json")["data"]["schedule"]
    page["pages"]["newer"] = None
    monkeypatch.setattr(lolesports, "get_schedule", lambda ids, token=None: page)
    got = lolesports.upcoming_matches(["worlds"], datetime(2026, 10, 15, 12, tzinfo=UTC),
                                      datetime(2026, 10, 15, 20, tzinfo=UTC))
    assert [e["match"]["id"] for e in got] == ["115660588583171786"]


def test_get_event_details_returns_event(monkeypatch):
    body = _load("event_details.json")
    monkeypatch.setattr(lolesports, "_get", lambda endpoint, **p: body["data"])
    event = lolesports.get_event_details("113475871524050783")
    assert event["id"] == "113475871524050783"
    assert [t["result"]["gameWins"] for t in event["match"]["teams"]] == [3, 2]


@pytest.mark.parametrize("name, expected", [
    ("Anubis Gaming", "anubis"), ("Gen.G Esports", "geng"), ("T1", "t1"),
    ("Team", "team"), ("Team Secret Whales", "teamsecretwhales"),
    ("T1 Esports Academy", "t1esportsacademy"),
])
def test_normalize_team_name(name, expected):
    assert lolesports.normalize_team_name(name) == expected


def test_academy_is_a_separate_team():
    assert lolesports.normalize_team_name("T1 Esports Academy") != lolesports.normalize_team_name("T1")


# ---------------------------------------------------------------------------
# resolve_team
# ---------------------------------------------------------------------------

AS_OF = datetime(2026, 10, 15, tzinfo=UTC)
TEAM_GAMES = pd.DataFrame({
    "date": pd.to_datetime(["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02", "2025-01-01"]),
    "teamid": ["oe:team:geng", "oe:team:t1", "oe:team:dk1", "oe:team:dk2", "oe:team:old"],
    "teamname": ["Gen.G", "T1", "Dplus KIA", "Dplus Kia", "Old Squad"],
})


@pytest.fixture(autouse=True)
def aliases(tmp_path, monkeypatch):
    path = tmp_path / "team_aliases.csv"
    path.write_text("lolesports_code,lolesports_name,oe_teamid\nHLE,Hanwha Life Esports,oe:team:hle\n")
    monkeypatch.setattr(lolesports, "ALIASES_PATH", path)


def _resolve(name, code):
    return lolesports.resolve_team(name, code, TEAM_GAMES, as_of=AS_OF)


def test_resolve_team_alias_hit():
    assert _resolve("Hanwha Life Esports", "HLE") == "oe:team:hle"


def test_resolve_team_normalised_hit():
    assert _resolve("Gen.G Esports", "GEN") == "oe:team:geng"


def test_resolve_team_miss(caplog):
    assert _resolve("Karmine Corp", "KC") is None
    assert "unmapped team KC/Karmine Corp" in caplog.text


def test_resolve_team_ambiguous(caplog):
    assert _resolve("Dplus KIA", "DK") is None
    assert "ambiguous" in caplog.text


def test_resolve_team_stale_team_is_ignored():
    assert _resolve("Old Squad", "OLD") is None
