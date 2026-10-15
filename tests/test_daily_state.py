"""daily_state on the OE fixture: new-game detection, completeness, same-day cut, idempotency, dry-run."""

from pathlib import Path

import pandas as pd
import pytest

import pipeline.daily_state as daily_state
import pipeline.elo as elo
from pipeline import oe

FIXTURES = Path(__file__).parent / "fixtures" / "oe"
LCK = "LOLTMNT03_339605"                 # 2026-01-22, the initial state
LPL_PARTIAL = "13080-13080_game_1"       # 2026-01-22
LPL = "13065-13065_game_1"               # 2026-02-03
MSI = "LOLTMNT01_423456"                 # 2026-07-12
BROKEN = "BROKEN"                        # 2026-02-04, one player row missing


@pytest.fixture
def oe_rows():
    base = oe.load_year(2026, FIXTURES)
    broken = base[base["gameid"] == LPL].assign(gameid=BROKEN, date=pd.Timestamp("2026-02-04"))
    return pd.concat([base, broken.iloc[1:]], ignore_index=True)


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch, oe_rows):
    """Committed state = the LCK game only (ELO + team_games); OE 'download' = oe_rows."""
    monkeypatch.setattr(daily_state, "RAW_DIR", tmp_path)
    monkeypatch.setattr(elo, "HISTORY_PATH", tmp_path / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", tmp_path / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", tmp_path / "player_elo_recent.parquet")
    monkeypatch.setattr(elo, "ROLE_NORMS_PATH", tmp_path / "role_norms.json")
    norms = elo.compute_role_norms(oe.player_rows(oe_rows), years=None)
    first = oe_rows[oe_rows["gameid"] == LCK]
    elo.build_elo(force=True, games=first, norms=norms)
    oe.team_rows(first).to_csv(tmp_path / "team_games_2026.csv", index=False)

    monkeypatch.setattr(oe, "list_files", lambda: {2026: "file-id"})
    monkeypatch.setattr(oe, "download_year", lambda year, file_id=None: None)
    monkeypatch.setattr(oe, "load_year", lambda year: oe_rows)
    return tmp_path


def _snapshot(tmp_path):
    return {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.suffix in (".csv", ".parquet")}


def test_ingests_complete_games_before_target_day(state, caplog):
    got = daily_state.run(date="2026-07-12")
    assert got == [LPL_PARTIAL, LPL]                      # MSI is same-day, BROKEN incomplete
    assert f"{BROKEN}: incomplete game" in caplog.text
    tg = daily_state.load_team_games()
    assert list(dict.fromkeys(tg["gameid"])) == [LCK, LPL_PARTIAL, LPL]
    expected = oe.team_rows(oe.load_year(2026))
    expected = expected[expected["gameid"].isin([LCK, LPL_PARTIAL, LPL])]
    assert tg["patch"].tolist() == expected["patch"].tolist()       # str, "16.01" stays "16.01"
    assert tg["starters"].tolist() == expected["starters"].tolist()
    recent = pd.read_parquet(elo.RECENT_PATH)
    assert set(recent["game_id"]) == {LCK, LPL_PARTIAL, LPL}


def test_same_day_game_waits_for_next_run(state):
    daily_state.run(date="2026-07-12")
    assert daily_state.run(date="2026-07-13") == [MSI]
    assert MSI in set(pd.read_parquet(elo.RECENT_PATH)["game_id"])


def test_second_run_is_a_noop(state):
    daily_state.run(date="2026-07-12")
    before = _snapshot(state)
    assert daily_state.run(date="2026-07-12") == []
    assert _snapshot(state) == before


def test_matches_a_full_rebuild(state, oe_rows):
    daily_state.run(date="2026-07-13")
    incremental = pd.read_parquet(elo.CURRENT_PATH)
    elo.build_elo(force=True, games=oe_rows[oe_rows["gameid"] != BROKEN],
                  norms=elo._load_norms())
    pd.testing.assert_frame_equal(incremental, pd.read_parquet(elo.CURRENT_PATH))


def test_dry_run_writes_nothing(state):
    before = _snapshot(state)
    assert daily_state.run(date="2026-07-12", dry_run=True) == [LPL_PARTIAL, LPL]
    assert _snapshot(state) == before


def test_blocked_download_fails_loudly_and_writes_nothing(state, monkeypatch):
    def blocked(year, file_id=None):
        raise RuntimeError("OE 2026: download file-id is not the OE CSV: Google Drive - Quota exceeded")
    monkeypatch.setattr(oe, "download_year", blocked)
    before = _snapshot(state)
    with pytest.raises(RuntimeError, match="Quota exceeded.*State NOT updated"):
        daily_state.run(date="2026-07-12")
    assert _snapshot(state) == before


def test_complete_gameids(oe_rows):
    no_result = oe_rows[oe_rows["gameid"] == LCK].assign(gameid="NORESULT")
    no_result.loc[no_result.index[0], "result"] = pd.NA
    two_tops = oe_rows[oe_rows["gameid"] == LCK].assign(gameid="TWOTOPS")
    two_tops.loc[two_tops["position"] == "jng", "position"] = "top"
    got = daily_state.complete_gameids(pd.concat([oe_rows, no_result, two_tops]))
    assert got == {LCK, LPL_PARTIAL, LPL, MSI}


def test_midnight_only_batch_keeps_full_timestamps(state, oe_rows, monkeypatch):
    midnight = oe_rows[oe_rows["gameid"] == LPL].assign(gameid="MIDNIGHT", date=pd.Timestamp("2026-02-10"))
    monkeypatch.setattr(oe, "load_year", lambda year: pd.concat([oe_rows[oe_rows["gameid"] == LCK], midnight]))
    assert daily_state.run(date="2026-02-11") == ["MIDNIGHT"]
    assert (state / "team_games_2026.csv").read_text().splitlines()[-1].split(",")[1] == "2026-02-10 00:00:00"
    assert pd.api.types.is_datetime64_any_dtype(daily_state.load_team_games()["date"])
