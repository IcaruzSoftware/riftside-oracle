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
