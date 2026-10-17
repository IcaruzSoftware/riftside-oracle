"""build_dataset: fails loudly on a missing or stale ELO history, builds normally on a fresh one."""

from pathlib import Path

import pandas as pd
import pytest

import pipeline.build_dataset as build_dataset
import pipeline.elo as elo
import pipeline.features as features
from pipeline import oe

FIXTURES = Path(__file__).parent / "fixtures" / "oe"


@pytest.fixture(scope="module")
def base():
    return oe.load_year(2026, FIXTURES)


@pytest.fixture(scope="module")
def norms(base):
    return elo.compute_role_norms(oe.player_rows(base), years=None)


@pytest.fixture
def games(base):
    """The fixture games + a copy dated 100 days later."""
    later = base.assign(gameid=base["gameid"] + "-c1", date=base["date"] + pd.Timedelta(days=100))
    return pd.concat([base, later], ignore_index=True)


@pytest.fixture(autouse=True)
def _point_paths(monkeypatch, tmp_path, games):
    monkeypatch.setattr(elo, "HISTORY_PATH", tmp_path / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", tmp_path / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", tmp_path / "player_elo_recent.parquet")
    monkeypatch.setattr(elo, "ROLE_NORMS_PATH", tmp_path / "role_norms.json")
    monkeypatch.setattr(features, "_cache", {})
    monkeypatch.setattr(build_dataset, "OUT_PATH", tmp_path / "feature_matrix.parquet")
    monkeypatch.setattr(build_dataset, "START_DATE", "2000-01-01")
    monkeypatch.setattr(build_dataset, "load_team_games", lambda: oe.team_rows(games))


def test_stale_history_raises(base, norms):
    elo.build_elo(force=True, games=base, norms=norms)       # history ends before the copies
    with pytest.raises(RuntimeError, match=r"stale.*elo\.py --force"):
        build_dataset.build_dataset(force=True)
    assert not build_dataset.OUT_PATH.exists()


def test_missing_history_raises():
    with pytest.raises(FileNotFoundError, match=r"elo\.py --force"):
        build_dataset.build_dataset(force=True)
    assert not build_dataset.OUT_PATH.exists()


def test_fresh_history_builds(games, norms):
    elo.build_elo(force=True, games=games, norms=norms)
    df = build_dataset.build_dataset(force=True)
    assert len(df) == games["gameid"].nunique()
    assert df["blue_team_id"].notna().all() and df["red_team_id"].notna().all()
    assert build_dataset.OUT_PATH.exists()
