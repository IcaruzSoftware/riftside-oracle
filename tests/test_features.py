"""Features on the OE fixture: point-in-time, train/live parity, group isolation, series math."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.elo as elo
import pipeline.features as features
from pipeline import oe

FIXTURES = Path(__file__).parent / "fixtures" / "oe"
LCK = "LOLTMNT03_339605"
DAY = pd.Timestamp("2026-08-10")         # the LCK game's 2nd copy is played on this day
GAME = f"{LCK}-c2"                       # game 2 of the series ...
GAME1 = "SERIES-G1"                      # ... game 1 is earlier the same day
LATER = "LOLTMNT01_423456-c2"            # MSI copy, 2027-01-28: HLE again, after DAY


@pytest.fixture(scope="module")
def base():
    return oe.load_year(2026, FIXTURES)


@pytest.fixture(scope="module")
def norms(base):
    return elo.compute_role_norms(oe.player_rows(base), years=None)


@pytest.fixture(autouse=True)
def _point_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(elo, "HISTORY_PATH", tmp_path / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", tmp_path / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", tmp_path / "player_elo_recent.parquet")
    monkeypatch.setattr(elo, "ROLE_NORMS_PATH", tmp_path / "role_norms.json")
    monkeypatch.setattr(features, "_cache", {})


def _games(base):
    """The 4 fixture games + 2 perturbed copies (dates +100/+200 days) + GAME1 on DAY."""
    rng = np.random.default_rng(0)
    frames = [base]
    for c in (1, 2):
        f = base.copy()
        f["gameid"] = f["gameid"] + f"-c{c}"
        f["date"] = f["date"] + pd.Timedelta(days=100 * c)
        f[oe._ELO_COLS] = f[oe._ELO_COLS] * rng.uniform(0.7, 1.3, (len(f), len(oe._ELO_COLS)))
        frames.append(f)
    g1 = base[base["gameid"] == LCK].copy()
    g1["gameid"], g1["date"] = GAME1, DAY + pd.Timedelta(hours=5)
    return pd.concat(frames + [g1], ignore_index=True)


def _absurd_from(games, day):
    """Every row dated on/after `day`: results flipped, all stats x1000."""
    g = games.copy()
    late = g["date"] >= day
    g.loc[late, "result"] = 1 - g.loc[late, "result"]
    cols = oe._ELO_COLS + oe._TEAM_COLS + ["gamelength"]
    g.loc[late, cols] = g.loc[late, cols] * 1000
    return g


def _feats(games, norms, game_id, live=False, elo_games=None, tg=None, **kw):
    """Build ELO from elo_games (default: games), then the features of game_id."""
    elo.build_elo(force=True, games=games if elo_games is None else elo_games, norms=norms)
    features._cache.clear()
    full = oe.team_rows(games)
    row = full[full["gameid"] == game_id].set_index("side")
    return features.build_feature_matrix(
        row.at["Blue", "teamid"], row.at["Red", "teamid"], row.at["Blue", "date"],
        {"league": row.at["Blue", "league"], "playoffs": row.at["Blue", "playoffs"]},
        full if tg is None else tg, live, game_id=None if live else game_id, **kw)


def _assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == pytest.approx(b[k], abs=1e-9), k


def test_point_in_time(base, norms):
    games = _games(base)
    absurd = _absurd_from(games, DAY)
    true_f = _feats(games, norms, GAME)
    _assert_same(true_f, _feats(absurd, norms, GAME))
    assert true_f["h2h_games"] == 2 and not np.isnan(true_f["blue_win_pct_last5"])
    # sanity: the absurd rows do reach a game after DAY (HLE's rolling stats include them)
    assert _feats(games, norms, LATER)["red_kills_pm_last5"] != _feats(absurd, norms, LATER)["red_kills_pm_last5"]


def test_training_equals_live(base, norms):
    games = _games(base)
    train = _feats(games, norms, GAME)
    before = games[games["date"] < DAY]
    live = _feats(games, norms, GAME, live=True, elo_games=before, tg=oe.team_rows(before))
    _assert_same(train, live)        # live starters (latest lineups) == the actual starters here
    assert not np.isnan(train["blue_league_mean_outcome_elo"]) and train["blue_roster_churn"] == 5


def test_training_equals_live_across_new_year(base, norms):
    # LCK-JAN is the lineups' first game of 2027, where elo applies the off-season pull.
    # Training's as-of value (LCK-JAN's pre-game snapshot) must equal live's (state built
    # up to the day before), i.e. the pull must not leak into the snapshot.
    games = pd.concat([base[base["gameid"] == LCK]] + [
        base[base["gameid"] == LCK].assign(gameid=gid, date=pd.Timestamp(d))
        for gid, d in (("LCK-DEC", "2026-12-30"), ("LCK-JAN", "2027-01-05"))], ignore_index=True)
    train = _feats(games, norms, "LCK-JAN")
    before = games[games["date"] < pd.Timestamp("2027-01-05")]
    live = _feats(games, norms, "LCK-JAN", live=True, elo_games=before, tg=oe.team_rows(before))
    _assert_same(train, live)
    assert train["diff_outcome_elo"] != 0


def test_player_form_window(base, norms, monkeypatch):
    games = _games(base)
    # Blue (HANJIN) starters have 2 games before DAY: the window reaches their first (init) game.
    f = _feats(games, norms, GAME)
    assert f["blue_form_outcome_elo"] == pytest.approx(f["blue_outcome_elo"] - elo.LEAGUE_PRIOR["LCK"])
    assert f["blue_form_general_elo"] == pytest.approx(f["blue_general_elo"] - 5 * elo.INITIAL_ELO)
    monkeypatch.setattr(features, "FORM_N", 1)
    f = _feats(games, norms, GAME)
    hist = pd.read_parquet(elo.HISTORY_PATH)
    prev = hist[(hist["game_id"] == f"{LCK}-c1") & (hist["side"] == "Blue")]
    assert f["blue_form_outcome_elo"] == pytest.approx(f["blue_outcome_elo"] - prev["pre_outcome_elo"].mean())


def test_live_missing_player_gets_init_value(base, norms):
    games = _games(base)
    before = games[games["date"] < DAY]
    new = ["oe:player:new-1", "", None, "oe:player:new-2", "oe:player:new-3"]
    f = _feats(games, norms, GAME, live=True, elo_games=before, tg=oe.team_rows(before),
               red_starters=new)
    # Only 5 players were last seen in LCK (HLE's last game was MSI) -> LCK prior.
    assert f["red_outcome_elo"] == elo.LEAGUE_PRIOR["LCK"]
    assert f["red_general_elo"] == elo.INITIAL_ELO * len(elo.SKILLS)
    assert f["red_rookie_count"] == 3 and np.isnan(f["red_form_outcome_elo"])


def test_live_and_training_same_keys(base, norms):
    games = _games(base)
    train = _feats(games, norms, GAME)
    live = _feats(games, norms, GAME, live=True, elo_games=games[games["date"] < DAY])
    unknown = features.build_feature_matrix("oe:team:nobody", "oe:team:nobody2", DAY,
                                            {"league": "LCK", "playoffs": 0},
                                            oe.team_rows(games), True)
    assert train.keys() == live.keys() == unknown.keys()
    assert len(train) > 100


def test_failing_group_leaves_others(base, norms, monkeypatch):
    games = _games(base)
    full = _feats(games, norms, GAME)

    def boom(g):
        raise RuntimeError("boom")
    monkeypatch.setattr(features, "get_team_rolling_stats", boom)
    partial = _feats(games, norms, GAME)
    rolling = {k for k in full if "_last5" in k or "_last10" in k}
    assert rolling and set(partial) == set(full) - rolling
    _assert_same(partial, {k: full[k] for k in partial})


@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_series_win_prob_closed_forms(p):
    assert features.series_win_prob(p, 1) == pytest.approx(p)
    assert features.series_win_prob(p, 3) == pytest.approx(p**2 * (3 - 2 * p))
    assert features.series_win_prob(p, 5) == pytest.approx(p**3 * (10 - 15 * p + 6 * p**2))
