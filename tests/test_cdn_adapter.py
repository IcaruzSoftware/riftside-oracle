"""CDN box-score adapter: derived TS% / PIE from basic stats on a real fixture."""

from pipeline.elo import load_cdn_game_players, _parse_iso_minutes


def test_iso_minutes():
    assert _parse_iso_minutes("PT39M12.00S") == 39.2
    assert _parse_iso_minutes("PT00M00.00S") == 0.0
    assert _parse_iso_minutes(None) == 0.0


def test_adapter_shape(finals_game):
    df = load_cdn_game_players(finals_game)
    # Same derived columns compute_skill_scores expects...
    for col in ["personId", "teamId", "minutes_float", "points",
                "trueShootingPercentage", "assistToTurnover", "reboundPercentage", "PIE"]:
        assert col in df.columns
    # ...and defensiveRating / tracking stay absent (handled by fallbacks).
    assert "defensiveRating" not in df.columns
    assert "speed" not in df.columns


def test_true_shooting(finals_game):
    df = load_cdn_game_players(finals_game)
    # Wembanyama: 19 PTS, 19 FGA, 5 FTA -> TS = 19 / (2*(19 + 0.44*5)) = 0.448113
    wemby = df[df.personId == "1641705"].iloc[0]
    assert wemby.trueShootingPercentage == pytest_approx(0.448113)


def test_player_name_from_cdn(finals_game):
    df = load_cdn_game_players(finals_game)
    assert "player_name" in df.columns
    wemby = df[df.personId == "1641705"].iloc[0]
    assert wemby.player_name == "Victor Wembanyama"


def test_pie_sums_to_one(finals_game):
    df = load_cdn_game_players(finals_game)
    assert df.PIE.sum() == pytest_approx(1.0)
    # Brunson (45 pts) is the game's top contributor.
    brunson = df[df.personId == "1628973"].iloc[0]
    assert brunson.PIE == df.PIE.max()
    assert brunson.PIE > 0.2


def pytest_approx(x, tol=1e-5):
    import pytest
    return pytest.approx(x, abs=tol)
