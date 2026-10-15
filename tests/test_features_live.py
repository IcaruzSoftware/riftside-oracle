"""Live-mode lineup selection from the ELO state."""

import pandas as pd

from pipeline.features import _live_team_lineup


def _row(pid, team_id, date):
    return {"player_id": pid, "team_id": team_id, "last_game_date": pd.Timestamp(date)}


def test_prefers_current_season_when_enough():
    rows = [_row(f"c{i}", 10, "2026-11-01") for i in range(8)]      # 8 current (2026-27)
    rows.append(_row("prev", 10, "2026-01-15"))                     # previous season
    lineup = _live_team_lineup(pd.DataFrame(rows), "10", "2026-27")
    assert len(lineup) == 8
    assert "prev" not in set(lineup["player_id"])


def test_fallback_adds_previous_season_but_excludes_retired():
    rows = [_row(f"c{i}", 10, "2026-11-01") for i in range(7)]      # 7 current (< 8)
    rows += [_row(f"p{i}", 10, "2026-01-15") for i in range(2)]     # 2 previous (2025-26)
    rows.append(_row("retired", 10, "2024-01-15"))                  # 2023-24 -> excluded
    rows.append(_row("otherteam", 99, "2026-11-01"))               # different team
    lineup = _live_team_lineup(pd.DataFrame(rows), "10", "2026-27")
    ids = set(lineup["player_id"])
    assert "retired" not in ids
    assert "otherteam" not in ids
    assert len(lineup) == 9   # 7 current + 2 previous
