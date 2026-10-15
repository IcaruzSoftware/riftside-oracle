"""Daily state: game-log append is idempotent and carries the columns features need."""

import pandas as pd

import pipeline.daily_state as ds

REQUIRED_COLS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
                 "WL", "PTS", "PLUS_MINUS", "AST", "REB", "TOV"]


def test_game_log_append_idempotent(monkeypatch, tmp_path, finals_game):
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(ds, "RAW_DIR", raw)

    rows = ds._game_log_rows(finals_game, "0042500405", "2026-06-13")
    path = raw / "game_log_playoffs_2025-26.csv"

    ds._append_game_logs({("playoffs", "2025-26"): rows}, dry_run=False)
    first = pd.read_csv(path)
    assert len(first) == 2                     # one row per team
    assert set(REQUIRED_COLS).issubset(first.columns)
    assert set(first["MATCHUP"]) == {"SAS vs. NYK", "NYK @ SAS"}

    # Re-applying the same game must not duplicate rows.
    ds._append_game_logs({("playoffs", "2025-26"): rows}, dry_run=False)
    second = pd.read_csv(path)
    assert len(second) == 2


def test_player_contribs_and_stats_carry_names(monkeypatch, tmp_path, finals_game):
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(ds, "RAW_DIR", raw)

    contribs = ds._player_contribs(finals_game)
    assert all(len(c) == 4 for c in contribs)
    names = {pid: name for pid, _, _, name in contribs}
    assert names["1641705"] == "Victor Wembanyama"

    ds._update_player_stats("2025-26", contribs, dry_run=False)
    df = pd.read_csv(raw / "player_season_stats_2025-26.csv")
    assert "PLAYER_NAME" in df.columns
    assert (df["PLAYER_NAME"] == "Victor Wembanyama").any()


def _cdn_game(gid, home_tri, home_id, away_tri, away_id, hs, aw):
    def team(tri, tid, score):
        return {
            "teamId": tid, "teamTricode": tri, "score": score,
            "statistics": {"assists": 20, "reboundsTotal": 40, "turnovers": 12},
            "players": [{"personId": f"{tid}_{i}",
                         "statistics": {"minutes": "PT20M00.00S", "points": 10}} for i in range(8)],
        }
    return {"gameId": gid, "gameStatus": 3,
            "homeTeam": team(home_tri, home_id, hs), "awayTeam": team(away_tri, away_id, aw)}


def test_run_revisits_days_before_last_logged(monkeypatch, tmp_path):
    import pipeline.elo as elo
    from pipeline import nba_cdn

    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(ds, "RAW_DIR", raw)

    # A game already logged on 2026-10-21 (day D+1).
    logged = _cdn_game("0022600002", "BOS", 1610612738, "MIA", 1610612748, 110, 100)
    pd.DataFrame(ds._game_log_rows(logged, "0022600002", "2026-10-21")).to_csv(
        raw / "game_log_regular_2026-27.csv", index=False)

    # Schedule: an unlogged final game on 2026-10-20 (day D) + the logged one on D+1.
    schedule = {"leagueSchedule": {"gameDates": [
        {"gameDate": "10/20/2026 00:00:00", "games": [
            {"gameId": "0022600001", "gameStatus": 3,
             "homeTeam": {"teamId": 1610612744, "teamTricode": "GSW", "score": 120},
             "awayTeam": {"teamId": 1610612747, "teamTricode": "LAL", "score": 115}}]},
        {"gameDate": "10/21/2026 00:00:00", "games": [
            {"gameId": "0022600002", "gameStatus": 3,
             "homeTeam": {"teamId": 1610612738, "teamTricode": "BOS", "score": 110},
             "awayTeam": {"teamId": 1610612748, "teamTricode": "MIA", "score": 100}}]},
    ]}}

    missed = _cdn_game("0022600001", "GSW", 1610612744, "LAL", 1610612747, 120, 115)
    monkeypatch.setattr(nba_cdn, "fetch_boxscore", lambda gid: missed if gid == "0022600001" else None)

    applied = []
    monkeypatch.setattr(elo, "update_elo", lambda games, loader=None: applied.extend(g for g, _ in games))
    monkeypatch.setattr(elo, "update_team_assignments", lambda games, loader=None: None)

    new_ids = ds.run(date="2026-10-21", schedule=schedule, dry_run=False)

    # The 2026-10-20 game (a day before the last logged game) was revisited & applied.
    assert "0022600001" in new_ids
    assert "0022600001" in applied
    log = pd.read_csv(raw / "game_log_regular_2026-27.csv")
    log["GAME_ID"] = log["GAME_ID"].astype(str).str.zfill(10)
    assert "0022600001" in set(log["GAME_ID"])
