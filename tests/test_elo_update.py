"""Incremental ELO (update_elo) equals a full rebuild, and is idempotent."""

import numpy as np
import pandas as pd

import pipeline.elo as elo

PLAYERS = [(f"p{i:02d}", 1 if i < 8 else 2) for i in range(16)]
ELO_COLS = ["general_elo"] + [f"{s}_elo" for s in elo.SKILLS]


def _frame(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for pid, tid in PLAYERS:
        mp  = float(rng.integers(12, 38))
        pts = float(rng.integers(0, 30))
        fga = float(rng.integers(3, 22))
        fta = float(rng.integers(0, 10))
        ast = float(rng.integers(0, 11))
        tov = float(rng.integers(0, 6))
        tsa = fga + 0.44 * fta
        rows.append({
            "personId": pid, "teamId": tid, "minutes_float": mp,
            "points": pts, "assists": ast, "turnovers": tov,
            "steals": float(rng.integers(0, 4)), "blocks": float(rng.integers(0, 4)),
            "reboundsTotal": float(rng.integers(0, 14)),
            "threePointersMade": float(rng.integers(0, 6)),
            "threePointersAttempted": float(rng.integers(0, 10)),
            "threePointersPercentage": float(rng.random()),
            "trueShootingPercentage": pts / (2 * tsa) if tsa > 0 else 0.0,
            "assistToTurnover": ast / tov if tov > 0 else ast,
            "reboundPercentage": float(rng.random() * 20),
            "PIE": float(rng.random() * 0.2),
        })
    return pd.DataFrame(rows)


def _point_paths(monkeypatch, tmp_path):
    proc = tmp_path / "proc"; proc.mkdir()
    monkeypatch.setattr(elo, "PROCESSED_DIR", proc)
    monkeypatch.setattr(elo, "HISTORY_PATH", proc / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", proc / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", proc / "player_elo_recent.parquet")


def _scenario(n=6):
    dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(n)]
    gids  = [f"004200{i:04d}" for i in range(n)]
    frames = {gids[i]: _frame(100 + i) for i in range(n)}
    index = pd.DataFrame({"GAME_ID": gids, "GAME_DATE": dates})
    return index, gids, dates, (lambda g: frames[g].copy())


def _read_current():
    return (pd.read_parquet(elo.CURRENT_PATH)
              .sort_values("player_id").reset_index(drop=True))


def test_update_elo_matches_full_rebuild(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    index, gids, dates, loader = _scenario(n=6)
    k = 2

    elo.build_elo(force=True, loader=loader, game_index=index)
    full = _read_current()

    elo.build_elo(force=True, loader=loader, game_index=index.iloc[:-k])
    later = [(gids[i], dates[i]) for i in range(len(gids) - k, len(gids))]
    elo.update_elo(later, loader=loader)
    incremental = _read_current()

    pd.testing.assert_frame_equal(
        full[["player_id"] + ELO_COLS], incremental[["player_id"] + ELO_COLS],
        atol=1e-9, check_like=True,
    )


def test_update_elo_is_idempotent(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    index, gids, dates, loader = _scenario(n=5)

    elo.build_elo(force=True, loader=loader, game_index=index.iloc[:-2])
    later = [(gids[i], dates[i]) for i in range(len(gids) - 2, len(gids))]
    elo.update_elo(later, loader=loader)
    once = _read_current()

    elo.update_elo(later, loader=loader)   # replay the same games
    twice = _read_current()

    pd.testing.assert_frame_equal(once[["player_id"] + ELO_COLS], twice[["player_id"] + ELO_COLS])


def test_name_column_from_trad_fields():
    df = pd.DataFrame({"firstName": ["Kent", None], "familyName": ["Bazemore", "Doe"]})
    names = elo._name_column(df)
    assert names.iloc[0] == "Kent Bazemore"
    assert names.iloc[1] == "Doe"  # missing first name -> just the last, stripped


def test_names_flow_to_state_files(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    frame = _frame(1)
    frame["player_name"] = "Name " + frame["personId"]
    index = pd.DataFrame({"GAME_ID": ["0022600001"], "GAME_DATE": [pd.Timestamp("2026-10-25")]})
    elo.build_elo(force=True, loader=lambda g: frame.copy(), game_index=index)

    cur = pd.read_parquet(elo.CURRENT_PATH)
    rec = pd.read_parquet(elo.RECENT_PATH)
    assert list(cur.columns)[:2] == ["player_id", "player_name"]
    assert list(rec.columns)[:4] == ["game_id", "game_date", "player_id", "player_name"]
    assert cur.set_index("player_id").loc["p00", "player_name"] == "Name p00"
    assert (rec["player_name"] != "").all()


def test_load_state_backward_compatible_without_name(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    # Build state, then strip player_name to simulate an older (name-less) state file.
    idx = pd.DataFrame({"GAME_ID": ["0022600001"], "GAME_DATE": [pd.Timestamp("2026-10-20")]})
    elo.build_elo(force=True, loader=lambda g: _frame(1), game_index=idx)
    pd.read_parquet(elo.CURRENT_PATH).drop(columns=["player_name"]).to_parquet(elo.CURRENT_PATH, index=False)
    pd.read_parquet(elo.RECENT_PATH).drop(columns=["player_name"]).to_parquet(elo.RECENT_PATH, index=False)

    # update_elo must load the name-less state fine and fill names for the new game.
    frame2 = _frame(2)
    frame2["player_name"] = "N " + frame2["personId"]
    elo.update_elo([("0022600002", "2026-10-27")], loader=lambda g: frame2.copy())

    cur = pd.read_parquet(elo.CURRENT_PATH).set_index("player_id")
    assert "player_name" in cur.columns
    assert cur.loc["p00", "player_name"] == "N p00"


def _mini(rows):
    return pd.DataFrame([{"personId": pid, "teamId": tid, "minutes_float": 20.0} for pid, tid in rows])


def test_update_team_assignments_never_moves_backwards(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    # Seed state from one regular-season game on 2026-10-25 (p00..p07 -> team 1).
    index = pd.DataFrame({"GAME_ID": ["0022600001"], "GAME_DATE": [pd.Timestamp("2026-10-25")]})
    elo.build_elo(force=True, loader=lambda g: _frame(1), game_index=index)
    assert _read_current().set_index("player_id").loc["p00", "team_id"] == 1

    # An OLDER preseason game must not move p00 to a new team or rewind its date...
    elo.update_team_assignments([("0012600001", "2026-10-05")],
                                loader=lambda g: _mini([("p00", 2), ("newbie", 2)]))
    cur = _read_current().set_index("player_id")
    assert cur.loc["p00", "team_id"] == 1
    assert pd.Timestamp(cur.loc["p00", "last_game_date"]) == pd.Timestamp("2026-10-25")
    # ...but a brand-new player from that game is still added.
    assert cur.loc["newbie", "team_id"] == 2
    assert pd.Timestamp(cur.loc["newbie", "last_game_date"]) == pd.Timestamp("2026-10-05")

    # A NEWER game does move the player.
    elo.update_team_assignments([("0012600002", "2026-10-26")],
                                loader=lambda g: _mini([("p01", 2)]))
    cur = _read_current().set_index("player_id")
    assert cur.loc["p01", "team_id"] == 2
    assert pd.Timestamp(cur.loc["p01", "last_game_date"]) == pd.Timestamp("2026-10-26")
