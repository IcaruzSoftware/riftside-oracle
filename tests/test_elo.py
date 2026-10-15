"""ELO on the OE fixture: skills, role norms, outcome Elo, init, incremental parity."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.elo as elo
from pipeline import oe

FIXTURES = Path(__file__).parent / "fixtures" / "oe"
LCK = "LOLTMNT03_339605"
LPL_PARTIAL = "13080-13080_game_1"
MSI = "LOLTMNT01_423456"
SKILL_COLS = [f"{s}_elo" for s in elo.SKILLS]
ELO_COLS = ["outcome_elo", "general_elo"] + SKILL_COLS


@pytest.fixture(scope="module")
def base():
    return oe.load_year(2026, FIXTURES)


@pytest.fixture(scope="module")
def norms(base):
    return elo.compute_role_norms(oe.player_rows(base), years=None)


@pytest.fixture(autouse=True)
def _point_paths(monkeypatch, tmp_path):
    proc = tmp_path / "proc"; proc.mkdir()
    monkeypatch.setattr(elo, "PROCESSED_DIR", proc)
    monkeypatch.setattr(elo, "HISTORY_PATH", proc / "player_elo.parquet")
    monkeypatch.setattr(elo, "CURRENT_PATH", proc / "player_elo_current.parquet")
    monkeypatch.setattr(elo, "RECENT_PATH", proc / "player_elo_recent.parquet")
    monkeypatch.setattr(elo, "ROLE_NORMS_PATH", proc / "role_norms.json")


def _games(base, copies=2, seed=0):
    """The 4 fixture games plus perturbed copies (new gameids, dates +100 days each)."""
    rng = np.random.default_rng(seed)
    frames = [base]
    for c in range(1, copies + 1):
        f = base.copy()
        f["gameid"] = f["gameid"] + f"-c{c}"
        f["date"] = f["date"] + pd.Timedelta(days=100 * c)
        f[oe._ELO_COLS] = f[oe._ELO_COLS] * rng.uniform(0.7, 1.3, (len(f), len(oe._ELO_COLS)))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _copy_game(base, src, gameid, date, new_players=False):
    """Game src re-dated as gameid; new_players: played by 10 brand-new playerids."""
    extra = base[base["gameid"] == src].copy()
    extra["gameid"] = gameid
    extra["date"] = pd.Timestamp(date)
    if new_players:
        is_p = extra["position"] != "team"
        extra.loc[is_p, "playerid"] = [f"oe:player:{gameid}-{i}" for i in range(is_p.sum())]
    return extra


def _new_players_game(base):
    """The MSI game one day later, played by 10 brand-new players."""
    return _copy_game(base, MSI, "NEW", "2026-07-13 08:32:42", new_players=True)


def _current():
    return pd.read_parquet(elo.CURRENT_PATH).set_index("player_id")


def _post(hist):
    """Post-game ELO per history row: the player's next pre-game snapshot, else current."""
    cur = _current()
    h = hist.sort_values(["game_date", "game_id"]).copy()
    for col in ["outcome_elo"] + SKILL_COLS:
        h[f"post_{col}"] = (h.groupby("player_id")[f"pre_{col}"].shift(-1)
                            .fillna(h["player_id"].map(cur[col])))
    return h


def test_rank_to_delta_sums_to_zero():
    for n in range(1, 13):
        assert sum(elo.rank_to_delta(r, n) for r in range(1, n + 1)) == 0


def test_compute_deltas_ties_and_nan():
    d = elo.compute_deltas(np.array([2.0, 1.0, 1.0, 0.0, np.nan]))
    assert d.sum() == pytest.approx(0) and d[1] == d[2] and d[4] == 0
    assert d[0] > d[1] > d[3]


def test_role_normalisation(base, norms):
    g = oe.player_rows(base[base["gameid"] == LCK]).copy()
    bot = g[g["position"] == "bot"].index[0]
    sup = g[g["position"] == "sup"].index[0]
    g.loc[sup, ["dpm", "damageshare"]] = g.loc[bot, ["dpm", "damageshare"]].to_numpy()
    s = elo.compute_skill_scores(g, norms)
    assert s.loc[sup, "score_damage"] > s.loc[bot, "score_damage"]   # same raw damage, support ranks higher

    r, n = g.loc[bot], norms["bot"]
    z = lambda stat, x: (x - n[stat]["mean"]) / n[stat]["std"]
    m = r["gamelength"] / 60
    combat = np.mean([z("kills_pm", r["kills"] / m), z("assists_pm", r["assists"] / m),
                      -z("deaths_pm", r["deaths"] / m)])
    assert s.loc[bot, "score_combat"] == pytest.approx(combat)


def test_partial_game_skips_laning_only(base, norms):
    hist = elo.build_elo(force=True, games=base, norms=norms)
    h = _post(hist)
    g = h[h["game_id"] == LPL_PARTIAL]
    assert len(g) == 10
    assert (g["post_laning_elo"] == g["pre_laning_elo"]).all()
    for s in ["combat", "damage", "economy", "vision"]:
        d = g[f"post_{s}_elo"] - g[f"pre_{s}_elo"]
        assert (d != 0).all() and d.sum() == pytest.approx(0)
    full = h[h["game_id"] == LCK]
    assert (full["post_laning_elo"] != full["pre_laning_elo"]).all()


def test_outcome_elo_zero_sum_between_sides(base, norms):
    # Precondition: zero-sum only holds when both sides share k. Every fixture player
    # is within their first NEW_GAMES games, so all get NEW_K_MULT (mixed case below).
    h = _post(elo.build_elo(force=True, games=base, norms=norms))
    assert h.groupby("player_id").size().max() <= elo.NEW_GAMES
    h["d"] = h["post_outcome_elo"] - h["pre_outcome_elo"]
    for _, g in h.groupby("game_id"):
        blue, red = g[g["side"] == "Blue"], g[g["side"] == "Red"]
        assert blue["d"].mean() == pytest.approx(-red["d"].mean())
    res = oe.player_rows(base).set_index(["gameid", "playerid"])["result"]
    h["result"] = h.set_index(["game_id", "player_id"]).index.map(res).astype(float)
    won = h["result"] == 1
    assert (h.loc[won, "d"] > 0).all() and (h.loc[~won, "d"] < 0).all()


def test_outcome_elo_exact_delta(base, norms):
    h = _post(elo.build_elo(force=True, games=base, norms=norms))
    rows = oe.player_rows(base).set_index(["gameid", "playerid"])
    key = h.set_index(["game_id", "player_id"]).index
    h["result"] = key.map(rows["result"]).astype(float)
    h["gd15"], h["len"] = key.map(rows["golddiffat15"]), key.map(rows["gamelength"])
    h["n_before"] = h.groupby("player_id").cumcount()
    uneven = 0
    for _, g in h.groupby("game_id"):
        blue = g["side"] == "Blue"
        r_blue = g.loc[blue, "pre_outcome_elo"].mean()
        r_red = g.loc[~blue, "pre_outcome_elo"].mean()
        e_blue = 1 / (1 + 10 ** ((r_red - r_blue) / 400))
        e = np.where(blue, e_blue, 1 - e_blue)
        gd = g.loc[blue, "gd15"].sum() if g["gd15"].notna().all() else np.nan
        k = (elo.K * elo.mov_mult(gd, g["len"].iloc[0])
             * np.where(g["n_before"] < elo.NEW_GAMES, elo.NEW_K_MULT, 1.0))
        np.testing.assert_allclose(g["post_outcome_elo"] - g["pre_outcome_elo"],
                                   k * (g["result"] - e), atol=1e-9)
        uneven += abs(e_blue - 0.5) > 0.1
    assert uneven >= 1   # the MSI game (LCK side vs 1350 newcomers) is far from 50/50


def test_outcome_elo_new_players_move_further(base, norms, monkeypatch):
    # NEW_GAMES=1: at MSI the HLE side (1 earlier LCK game) has plain k, the BLG side
    # (first game) NEW_K_MULT * k, so the game is not zero-sum.
    monkeypatch.setattr(elo, "NEW_GAMES", 1)
    h = _post(elo.build_elo(force=True, games=base, norms=norms))
    g = h[h["game_id"] == MSI].assign(d=lambda x: x["post_outcome_elo"] - x["pre_outcome_elo"])
    seen = g["player_id"].isin(h.loc[h["game_id"] == LCK, "player_id"])
    assert seen.sum() == 5
    assert g.loc[~seen, "d"].mean() == pytest.approx(-elo.NEW_K_MULT * g.loc[seen, "d"].mean())


def test_mov_mult():
    assert elo.mov_mult(elo.MOV_GD_REF, 1000) == pytest.approx(1.0)   # gd15 wins over length
    assert elo.mov_mult(8000, 0) > elo.mov_mult(-3000, 0) > elo.mov_mult(0, 0) > 0
    assert elo.mov_mult(np.nan, elo.MOV_LEN_REF) == pytest.approx(1.0)
    assert elo.mov_mult(np.nan, 1200) > 1 > elo.mov_mult(np.nan, 2400)    # short win = dominant
    assert elo.mov_mult(np.nan, np.nan) == elo.mov_mult(np.nan, 0) == 1.0                  # no usable margin


def _new_year_games(base):
    """LCK game (Jan 2026), the same lineups on 2026-12-30 and on 2027-01-05."""
    return pd.concat([base[base["gameid"] == LCK],
                      _copy_game(base, LCK, "LCK-DEC", "2026-12-30"),
                      _copy_game(base, LCK, "LCK-JAN", "2027-01-05")], ignore_index=True)


def test_new_year_regression_toward_league_mean(base, norms):
    # 10 LCK players, LCK mean stays 1600 (zero-sum, equal k). At LCK-JAN each rating is
    # first pulled REGRESS toward 1600, then updated; the pre-game snapshot is unpulled.
    games = _new_year_games(base)
    elo.build_elo(force=True, games=games[games["gameid"] != "LCK-JAN"], norms=norms)
    end_2026 = _current()["outcome_elo"]
    mean = end_2026.mean()
    assert mean == pytest.approx(elo.LEAGUE_PRIOR["LCK"]) and end_2026.std() > 0

    h = elo.build_elo(force=True, games=games, norms=norms)
    jan = h[h["game_id"] == "LCK-JAN"].set_index("player_id")
    np.testing.assert_allclose(jan["pre_outcome_elo"], end_2026.reindex(jan.index))

    pulled = end_2026.reindex(jan.index) + elo.REGRESS * (mean - end_2026.reindex(jan.index))
    blue = (jan["side"] == "Blue").to_numpy()
    e_blue = 1 / (1 + 10 ** ((pulled[~blue].mean() - pulled[blue].mean()) / 400))
    rows = oe.player_rows(games[games["gameid"] == "LCK-JAN"]).set_index("playerid").reindex(jan.index)
    k = elo.K * elo.mov_mult(rows.loc[blue, "golddiffat15"].sum(), rows["gamelength"].iloc[0]) * elo.NEW_K_MULT
    expect = pulled + k * (rows["result"] - np.where(blue, e_blue, 1 - e_blue))
    np.testing.assert_allclose(_current()["outcome_elo"].reindex(jan.index), expect)


def test_new_year_update_elo_matches_full_rebuild(base, norms, monkeypatch):
    games = _new_year_games(base)
    elo.build_elo(force=True, games=games, norms=norms)
    full_cur, full_rec = _current(), pd.read_parquet(elo.RECENT_PATH)
    elo.build_elo(force=True, games=games[games["gameid"] != "LCK-JAN"], norms=norms)
    elo.update_elo(games[games["gameid"] == "LCK-JAN"])
    pd.testing.assert_frame_equal(full_cur, _current(), check_like=True, atol=1e-9)
    pd.testing.assert_frame_equal(full_rec, pd.read_parquet(elo.RECENT_PATH), check_like=True, atol=1e-9)
    # ... and the pull really happened: without it the state differs.
    monkeypatch.setattr(elo, "REGRESS", 0.0)
    elo.build_elo(force=True, games=games, norms=norms)
    assert (_current()["outcome_elo"] - full_cur["outcome_elo"]).abs().min() > 1e-6


def test_new_player_league_prior_and_league_mean(base, norms):
    # 5th game: an MSI game of 10 brand-new players. After the 4 fixture games, exactly
    # 10 players last played at MSI (the LCK side + 5 MSI newcomers), so they get that mean.
    extra = _new_players_game(base)

    elo.build_elo(force=True, games=base, norms=norms)
    cur = _current()
    msi_mean = cur.loc[cur["league"] == "MSI", "outcome_elo"]
    assert len(msi_mean) == 10 and msi_mean.mean() != pytest.approx(elo.DEFAULT_PRIOR)

    hist = elo.build_elo(force=True, games=pd.concat([base, extra]), norms=norms).set_index("game_id")
    assert (hist.loc[LCK, "pre_outcome_elo"] == elo.LEAGUE_PRIOR["LCK"]).all()
    msi_new = hist.loc[MSI][hist.loc[MSI, "league"] == "MSI"]
    assert (msi_new[msi_new["side"] == "Red"]["pre_outcome_elo"] == elo.DEFAULT_PRIOR).all()
    assert hist.loc["NEW", "pre_outcome_elo"].to_numpy() == pytest.approx(msi_mean.mean())


def test_league_mean_follows_players_who_leave(base, norms):
    # LCK game (10 LCK) -> MSI: the LCK Red side moves to MSI (LCK 5, MSI 10)
    # -> that side plays LCK again with the MSI-shifted ratings (LCK 10, mean != 1600; MSI 5).
    games = pd.concat([base[base["gameid"].isin([LCK, MSI])],
                       _copy_game(base, LCK, "LCK2", "2026-07-13")], ignore_index=True)
    elo.build_elo(force=True, games=games, norms=norms)
    cur = _current()
    lck = cur.loc[cur["league"] == "LCK", "outcome_elo"]
    assert len(lck) == 10 and lck.mean() != pytest.approx(elo.LEAGUE_PRIOR["LCK"])
    assert (cur["league"] == "MSI").sum() == 5

    games = pd.concat([games,
                       _copy_game(base, LCK, "NEWLCK", "2026-07-14", new_players=True),
                       _copy_game(base, MSI, "NEWMSI", "2026-07-15", new_players=True)])
    hist = elo.build_elo(force=True, games=games, norms=norms).set_index("game_id")
    assert hist.loc["NEWLCK", "pre_outcome_elo"].to_numpy() == pytest.approx(lck.mean(), abs=1e-9)
    assert (hist.loc["NEWMSI", "pre_outcome_elo"] == elo.DEFAULT_PRIOR).all()   # MSI dropped to 5


def test_nan_playerid_is_skipped(base, norms):
    g = base[base["gameid"] == LCK].copy()
    dropped = g.loc[(g["side"] == "Blue") & (g["position"] == "mid"), "playerid"].iloc[0]
    g.loc[(g["side"] == "Blue") & (g["position"] == "mid"), "playerid"] = np.nan
    hist = elo.build_elo(force=True, games=g, norms=norms)
    cur = _current()
    assert len(hist) == 9 and dropped not in set(hist["player_id"]) and dropped not in cur.index
    assert len(cur) == 9
    assert (cur[SKILL_COLS].drop(columns="laning_elo") != elo.INITIAL_ELO).any(axis=1).all()
    assert (cur["outcome_elo"] != elo.LEAGUE_PRIOR["LCK"]).all()


def test_update_elo_matches_full_rebuild(base, norms):
    # The cutoff falls between MSI and NEW, so update_elo has to init NEW's players
    # from the MSI league mean rebuilt from the state files.
    # 6 copies: the LCK Red side plays 14 games, so recent must trim them to RECENT_N.
    games = pd.concat([_games(base, copies=6), _new_players_game(base)], ignore_index=True)
    cutoff = pd.Timestamp("2026-07-13")

    elo.build_elo(force=True, games=games, norms=norms)
    full_cur, full_rec = _current(), pd.read_parquet(elo.RECENT_PATH)

    elo.build_elo(force=True, games=games[games["date"] < cutoff], norms=norms)
    elo.update_elo(games[games["date"] >= cutoff])
    inc_cur, inc_rec = _current(), pd.read_parquet(elo.RECENT_PATH)

    pd.testing.assert_frame_equal(full_cur, inc_cur, check_like=True, atol=1e-9)
    pd.testing.assert_frame_equal(full_rec, inc_rec, check_like=True, atol=1e-9)
    busy = full_cur.index[full_cur["games_played"] > elo.RECENT_N]
    assert len(busy) == 5
    assert (inc_rec[inc_rec["player_id"].isin(busy)].groupby("player_id").size() == elo.RECENT_N).all()


def test_update_elo_is_idempotent(base, norms):
    games = _games(base)
    cutoff = pd.Timestamp("2026-06-01")
    elo.build_elo(force=True, games=games[games["date"] < cutoff], norms=norms)
    elo.update_elo(games[games["date"] >= cutoff])
    once_cur, once_rec = _current(), pd.read_parquet(elo.RECENT_PATH)

    elo.update_elo(games)   # everything already applied
    pd.testing.assert_frame_equal(once_cur, _current())
    pd.testing.assert_frame_equal(once_rec, pd.read_parquet(elo.RECENT_PATH))


def test_update_elo_requires_role_norms(base):
    with pytest.raises(FileNotFoundError, match="role_norms"):
        elo.update_elo(base)
