"""
Player ELO from Oracle's Elixir (OE) player rows.

Design
------
Each player has two independent rating families, both updated every game.

Performance: 5 skill ELOs + general ELO (sum of the skills). Per game, per skill:
  1. Component stats (rates use minutes = gamelength / 60 where OE gives counts).
  2. Role normalisation with the frozen role_norms.json:
       z = (x - mean[position, stat]) / std[position, stat]   (inverted where noted)
  3. Composite = mean of the component z's; NaN if any component is NaN
     (a `partial` game has no @15 stats, so laning is skipped for that game).
  4. Rank the valid players 1..N (1 = best; ties -> average rank, rounded).
  5. rank_to_delta(rank, N), then mean-subtract so the skill is exactly zero-sum.
     Fewer than 2 valid players -> no update for that skill.
  All 10 players play the whole game: no minutes weighting, no qualification.

Outcome: classic Elo on the game result. Team rating R = mean outcome_elo of the
side's rated starters, E = 1 / (1 + 10^((R_opp - R) / 400)), each starter
+= K * (result - E). Skipped if a side has no rated starters. A new player starts
at the current mean outcome_elo of players whose last game was in the same OE
league (if >= MIN_LEAGUE_PLAYERS such players), else LEAGUE_PRIOR / DEFAULT_PRIOR.
This is what carries strength across regions (internationals, transfers).

Rows without a playerid are skipped (no rating, not ranked, not snapshotted);
the rest of their game is processed normally.

Skills
------
  combat   kills/min + assists/min + deaths/min (inverted)
  damage   dpm + damageshare
  economy  earned gpm + earnedgoldshare + cspm
  laning   golddiffat15 + xpdiffat15 + csdiffat15   [NaN in partial games -> skipped]
  vision   vspm + wcpm + controlwardsbought/min

Outputs
-------
  data/processed/player_elo.parquet          (not committed)
    One row per rated player per game: the PRE-game snapshot (HISTORY_COLS).
  data/processed/player_elo_current.parquet  (committed live state, post-game)
    One row per player (CURRENT_COLS); team/position/league/name follow the
    latest game.
  data/processed/player_elo_recent.parquet   (committed live state)
    Last RECENT_N pre-game snapshots per player, same columns as the history.
  data/processed/role_norms.json             (committed, frozen)
    {position: {stat: {"mean", "std"}}} over NORM_YEARS `complete` rows. Written
    by build_elo on its first run or with force; update_elo only reads it, which
    keeps incremental updates equal to a full rebuild.

Run
---
  python src/pipeline/elo.py                # build if missing, print top 10
  python src/pipeline/elo.py --force        # rebuild from every OE year (and the norms)
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import oe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH    = PROCESSED_DIR / "player_elo.parquet"          # full history (not committed)
CURRENT_PATH    = PROCESSED_DIR / "player_elo_current.parquet"  # committed live state (post-game)
RECENT_PATH     = PROCESSED_DIR / "player_elo_recent.parquet"   # committed live state (pre-game tail)
ROLE_NORMS_PATH = PROCESSED_DIR / "role_norms.json"             # committed, frozen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_ELO = 1000.0
RECENT_N    = 11        # pre-game snapshots kept per player in the "recent" state file
                        # (the form feature looks back up to 10 games -> needs 11 rows)
NORM_YEARS  = range(2022, 2026)

SKILLS = ["combat", "damage", "economy", "laning", "vision"]
COMPONENTS = {
    "combat":  ["kills_pm", "assists_pm", "deaths_pm"],
    "damage":  ["dpm", "damageshare"],
    "economy": ["earned gpm", "earnedgoldshare", "cspm"],
    "laning":  ["golddiffat15", "xpdiffat15", "csdiffat15"],
    "vision":  ["vspm", "wcpm", "controlwardsbought_pm"],
}
RATES    = {"kills_pm": "kills", "assists_pm": "assists", "deaths_pm": "deaths",
            "controlwardsbought_pm": "controlwardsbought"}   # per-minute from raw counts
INVERTED = {"deaths_pm"}
STATS    = [s for comps in COMPONENTS.values() for s in comps]

K = 24.0                    # outcome Elo K-factor (tuning knob)
MIN_LEAGUE_PLAYERS = 10     # league-mean init needs this many players last seen there
DEFAULT_PRIOR = 1350.0
# ponytail: hand-set tiers, fit from international results if it matters
LEAGUE_PRIOR = {
    "LCK": 1600.0, "OGN": 1600.0, "LPL": 1600.0,
    "LEC": 1500.0, "EU LCS": 1500.0, "LCS": 1500.0, "NA LCS": 1500.0, "LTA N": 1500.0,
    "LTA": 1500.0,
    "LCP": 1450.0, "PCS": 1450.0, "LMS": 1450.0, "CBLOL": 1450.0, "LTA S": 1450.0,
}
INTERNATIONAL = {"WLDs", "MSI", "FST", "EWC", "MSC"}   # cross-region signal; K unchanged in v1

HISTORY_COLS = (
    ["game_id", "game_date", "player_id", "player_name", "team_id", "position", "league",
     "side", "pre_outcome_elo", "pre_general_elo"]
    + [f"pre_{s}_elo" for s in SKILLS]
)
META_COLS = ["player_name", "team_id", "position", "league", "last_game_date", "games_played"]
CURRENT_COLS = (["player_id"] + META_COLS + ["outcome_elo", "general_elo"]
                + [f"{s}_elo" for s in SKILLS])


# ---------------------------------------------------------------------------
# Role normalisation + skill scores
# ---------------------------------------------------------------------------

def _stats(players: pd.DataFrame) -> pd.DataFrame:
    """The STATS component columns (raw OE stats + per-minute rates)."""
    minutes = players["gamelength"] / 60.0
    return pd.DataFrame({s: players[RATES[s]] / minutes if s in RATES else players[s]
                         for s in STATS}, index=players.index)


def compute_role_norms(players: pd.DataFrame, years=NORM_YEARS) -> dict:
    """
    {position: {stat: {"mean", "std"}}} over `complete` player rows with a playerid,
    restricted to OE `year` in years (years=None: all rows, e.g. a test fixture).
    """
    p = players[(players["datacompleteness"] == "complete") & players["playerid"].notna()]
    if years is not None:
        p = p[p["year"].isin(list(years))]
    g = _stats(p).groupby(p["position"])
    mean, std = g.mean(), g.std()
    return {pos: {s: {"mean": float(mean.at[pos, s]), "std": float(std.at[pos, s])}
                  for s in STATS}
            for pos in mean.index}


def compute_skill_scores(players: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """
    score_{skill} per row: mean of the role-normalised component z's (deaths
    inverted). NaN if any component is NaN or the position has no norms; a NaN
    score excludes the player from that skill's ranking.
    """
    stats = _stats(players)
    z = {}
    for s in STATS:
        mean = players["position"].map({pos: n[s]["mean"] for pos, n in norms.items()})
        std  = players["position"].map({pos: n[s]["std"] for pos, n in norms.items()})
        z[s] = (stats[s] - mean) / std * (-1.0 if s in INVERTED else 1.0)
    z = pd.DataFrame(z, index=players.index)
    return pd.DataFrame({f"score_{k}": z[comps].mean(axis=1, skipna=False)
                         for k, comps in COMPONENTS.items()}, index=players.index)


def _load_norms() -> dict:
    if not ROLE_NORMS_PATH.exists():
        raise FileNotFoundError(f"{ROLE_NORMS_PATH} not found - run `python src/pipeline/elo.py "
                                "--force` once to compute the frozen role norms")
    return json.loads(ROLE_NORMS_PATH.read_text())


# ---------------------------------------------------------------------------
# Rank -> delta
# ---------------------------------------------------------------------------

def rank_to_delta(rank: int, n: int) -> int:
    """
    Convert 1-indexed rank to ELO delta on a dynamic zero-centered scale.

    Odd  N=15: rank 1 → +7, rank 8 → 0,  rank 15 → -7
    Even N=20: rank 1 → +10, rank 10 → +1, rank 11 → -1, rank 20 → -10

    Sum of all deltas is always 0 (zero-sum per game per skill).
    """
    if n % 2 == 1:          # odd — single middle player gets 0
        return (n + 1) // 2 - rank
    else:                   # even — no zero, gap between the two middle ranks
        half = n // 2
        return half - rank + (1 if rank <= half else 0)


def compute_deltas(scores: np.ndarray) -> np.ndarray:
    """
    Zero-sum skill deltas for one game's scores (NaN = not ranked -> delta 0).
    Rank 1 = highest score; ties -> average rank rounded half-to-even (as the
    pandas rank().round() it replaces). Fewer than 2 valid scores -> all zeros.
    """
    out = np.zeros(len(scores))
    valid = ~np.isnan(scores)
    n = int(valid.sum())
    if n < 2:
        return out
    x = scores[valid]
    avg_rank = (x[None, :] > x[:, None]).sum(1) + ((x[None, :] == x[:, None]).sum(1) + 1) / 2
    raw = np.array([rank_to_delta(int(r), n) for r in np.round(avg_rank)], dtype=float)
    out[valid] = raw - raw.mean()
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    # lsum/lcnt: running outcome_elo sum/count of players by their LAST game's league.
    return {"skills": {}, "outcome": {}, "meta": {},
            "lsum": defaultdict(float), "lcnt": defaultdict(int)}


def _replay(st: dict, players: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """
    Apply OE player rows to the state in place, game by game in (date, gameid)
    order. Returns the pre-game snapshots (HISTORY_COLS) of the processed games.
    Games without a result yet are skipped.
    """
    # 12 (gameid, playerid) pairs in 2014-2026 are the same id in two slots (data errors).
    p = (players[players["playerid"].notna()]
         .drop_duplicates(["gameid", "playerid"])
         .sort_values(["date", "gameid"], kind="stable")
         .reset_index(drop=True))
    scores = compute_skill_scores(p, norms).to_numpy()
    gid    = p["gameid"].to_numpy()
    pids   = p["playerid"].tolist()
    names  = p["playername"].tolist()
    teams  = p["teamid"].astype(object).where(p["teamid"].notna(), None).tolist()
    pos    = p["position"].tolist()
    league = p["league"].tolist()
    dates  = p["date"].tolist()
    res    = p["result"].to_numpy(dtype=float, na_value=np.nan)
    blue   = (p["side"] == "Blue").to_numpy()

    pre  = np.full((len(p), 1 + len(SKILLS)), np.nan)
    keep = np.zeros(len(p), dtype=bool)
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]) if len(p) else np.array([], int)
    ends   = np.r_[starts[1:], len(p)]
    skills, outcome, meta, lsum, lcnt = (st["skills"], st["outcome"], st["meta"],
                                         st["lsum"], st["lcnt"])

    for a, b in tqdm(zip(starts, ends), total=len(starts), desc="ELO  games"):
        r = res[a:b]
        if np.isnan(r).any():
            continue                                    # no result yet
        ps, lg, date = pids[a:b], league[a], dates[a]

        # New players: league-mean (or prior) outcome init, before the game is processed.
        if any(q not in outcome for q in ps):
            init = (lsum[lg] / lcnt[lg] if lcnt[lg] >= MIN_LEAGUE_PLAYERS
                    else LEAGUE_PRIOR.get(lg, DEFAULT_PRIOR))
            for q in ps:
                if q not in outcome:
                    outcome[q] = init
                    skills[q] = np.full(len(SKILLS), INITIAL_ELO)

        o = np.array([outcome[q] for q in ps])
        S = np.array([skills[q] for q in ps])
        pre[a:b, 0], pre[a:b, 1:], keep[a:b] = o, S, True

        S = S + np.column_stack([compute_deltas(scores[a:b, k]) for k in range(len(SKILLS))])

        bl = blue[a:b]
        o_new = o
        # Needs both sides rated and consistent results (3 OE games have both sides at 0).
        if bl.any() and not bl.all() and r[bl][0] + r[~bl][0] == 1:
            e_blue = 1.0 / (1.0 + 10.0 ** ((o[~bl].mean() - o[bl].mean()) / 400.0))
            o_new = o + K * (r - np.where(bl, e_blue, 1.0 - e_blue))

        for i, q in enumerate(ps):
            old = meta.get(q)
            if old is not None:
                lsum[old["league"]] -= o[i]
                lcnt[old["league"]] -= 1
            if old is None or date >= old["last_game_date"]:
                name = names[a + i]
                if not isinstance(name, str) or name == "MISSING":
                    name = (old and old["player_name"]) or (name if isinstance(name, str) else "")
                meta[q] = {"player_name": name, "team_id": teams[a + i], "position": pos[a + i],
                           "league": lg, "last_game_date": date,
                           "games_played": (old["games_played"] if old else 0) + 1}
            else:
                old["games_played"] += 1
            lsum[meta[q]["league"]] += o_new[i]
            lcnt[meta[q]["league"]] += 1
            outcome[q] = float(o_new[i])
            skills[q] = S[i]

    hist = pd.DataFrame({
        "game_id": p["gameid"], "game_date": p["date"], "player_id": p["playerid"],
        "player_name": p["playername"].fillna(""), "team_id": p["teamid"],
        "position": p["position"], "league": p["league"], "side": p["side"],
        "pre_outcome_elo": pre[:, 0], "pre_general_elo": pre[:, 1:].sum(1),
        **{f"pre_{s}_elo": pre[:, i + 1] for i, s in enumerate(SKILLS)},
    })
    return hist[keep].reset_index(drop=True)


def _tail_recent(hist: pd.DataFrame) -> pd.DataFrame:
    """Return the last RECENT_N pre-game snapshots per player, chronologically."""
    return (hist.sort_values(["game_date", "game_id", "player_id"])
                .groupby("player_id", group_keys=False)
                .tail(RECENT_N)
                .reset_index(drop=True))


def _write_state(st: dict, recent: pd.DataFrame) -> pd.DataFrame:
    """Write the two committed live-state files (post-game current + recent tail)."""
    current = pd.DataFrame(
        [{"player_id": q, **m, "outcome_elo": st["outcome"][q],
          "general_elo": float(st["skills"][q].sum()),
          **{f"{s}_elo": float(st["skills"][q][i]) for i, s in enumerate(SKILLS)}}
         for q, m in st["meta"].items()],
        columns=CURRENT_COLS)
    current["last_game_date"] = pd.to_datetime(current["last_game_date"])
    current = current.sort_values("player_id").reset_index(drop=True)
    current.to_parquet(CURRENT_PATH, index=False)
    logger.info("Saved current ELO (post-game): %d players → %s", len(current), CURRENT_PATH)

    recent.to_parquet(RECENT_PATH, index=False)
    logger.info("Saved recent ELO snapshots: %d rows → %s", len(recent), RECENT_PATH)
    return current


def _load_state() -> tuple[dict, pd.DataFrame]:
    """The replay state + recent snapshots from the committed state files."""
    st = _empty_state()
    if CURRENT_PATH.exists():
        for r in pd.read_parquet(CURRENT_PATH).to_dict("records"):
            q = r["player_id"]
            st["skills"][q] = np.array([r[f"{s}_elo"] for s in SKILLS], dtype=float)
            st["outcome"][q] = float(r["outcome_elo"])
            st["meta"][q] = {c: r[c] for c in META_COLS}
            st["lsum"][r["league"]] += r["outcome_elo"]
            st["lcnt"][r["league"]] += 1
    recent = (pd.read_parquet(RECENT_PATH) if RECENT_PATH.exists()
              else pd.DataFrame(columns=HISTORY_COLS))
    return st, recent


def _load_all_players() -> pd.DataFrame:
    years = sorted(int(f.name[:4]) for f in oe.OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    if not years:
        raise FileNotFoundError(f"No OE year files in {oe.OE_DIR} - run bootstrap_data.py")
    logger.info("Loading OE years %d-%d", years[0], years[-1])
    return pd.concat([oe.player_rows(oe.load_year(y)) for y in years], ignore_index=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def build_elo(force: bool = False, games: pd.DataFrame | None = None,
              norms: dict | None = None) -> pd.DataFrame:
    """
    Replay every game from scratch and write the history, both state files and
    (first run, force, or norms supplied) role_norms.json.

    Args:
        force: rebuild even if the history exists, and recompute the role norms.
        games: OE rows to replay (team rows are ignored); None = every year in oe.OE_DIR.
        norms: role norms to use instead of loading/computing them (tests).

    Returns the pre-game snapshot history (HISTORY_COLS).
    """
    if games is None:
        if HISTORY_PATH.exists() and not force:
            logger.info("ELO already built — loading from %s", HISTORY_PATH)
            return pd.read_parquet(HISTORY_PATH)
        games = _load_all_players()
    players = oe.player_rows(games)

    if norms is None and ROLE_NORMS_PATH.exists() and not force:
        norms = _load_norms()
    else:
        norms = norms or compute_role_norms(players)
        ROLE_NORMS_PATH.write_text(json.dumps(norms, indent=1))
        logger.info("Saved role norms → %s", ROLE_NORMS_PATH)

    st = _empty_state()
    hist = _replay(st, players, norms)
    if hist.empty:
        raise RuntimeError("No ELO records generated — verify the OE files have results")

    hist.to_parquet(HISTORY_PATH, index=False)
    logger.info("Saved ELO history: %d records, %d games → %s",
                len(hist), hist["game_id"].nunique(), HISTORY_PATH)
    _write_state(st, _tail_recent(hist))
    return hist


def update_elo(games: pd.DataFrame) -> pd.DataFrame:
    """
    Incrementally continue the saved state with OE rows (idempotent): gameids
    already in player_elo_recent.parquet are skipped, so re-running is a no-op.
    Uses the frozen role_norms.json (raises if missing).

    Returns the refreshed post-game current-state DataFrame.
    """
    norms = _load_norms()
    st, recent = _load_state()
    new = oe.player_rows(games)
    new = new[~new["gameid"].isin(set(recent["game_id"]))]
    # ponytail: arrival-order updates for late OE rows; full rebuild re-sorts
    hist = _replay(st, new, norms)
    if not hist.empty:
        recent = _tail_recent(hist if recent.empty else pd.concat([recent, hist], ignore_index=True))
    current = _write_state(st, recent)
    logger.info("update_elo: applied %d new game(s); state now %d players",
                hist["game_id"].nunique(), len(current))
    return current


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build player ELO from Oracle's Elixir data")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    t0 = time.monotonic()
    hist = build_elo(force=args.force)
    cur = pd.read_parquet(CURRENT_PATH)
    print(f"\nDone. {hist['game_id'].nunique():,} games, {len(hist):,} records, "
          f"{len(cur):,} players.")
    print("\nTop 10 by outcome_elo:")
    print(cur.nlargest(10, "outcome_elo")[
        ["player_name", "team_id", "league", "last_game_date", "games_played",
         "outcome_elo", "general_elo"]].to_string(index=False))
    print(f"\nRuntime {time.monotonic() - t0:.1f}s")
