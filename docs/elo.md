# Player ELO system

Implementation: `src/pipeline/elo.py`. See also [architecture.md](architecture.md) for
where ELO fits in the overall pipeline and [data.md](data.md) for the committed file
formats.

## Overview

Every player carries two independent rating families, both updated after every game
from the Oracle's Elixir (OE) player rows:

- **Performance ELO**: 5 skill ELOs plus `general_elo`, their sum. It measures how well a
  player played relative to the other nine players in the game, regardless of the result.
- **Outcome ELO**: one classic Elo rating per player, driven by the game result. It
  carries strength across regions, which the performance ELO cannot, because it is only
  zero-sum inside one game.

The model uses both, as team means over the starters and as per-role lane differences
(see [model.md](model.md)).

## Performance ELO

### Skills

Each skill is the mean of 2-3 role-normalised component stats. `minutes` is
`gamelength / 60`; OE column names are exact.

| Skill | Components |
|---|---|
| `combat` | `kills`/min, `assists`/min, `deaths`/min (inverted) |
| `damage` | `dpm`, `damageshare` |
| `economy` | `earned gpm`, `earnedgoldshare`, `cspm` |
| `laning` | `golddiffat15`, `xpdiffat15`, `csdiffat15` |
| `vision` | `vspm`, `wcpm`, `controlwardsbought`/min |

`deaths/min` is inverted (z-score negated) so that more is better in every component.
Rates are built from raw counts (`kills`, `assists`, `deaths`, `controlwardsbought`)
divided by game minutes; the other components are used as OE gives them.

### Role normalisation

Raw stats are not comparable across roles (a support's damage against a carry's). Each
component is converted to a z-score within the player's `position`:

```
z = (x - mean[position, stat]) / std[position, stat]
```

The means and standard deviations are in `src/data/processed/role_norms.json`, computed
once over the `complete` OE rows with a `playerid` from the years `NORM_YEARS`
(2022-2025) and **frozen**. `build_elo` writes the file on its first run or with
`--force`; `update_elo` only reads it (and raises if it is missing). Freezing keeps an
incremental update identical to a full rebuild.

A skill's composite is the mean of its component z-scores. If any component is NaN, or the
position has no norms, the composite is NaN and the player is not ranked in that skill for
that game.

### Laning on partial games

OE rows flagged `partial` have no @15 stats. All three `laning` components are then NaN,
so the laning skill is skipped for every player in that game: no laning ELO changes. The
other four skills are unaffected. Across all leagues partial games are 17-20% of games in
2016-18 and 1.5% in 2026. In the major leagues they were 30% in 2025 (LPL) and 4.5% in
2026.

### Ranking and delta

For each skill, the players with a valid composite (normally all 10) are ranked 1..N
(1 = best) and each receives `rank_to_delta(rank, N)`:

```
ODD N  (e.g. 15):  +7 +6 +5 +4 +3 +2 +1  0 -1 -2 -3 -4 -5 -6 -7
EVEN N (e.g. 10):  +5 +4 +3 +2 +1 -1 -2 -3 -4 -5
```

Ties receive the average of the tied ranks, rounded half-to-even (two players tied for
3rd have average rank 3.5, which rounds to 4, so both get rank 4). With ties the rounded ranks can break the sum, so the mean of the
deltas is subtracted afterwards; the skill is then exactly zero-sum per game. With
fewer than 2 valid players, the skill gets no update.

All 10 players play the whole game, so there is no minutes weighting and no
qualification threshold. A skill's ELO changes only through the rank in the game, never
through the result.

### General ELO

```
general_elo = combat + damage + economy + laning + vision
```

Every skill starts at `INITIAL_ELO = 1000.0`, so a new player's `general_elo` is 5000.
`general_elo` is derived (`pre_general_elo` in the history, `general_elo` in the current
state) and is not updated independently. Performance ELO has no cross-region anchor: a
1100 in the LCK and a 1100 in the LEC are different things, which is what the outcome ELO
is for.

## Outcome ELO

Classic Elo on the game result, per player, on top of the performance skills.

- **Team rating** `R` = mean `outcome_elo` of the side's starters.
- **Expected score** `E = 1 / (1 + 10^((R_opp - R) / 400))`.
- **Update**, for each starter: `outcome_elo += k * (result - E)`, with
  `k = K * mov_mult * (NEW_K_MULT if the player has fewer than NEW_GAMES games else 1)`.
- The update is skipped if a side has no rated starters, or if the two results are not
  consistent (3 OE games have both sides marked as losers).

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `K` | 24.0 | base K-factor |
| `NEW_K_MULT` / `NEW_GAMES` | 2.0 / 20 | K multiplier for a player's first 20 games |
| `REGRESS` | 0.2 | new-year pull toward the league mean |
| `MIN_LEAGUE_PLAYERS` | 10 | players needed for a league mean to be used |
| `DEFAULT_PRIOR` | 1350.0 | init for leagues not in `LEAGUE_PRIOR` |
| `MOV_GD_REF` | 2289.0 | mean absolute team gold diff @15 |
| `MOV_LEN_REF` | 1890.0 | median game length in seconds |

`REGRESS`, `NEW_K_MULT` and the margin multiplier were chosen on walk-forward validation
windows that all lie before the test slice (see [model.md](model.md#evaluation-protocol)).
Other K values (16/24/32/48) and an international K multiplier did not help.

### Margin of victory

`mov_mult(gd15, gamelength)` scales K by how decisive the game was:

```
gd15 known:    sqrt((|team gold diff @15| + 500) / (MOV_GD_REF + 500))
gd15 missing:  sqrt(MOV_LEN_REF / gamelength)        (partial games; shorter = more decisive)
neither:       1.0
```

The team gold diff is the blue starters' `golddiffat15` summed; it is NaN if any player
in the game lacks it. Even games still move ratings because of the +500.

### Initialisation of new players

A player with no rating starts at the current mean `outcome_elo` of the players whose
**last game** was in the same OE league code, if at least `MIN_LEAGUE_PLAYERS` such
players exist. Otherwise: `LEAGUE_PRIOR.get(league, DEFAULT_PRIOR)`. The league mean is
how strength moves across regions: a transfer or an international event feeds the
player's rating into the league he now plays in.

`LEAGUE_PRIOR` is a hand-set tier table, including the historical OE codes:

| Prior | OE league codes |
|---|---|
| 1600 | `LCK`, `OGN`, `LPL` |
| 1500 | `LEC`, `EU LCS`, `LCS`, `NA LCS`, `LTA N`, `LTA` |
| 1450 | `LCP`, `PCS`, `LMS`, `CBLOL`, `LTA S` |
| 1350 | everything else (`DEFAULT_PRIOR`) |

Skill ELOs of a new player start at `INITIAL_ELO`.

### Off-season regression (`REGRESS`)

At a player's first game of a new calendar year, if that game is in a domestic league with
at least `MIN_LEAGUE_PLAYERS` rated players, `outcome_elo` first moves `REGRESS` (20%) of
the way toward that league's mean. If the year's first game is international or in a small
league, there is no pull that year (rare).

The pull is applied inside that game's update, **after** its pre-game snapshot, so the
snapshot equals the previous post-game state. Training (which reads snapshots) and live
prediction (which reads the current state) therefore see the same values; a test covers
this.

### Faster placement for new players (`NEW_K_MULT`)

A player's first `NEW_GAMES` (20) games use `K * NEW_K_MULT`. K is applied per player, so a
game between a new and an established side is **not zero-sum**: the new side moves twice as
far. It is zero-sum when both sides have the same multiplier per player.

### International events

`INTERNATIONAL = {WLDs, MSI, FST, EWC, MSC}`. They use the same K and update rule. They
are where cross-region strength is exchanged, and they never trigger the new-year pull.
They are not a tier in `LEAGUE_PRIOR`, so `league_tier` is 1350 for them
(`is_international` separates them in the features).

## Players without a `playerid`

Rows with no `playerid` (0.3-1.9% of rows per year, essentially all in minor leagues such
as LJL, LPLOL, LIT, LAS; 0% in the major leagues) are skipped: no rating, not ranked,
no snapshot. The rest of their game is processed normally, with the missing slot absent
from the team mean. Twelve `(gameid, playerid)` pairs where one id fills two slots are
de-duplicated.

## Output files

Under `src/data/processed/`:

- `player_elo.parquet`: **not committed**, full history, one row per rated player per
  game. Rebuilt by `elo.py --force`. Stores the ELO **before** the game, the value that
  was known going into it. Columns: `game_id`, `game_date`, `player_id`, `player_name`,
  `team_id`, `position`, `league`, `side`, `pre_outcome_elo`, `pre_general_elo`,
  `pre_{skill}_elo` for the 5 skills.
- `player_elo_current.parquet`: **committed live state**. One row per player, post-game
  values after their last game: `player_id`, `player_name`, `team_id`, `position`,
  `league`, `last_game_date`, `games_played`, `outcome_elo`, `general_elo`,
  `{skill}_elo` for the 5 skills. Team, position, league and name follow the latest game.
  Live predictions read it for each player's rating.
- `player_elo_recent.parquet`: **committed live state**. The last `RECENT_N = 11`
  pre-game snapshots per player, same columns as the history. Feeds the live "form"
  feature, which looks back up to 10 games.
- `role_norms.json`: **committed, frozen**, see above.

## Entry points

- `build_elo(force=False, games=None, norms=None)`: replays every OE game since 2014 in
  `(date, gameid)` order, writes the history, both state files and (first run or
  `--force`) `role_norms.json`. Run with `python src/pipeline/elo.py --force`. The ELO
  warms up on 2014-2016; training data starts in 2017 ([model.md](model.md)). Games
  without a result yet are skipped.
- `update_elo(games)`: incremental. Takes OE rows, loads the committed state, replays
  only the games not yet in `player_elo_recent.parquet`, and rewrites both state files.
  - **Idempotent**: gameids already in the recent snapshots are skipped, so replaying the
    same games is a no-op.
  - `daily_state.py` only passes gameids that are new to the `team_games` files.
  - **Late rows** (OE sometimes adds a game a day or more late, LPL especially) are
    applied in arrival order, not date order. A later full rebuild re-sorts them, so the
    two can differ slightly for such games.
  - A player's `last_game_date` / team / league only move forward; a late game older than
    that adds to `games_played` but does not rewrite them.

`tests/test_elo.py` asserts that `update_elo` applied incrementally reproduces a full
`build_elo` rebuild over the same in-order games, including across a new year, and that
re-applying the games changes nothing.

## Using ELO in features

[model.md](model.md) describes the features. In short, `features.py` takes each team's
**starters** (5 players in role order) and builds, per team, the mean of `outcome`,
`general` and the 5 skill ELOs, plus per-role differences of `general` and `outcome`.
Training reads each starter's first pre-game snapshot on or after the game's day; live
reads `player_elo_current.parquet`.

## Design decisions and limitations

**No team-level ELO.** A team's strength comes from its starters. A lineup change moves
the team rating automatically. A team-level Elo was tried and gave no measurable gain.

**Ranking within the game.** Performance is relative to the nine other players in the game,
not to the league, and not role-agnostic: the role z-scores remove the role effect first.

**No time decay in the skills.** Skill ELOs from years ago count like last week's. The
outcome ELO has the yearly pull toward the league mean; the model's rolling windows and
form features add recency.

**Small leagues and priors.** The `LEAGUE_PRIOR` tiers are hand-set. Cross-region strength
rests on the league mean init, international games (a few hundred per year) and transfers.

### Known nits

- `mov_mult` treats comebacks as dominant wins (it uses the @15 gold diff, whichever side
  went on to win).
- A side with a missing `playerid` gets a partial gold-diff sum.
- The new-year pull is skipped for the year if a player's first game of that year is
  international.

### Odd but plausible observations

- The mean `outcome_elo` of players whose last game was in the LPL (about 1615) is below the
  LEC (about 1690). One reason: many stars' last game is Worlds (as of 2026-10-15), and the league
  key is the last game's league.
- Two EMEA Masters players reach the overall `outcome_elo` top 10 (about 1920).
- Ratings drift up slowly over the years (about 1450 to 1520). Not investigated further;
  the non-zero-sum updates of new players are one candidate.
