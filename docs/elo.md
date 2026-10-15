# Player ELO system

Implementation: `src/pipeline/elo.py`. See also [architecture.md](architecture.md) for
where ELO fits in the overall pipeline and [data.md](data.md) for the committed file
formats.

## Overview

Each player carries 7 independent skill ELOs and 1 general ELO.
The general ELO is the live sum of all 7 skill ELOs.

After every game, every qualifying player's skill ELOs are updated
based on how they ranked against the other players in that game —
not on whether their team won or lost.

This is intentional. A player who scores 30 on a losing team
performed better than most players that night and gains ELO.
A player who scores 6 and turns the ball over on a winning team
performed poorly and loses ELO. Team outcome is irrelevant.

The model then uses these ELOs (and their differentials between
matchups) as features — letting XGBoost decide which skills
matter most for predicting game outcomes.

## Qualification

A player must have played >= 10 minutes (`MIN_MINUTES`) to be included in a
game's ranking for any skill. Players below this threshold receive
no ELO update for that game.

10 minutes was chosen as the minimum meaningful sample for per-36
normalized stats. Below this threshold, counting stats (assists,
steals, etc.) become too noisy to rank fairly.

## The 7 skills

Each skill is a composite of 2-3 related stats, computed as the
sum of z-scored components within that game. Z-scoring puts all
components on the same scale before combining, regardless of
their original units.

1. **SCORING** — points per 36 minutes + true shooting percentage.
   Captures both volume (how much you score) and efficiency (how
   well you convert possessions). A high-volume inefficient scorer
   and a low-volume efficient scorer will rank similarly.

2. **PLAYMAKING** — assists per 36 minutes + assist-to-turnover ratio.
   Captures both how many plays you create and how safely you
   do it. A player who dishes 10 assists but also has 6 turnovers
   scores lower than one with 7 assists and 1 turnover.

3. **DEFENSE** — defensive rating (inverted) + steals per 36 + blocks per 36.
   Defensive rating is inverted (lower = better, so the z-score is negated)
   so that all three components point in the same direction.
   Captures both team-level defensive impact and individual disruptions.
   (In the daily/live pipeline, defensive rating is unavailable — see
   "CDN adapter" below; the skill falls back to steals + blocks only.)

4. **REBOUNDING** — rebound percentage (from the advanced box score, or
   derived from basics in the CDN adapter).
   Uses percentage rather than raw count so players who share
   the floor with strong rebounders aren't unfairly penalized.

5. **EFFICIENCY** — PIE (Player Impact Estimate).
   PIE is the NBA's own composite stat. It measures the share of
   all game events (points, rebounds, assists, steals, blocks,
   turnovers, fouls) that a player was responsible for relative
   to their team and opponent. It already accounts for context,
   making it a clean single-signal efficiency measure.

6. **HUSTLE** — speed (mph) + distance (miles) + touches per 36.
   Requires `BoxScorePlayerTrackV3` tracking data. Skipped entirely for
   games where tracking data is absent (generally pre-2016 stats.nba.com
   games, or any game processed through the CDN adapter — see below).
   Players' hustle ELOs simply do not update for those games.

7. **THREE POINT** — 3-pointers made per 36 + 3-point percentage.
   Only computed for players who attempted at least 1 three-pointer
   in the game. A player who shoots zero threes is not ranked in
   this skill for that game — their three_point ELO does not update.
   This prevents players who never attempt threes from being
   penalized for a skill they are not deployed to use.

## The ranking system

After computing composite skill scores, players are ranked 1..N
within the game (1 = best score). The ELO delta is determined
entirely by rank using a dynamic zero-centered scale (`rank_to_delta`).

For N players, the scale is:

```
ODD N (e.g. N=15):
  The single middle player (rank 8) gets delta = 0.
  Each rank above the middle gains +1 more.
  Each rank below the middle loses -1 more.

  rank:   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  delta: +7  +6  +5  +4  +3  +2  +1   0  -1  -2  -3  -4  -5  -6  -7
  sum: 0

EVEN N (e.g. N=20):
  No single middle player. The top half gains, bottom half loses.
  There is no zero — the two middle ranks get +1 and -1.

  rank:   1   2  ...  10  11  ...  20
  delta: +10  +9  ...  +1  -1  ...  -10
  sum: 0
```

The scale adapts dynamically to however many players qualified
that game. A game with 14 qualifiers uses a ±7 max scale.
A game with 22 qualifiers uses a ±11 max scale. This means a
dominant performance is always rewarded relative to the field,
regardless of how deep the rotation was.

Ties in ranking receive the average rank of the tied positions,
rounded to the nearest integer. Two players tied for 3rd both
get rank 3 (delta = same as 3rd place) rather than one getting
3rd and one getting 4th.

## Minutes weighting

A player who plays 10 minutes and happens to have 0 turnovers
would rank #1 in ball security on a per-36 basis. That is a
legitimate data point — but it is a less reliable signal than
a player who goes 36 minutes with 0 turnovers.

To account for this, the raw rank delta is scaled by the
square root of the player's minutes fraction:

```
scaled_delta = raw_delta × sqrt(minutes / 36)
```

Square root (rather than linear) is used because it provides
a smooth taper that is not overly harsh on legitimate rotation
players. Examples:

```
36 minutes  → × 1.000  (full weight)
25 minutes  → × 0.833
18 minutes  → × 0.707
10 minutes  → × 0.527
```

A starter who ranks #1 with 36 minutes gains full ELO.
A bench player who ranks #1 with 10 minutes gains about half.

## Zero-sum preservation

Scaling by minutes breaks the zero-sum property because
high-minute players dominate the weighting. To restore it,
the mean of all scaled deltas within a game is subtracted:

```
final_delta = scaled_delta - mean(all scaled deltas in this game)
```

This recenters the game to exactly zero-sum regardless of how
minutes were distributed that night. The relative differences
between players are preserved — only the baseline shifts.

As a result, ELO points are conserved globally. There is no
inflation or deflation over time. A player can only gain ELO
that another player lost in the same game.

## General ELO

The general ELO is not stored directly — it is computed on
the fly as the sum of the 7 skill ELOs:

```
general_elo = scoring + playmaking + defense +
              rebounding + efficiency + hustle + three_point
```

At initialization, every skill starts at 1000, so every new
player's general ELO starts at 7000. After thousands of games,
elite players will be significantly above 7000 and poor players
below.

The model sees both the individual skill ELOs and the general
ELO as separate features, letting it learn which skills are
most predictive of game outcomes.

## Initial state and convergence

All players start at `INITIAL_ELO = 1000` for every skill.

New players entering the league are initialized at 1000.
Rookies are therefore slightly overrated in the first few games
until their ELO converges to their true level. This is an
accepted tradeoff — ELO systems require some history to be
accurate, and the bootstrap covers 2015-16 onward for most players.

The model's training data spans games from 2015-16 onward, so by
the time it predicts games from 2020+ the ELO values have had
5+ seasons to converge for established players.

## Output files

Three files live under `data/processed/` (paths relative to `src/`):

`player_elo.parquet` — **not committed**, full history, one row per player per
game. Rebuilt from raw box scores by `elo.py --force`; too large and
data-dependent to commit. Stores the ELO snapshot BEFORE the game was
processed — the correct value to use as a training feature, since it is what
was known going into the game.

Columns: `game_id`, `game_date`, `player_id`, `player_name`, `pre_general_elo`,
`pre_scoring_elo`, `pre_playmaking_elo`, `pre_defense_elo`,
`pre_rebounding_elo`, `pre_efficiency_elo`, `pre_hustle_elo`,
`pre_three_point_elo`. (`player_name` is a human-readable "First Last" for
skimming the data — it is never used in the model.)

`player_elo_current.parquet` — **committed live state**. One row per player:
their post-game ELO after the last game they appeared in, plus `player_name`,
`team_id` and `last_game_date` (columns: `player_id`, `player_name`, `team_id`,
`last_game_date`, `general_elo`, `{skill}_elo`×7). This is what live predictions
read for a team's current roster and ELO (there is no future box score to read a
lineup from).

`player_elo_recent.parquet` — **committed live state**. The last `RECENT_N`
(11) pre-game snapshots per player, in the same shape as `player_elo.parquet`.
Feeds the live "player form" feature (ELO trajectory over the last 10 games),
since the full history file isn't shipped to the daily-pipeline runner.

## State files, incremental updates, and the CDN adapter

The three files above are produced by three entry points, all in `elo.py`:

- `build_elo(force=False, loader=load_game_players, game_index=None)` — full
  rebuild from scratch, iterating every game chronologically via the
  stats.nba.com box-score loader. Writes all three files. Run with
  `python src/pipeline/elo.py --force`.
- `update_elo(games, loader=None)` — incremental: loads the current state from
  `player_elo_current.parquet` / `player_elo_recent.parquet`, applies only the
  given `(game_id, game_date)` pairs that aren't already reflected in the
  recent-snapshot file, and rewrites both state files. Defaults to the CDN
  loader. Used by `daily_state.py` for newly-completed stateful games
  (regular season / playoffs / play-in / Cup final). Idempotent — replaying
  the same games is a no-op.
- `update_team_assignments(games, loader=None)` — refreshes only `team_id` +
  `last_game_date` in the state, without touching any ELO value. Used for
  preseason games, which should move a player onto their new team's roster
  for live-lineup purposes but must never affect skill ELO.

`tests/test_elo_update.py` asserts `update_elo` applied incrementally produces
byte-identical ELO to a full `build_elo` rebuild over the same games, and that
re-applying the same games is a no-op.

### CDN adapter

The daily pipeline never has access to stats.nba.com's advanced/tracking box
scores — only cdn.nba.com's `liveData` box score, which carries basic counting
stats. `load_cdn_game_players()` derives the same derived columns
`compute_skill_scores()` expects, from basics alone:

```
trueShootingPercentage = PTS / (2 · (FGA + 0.44 · FTA))
assistToTurnover       = AST / TO
reboundPercentage      = 100 · TRB · (TeamMin/5) / (Min · (TeamTRB + OppTRB))
PIE                    = player game-impact numerator / game total
```

`defensiveRating` and the tracking stats (`speed`, `distance`, `touches`) are
simply absent from the CDN feed. `compute_skill_scores()` already degrades
gracefully for missing columns: the defense skill's `defensiveRating`
component gets median/constant-filled (z-score contributes 0), so **defense
effectively becomes steals + blocks only** for CDN-driven games. Hustle has no
fallback — `score_hustle` is `NaN` for every player, so no player's hustle ELO
updates on games processed through the CDN adapter (it stays frozen at
whatever it was after the last stats.nba.com game with tracking data).

Historical ELO from the bootstrap (`load_game_players()`, reading
`boxscore_traditional_*.json` + `boxscore_advanced_*.json` +
`boxscore_tracking_*.json`) uses the real stats.nba.com advanced and tracking
data and is unaffected by any of this — defense and hustle work as designed
for every bootstrapped game where tracking data exists.

## Using ELO in features.py

`features.py` never reads the parquet files directly for each game — it
caches them once at module level and reads through `get_elo_features()`:

- **Historical / training** (`use_current=False`): looks up the game's
  pre-game ELO snapshot from `player_elo.parquet` by `game_id`, restricted to
  players who appeared in that game's box score minus that game's inactives
  (`boxscore_summary`'s `InactivePlayers`) — a real subtraction for the data
  the committed model was trained on, but a no-op on the re-downloaded
  bootstrap dataset, which doesn't have `boxscore_summary` files (see
  [data.md](data.md)).
- **Live prediction** (`use_current=True`): builds each team's roster from
  `player_elo_current.parquet` (players whose latest appearance was for that
  team; falls back to also including previous-season appearances if the
  current season has fewer than 8 such players, never further back than that
  — see `_live_team_lineup()`), using each player's post-game ELO.

In both modes, each team's composite ELO per skill is the **PPG-weighted**
average of its active players' ELOs (higher season PPG = more assumed
playing time = more influence on the team composite), computed by
`_weighted_team_elo()`. The final features are `home_{skill}_elo`,
`away_{skill}_elo`, and `elo_{skill}_diff` (home − away) for all 7 skills
plus `general` — 24 features per game.

## Design decisions and limitations

**No team-level ELO.** Team ELO was deliberately excluded. A team's strength
is derived entirely from which players are on the court that night. This
forces the model to account for injuries and roster changes automatically
rather than averaging over them.

**Ranking within the game, not league-wide.** Players are ranked against the
14-22 others who played that night, not against the entire league. This means
a good performance on a night where most players underperformed is rewarded
the same as a good performance on a high-scoring night. It is context-relative,
not context-neutral.

**No decay / time weighting.** ELO from 3 seasons ago counts equally to ELO
from last week. This is a simplification. A future improvement would be to
apply exponential decay to old games so recent form matters more. The rolling
feature windows and the ELO-trajectory "form" feature in `features.py`
partially compensate for this.

**Per-36 normalization for counting stats only.** Percentages (TS%, 3P%,
`reboundPercentage`) are not per-36'd — they are already rate stats. Only
counting stats (points, assists, steals, blocks, 3PM, touches) are normalized
to per-36-minutes before ranking.

**Three-point skill is opt-in.** Players who do not attempt threes do not
participate in the three_point skill ranking. Their three_point ELO stays at
whatever value it accumulated from games where they did attempt threes (or
1000 if they never have). This is intentional — a center who never shoots
threes should not have their ELO penalized for a skill they are not asked to
use.
