# Model

Binary XGBoost classifier predicting the probability that the **blue side wins one
game**, Platt-calibrated. A side-neutral game probability and a series formula turn that
into the series win probability the site shows.

Implementation: `src/pipeline/train.py` (training), `src/pipeline/build_dataset.py`
(feature matrix), `src/pipeline/features.py` (feature engineering),
`src/pipeline/predict.py` (inference). See [architecture.md](architecture.md) for where
this fits in the pipeline and [elo.md](elo.md) for the ELO inputs.

## Orientation

Every row is one game, blue side against red side, label `blue_win` (1 = blue won).
Features come as `blue_*`, `red_*` and `diff_*` (blue minus red). Blue-side advantage is
learned from the label, so there is no explicit side feature. The prediction is made
**before the draft**; picks and bans are not features.

`build_feature_matrix(blue_team_id, red_team_id, game_date, context, team_games_df,
use_current_elo, game_id=None, blue_starters=None, red_starters=None)` returns a flat
`dict[str, float]`. `context` is `{"league": OE code, "playoffs": 0/1}`.

## Features (136)

The committed model has 136 features (`feature_names` in `src/models/xgb_model.pkl`).
Nine groups, one `get_*_features()` each:

| Group | Function | Features | Count |
|---|---|---|---|
| ELO | `get_elo_features` | per team, the mean over the 5 starters of `outcome`, `general` and the 5 skill ELOs (7 values) as `blue_`/`red_`/`diff_`; plus per-role lane diffs `diff_{top,jng,mid,bot,sup}_{general,outcome}_elo` | 31 |
| Rolling team stats | `get_team_rolling_stats` | over the last 5 and last 10 games: win%, mean `golddiffat15`, game length, kills/min, deaths/min, first blood / dragon / herald / tower / baron rates, dragons per game (11 stats x 2 windows x blue/red/diff) | 66 |
| Side splits | `get_side_splits` | win% on blue and on red over the last 365 days (both teams), plus `*_win_pct_own_side` (blue team on blue, red team on red) | 9 |
| Rest | `get_rest_features` | `days_since_last`, `games_last_7` | 6 |
| Head-to-head | `get_head_to_head_features` | `h2h_games`, `h2h_blue_wins`, `h2h_blue_win_pct` over the last 365 days | 3 |
| Roster | `get_roster_features` | `roster_churn`, `rookie_count` | 6 |
| Player form | `get_player_form_features` | `form_outcome_elo`, `form_general_elo` | 6 |
| Streak | `get_streak_features` | signed current win/loss streak | 3 |
| Context | `get_context_features` | `is_international`, `playoffs`, `league_tier`, `league_mean_outcome_elo` (blue/red/diff) | 6 |

Group details:

- **ELO**: each starter's ELO vector as of the start of the game's day (see
  point-in-time below). The team value is a NaN-skipping mean over the starters; the lane
  diffs compare the two players of each role directly. The skills are defined in
  [elo.md](elo.md).
- **Rolling stats, side splits, rest, h2h**: all use the team's rows from `team_games`
  (all leagues and internationals), restricted to earlier days. Means skip NaN
  (`golddiffat15` is missing in partial games). `days_since_last` is NaN for a team with no
  earlier game; `games_last_7` counts games in the 7 days before the game's day.
- **Roster**: `roster_churn` is the number of starters shared by the team's **last two
  lineups before the day** (0-5; NaN with fewer than two earlier games). It is computed the
  same way in training and live. `rookie_count` is the number of starters with fewer than
  `ROOKIE_GAMES = 20` games in the ELO.
- **Player form**: mean over the starters of (ELO now minus the pre-game ELO 10 games
  earlier, or at the first game if fewer than 10) for `outcome_elo` and `general_elo`.
  Starters without an earlier game are left out; NaN if none remain. Live reads the last 11
  snapshots in `player_elo_recent.parquet`.
- **Streak**: `+k` after k wins in a row, `-k` after k losses.
- **Context**: `playoffs` is OE's flag in training and a `blockName` rule live (see
  limitations). `league_tier` is the `LEAGUE_PRIOR` tier of the game's league (1350 for
  internationals and unlisted leagues). `league_mean_outcome_elo` is, per team, the mean strength of
  its home league: a team's home league is its most recent domestic league within 365 days, the
  league's strength is the mean over its teams of the mean `outcome_elo` of each team's last
  lineup. It tells the model how strong the league a team's ratings were earned in is.

Each group is wrapped in its own try/except in `build_feature_matrix()`: a failure logs a
warning and leaves that group's keys out. `predict.py` fills missing keys with NaN, which
XGBoost treats as missing.

## Point-in-time rule and train/live differences

**Day level.** A game on calendar day `d` (UTC) sees only `team_games` rows from days
before `d`, and each player's ELO as of the start of day `d`:

- training: the player's first pre-game snapshot on or after `d` in `player_elo.parquet`
  (else the current value);
- live: the player's row in `player_elo_current.parquet`.

So game 2 of a series never sees game 1 in training, which matches a pre-series live
prediction. Same-day games do not see each other either.

**About 1.1% of games belong to series that cross UTC midnight.** Their later games fall on
the next UTC day, so in training they see the earlier games of the series; a live prediction
made before the series cannot.

**Rosters.** This is the one structural difference between training and live:

- Training uses the 5 players who actually played the game (`team_games` row of the
  `game_id`). Lineups are usually known before the match, so this leaks little, but a
  same-day substitution is visible only in training.
- Live uses the team's latest lineup before the day (its most recent `team_games` row), with
  ELO from `player_elo_current.parquet`. Same-day substitutions are invisible.

**An empty starter slot** (missing `playerid`) is handled differently:

- Training leaves it out of the team means (the lane diffs for that role are NaN).
- Live fills it, and any player unknown to the ELO state, with the init value: `outcome_elo`
  is the mean of the current players last seen in the team's home league if at least
  `MIN_LEAGUE_PLAYERS` (10) exist, else that league's `LEAGUE_PRIOR`; every skill is 1000.

Everything else reads the same `team_games` and ELO state in both modes, so there is no
other train/serve gap. `team_games_*.csv` is committed state written by `daily_state.py`
(see [data.md](data.md)).

## Dataset (`build_dataset.py`)

One row per OE game from `START_DATE = "2017-01-01"` onward (ELO warms up on 2014-2016),
across **all leagues**: more data, and players from minor leagues feed the majors. About
93,800 games; games where either side has no `teamid` are skipped. Output:
`feature_matrix.parquet` (not committed), meta columns `game_id`, `game_date`, `league`,
`blue_team_id`, `red_team_id`, `blue_win` plus the 136 features. Rebuild time is roughly a
minute.

## Training (`train.py`)

1. **Chronological split**, no shuffling: train (first 70%) -> calibration (next 15%) ->
   test (most recent 15%).
2. **Optuna** search (default 50 trials, `--trials N`) on the **train slice only**, scored by
   the mean **log loss** over a 5-fold `TimeSeriesSplit` (`CV_FOLDS = 5`). The search space is
   `n_estimators` 300-1500, `max_depth` 3-8, `learning_rate` 0.01-0.2 (log), `subsample` 0.6-1,
   `colsample_bytree` 0.5-1, `min_child_weight` 1-10, `gamma` 0-2, `reg_alpha` and
   `reg_lambda` 1e-4-10 (log). The calibration and test slices are not seen by the search.
3. **Final XGBoost model** fit on the train slice with the best params (`hist`,
   `random_state=42`, no early stopping).
4. **Platt calibration**: a `LogisticRegression(C=1.0)` fit on the base model's raw
   probabilities over the calibration slice.
5. **Evaluation** on the test slice: accuracy, AUC, Brier, log loss, a 10-bin calibration
   curve, for all leagues and for the **major** slice (`MAJOR`: LCK, LPL, LEC, LCS, CBLOL, LCP,
   LTA N, LTA S, LTA, WLDs, MSI, FST), each against two baselines:
   - always blue (p = the train slice's blue rate),
   - the team with the higher mean `outcome_elo` (`p = 1 / (1 + 10^(-diff/400))`, NaN diff
     counted as 0).
6. **Gate.** `GATE: PASS` only if, on the major slice, the calibrated model has lower log loss
   **and** higher accuracy than the outcome-ELO baseline. On FAIL, stop and report before
   shipping a model.
7. Saves `models/xgb_model.pkl` (`{base_model, platt, feature_names}`),
   `data/processed/train_metrics.json` and `data/processed/feature_importance.csv`.

## From game probability to series probability (`predict.py`)

Sides are unknown before a match, so the model is run twice and averaged. With team1 =
`teams[0]`:

```
p1     = P(team1 wins a game on blue)      features: team1 blue, team2 red
p2     = 1 - P(team2 wins a game on blue)  features: team2 blue, team1 red
p_game = (p1 + p2) / 2                      side-neutral
series = series_win_prob(p_game, best_of)   first to ceil(bo/2) wins
```

Closed forms: Bo1 `p`, Bo3 `p^2 (3 - 2p)`, Bo5 `p^3 (10 - 15p + 6p^2)`. This assumes
**independent, identically distributed games**: no momentum, no side alternation, no
adaptation between games. Compounding a per-game `p` amplifies small calibration errors,
most visibly in Bo5s. Supported formats are Bo1, Bo3 and Bo5. The pick is team1 if the series
probability is at least 0.5; `confidence` is the larger of the two series probabilities.

**SHAP.** Per-feature contributions come from XGBoost's own `pred_contribs` on the base
model (TreeSHAP; the same values as the `shap` package, much faster to import). They are in
log-odds **before** the Platt step, for the team1-on-blue orientation, so a positive value
pushes toward team1. The top 10 by absolute value are stored in `shap_values`.

**Live context.** The lolesports league slug maps to an OE league code. `playoffs` is 1 for a
domestic match whose `blockName` matches `playoff|final|knockout|play.?in` and always 0 for
internationals.

## Results

Test slice: games from **2025-05-20 to 2026-10-14**, the most recent 15%. Nothing was tuned on
it (see [evaluation protocol](#evaluation-protocol)).

### Series accuracy (the metric that matters)

Series are games grouped by day and team pair; the best-of is inferred from the winner's game
count. The series probability is `series_win_prob` of the side-neutral `p`. "Phase 3 model" is
the model before the improvement round (plain outcome ELO). Baseline: the team with the higher
outcome ELO wins.

| | Model | Phase 3 model | Baseline: higher outcome ELO |
|---|---|---|---|
| All leagues (6,260 series) | **70.2%** | 70.4% | 69.8% |
| All: Bo1 / Bo3 / Bo5 | 67.4 / 72.4 / 68.7 | 67.5 / 72.7 / 68.7 | 67.2 / 71.9 / 68.6 |
| Major leagues (1,453 series) | **69.9%** | 69.3% | 70.0% |
| Major: Bo1 (146) / Bo3 (904) / Bo5 (403) | 61.6 / 72.7 / 66.5 | 59.6 / 72.8 / 65.0 | 61.6 / 72.2 / 68.0 |

### Game level

| | Accuracy | Log loss |
|---|---|---|
| All leagues: model | 65.3% | 0.618 |
| All leagues: outcome-ELO baseline | 64.9% | 0.626 |
| All leagues: always blue | 53.4% | - |
| Major leagues: model | 63.8% | 0.639 |
| Major leagues: outcome-ELO baseline | 63.2% | 0.641 |

The gate reads PASS.

### Confidence intervals (major leagues, paired bootstrap, 95%)

- Series log loss vs the Phase 3 model: -0.006, CI [-0.011, -0.001]. Significantly better.
- Series accuracy vs the Phase 3 model: +0.55 pp, CI [-0.55, +1.65]. Not significant.
- Against the plain outcome-ELO baseline the model is **level on major-league series**.
  Across all leagues it is significantly better in log loss.

The site's "About the model" text hardcodes 70.2% series (6,260), 69.9% major (1,453), 65.3%
game accuracy, log loss 0.618 and the 69.8% baseline (`web/app/(main)/page.tsx`); it does not
read `train_metrics.json`. Update it after any retrain. The site's `running_record`
(counted from `RECORD_START`, see [operations.md](operations.md)) is the live number.

### Honest assessment

- Per single game, before the draft, about 64-65% is close to the ceiling with this data. The
  draft is unknown and many matchups are coin flips; 70% per game is not realistic.
- At series level about 70% is where the model is, and it is what the public record counts.
- On the major leagues almost all predictive power is in the outcome ELO. The XGBoost layer adds
  little there, though it helps across all leagues.
- In major-league Bo5s the series probabilities are slightly worse calibrated than the plain
  ELO's: series log loss 0.637 against 0.620. It affects the displayed probability, not who is
  picked. An optional future fix is to shrink `p` before compounding, chosen on validation.

### What was tried without gain

Tried in the improvement round and not adopted (no measurable gain on the major leagues):

- K values (16/24/32/48) and an international K multiplier
- a team-level Elo
- strength-of-schedule-adjusted form, residual form, patch form
- lineup activity
- a player margin rating
- champion meta-share and champion pool breadth (would also need extra committed state)
- objective-control rates
- MAJOR or recency sample weights
- top-k feature pruning (worse)
- logistic regression, and an XGBoost + logistic regression ensemble
- separate calibration for the major leagues

What was adopted: the refined outcome ELO (new-year pull `REGRESS`, margin multiplier,
`NEW_K_MULT`; see [elo.md](elo.md)) and Optuna on the train slice minimising log loss.

### Evaluation protocol

- Choices are made on validation only: walk-forward validation over 3 windows between
  2021-07 and 2025-05 (all before the test slice), with fixed hyperparameters, 3 seeds and a
  paired bootstrap.
- The test slice was looked at **twice**. The second look followed a correctness fix (a
  look-ahead in the new-year pull, found in review) with no tuning on test; the validation
  gain was unchanged by the fix.
- Reports lead with series accuracy (all leagues and major leagues, by Bo1/Bo3/Bo5), then
  game-level log loss.

## Known limitations

- **No draft.** The model is pre-draft; the draft swings games and is invisible to it. The
  ceiling is well below what draft-aware or live models reach.
- **Bo5 calibration.** See the assessment above.
- **Cross-region strength** rests on international games (a few hundred per year), player
  transfers and hand-set league priors.
- **Small samples and roster churn.** 100-200 games per split per major league. Rosters change
  every off-season, so early-season predictions are the weakest.
- **Live roster = last-known starters.** Same-day substitutions are invisible; training saw the
  real lineup, so live accuracy can differ from the backtest.
- **Playoffs flag.** Live `playoffs` comes from the lolesports `blockName`, training from OE's
  flag, and the two disagree in some leagues: "Regional Qualifier" is 1 in OE for LCK and LCP,
  "Play-Ins" is 0 in OE for CBLOL and LCP. The effect on the side-neutral `p` is small.
- **Data lag.** OE refreshes around 07:05 UTC, so a team playing on consecutive days is
  predicted without the previous day, and LCK/LPL matches between about 05:00 and 09:00 UTC are
  predicted on state missing the previous day. LPL rows can also lag and lack early-game stats.
- **Forfeits** (0-0 `gameWins`) never resolve automatically; see
  [operations.md](operations.md).
- **iid series maths** and the day-level UTC-midnight effect described above.

## Retraining

Raw OE data (not in git, see [data.md](data.md)) must be in `src/data/raw/oe/`.

```bash
python src/pipeline/elo.py --force                  # always first: build_dataset reads the uncommitted player_elo.parquet
python src/pipeline/daily_state.py --init           # only if committing the ELO state: delete team_games_*.csv first
python src/pipeline/build_dataset.py --force        # rebuild feature_matrix.parquet
python src/pipeline/train.py --trials 100 --force   # retrain; check the GATE line
```

Always rebuild the ELO before `build_dataset.py`: a `player_elo.parquet` older than the local
OE files makes later games fall back to the current ELO (a silent look-ahead). Commit the ELO
state only together with a fresh `--init` from the same OE files; otherwise `team_games`
desyncs and the missing games are never picked up (`daily_state` only passes gameids that are
new to `team_games`). `elo.py --force` also recomputes `role_norms.json`.

Then commit `src/models/xgb_model.pkl` and update the hardcoded results in
`web/app/(main)/page.tsx`, `README.md`, the Results section above and the model numbers in
`HANDOFF.md`. Without `--force`, `train.py` loads the existing model.

Adding a feature: write a `get_*_features()` in `features.py`, add it to the group list in
`build_feature_matrix()`, retrain, and extend `FEATURE_LABELS` in `web/components/GameCard.tsx`. A feature
that exists only in one of training or live reads as missing in the other.

`requirements.txt` pins `numpy==2.1.3`, `pandas==2.2.3`, `scikit-learn==1.8.0` and
`xgboost==3.2.0`. The committed pickle loads with exactly these versions (checked in the
project venv). Do not bump them without retraining, or loading the pickle can warn or break;
re-pin to whatever versions you train with and keep the header comment in `requirements.txt`
current.
