# Model

Binary XGBoost classifier predicting home-team win probability, Platt-calibrated.
Implementation: `src/pipeline/train.py` (training), `src/pipeline/build_dataset.py`
(feature matrix), `src/pipeline/features.py` (feature engineering),
`src/pipeline/predict.py` (inference). See [architecture.md](architecture.md) for
where this fits in the pipeline and [elo.md](elo.md) for the ELO feature group.

## Features

`build_feature_matrix()` in `features.py` returns a flat `dict[str, float]`,
~100 keys per game, prefixed `home_`/`away_` where applicable, with `diff_*`
or `*_diff` differential columns for the key metrics. Nine feature groups:

| Group | Function | Features |
|---|---|---|
| ELO | `get_elo_features` | `home_/away_/elo_diff_` × 7 skills + general (24 features) — see [elo.md](elo.md) |
| Rolling team stats | `get_team_rolling_stats` | last-5 / last-10 pts, opp pts, net rating, win% + last-5 AST/REB/TOV, plus home−away diffs |
| Home/away splits | `get_home_away_splits` | season-to-date win% / pts / opp pts, split by home vs away |
| Rest & schedule | `get_rest_features` | days rest, back-to-back flag, games in last 7 days, rest/b2b diffs |
| Head-to-head | `get_head_to_head_features` | this-season H2H wins/losses/win% between the two teams |
| Player availability | `get_player_availability_features` | star player out, missing PPG, availability score (0-1) — see caveat below |
| Efficiency differential | `get_efficiency_differential_features` | last-10 offense-vs-defense matchup diffs |
| Player form | `get_player_form_features` | ELO trajectory (general ELO change) over the top-3 PPG players' last 10 games |
| Streaks | `get_streak_features` | current win/loss streak, last-5 win% |

Each group is wrapped in its own try/except in `build_feature_matrix()` — a
failure in one group logs a warning and leaves those keys out rather than
failing the whole prediction (missing keys read as `0.0` downstream via
`feature_dict.get(f, 0.0)` in `predict.py`).

Head-to-head and home/away split features use **all games since 2015-16**,
not just the current season — this is the same in training and live, so it
isn't a source of train/live mismatch.

## Training procedure (`train.py`)

1. **Chronological split**, no shuffling — the model must never train on
   future games relative to what it's tested on:
   `train (70%) → calibration (15%) → test (15%, most recent games)`.
2. **Optuna hyperparameter search** (default 50 trials, `--trials N` to
   change) over the train+calibration portion (85%), scored by mean AUC
   across a 5-fold `TimeSeriesSplit` (`CV_FOLDS = 5`).
3. **Final XGBoost model** retrained on the train-only portion (70%) with the
   best params found.
4. **Platt calibration** — a `LogisticRegression` fit on the base model's raw
   probabilities over the held-out calibration slice (15%), so `predict.py`'s
   probabilities are well-calibrated, not just well-ranked.
5. **Evaluation** on the final 15% test slice (accuracy, AUC-ROC, Brier score,
   log loss, a 10-bin calibration curve).
6. Saves `models/xgb_model.pkl` (`{base_model, platt, feature_names}`),
   `data/processed/train_metrics.json`, and
   `data/processed/feature_importance.csv`.

Run with `python src/pipeline/train.py` (`--trials N`, `--force` to retrain
over an existing model).

## Reported metrics

From the site's "About the model" section (`web/app/(main)/page.tsx`, hardcoded):
on 2,116 held-out test games (chronologically most recent, from a 14,108-game
training run) —

- **Accuracy: 67.5%**
- **AUC-ROC: 0.729**
- **Brier score: 0.209** (lower is better; 0.25 = coin-flip)
- **Log loss: 0.606** (random guessing = 0.693)

Training data spans 2015-16 through 2025-26 (14,108 games at the point these
numbers were produced: ~9,875 train / 2,116 calibration / 2,116 test).

**These numbers are optimistic relative to what the live pipeline can
actually deliver — see the caveats below before quoting them as "current
accuracy."** The site's own `running_record` table (tracked from
`RECORD_START`, see [operations.md](operations.md)) is the honest, live number.

## Honest caveats

**Training features leak information live predictions don't have.**

- **ELO lineups**: for a historical/training game, `get_elo_features()` uses
  the exact players who played ≥10 minutes *in that specific game*
  (`player_elo.parquet`, snapshotted from the real box score). For a live
  prediction, there is no future box score — the roster comes from
  `player_elo_current.parquet` (whoever's *most recent* appearance was for
  that team). This is a reasonable approximation but isn't the same as
  knowing the actual game-day rotation.
- **PPG weights and star-out detection**: both modes use `player_stats_df`
  (season-to-date PPG) to weight players and flag a 20+ PPG player as "star
  out." In training, `build_dataset.py` passes the **full final-season**
  stats file for every game in that season — including games that happened
  *after* the game being featured. That's look-ahead bias. In live mode the
  season file only contains games actually played so far, so it's a strictly
  weaker (but honest) signal.
- **Player availability: real in training, neutral live.** The
  **committed model** was trained on the original dataset (`collect.py`,
  which does fetch `boxscore_summary`), so `star_player_out` / `missing_ppg` /
  `availability_score` carried real signal during training. There's no live
  injury feed, though, so these features are always neutral (0 / 0.0 / 1.0)
  at inference time — a genuine train/serve mismatch for this feature group
  specifically, separate from the leakage points above. (The *re-downloaded*
  bootstrap dataset used for the incident recovery doesn't have
  `boxscore_summary` at all — see [data.md](data.md) — so this would also
  need fixing before any future retrain on that data, or the new training set
  would lose this signal entirely rather than just facing a serve-time gap.)
- **The Optuna search saw the calibration slice.** The hyperparameter search
  in step 2 above cross-validates over train+calibration combined (85%),
  so the reported test metrics (from the untouched final 15%) are clean, but
  the calibration slice itself wasn't fully held out from model selection.

**Daily ELO updates are weaker than the historical bootstrap.** The CDN box
score used for live ELO updates has no `defensiveRating` or tracking data, so
(per [elo.md](elo.md)) the defense skill degrades to steals+blocks only and
the hustle skill freezes for every game processed after the bootstrap. Skill
ELOs computed from the historical stats.nba.com data remain unaffected.

**Net effect:** expect live accuracy measurably below the 67.5% backtest
number. The `running_record` table is the number to trust for "how is it
actually doing."

## Recommended next step

Retrain with live-equivalent features — i.e. change `build_dataset.py` /
`features.py` so the training pass mirrors what live prediction actually
sees (roster from ELO-state appearances rather than the game's real box
score, and a point-in-time season-stats cutoff rather than the full season).
This requires the bootstrapped raw data plus rerunning `build_dataset.py`
and `train.py`. Note `scikit-learn==1.8.0` is pinned in `requirements.txt`
because the committed pickle was saved with it — bump it only alongside a
retrain, or `pickle.load` will raise `InconsistentVersionWarning` (or worse).
