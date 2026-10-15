# Data reference

Every file the pipeline reads or writes, what's in it, and which columns the
code actually relies on. See [architecture.md](architecture.md) for how these
files flow between pipeline stages, and [operations.md](operations.md) for the
bootstrap procedure that produces the historical ones.

## Directory layout (`src/data/`)

```
src/data/
  raw/         downloaded box scores, game logs, season stats (.gitignore'd except the whitelist below)
  processed/   ELO parquet files, training feature matrix (.gitignore'd except the whitelist below)
```

**Never commit raw or processed data manually.** `.gitignore` ignores
`src/data/raw/*` and `src/data/processed/*` wholesale, then re-includes
(`!`) only the four committed "live state" files listed below. Everything
else regenerates from a bootstrap or a daily run.

## `src/data/raw/` — downloaded / appended inputs

| File(s) | Committed? | Written by | Columns the code reads |
|---|---|---|---|
| `game_log_regular_{season}.csv`, `game_log_playoffs_{season}.csv` | Yes | `bootstrap_data.py` (initial), `daily_state.py` (daily append) | `GAME_ID`, `GAME_DATE`, `TEAM_ID`, `TEAM_ABBREVIATION`, `MATCHUP`, `WL`, `PTS`, `PLUS_MINUS`, `AST`, `REB`, `TOV` |
| `player_season_stats_{season}.csv` | Yes | `bootstrap_data.py` (initial), `daily_state.py` (daily update) | `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID`, `GP`, `PTS` (bootstrap files carry the full LeagueDashPlayerStats column set; `daily_state.py` fills these) |
| `boxscore_traditional_{game_id}.json` | No | `bootstrap_data.py` / `collect.py` | `PlayerStats` dataset: `personId`, `minutes`, `points`, `assists`, `steals`, `blocks`, `reboundsTotal`, `turnovers`, `threePointersMade/Attempted/Percentage`, `teamId` |
| `boxscore_advanced_{game_id}.json` | No | same | `PlayerStats`: `personId`, `defensiveRating`, `trueShootingPercentage`, `usagePercentage`, `assistToTurnover`, `reboundPercentage`, `PIE` |
| `boxscore_tracking_{game_id}.json` | No | same | `PlayerStats`: `personId`, `speed`, `distance`, `touches` (absent pre-2016 or if the endpoint returned nothing — hustle skill is skipped for that game) |
| `boxscore_summary_{game_id}.json` | No | `collect.py` only (**not** fetched by `bootstrap_data.py`) | `InactivePlayers`: `personId`, `teamId` |
| `bootstrap.log`, `missing_boxscores.txt` | No | `bootstrap_data.py` | operational logs, not data |

`MATCHUP` encodes home/away: `"XXX vs. YYY"` = home team row, `"XXX @ YYY"` =
away team row. This is how `build_dataset.py`, `predict.py`, and
`features.py`'s `split_home_away()` all determine which side a row is on —
there's no separate `is_home` column anywhere.

**Gotcha (matters for a future retrain, not for the committed model):** the
**committed** `models/xgb_model.pkl` was trained on the *original* dataset,
collected with `collect.py` — which does fetch `boxscore_summary` (it's in
`BOXSCORE_ENDPOINTS_CORE`) — so that training run had real inactive-player
data and the model learned genuine availability effects. The *re-downloaded*
bootstrap dataset (`bootstrap_data.py`, built after the data-loss incident)
only fetches `traditional`, `advanced`, and `tracking` — **no**
`boxscore_summary_*.json` files exist for it. `features.py`'s
`_load_inactive_ids()` returns an empty set when the file is missing, so if
someone runs `build_dataset.py` on the bootstrapped data as-is, the new
training set's availability features (`star_player_out`, `missing_ppg`,
`availability_score`) would come out neutral, unlike the committed model's
training data. Fix before retraining: add `"summary"` to `bootstrap_data.py`'s
`_WANTED` endpoints and re-run phase 2 for it (~14k extra calls, ≈2.5h).

Separately, and true regardless of any of this: **live predictions have no
injury feed**, so availability features are always neutral at inference time.
Since the committed model *did* learn from real availability data, this is a
genuine train/serve mismatch worth keeping in mind when reading a live
prediction's SHAP breakdown — the model may be underweighting or misreading
matchups where availability would normally matter.

`collect.py` (the original, full collector — as opposed to `bootstrap_data.py`,
which was written later to re-fetch only what `elo.py` needs) also pulls
several endpoints that **no pipeline code currently reads**: `player_matchups_*.csv`,
`hustle_stats_*.csv`, `synergy_*_*.csv`, `pt_stats_*_*.csv`, `pt_defend_*.csv`,
`lineups_5man_*.csv`, `player_info_{id}.json`, and the `fourfactors` / `misc` /
`matchups` / `hustle` box score types. They exist as raw material for future
feature work, not as something `elo.py` or `features.py` depends on today.

## `src/data/processed/` — pipeline outputs

| File | Committed? | Written by | Purpose |
|---|---|---|---|
| `player_elo.parquet` | **No** | `elo.py` (`build_elo`) | Full pre-game ELO history, one row per player per game. Used by `features.py` for historical/training features. |
| `player_elo_current.parquet` | **Yes** | `elo.py` (`build_elo`, `update_elo`, `update_team_assignments`) | Post-game ELO per player + `player_name` + `team_id` + `last_game_date`. Live roster/ELO source. |
| `player_elo_recent.parquet` | **Yes** | `elo.py` (`build_elo`, `update_elo`) | Last 11 pre-game ELO snapshots per player (incl. `player_name`). Feeds the live "form" feature. |
| `feature_matrix.parquet` | No | `build_dataset.py` | One row per historical game, ~100 feature columns + `home_win` label. Training input. |
| `train_metrics.json`, `feature_importance.csv` | No | `train.py` | Training run metrics / ranked feature importance. |
| `shap_beeswarm.png`, `shap_summary.json` | No | `shap_export.py` | Global SHAP plots for a possible model-explainability subpage. **Not currently invoked by any script** (`train.py` doesn't call it, and `predict.py` computes its own top-10 SHAP inline via `_shap_top_features()` rather than `shap_export.export_shap_for_prediction()`). Would need to be run manually — see the old snippet this replaced in README history. |

Full column list for `player_elo*.parquet`: `game_id`, `game_date`,
`player_id`, `player_name`, `pre_general_elo` / `general_elo`, and
`pre_{skill}_elo` / `{skill}_elo` for each of the 7 skills in
[elo.md](elo.md). `player_elo_current.parquet` additionally has `team_id` and
`last_game_date` (its columns are `player_id`, `player_name`, `team_id`,
`last_game_date`, `general_elo`, `{skill}_elo`×7). `player_name` is a
human-readable "First Last" for eyeballing the data; it carries no weight in the
model.

## `src/models/`

`xgb_model.pkl` — **committed** (the `.gitignore` ignores `src/models/*.pkl`
then re-includes this one file). A pickled dict: `{"base_model": XGBClassifier,
"platt": LogisticRegression, "feature_names": [...]}`. Saved with
scikit-learn 1.8.0 and xgboost 3.2.0 — see [model.md](model.md) for the
training procedure and `requirements.txt` for why the scientific stack is
pinned.

## Supabase schema

Defined in `src/supabase/schema.sql` (fresh setup) and
`src/supabase/migrations/001_game_time_and_rls.sql` (brings an existing,
pre-migration database up to the same shape — see
[operations.md](operations.md) for when to apply it).

**`predictions`** — one row per game per day, keyed by unique `game_id`.

| Column | Notes |
|---|---|
| `id` | UUID PK |
| `game_id` | unique, 10-digit NBA game ID |
| `game_date`, `game_time_utc` | ET calendar date; scheduled tip-off in UTC (from the CDN schedule) |
| `home_team`, `away_team`, `home_team_id`, `away_team_id` | abbreviation + NBA team ID |
| `predicted_winner` (`home`\|`away`), `predicted_team` (abbreviation) | |
| `home_win_prob`, `away_win_prob`, `confidence` | Platt-calibrated probabilities |
| `actual_winner`, `correct` | `NULL` until `evaluate.py` resolves the game |
| `created_at` | |

**`shap_values`** — one row per feature per prediction (top 10, replaced
delete-then-insert on every re-prediction): `prediction_id` (FK, cascade
delete), `feature_name`, `shap_value`, `feature_value`.

**`running_record`** — single row (`id = 1`, enforced by the primary key
default): `total_correct`, `total_incorrect`, `accuracy`, `last_updated`.
Recomputed by `evaluate.py` on every run over predictions with
`game_date >= RECORD_START` (see [model.md](model.md) and
[operations.md](operations.md) for what `RECORD_START` excludes and why).

**`model_metadata`** — one row per trained model version
(`model_version`, `trained_on_date`, `seasons_covered`, `accuracy_on_test`,
`auc_roc`, `brier_score`, `feature_list`, `best_params`, `notes`). Defined in
the schema but **nothing currently writes to it** — `train.py` saves metrics
to `train_metrics.json` locally, not to this table. Wire this up if you want
model-version history in Supabase.

**Row Level Security:** all four tables have RLS enabled with a public
`SELECT`-only policy (`USING (true)`, no insert/update/delete policy). The
pipeline connects with the Supabase **service-role key**, which bypasses RLS
entirely; the anon key used by the website is restricted to reads by these
policies alone.

## NBA data sources

Two completely different NBA data sources are used, for two different jobs:

1. **stats.nba.com** (via the `nba_api` package) — used only for the
   one-time historical bootstrap (`bootstrap_data.py` / `collect.py`).
   Provides the advanced and tracking box scores (`defensiveRating`,
   `trueShootingPercentage`, `PIE`, `speed`/`distance`/`touches`, etc.) that
   the CDN does not have. Works with `nba_api`'s default browser-like headers
   from the owner's home PC, but **times out from GitHub-hosted runners**
   (confirmed), so the bootstrap is a local/overnight job — never CI.

2. **cdn.nba.com** (via `src/pipeline/nba_cdn.py`, plain `requests` with the
   browser-style header constant `CDN_HEADERS`; `nba_api`'s live header set is
   outdated and gets a 403) — used by the daily
   pipeline (`daily_state.py`, `predict.py`, `evaluate.py`). Two endpoints:
   - `scheduleLeagueV2.json` — the full season schedule (`fetch_schedule()` /
     `games_for_date()`).
   - `liveData/boxscore/boxscore_{game_id}.json` — one game's live box score
     (`fetch_boxscore()`), basic counting stats only. Exists from the
     **2019-20 season onward**; older game IDs return 403/404.

   **Confirmed: cdn.nba.com also returns 403 from GitHub-hosted runners** even
   with `CDN_HEADERS` (NBA blacklists cloud IP ranges), while it works direct
   from a home PC. To keep NBA data, the daily jobs route these CDN calls
   through static residential proxies set in the `NBA_PROXIES` secret
   (comma-separated `http(s)://user:pass@host:port`; URL-encode special
   characters in the password). `nba_cdn` reads `NBA_PROXIES`, validates each
   entry, fails over across them sticky to the last that worked, and on
   all-fail raises a credential-free `proxy #i (host:port): <reason>` summary.
   Unset/empty → direct connection (local dev, unchanged). Check reachability
   with `python src/pipeline/nba_cdn.py --check` or the `check_nba_proxies.yml`
   workflow — see [operations.md](operations.md).

### Game-ID prefixes

The first 3 of the 10 digits in an NBA game ID encode the game type
(`nba_cdn.game_prefix()`):

| Prefix | Type | Stateful (updates ELO / can be predicted)? |
|---|---|---|
| `001` | Preseason | No — team-assignment refresh only (`update_team_assignments`) |
| `002` | Regular season | Yes |
| `003` | All-Star | No — ignored entirely |
| `004` | Playoffs | Yes |
| `005` | Play-in | Yes |
| `006` | NBA Cup final | Yes |

`nba_cdn.is_stateful()` returns true for `{002, 004, 005, 006}` — these are
the games `daily_state.py` folds into the ELO state and `predict.py` /
`evaluate.py` act on.
