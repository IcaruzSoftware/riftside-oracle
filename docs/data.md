# Data reference

Every file and table the pipeline reads or writes. See
[architecture.md](architecture.md) for how they flow between stages and
[operations.md](operations.md) for the procedures that produce them.

## Directory layout (`src/data/`)

```
src/data/
  raw/
    oe/            Oracle's Elixir year CSVs (gitignored, ~856 MB)
    team_games_*.csv   committed live state (one file per year)
  processed/       ELO parquet files, role norms, training matrix (gitignored except the whitelist below)
```

**Never commit raw or processed data manually.** `.gitignore` ignores
`src/data/raw/*` and `src/data/processed/*` wholesale (the *contents*, not the
directories: git cannot re-include a file whose parent directory is itself
ignored, so a `dir/` pattern would make the `!` negations silently do
nothing), then re-includes only the files listed under
[Committed live state](#committed-live-state). Never add a new generated file
under `src/data/` without adding its negation. The daily workflow runs
`git add src/data/`, so anything not whitelisted stays out.

## Committed live state

These files plus the model are everything the daily runner and the tests need
beyond the code. Without the raw OE files the tests and live predictions still
work; rebuilding history or retraining needs the raw data.

| File | Written by | Purpose |
|---|---|---|
| `src/data/raw/team_games_{year}.csv` (2014 to 2026) | `daily_state.py` (`--init` once, then the daily append) | Team game log: 2 rows per game, one per team. About 206k rows in total (~79 MB, largest file ~10 MB). Do not hand-edit. |
| `src/data/processed/player_elo_current.parquet` | `elo.py` (`build_elo`, `update_elo`) | Post-game ELO per player. Live roster and ELO source. |
| `src/data/processed/player_elo_recent.parquet` | same | Last 11 pre-game snapshots per player. Live "form" source, and the idempotence guard of `update_elo`. Committed in full every day, no pruning. |
| `src/data/processed/role_norms.json` | `elo.py` (`build_elo`, first run or `--force`) | Per position × stat mean/std for role normalisation. **Frozen**: `update_elo` only reads it, which keeps incremental updates equal to a full rebuild. |
| `src/models/xgb_model.pkl` | `train.py` | The calibrated model (see [Model file](#model-file)). |

### `team_games_{year}.csv`

The team-row subset of OE, in `oe.TEAM_GAMES_COLS` order. The file year is the
year of the game's `date`, which can differ from OE's `year` column.

| Column | Notes |
|---|---|
| `gameid`, `date`, `league`, `year`, `split`, `playoffs`, `game`, `patch` | OE fields. `date` is `YYYY-MM-DD HH:MM:SS`, naive, treated as UTC. `game` is the game number in the series. |
| `side` | `Blue` or `Red`. The only home/away signal; there is no separate flag. |
| `teamid`, `teamname` | OE ids (`oe:team:<hex>`). `teamid` can be empty (0.3 to 3% of rows per year, 0% in the major leagues). |
| `result` | 1 = this team won. |
| `gamelength` | seconds |
| `teamkills`, `teamdeaths` | |
| `golddiffat15`, `xpdiffat15`, `csdiffat15` | empty in `partial` games |
| `firstblood`, `firstdragon`, `firstherald`, `firsttower`, `firstbaron` | 0/1, empty when OE lacks them |
| `dragons`, `barons`, `towers`, `opp_towers` | |
| `starters` | The side's 5 `playerid`s in role order `top\|jng\|mid\|bot\|sup`, `\|`-joined. Always 5 slots; a missing id is an empty string, so a slot's position is its role (`oe:player:a\|\|oe:player:c\|oe:player:d\|oe:player:e` has no jungler). This is how live mode knows a team's last lineup without the raw files. |

Only complete games are written: 12 rows, 5 roles per side, results present
(`daily_state.complete_gameids`). A few minor-league games are in
`team_games` but not in the ELO because none of their players has a
`playerid`.

### `player_elo_current.parquet`

One row per `playerid` (`elo.CURRENT_COLS`): `player_id`, `player_name`,
`team_id`, `position`, `league`, `last_game_date`, `games_played`,
`outcome_elo`, `general_elo`, and `combat_elo`, `damage_elo`, `economy_elo`,
`laning_elo`, `vision_elo`. `team_id`, `position`, `league` and `player_name`
follow the player's latest game. `general_elo` is the sum of the five skills.

### `player_elo_recent.parquet`

The last `RECENT_N = 11` pre-game snapshots per player, in the
`elo.HISTORY_COLS` shape: `game_id`, `game_date`, `player_id`, `player_name`,
`team_id`, `position`, `league`, `side`, `pre_outcome_elo`, `pre_general_elo`,
`pre_{combat,damage,economy,laning,vision}_elo`.

### `role_norms.json`

`{position: {stat: {"mean", "std"}}}` for the five positions and the 14
component stats of the skills, computed over the `complete` player rows of the
years 2022 to 2025 (`elo.NORM_YEARS`).

### Model file

`src/models/xgb_model.pkl` is committed (`.gitignore` ignores
`src/models/*.pkl`, then re-includes this one file). A pickled dict:
`{"base_model": XGBClassifier, "platt": LogisticRegression, "feature_names":
[...]}`. It was saved with scikit-learn 1.8.0 and xgboost 3.2.0; both are
pinned in `requirements.txt`, so do not bump scikit-learn without retraining.
Training and features: [model.md](model.md).

## Not committed

| File | Written by | Why not |
|---|---|---|
| `src/data/raw/oe/*_LoL_esports_match_data_from_OraclesElixir.csv` | `bootstrap_data.py`, `daily_state.py` (current year, as a cache) | 13 files (2014 to 2026), ~856 MB; re-downloadable. |
| `src/data/raw/oe/*.part` | `oe.download_year` | Temporary file during a download. |
| `src/data/processed/player_elo.parquet` | `elo.py` | Full pre-game ELO history, one row per rated player per game (`HISTORY_COLS`). Training input for the ELO features. Rebuilt in about 21 s. |
| `src/data/processed/feature_matrix.parquet` | `build_dataset.py` | One row per game since 2017 (93,753 at the last build), `META_COLS` + the feature columns. Training input. |
| `src/data/processed/train_metrics.json`, `feature_importance.csv` | `train.py` | Training run metrics and ranked feature importance. |

## OE: Oracle's Elixir

Free yearly CSVs by Tim Sevenhuysen, 2014 to 2026, in a public Google Drive
folder (id `1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH`, `oe.FOLDER_ID`). OE refreshes
once a day, around 07:05 UTC, with about a day of lag. The game statistics are
Riot's property; the site credits OE and carries the Riot disclaimer.

Files are named `{year}_LoL_esports_match_data_from_OraclesElixir.csv` and
land in `src/data/raw/oe/`. A full scripted download is roughly 856 MB; the
current-year file is ~78 MB. `oe.load_year()` keeps 43 of the ~165 columns
(`oe.NEEDED_COLS`), reindexed so a column absent in an older year is NaN.

### Shape and quirks

- **12 rows per game**: 10 player rows (`participantid` 1 to 10, `position` in
  `top, jng, mid, bot, sup`) and 2 team rows (`participantid` 100/200,
  `position == "team"`). `side` is `Blue` or `Red`, `result` 0 or 1,
  `gamelength` is in seconds. This held across a full audit of all 13 files.
- **Ids are OE's own**: `playerid` is `oe:player:<hex>` (stable 2014 to 2026),
  `teamid` is `oe:team:<hex>`. `gameid` is the Riot platform id (for example
  `LOLTMNT05_228413`) or `13057-13057_game_1` for LPL.
- **Missing ids**: 0.3 to 1.9% of rows per year have no `playerid`, 0.3 to 3%
  no `teamid`; both are 0% in the major leagues. Rows without a `playerid`
  are skipped by the ELO (no rating, no snapshot; the rest of the game is
  processed). `playername == "MISSING"` can occur: always use the id, never
  the name.
- **`datacompleteness`** is `complete` or `partial`. `partial` games lack the
  `*at10/15/20/25` early-game stats (so the `laning` skill is skipped for that
  game). Share of partial games: 17 to 20% in 2016 to 2018, 1.5% in 2026; in
  the major leagues 30% in 2025 (LPL, until about January 2026) and 4.5% in
  2026.
- **Dates are not sorted within a file; always sort.** The `year` column can
  say 2027 inside the 2026 file; use `date`.
- **Consistency**: 3 games have both sides marked as losers and 12 have the
  same `teamid` on both sides; both are handled. 12 `(gameid, playerid)`
  pairs are the same id in two slots and are de-duplicated in the ELO replay.
- **Coverage**: 2014 has only NA, EU and Worlds; Korea starts in 2015 and
  China in 2016.
- **League codes change over time** (`league` is OE's short code; build sets
  from the data, do not assume):
  - `OGN` (2015) became `LCK`.
  - `NA LCS` became `LCS` (2019), then `LTA N` (2025), then `LCS` again (2026).
  - `EU LCS` became `LEC`.
  - `CBLOL` became `LTA S` (2025), then `CBLOL` again.
  - `LMS` became `PCS` (2020); `LCP` exists since 2025.
  - Internationals: `WLDs`, `MSI`, `FST`, `EWC`, `MSC` (`elo.INTERNATIONAL`).
  - Others seen: `EM` (EMEA Masters), `DCup`.
  The code maps these in `elo.LEAGUE_PRIOR` / `elo.INTERNATIONAL`,
  `train.MAJOR` and `predict.OE_LEAGUE`.

## lolesports API

Unofficial and undocumented, so every call raises on HTTP or shape errors.

- Base `https://esports-api.lolesports.com/persisted/gw/`, header `x-api-key`
  set to the public key lolesports.com itself sends (`lolesports.API_KEY`; it is
  not confidential), always `hl=en-US`.
- `getLeagues` → `data.leagues[]` with `id`, `slug`, `name`, `region`. Slugs
  used (`predict.PREDICT_LEAGUES`): `lck, lpl, lec, lcs, cblol-brazil, lcp,
  worlds, msi, first_stand`.
- `getSchedule?leagueId=<id>[,<id>...][&pageToken=...]` →
  `data.schedule.{pages:{older,newer}, events[]}`. An event has `startTime`,
  `state` (`unstarted`, `inProgress`, `completed`), `type` (only `match`
  matters; `show` events exist too), `blockName`, `league{name,slug}` and
  `match{id, teams[], strategy{type:"bestOf", count}}`. Each team has `name`,
  `code`, `image`, `result{outcome,gameWins}`. Upcoming bracket matches have
  teams named `TBD`; they are skipped and picked up by a later run.
- `getEventDetails?id=<matchId>` → `data.event` with `match.{strategy, teams,
  games[]}`. The live endpoint carries `gameWins` but no `outcome`, so
  `evaluate.py` takes the winner as the team whose `gameWins` reached
  ⌈best_of / 2⌉. An unknown id returns `event: null`, which counts as
  unresolved.
- `blockName` is the only stage signal at prediction time. `predict.py` sets
  `playoffs = 1` for a domestic match whose block name matches
  `playoff|final|knockout|play.?in`; internationals are always 0.

## `src/pipeline/team_aliases.csv`

Columns `lolesports_code,lolesports_name,oe_teamid`; matched on the exact
`(code, name)` pair. Tracked as normal source (not under `src/data/`). Add a
row when `resolve_team` logs an unmapped team (see
[operations.md](operations.md#when-a-job-fails)). Currently 11 aliases.

## Not used (deliberately)

- **Riot developer API**: solo queue only, no pro games.
- **Leaguepedia Cargo API**: anonymous queries were rate-limited immediately
  when tried; OE already covers what is needed.
- **gol.gg**: would be scraping.
- **lolesports live-stats feed** (`feed.lolesports.com`): 10-second in-game
  frames, not needed for pre-match prediction.

## Data access rules

- **OE via Google Drive.** With `GOOGLE_DRIVE_API_KEY` set (the owner's API
  key; a GitHub secret in CI), `oe.list_files()` uses the Drive API v3
  `files.list` (`'<FOLDER_ID>' in parents and trashed=false`, shared drives
  included) and
  `oe.download_year()` uses `https://www.googleapis.com/drive/v3/files/<id>?alt=media`.
  The key goes only in the `X-Goog-Api-Key` header, never in a URL, exception
  or log. One API redirect to an https `*.googleusercontent.com` /
  `*.googleapis.com` host is followed with a plain request without the key;
  any other redirect is an error. Without a key (unset, empty or
  blank) both use the public links: the
  embedded folder view HTML and the no-auth download endpoint
  `https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t`.
  Either way the body must start with `gameid,`. A Drive API quota error
  (`downloadQuotaExceeded`, `quotaExceeded`, `rateLimitExceeded`, ..., HTTP 429
  or `RESOURCE_EXHAUSTED`) falls back to the public link once, for the listing
  and the download separately; an invalid key, a missing file or any other API
  error fails with its own message and no fallback. Drive can answer scripted public full downloads
  with a "Quota exceeded" page for up to 24 hours (it did on 2026-10-15, even
  for the 2014 file, while small Range requests still worked). The per-file
  download quota may also apply to API downloads, so the key makes a block less
  likely but not impossible. We do **not** chunk around the quota with Range
  requests, because that would circumvent Google's limit. OE's old
  S3 mirror is gone. A blocked download is a loud failure: `daily_state.py`
  raises, the state stays as committed, `evaluate` and `predict` still run on
  the committed state, the job exits non-zero, and the next day's run catches
  up through `LOOKBACK_DAYS = 14`. A full historical download is therefore
  best done by hand from the Drive folder in a browser. `oe.py --check` says
  which path it uses (Drive API or public link) and starts a real (non-Range)
  download, so it detects the block.
- **lolesports API**: works from GitHub-hosted runners and from a home PC with
  the public key. No proxy is needed or used (`check_sources.yml` passed on a
  GitHub runner).

## Supabase schema

Defined in `src/supabase/schema.sql`, run once in the SQL editor of a new
project ([operations.md](operations.md#applying-a-supabase-migration)). There
is no `migrations/` directory.

**`predictions`**: one row per lolesports match (series), unique `match_id`.

| Column | Notes |
|---|---|
| `id` | UUID PK |
| `match_id` | lolesports match id, unique, the upsert key |
| `league_slug`, `league_name`, `block_name` | e.g. `lck`, `LCK`, `Playoffs` |
| `start_time_utc` | scheduled start, `TIMESTAMPTZ` |
| `match_date` | UTC date of `start_time_utc` (the site groups by the viewer's local date instead) |
| `best_of` | 1, 3 or 5 (CHECK) |
| `team1_code`, `team1_name`, `team1_image`, `team2_*` | straight from lolesports; team1 = `teams[0]` |
| `team1_game_prob` | side-neutral per-game probability for team1 |
| `team1_series_prob`, `team2_series_prob` | series probabilities, sum to 1 |
| `predicted_team` | team code of the pick |
| `confidence` | `max(series prob)`, 0 to 1 (CHECK) |
| `actual_winner`, `score`, `correct` | `NULL` until `evaluate.py` resolves the match. `score` is `team1-team2` game wins, e.g. `3-1`. |
| `created_at` | |

Indexes on `match_date DESC` and `correct`. `predict.py` upserts on `match_id`
and never touches a row that already has `actual_winner`.

**`shap_values`**: one row per feature per prediction (top 10 by |value|,
replaced delete-then-insert on every re-prediction): `prediction_id` (FK,
cascade delete), `feature_name`, `shap_value`, `feature_value`. Computed from
the team1-blue orientation; positive pushes toward team1.

**`running_record`**: single row (`id = 1`, seeded with zeroes by the schema):
`total_correct`, `total_incorrect`, `accuracy`, `last_updated`. Recomputed by
`evaluate.py` on every run over predictions with `match_date >= RECORD_START`
(`"2026-10-15"` in `evaluate.py`), counted server-side so Supabase's 1000-row
default cannot undercount.

**Row Level Security**: enabled on all three tables with a public
`SELECT`-only policy (`USING (true)`, no write policy). The pipeline connects
with the **service-role key**, which bypasses RLS; the anon key used by the
website is read-only through these policies alone.

**Table privileges (GRANTs)**: newer Supabase projects no longer grant table
privileges by default, and without them even the service role fails with
`42501 permission denied for table predictions`. `schema.sql` ends with
explicit grants: `USAGE` on schema `public` to `anon, authenticated,
service_role`; full CRUD on the three tables to `service_role`; `SELECT` to
`anon, authenticated`.
