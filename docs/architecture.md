# Architecture

## Components

```
                     ┌──────────────────────────────────────────────────┐
                     │              One-time / rare rebuild              │
                     │  bootstrap_data.py → src/data/raw/oe/*.csv        │
                     │  elo.py --force    → player_elo*.parquet,         │
                     │                      role_norms.json              │
                     │  daily_state.py --init → team_games_*.csv         │
                     │  build_dataset.py  → feature_matrix.parquet       │
                     │  train.py          → models/xgb_model.pkl         │
                     └──────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────── Daily automation (GitHub Actions) ────────────────────────────────┐
│                                                                                                    │
│  daily_evaluate.yml (daily_run.py --evaluate) — two independent steps, both always attempted:     │
│    step 1  lolesports getEventDetails ─▶ evaluate.py ─▶ Supabase: predictions.actual_winner /      │
│            score / correct, running_record                                                         │
│    step 2  Oracle's Elixir (Drive) ─▶ daily_state.py ─▶ team_games_{year}.csv (append)             │
│            ─▶ elo.update_elo ─▶ player_elo_current / player_elo_recent.parquet                     │
│    then commit src/data/ (even if a step failed)                                                   │
│                                                                                                    │
│  daily_predict.yml (daily_run.py --predict):                                                       │
│    lolesports getSchedule ─▶ predict.py ─▶ features.py (live mode, committed state)               │
│    ─▶ xgb_model.pkl ─▶ Supabase: predictions, shap_values                                          │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
         Supabase (Postgres, RLS)  ──▶  web/ (Next.js static export)  ──▶  manual build+upload  ──▶  Plesk/nginx
```

Two independent public sources feed this (details in [data.md](data.md)):
**Oracle's Elixir** (OE), yearly match-data CSVs on Google Drive, is the only
source of game statistics, for the historical rebuild and for the daily
update alike; the **lolesports API** supplies the schedule and the results.
There is no shared team id between them, see
[The lolesports to OE team join](#the-lolesports-to-oe-team-join).

## Data flow, end to end

1. **Bootstrap** (`bootstrap_data.py`, one-time / rare) downloads every OE
   year file (2014 onward) into `src/data/raw/oe/` (gitignored).
2. **ELO** (`elo.py`, `build_elo`) replays every game chronologically. It
   writes the full pre-game history (`player_elo.parquet`, not committed),
   the two committed live-state files (`player_elo_current.parquet`,
   `player_elo_recent.parquet`) and the frozen `role_norms.json`.
3. **Team game log** (`daily_state.py --init`) writes the committed
   `team_games_{year}.csv` (two rows per game, one per team) from the local OE
   files. After that only the daily run appends to them.
4. **Feature matrix** (`build_dataset.py`) walks every game since 2017-01-01
   across all leagues and calls `features.py`'s `build_feature_matrix()` in
   **training mode**, producing `feature_matrix.parquet`.
5. **Train** (`train.py`) fits the XGBoost model on that matrix, Platt
   calibrates it and writes the committed `models/xgb_model.pkl`.
6. **Evaluate** (`evaluate.py`, first step of `daily_run.py --evaluate`)
   resolves `predictions` rows that have started but have no result, using
   lolesports `getEventDetails`, then recomputes `running_record` from
   `RECORD_START` onward.
7. **Daily state** (`daily_state.py`, second step of
   `daily_run.py --evaluate`; it still runs if step 6 raised) downloads the
   current OE year file, finds games not yet in `team_games_*.csv`, runs
   `elo.update_elo` on them and appends their team rows.
8. **Predict** (`predict.py`, `daily_run.py --predict`) fetches the
   lolesports schedule for the next 36 hours, builds features in **live mode**
   for each match in both side orientations, averages them into a
   side-neutral per-game probability, compounds that into a series
   probability, and upserts `predictions` + `shap_values` to Supabase.
9. **Web** (`web/`, Next.js static export) reads `predictions`, `shap_values`
   and `running_record` directly from Supabase in the browser (anon key,
   RLS-restricted to `SELECT`) and renders the site and the `/card` iframe
   widget.

The model predicts **one game, blue side vs red side**. A series
(Bo1/Bo3/Bo5) probability is derived from it (`features.series_win_prob`,
first to ⌈n/2⌉ with independent games). The public record counts **series**.

## Module-by-module responsibilities

**`src/pipeline/`**

- `oe.py` — Oracle's Elixir client. `list_files()` resolves year → Drive file
  id by parsing the public embedded folder view (a new year means a new id, so
  nothing is hardcoded). `download_year()` is atomic (temp file +
  `os.replace`), reuses a local file younger than `MAX_AGE_HOURS = 20`, and
  raises if Drive answers with anything but the CSV (the "Quota exceeded"
  page). `load_year()` reads one year into the 43 columns of `NEEDED_COLS`
  (columns a year lacks become NaN) with normalised dtypes. `player_rows()` /
  `team_rows()` split the 12 rows per game; `team_rows()` also builds the
  `starters` column. `--check` tests folder listing plus a real download start.
- `lolesports.py` — lolesports API client (`getLeagues`, `getSchedule`,
  `getEventDetails`), `upcoming_matches()` (unstarted `match` events in a
  window, no `TBD` teams, de-duplicated by match id) and `resolve_team()`. All
  calls raise on HTTP or shape errors instead of returning an empty result.
  `--check` calls `getLeagues`.
- `team_aliases.csv` — manual `(lolesports_code, lolesports_name) → oe_teamid`
  overrides read by `resolve_team()`. Tracked source, not under `src/data/`.
- `elo.py` — the player ELO system: five role-normalised skill ELOs plus the
  outcome ELO, full rebuild (`build_elo`) and incremental update
  (`update_elo`), the frozen `role_norms.json`. Full detail in
  [elo.md](elo.md).
- `features.py` — `build_feature_matrix()` for both training and live mode,
  nine feature groups each in its own try/except, plus `series_win_prob()`.
  See [model.md](model.md) for the feature list.
- `build_dataset.py` — builds `feature_matrix.parquet` (all leagues, games
  from `START_DATE = "2017-01-01"`, label `blue_win`).
- `train.py` — Optuna + XGBoost + Platt calibration, test-slice baselines and
  the gate against the outcome-ELO baseline. See [model.md](model.md).
- `daily_state.py` — brings the committed state (`team_games_*.csv`, ELO files)
  up to date from OE. `run(date=None, dry_run=False)`; `--init` is the
  one-time creation of the CSVs.
- `predict.py` — `predict_upcoming()` for the daily pipeline, plus an ad-hoc
  CLI for one match (`--match-id`) or a team pair (`--team1 --team2 --bo`).
- `evaluate.py` — resolves started predictions against lolesports results and
  recomputes `running_record`.

**`src/scripts/`**

- `bootstrap_data.py` — downloads every OE year file into `src/data/raw/oe/`
  (about 856 MB, a few minutes; files younger than 20 h are skipped).
- `daily_run.py` — the GitHub Actions entry point. `--evaluate` runs
  `evaluate()` then `daily_state.run()`; `--predict` runs `predict_upcoming()`;
  no flag runs all three. Every step is attempted even if an earlier one
  failed, and the exit code is 1 if any failed.

**`web/`** — Next.js 15 (App Router, static export, `output: "export"`).

- `app/(main)/page.tsx` is the public site: today's predictions, upcoming
  matches, running record, recent predictions, "About the model" (metrics
  hardcoded). `app/(main)/demo/page.tsx` renders the components with mock data.
  `app/(main)/layout.tsx` holds the header and footer with the Riot disclaimer
  and the data credit.
- `app/card/page.tsx` is the standalone 420×260 iframe widget meant to be embedded
  on the portfolio site.
- `lib/supabase.ts` is the only place that talks to Supabase (anon key,
  browser-side, since a static export has no server runtime). It exports
  `null` without the env vars, so a build without `web/.env.local` still
  succeeds and shows empty states. `lib/types.ts` mirrors the `predictions`
  row; `lib/date.ts` groups matches by the viewer's local date of
  `start_time_utc`; `lib/mock.ts` holds the demo rows.
- `components/`: `GameCard` (league · Bo{n}, series-win split bar, score once
  resolved, top 5 of the stored 10 SHAP features with readable labels),
  `RecentPredictionsTable`, `RecordStats`, `TeamLogo`.
- There is no team table: names and logos come from the prediction row.
  `TeamLogo` rewrites `http://static.lolesports.com/` to `https://` and falls
  back to a grey circle with the team code.

## The lolesports to OE team join

lolesports identifies teams by `name` / `code` / `image`, OE by `teamid` /
`teamname`; there is no shared id. `lolesports.resolve_team(name, code)`
returns an OE `teamid` or `None`:

1. **Alias.** An exact match on the `(code, name)` pair in
   `src/pipeline/team_aliases.csv`.
2. **Normalised exact match.** `normalize_team_name()` casefolds, drops
   non-alphanumerics and strips trailing `esports`, `gaming`, `team`, `club`
   words (`"Gen.G Esports"` → `geng`). The result must equal the normalised
   `teamname` of **exactly one** OE `teamid` among the team rows of the 365
   days before `as_of`. `predict.py` passes the start of the prediction window
   (now, or 00:00 UTC on `--date`), not the match start.
3. Otherwise it logs `unmapped team <code>/<name>` (with `(ambiguous: n OE
   teamids)` when several matched) and returns `None`. The match is then
   skipped, never matched fuzzily.

`academy` is deliberately **not** a stripped suffix: academy rosters are
separate teams (`T1 Esports Academy` is not `T1`). Results never need the
join, because `evaluate.py` reads winners straight from lolesports.

## Live vs. training feature paths (why they differ)

Both paths call the same `build_feature_matrix()` on the same data shape. The
`use_current_elo` flag and the roster source are the only differences.

**Point in time is day level.** A game on calendar day *d* sees only
`team_games` rows from days before *d* and player ELO "as of the start of day
*d*". A live prediction is made before the series with state up to the
previous day, so in training game 2 of a series must not see game 1 either.

| | Training (`use_current_elo=False`) | Live (`use_current_elo=True`) |
|---|---|---|
| ELO source | `player_elo.parquet`: each player's first pre-game snapshot on or after day *d* (else the current value) | `player_elo_current.parquet`: the post-game state |
| Roster | The 5 players who actually played this `game_id` (from the `starters` of that game's `team_games` row) | The `starters` of the team's latest `team_games` row before day *d* |
| Missing player or empty slot | Left out of the team means (its lane diffs are NaN) | Filled with the init value: the mean outcome ELO of current players last seen in the team's home league if at least 10, else the league prior; every skill ELO 1000 |
| Form trajectory | History snapshots from `player_elo.parquet` | `player_elo_recent.parquet`, the last 11 snapshots per player |
| `team_games` source | All OE years in memory | The committed `team_games_*.csv` |

Reasons for the difference: at prediction time there is no future box score to
read a lineup from, and only the small committed state (not the raw OE files
or the full ELO history) is available to the daily runner. Training uses the
real lineup, which is usually announced before the match, so this leaks little
but makes the backtest slightly optimistic. Details and the other caveats are
in [model.md](model.md).

Everything else (rolling stats, side splits, rest, head-to-head, roster churn,
streaks, context) reads the same `team_games` frame in both modes. Because
`daily_state.py` only ingests games dated strictly before the target day, the
ELO state never holds day *d* while live features cut `team_games` at days
before *d*.
