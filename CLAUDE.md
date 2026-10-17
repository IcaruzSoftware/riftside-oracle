# CLAUDE.md

Entry point for an AI session working in this repo. Read this first, then
**[HANDOFF.md](HANDOFF.md) for current status, decision log and open points**;
this file links to `docs/` for depth instead of duplicating it. Human-facing
overview: [README.md](README.md).

Forked from courtside-oracle (NBA).

Working agreements:

- The owner pushes and deploys; agents commit per phase/change.
- Model choices are made on validation only; the test slice is looked at once
  ([docs/model.md](docs/model.md#evaluation-protocol)).

## What this is

Riftside Oracle predicts professional League of Legends **series** (Bo1/Bo3/Bo5)
before they start. A per-player ELO system (5 role-normalised performance skills
plus an outcome ELO) feeds an XGBoost classifier (Platt-calibrated) that predicts
one game, blue side vs red side; the series probability is derived from it.
Predictions run daily via GitHub Actions, are stored in Supabase, and are shown
on a Next.js site (`https://riftside-oracle.gerritvisser.de`, subdomain not yet
confirmed) and a 420×260 iframe widget (`/card`) meant to be embedded on
https://gerritvisser.de from `IcaruzSoftware/portfolio` (embed not set up yet;
see HANDOFF.md).

The public record counts **series**, not games.

## Data flow

```
bootstrap (Oracle's Elixir yearly CSVs, Drive) → elo.py → committed ELO state
                                                    │
daily: OE current-year CSV → daily_state.py → team_games_*.csv + elo.update_elo (incremental)
daily: lolesports getEventDetails → evaluate.py → Supabase (resolve predictions, running_record)
daily: lolesports getSchedule → predict.py → features.py (live) → xgb_model.pkl → Supabase
                                                    │
        Supabase (RLS) → web/ (Next.js static export) → manual build+upload → Plesk/nginx
```

Training and live use the same OE data shape and the same `features.py`
functions. Details: [docs/architecture.md](docs/architecture.md) (module by
module), [docs/data.md](docs/data.md) (every file/table),
[docs/elo.md](docs/elo.md), [docs/model.md](docs/model.md) (features, training,
metrics, caveats), [docs/operations.md](docs/operations.md) (runbook).

## Repo map

```
src/pipeline/
  oe.py               Oracle's Elixir: resolve file ids from the Drive folder, download, load/normalise, player/team rows, --check
  lolesports.py       lolesports API: leagues, schedule, event details, resolve_team (aliases + normalised name), --check
  team_aliases.csv    manual lolesports code/name -> OE teamid overrides (tracked source, not state)
  elo.py              player ELO: 5 skill ELOs + outcome ELO, build_elo / update_elo, role norms
  daily_state.py      daily catch-up: OE rows -> team_games_*.csv + incremental ELO (--init writes team_games once; refuses if they exist)
  features.py         build_feature_matrix() (training and live), series_win_prob()
  build_dataset.py    walks OE history -> feature_matrix.parquet (training input)
  train.py            Optuna + XGBoost + Platt calibration -> models/xgb_model.pkl, baselines + gate
  predict.py          predict one match / ad-hoc series (CLI) or predict_upcoming() (daily pipeline)
  evaluate.py         resolve pending predictions via lolesports, recompute running_record
src/scripts/
  bootstrap_data.py   download every OE year file into src/data/raw/oe/
  daily_run.py        GitHub Actions entry point: --predict / --evaluate / both
src/supabase/
  schema.sql          tables (predictions, shap_values, running_record), RLS, explicit GRANTs; run once in the SQL editor
.github/workflows/
  daily_evaluate.yml  07:45 UTC — resolve results + advance committed state
  daily_predict.yml   after Daily Evaluate (any conclusion) + crons 03:00 and 15:00 UTC — predict the next 36h
  keepalive.yml       daily — ping Supabase, re-enable workflows GitHub auto-disabled
  check_sources.yml   manual — check OE and lolesports reachability from a runner
web/                  Next.js 15, static export; manual build+upload to the owner's Plesk server (no auto-deploy); app/(main)/ = site (+ /demo), app/card/ = iframe widget
tests/                pytest; conftest.py has a FakeSupabase; fixtures/ holds trimmed OE CSV and lolesports JSON
```

## Running locally

Python **3.12** (matches CI; the pins don't install on 3.13+). Use a local
`.venv` — it's gitignored.

```bash
pip install -r requirements.txt pytest                    # pytest is not in requirements.txt
python -m pytest                                           # 121 tests, via pytest.ini
python src/scripts/daily_run.py                             # evaluate + state update + predict (UTC date)
python src/scripts/daily_run.py --predict --dry-run         # print predictions, write nothing
python src/scripts/daily_run.py --evaluate --dry-run
python src/pipeline/daily_state.py --dry-run --date 2026-11-05
python src/pipeline/evaluate.py --dry-run
python src/pipeline/predict.py --match-id 115660588583171786
python src/pipeline/predict.py --team1 "Gen.G Esports" --team2 T1 --bo 5 [--league lck]
python src/pipeline/oe.py --check                           # OE folder listing + real download test
python src/pipeline/lolesports.py --check
python src/scripts/bootstrap_data.py                        # download all OE year files (see Data access)
python src/pipeline/elo.py --force                          # full ELO rebuild from src/data/raw/oe/
python src/pipeline/daily_state.py --init                   # team_games_*.csv from the same OE files (delete the existing ones first)
python src/pipeline/build_dataset.py --force                # rebuild feature_matrix.parquet
python src/pipeline/train.py --trials 100 --force           # retrain
```

`SUPABASE_URL` / `SUPABASE_KEY` (service-role key) are required for anything
that isn't `--dry-run`. Without the raw OE files (gitignored) the tests and
live predictions still work from the committed state; rebuilding ELO or
retraining needs them. `web/`: `cd web && npm install && npm run dev`
(`/demo` shows labelled mock rows; `?demo=1` on any page does too, but only under `npm run dev`).

## Committed live-state contract

`.gitignore` ignores `src/data/raw/*` and `src/data/processed/*` wholesale,
then re-includes exactly these with `!` negations — **never commit anything
else under `src/data/`,** and never add a new generated file there without
also adding its negation:

- `src/data/raw/team_games_*.csv` — one row per team per game, one file per year (appended daily by `daily_state.py`; don't hand-edit)
- `src/data/processed/player_elo_current.parquet` — post-game ELO snapshot per player
- `src/data/processed/player_elo_recent.parquet` — last 11 pre-game snapshots per player (committed in full, no pruning)
- `src/data/processed/role_norms.json` — role normalisation means/stds over 2022–2025; frozen for incremental updates, recomputed by every `elo.py --force`

`src/models/xgb_model.pkl` is committed (its own negation). Not committed:
the OE raw CSVs (`src/data/raw/oe/`, ~856 MB), `player_elo.parquet` (full
history) and `feature_matrix.parquet`. File-by-file table:
[docs/data.md](docs/data.md).

## Data access rules

- **Oracle's Elixir (OE)** — the only stats source, for history *and* the
  daily job. One CSV per year in a public Google Drive folder (id in
  `oe.FOLDER_ID`); `oe.py` resolves file ids from the Drive API listing with a
  key, else by parsing the embedded folder view, so a new year's file needs no
  code change. OE refreshes about 07:05
  UTC and lags about a day; the current-year file is ~78 MB. Local files
  younger than 20 h are reused.
- **Drive API key:** with `GOOGLE_DRIVE_API_KEY` set (the owner's own key, a
  GitHub secret), `oe.py` lists the folder and downloads through the Drive API
  v3 (`files.list`, `files.get?alt=media`), the key sent only in the
  `X-Goog-Api-Key` header (never in a URL or log; one redirect to an https
  `*.googleusercontent.com` / `*.googleapis.com` host is followed without the key).
  Unset, empty or blank = the public links (folder view HTML +
  `drive.usercontent.google.com`), as in the tests. Locally, set the variable
  in the shell to use the API for `bootstrap_data.py` / `daily_state.py`. An
  API quota error (403 quota reasons, 429, `RESOURCE_EXHAUSTED`) falls back to
  the public link once, for the listing and the download separately.
- **Drive quota:** on 2026-10-15 every scripted full download got Drive's
  "Quota exceeded" page. The per-file download quota may still apply to API
  downloads. Do **not** chunk around it with Range requests. A
  blocked download makes the state update fail loudly (the run exits 1), but
  evaluate and predict still run on the committed state, and the next day's
  run catches up through `LOOKBACK_DAYS = 14`. For a local rebuild, download
  the files by hand from the Drive folder into `src/data/raw/oe/`, keeping
  the original names (`2026_LoL_esports_match_data_from_OraclesElixir.csv`).
- **lolesports API** (`esports-api.lolesports.com`) — schedule and results.
  Unofficial, so every call fails loudly. Uses the public key the site itself
  sends (`lolesports.API_KEY`, not a secret). Both sources work from GitHub
  runners; **no proxy is needed** (`check_sources.yml` passed).
- Not used on purpose: the Riot developer API (solo queue only) and
  Leaguepedia Cargo (rate-limited).

## Invariants / gotchas

- `daily_state.py` ingests only games dated strictly **before** the target
  day (UTC); same-day games wait for the next run, so the ELO state never
  holds day d while live features cut `team_games` at day < d. It rescans
  from 14 days before the last logged game, so late OE rows still arrive.
- Only gameids **new to `team_games`** reach `elo.update_elo`, and the ELO
  update runs before the CSV append (a failed append is retried safely:
  `update_elo` skips games it already holds). Incomplete games (not 12 rows,
  5 roles per side, results present) are skipped and picked up later.
- `update_elo` applies late OE rows in **arrival order**, not date order; a
  full rebuild re-sorts. Incremental over in-order games equals a full build.
- Point-in-time is day-level: a game on day d sees only `team_games` rows
  from before d and ELO as of the start of d.
- `predict.py` only predicts unstarted matches inside the 36 h window,
  skips `TBD` and unresolved teams (logs a warning), and **never overwrites a
  row that has `actual_winner`**. Unstarted rows are refreshed.
- `evaluate.py` isolates each match (one failing lookup doesn't stop the
  others; it raises once at the end), counts an unknown match id as
  unresolved, and **recomputes `running_record` on every run**. Forfeits
  (0-0) never resolve; they only get the 7-day warning.
- `daily_run.py` attempts every step (evaluate, state update, predict) even
  if an earlier one failed, and exits non-zero if any did.
- `RECORD_START` (`evaluate.py`, currently `"2026-10-15"`) is the first match
  date counted in the public running record.
- `role_norms.json` is **frozen** for incremental updates: never recompute it
  there or they stop matching a rebuild. Every `elo.py --force` **recomputes**
  it, which changes the ELO state, so a retrain follows.
- `scikit-learn==1.8.0` and `xgboost==3.2.0` match the pickled
  `xgb_model.pkl`; don't bump either without retraining.
- Players with a missing `playerid` are skipped in the ELO (the rest of the
  game is processed). Never use `playername` as a key.
- Team aliases: lolesports and OE share no team id. `resolve_team` checks
  `team_aliases.csv` first, then an exact normalised-name match against
  recent OE teams — never fuzzy. A sponsor rename shows up as
  `WARNING unmapped team` and a skipped match.
- Playoffs context in live mode comes from the lolesports `blockName`
  (domestic leagues only); OE flags it differently in a few leagues.

## Where external things live

- **Supabase** project `pdblqlbgrelyqjgkpdte` (free tier; pauses after ~7
  days idle, kept awake by `keepalive.yml`). Service-role key in GitHub
  secrets (`SUPABASE_URL`, `SUPABASE_KEY`); the anon key goes into
  `web/.env.local` (`NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY`)
  at build time — **never the same key in both places**. Newer Supabase
  projects grant no table privileges by default, so `schema.sql` ends with
  explicit GRANTs.
- **Google Drive API key** (owner's Google Cloud project, restricted to the
  Drive API) in the GitHub secret `GOOGLE_DRIVE_API_KEY`, passed to
  `daily_evaluate.yml` and `check_sources.yml`; optional (empty = public links).
  How to create it: [docs/operations.md](docs/operations.md#creating-the-drive-api-key).
- **The owner's Plesk server** (nginx) is meant to serve the static export (not
  deployed yet). No auto-deploy — see
  [docs/operations.md](docs/operations.md#deploying-the-web-app). For `/card`
  to be iframed on `gerritvisser.de`, the CSP `frame-ancestors` header must be
  set in the server config when deploying (not by the Next.js app).
- **Portfolio embed**: `/card` is meant to be iframed from
  `IcaruzSoftware/portfolio` (owner step, see HANDOFF.md). Don't break the
  card's markup or query interface; check that repo for any breaking change.
- **GitHub Actions**: public repo, so GitHub auto-disables scheduled
  workflows after 60 days of no repo activity; `keepalive.yml` re-enables the
  two daily workflows (and itself) via `gh workflow enable` on every run. Bot identity:
  `riftside-oracle[bot]`.

## Known limitations

- **Pre-draft only.** Picks and bans are unknown before the match and swing
  games; the model can't see them. ~70% series accuracy is about the ceiling
  with this data.
- Cross-region strength rests on international games (a few hundred per
  year), player transfers and hand-set league priors (`elo.LEAGUE_PRIOR`).
- Small samples and roster churn: early-season predictions (Nov–Jan
  off-season changes) are the weakest.
- Live roster = the team's last-known starters; same-day substitutions are
  invisible. Training uses the actual lineups.
- OE lags about a day: a team playing on consecutive days is predicted
  without yesterday's game. LPL rows can lag more and lack early-game stats.
- The lolesports API is unofficial and can change without notice.
- Series maths assumes independent games (no momentum, no side alternation).
  In major-league Bo5s the series probabilities are slightly worse
  calibrated than plain ELO's.
- `web/app/(main)/page.tsx` hardcodes the model metrics text; it does not
  read them from anywhere.
- More small items and the accepted Phase 4 limitations:
  [HANDOFF.md](HANDOFF.md) (open points).

## Web rules

The site follows Riot's fan-project rules: keep the Riot disclaimer and the
"Match data: Oracle's Elixir and lolesports" credit in the footer, and add
**no betting language, odds or affiliate links** anywhere.

## Common changes

- **Add a team alias** — add a row to `src/pipeline/team_aliases.csv`
  (`lolesports_code,lolesports_name,oe_teamid`; find the id in
  `team_games_*.csv`). Check with `predict.py --dry-run`.
- **Add a feature** — add a `get_*_features()` function in `features.py`,
  wire it into `build_feature_matrix()` (training and live call the same
  function), then **you must retrain**, in this order: `elo.py --force` →
  (if committing the ELO state: delete `team_games_*.csv`, then
  `daily_state.py --init` from the same OE files) → `build_dataset.py --force`
  → `train.py --trials 100 --force`. Always rebuild the ELO before
  `build_dataset`: it reads the uncommitted `player_elo.parquet`, and a stale
  one silently falls back to the current ELO for later games (look-ahead).
  Commit the ELO state only together with a fresh `--init`, otherwise
  `team_games` desyncs from the ELO and the missing games are never picked up
  (`daily_state` only passes gameids new to `team_games`). Commit the new
  `models/xgb_model.pkl`. A live-only feature with no training equivalent
  is never a training column (so the model ignores it), and a training-only
  feature reads as missing (NaN) live. If a retrain moves the
  pins, update `requirements.txt` too.
  After a retrain, update the hardcoded results in `web/app/(main)/page.tsx`,
  README.md, docs/model.md (Results) and the HANDOFF.md model numbers. For a
  new feature also extend `FEATURE_LABELS` in `web/components/GameCard.tsx`.
- **Change a job's schedule** — edit the `cron:` line in the relevant
  `.github/workflows/*.yml` (times are UTC). `daily_predict.yml` also follows
  `daily_evaluate.yml` via `workflow_run`.
- **Change the predicted leagues** — `PREDICT_LEAGUES` and `OE_LEAGUE` in
  `predict.py`.
- **Apply schema changes** — SQL editor, see
  [docs/operations.md](docs/operations.md#applying-a-supabase-migration).
- **Deploy the web app** — not push-to-deploy: `cd web && npm run build`
  (with `web/.env.local` set), then manually upload `web/out/`'s contents to
  the subdomain on the Plesk server — see
  [docs/operations.md](docs/operations.md#deploying-the-web-app).
- **Rebuild ELO / bootstrap from scratch** — see
  [docs/operations.md](docs/operations.md#full-historical-bootstrap--elo-rebuild);
  commit only the state files listed above, never raw data.
