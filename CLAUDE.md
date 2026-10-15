# CLAUDE.md

Entry point for an AI session working in this repo. Read this first; it links
to `docs/` for depth instead of duplicating it. Human-facing overview:
[README.md](README.md).

## What this is

Courtside Oracle predicts NBA game outcomes: a custom per-player ELO system
feeds an XGBoost classifier (Platt-calibrated). Predictions run daily via
GitHub Actions, are stored in Supabase, and are shown on a Next.js site
(https://courtside-oracle.gerritvisser.de) and a 420×260 iframe widget
embedded on https://gerritvisser.de (separate repo: `IcaruzSoftware/portfolio`).

## Data flow

```
bootstrap (stats.nba.com, one-time) → elo.py → committed ELO state
                                                       │
daily: cdn.nba.com → daily_state.py → state files → elo.py (incremental)
daily: cdn.nba.com → evaluate.py    → Supabase (resolve predictions, running_record)
daily: cdn.nba.com → predict.py     → features.py (live) → xgb_model.pkl → Supabase
                                                       │
                        Supabase (RLS) → web/ (Next.js static export) → manual build+upload → Plesk/nginx
```

Details: [docs/architecture.md](docs/architecture.md) (module-by-module +
why the live and training feature paths differ),
[docs/data.md](docs/data.md) (every file/table),
[docs/elo.md](docs/elo.md) (the ELO system),
[docs/model.md](docs/model.md) (features, training, metrics, caveats),
[docs/operations.md](docs/operations.md) (runbook).

## Repo map

```
src/pipeline/
  collect.py          full nba_api collector (proxy-pool aware) — superseded by bootstrap_data.py for the actual rebuild
  elo.py              player ELO: scoring, ranking, build_elo/update_elo/update_team_assignments, CDN adapter
  nba_cdn.py          cdn.nba.com client: schedule, box score, game-ID prefix helpers, team id<->tricode
  daily_state.py      daily catch-up: CDN box scores -> game logs/player stats/ELO (replaces removed update_season.py)
  features.py         build_feature_matrix() — training and live feature paths
  build_dataset.py    walks history -> feature_matrix.parquet (training input)
  train.py            Optuna + XGBoost + Platt calibration -> models/xgb_model.pkl
  predict.py          predict one matchup (CLI) or predict_todays_games() (daily pipeline)
  evaluate.py         resolve pending predictions vs CDN results, recompute running_record
  seed_finals.py      one-time historical seed — already run, do not re-run
  shap_export.py      global SHAP plot/summary utility — not wired into the daily pipeline
src/scripts/
  bootstrap_data.py   resumable stats.nba.com downloader for the historical rebuild
  daily_run.py        GitHub Actions entry point: --predict / --evaluate / both
src/supabase/
  schema.sql            fresh-setup schema (tables + RLS), run once in the SQL editor
  migrations/001_*.sql  brings a pre-existing DB up to schema.sql's shape (game_time_utc + RLS)
.github/workflows/
  daily_predict.yml   runs after daily_evaluate.yml completes — predict today's games
  daily_evaluate.yml  3:00 AM ET — resolve results + advance committed state
  keepalive.yml       daily — ping Supabase, re-enable workflows GitHub auto-disabled
web/                  Next.js 15, static export; manual build+upload to the owner's Plesk server (no auto-deploy); app/(main)/ = site, app/card/ = iframe widget
tests/                pytest; conftest.py has a FakeSupabase + a tiny built-from-fixture live state
```

## Running locally

Python **3.12** (matches CI; use a local `.venv` — it's gitignored, not committed).

```bash
pip install -r requirements.txt
python -m pytest                                          # tests/, via pytest.ini
python src/scripts/daily_run.py                            # evaluate + predict, today (ET)
python src/scripts/daily_run.py --predict --dry-run --date 2026-06-13
python src/scripts/daily_run.py --evaluate --dry-run
python src/pipeline/daily_state.py --dry-run --date 2026-11-05
python src/pipeline/evaluate.py --dry-run
python src/pipeline/predict.py --game-id 0042500237
python src/pipeline/predict.py --home NYK --away OKC --date 2026-06-05
python src/pipeline/elo.py --force                          # full ELO rebuild from src/data/raw/
python src/pipeline/build_dataset.py --force                # rebuild feature_matrix.parquet
python src/pipeline/train.py --trials 100 --force            # retrain
python src/scripts/bootstrap_data.py --phase 2 --limit 3      # smoke-test the box-score downloader
```

`SUPABASE_URL` / `SUPABASE_KEY` (service-role key) are required for anything
that isn't `--dry-run`. `web/`: `cd web && npm install && npm run dev`.

## Committed live-state contract

`.gitignore` ignores `src/data/raw/*` and `src/data/processed/*` wholesale,
then re-includes exactly four files with `!` negations — **never commit
anything else under `src/data/`,** and never add a new generated file there
without also adding its negation:

- `src/data/raw/game_log_*.csv` — one row per team per game (appended daily by `daily_state.py`)
- `src/data/raw/player_season_stats_*.csv` — one row per player per season (updated daily)
- `src/data/processed/player_elo_current.parquet` — post-game ELO snapshot per player
- `src/data/processed/player_elo_recent.parquet` — last-11 pre-game ELO snapshots per player

`src/models/xgb_model.pkl` **is committed** (its own negation in
`.gitignore`) — the old "model not committed" caveat from an earlier README
is no longer true. `player_elo.parquet` (full history) and
`feature_matrix.parquet` are deliberately **not** committed — see
[docs/data.md](docs/data.md) for the full file-by-file table and why.

## NBA data access rules

Two unrelated sources, never mixed in one run:

- **stats.nba.com** (via `nba_api`) — bootstrap only. Works from the owner's
  home PC with `nba_api`'s default browser-like headers; **times out from
  GitHub-hosted runners** (confirmed), so the full rebuild is a local/overnight
  job, never CI.
- **cdn.nba.com** (via `src/pipeline/nba_cdn.py`) — the entire daily pipeline.
  Needs the browser-style JSON header set in `nba_cdn.CDN_HEADERS` (modern UA,
  `Origin`/`Referer` `https://www.nba.com`, `Sec-Fetch-*`) — a plain request,
  and even `nba_api`'s own (outdated) live header set, gets an Akamai 403.
  **Confirmed: cdn.nba.com returns 403 from GitHub-hosted runners even with
  `CDN_HEADERS`** (NBA blacklists cloud IP ranges), while it works direct from a
  home PC. So the daily jobs route the CDN calls through static residential
  proxies configured in the `NBA_PROXIES` secret (comma-separated
  `http(s)://user:pass@host:port`; URL-encode special characters in the password
  or urllib3 fails to parse it). Unset/empty → direct connection (local dev,
  unchanged). `nba_cdn` fails over across the proxies in order, sticky to the
  last one that worked, and on all-fail raises a summary that names each
  `host:port` but never credentials. Box scores exist from the 2019-20 season
  onward. `fetch_boxscore()` treats a 404 as normal (game not played yet →
  `None`) and a 403 as an IP block (failover trigger, loud on total failure,
  never silently skipped). Verify proxy reachability with
  `python src/pipeline/nba_cdn.py --check` (or the manual `check_nba_proxies.yml`
  workflow) — see [docs/operations.md](docs/operations.md).

## Invariants / gotchas

- Game-ID prefix encodes type: `001` preseason, `002` regular season, `003`
  all-star, `004` playoffs, `005` play-in, `006` Cup final. Only
  `{002,004,005,006}` are "stateful" (`nba_cdn.is_stateful`) — update ELO and
  get predicted. All-star games are ignored entirely; preseason only updates
  a player's team assignment (never ELO), and only if the new appearance is
  newer than what's on file.
- `daily_state.py` rescans from **7 days before the last logged game**
  through the target date, not just the target date — so a box score that
  wasn't available yet on its game day still gets picked up later. Idempotent
  either way: games already in the game logs / ELO state are skipped.
- `predict.py` only predicts games that haven't started yet and never
  overwrites a prediction that already has a result — a late/partial re-run
  can't clobber a resolved pick.
- `evaluate.py` recomputes `running_record` on every run (not only when a
  prediction was just resolved), and the state update in `daily_run.py
  --evaluate` still runs even if `evaluate()` itself raised.
- `RECORD_START` (`evaluate.py`, currently `"2026-10-01"`) is the only thing
  keeping the 5 retroactive `seed_finals.py` rows out of the public running
  record. Don't re-run `seed_finals.py`.
- MATCHUP string (`"XXX vs. YYY"` = home, `"XXX @ YYY"` = away) is the only
  home/away signal in the game logs — there's no dedicated column.
- The **committed model** was trained on the original dataset (`collect.py`,
  which does fetch `boxscore_summary_*.json`), so it learned real
  availability effects. The *re-downloaded* bootstrap dataset (`bootstrap_data.py`)
  does **not** fetch `boxscore_summary` — only if/when someone retrains on
  that data would availability features go neutral for training too (fix:
  add `"summary"` to `bootstrap_data.py`'s wanted endpoints and re-run phase 2
  for it before retraining, ~14k extra calls ≈ 2.5h). See [docs/data.md](docs/data.md).
- `scikit-learn==1.8.0` is pinned because `xgb_model.pkl` was pickled with
  it — don't bump it without retraining.

## Where external things live

- **Supabase** project `pxvimjflsishbfrublwp` (free tier; pauses after ~7
  days idle, kept awake by `keepalive.yml`). Service-role key in GitHub
  secrets (`SUPABASE_KEY`); anon key goes into `web/.env.local`
  (`NEXT_PUBLIC_SUPABASE_ANON_KEY`) at build time — never the same key in
  both places.
- **NBA proxies**: GitHub secret `NBA_PROXIES` (comma-separated
  `http(s)://user:pass@host:port`, 1–3 static residential proxies) routes the
  daily CDN calls past NBA's cloud-IP block. Passed to `daily_predict.yml`,
  `daily_evaluate.yml`, and `check_nba_proxies.yml`. The old `PROXIES` secret
  (nba_api proxy-pool, bootstrap only) is obsolete and unused by the pipeline.
- **The owner's Plesk server** (nginx; `courtside-oracle.gerritvisser.de` and
  `gerritvisser.de` both resolve to it) serves the static export. No
  auto-deploy — see [docs/operations.md](docs/operations.md#deploying-the-web-app)
  for the manual build + upload procedure. The CSP `frame-ancestors` header
  that lets `/card` be iframed on `gerritvisser.de` is set in the server
  config, not by the Next.js app (a static export can't set response headers
  itself).
- **Portfolio embed**: `/card` is iframed from `IcaruzSoftware/portfolio` — a
  breaking change to its markup/query needs a check in that repo too.
- **GitHub Actions**: public repo, so GitHub auto-disables scheduled
  workflows after 60 days with no repo activity; `keepalive.yml` re-enables
  all three daily via `gh workflow enable` on every run.

## Known limitations

- No live injury feed — player-availability features are always neutral for
  live predictions. This is a genuine train/serve mismatch: the **committed
  model** trained on real availability data (see above), so live predictions
  are missing a signal the model actually learned to use.
- Live ELO updates (CDN box score) lack `defensiveRating` and tracking data:
  defense degrades to steals+blocks only, and hustle ELO freezes for every
  post-bootstrap game. Historical/bootstrap ELO is unaffected.
- Training features are more optimistic than live: training ELO rosters are
  each game's actual ≥10-minute participants, live rosters are approximated
  from the ELO state; training PPG/star-out weights use the full final
  season, live uses season-to-date only. Expect live accuracy below the
  67.5% backtest. Full breakdown: [docs/model.md](docs/model.md).
- `web/app/(main)/page.tsx` hardcodes the model metrics text (67.5%/0.729/
  0.209/0.606) — it does not read `model_metadata` (which nothing writes to
  anyway).
- Head-to-head and home/away-split features use all games since 2015-16, not
  just the current season — consistent between training and live, just worth
  knowing when reading a small early-season sample.

## Common changes

- **Change a job's schedule** — edit the `cron:` line in the relevant
  `.github/workflows/*.yml` (times are UTC; comments note the ET equivalent).
  `daily_predict.yml` has no cron of its own — it follows `daily_evaluate.yml`
  via `workflow_run`.
- **Add a feature** — add a `get_*_features()` function in `features.py`,
  wire it into `build_feature_matrix()` (both training and live call the same
  function unless it needs `use_current_elo` branching), then **you must
  retrain**: `build_dataset.py --force` → `train.py --force`, and commit the
  new `models/xgb_model.pkl`. A live-only feature with no training equivalent
  will silently read as `0.0` in training and vice versa.
- **Apply a Supabase migration** — SQL editor, see
  [docs/operations.md](docs/operations.md#applying-a-supabase-migration).
  `001` must land before deploying any web build that queries
  `game_time_utc`.
- **Deploy the web app** — not push-to-deploy: `cd web && npm run build`
  (with `web/.env.local` set), then manually upload `web/out/`'s contents to
  the subdomain on the Plesk server — see
  [docs/operations.md](docs/operations.md#deploying-the-web-app).
- **Rebuild ELO / bootstrap from scratch** — see
  [docs/operations.md](docs/operations.md#full-historical-bootstrap--elo-rebuild)
  (~7h unattended); commit only the state files listed above, never raw data.
