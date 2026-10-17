# Operations runbook

## Local setup

Python **3.12** is required (the pinned scientific stack does not install on
3.13+; CI uses 3.12).

```bash
git clone https://github.com/IcaruzSoftware/riftside-oracle.git && cd riftside-oracle
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest
```

**Windows notes** (from the 2026-10-15 PC):

- If `uv` is not installed, create the venv with the launcher:
  `py -3.12 -m venv .venv`, then
  `.venv\Scripts\python -m pip install -r requirements.txt pytest`.
- If Windows Application Control blocks pyarrow 26, install
  `pip install pyarrow==21.0.0` into the venv by hand. `requirements.txt` stays
  unchanged (`pyarrow>=15.0.0`) and CI is unaffected.

The committed state ([data.md](data.md#committed-live-state)) is enough to run
the tests and live predictions. **The raw OE data is not in git** (13 CSVs,
~856 MB, gitignored). To rebuild history or retrain, download the files by
hand from the Drive folder
<https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH>
into `src/data/raw/oe/`, keeping the original names (for example
`2026_LoL_esports_match_data_from_OraclesElixir.csv`). `bootstrap_data.py`
does the same, but Drive often blocks scripted full downloads (see
[Drive quota](#when-a-job-fails)). To make `bootstrap_data.py` and
`daily_state.py` use the Drive API locally, set the key in the shell first
(`export GOOGLE_DRIVE_API_KEY=...`, PowerShell `$env:GOOGLE_DRIVE_API_KEY = "..."`);
the per-file download quota may still block it.

## Manual commands

```bash
python -m pytest                                              # tests/, via pytest.ini
python src/scripts/daily_run.py                               # evaluate + state update + predict
python src/scripts/daily_run.py --predict --dry-run --date 2026-10-16
python src/scripts/daily_run.py --evaluate --dry-run
python src/pipeline/daily_state.py --dry-run --date 2026-11-05
python src/pipeline/evaluate.py --dry-run
python src/pipeline/predict.py --dry-run                      # next 36h of PREDICT_LEAGUES matches
python src/pipeline/predict.py --match-id 115660588583171786  # one lolesports match
python src/pipeline/predict.py --team1 "Gen.G Esports" --team2 T1 --bo 5 --league lck
python src/pipeline/oe.py --check
python src/pipeline/lolesports.py --check
```

`SUPABASE_URL` / `SUPABASE_KEY` (service-role key) are required for anything
that is not `--dry-run`. `--date` is a UTC date (`YYYY-MM-DD`); for
`daily_state.py` it is the target day, and only games dated **before** it are
ingested. Ad-hoc `predict.py` takes team **names** as on lolesports or OE.

## The four GitHub Actions workflows

All live in `.github/workflows/` and run on `ubuntu-latest` with Python 3.12.
Matches run around the clock (LCK/LPL roughly 08:00 to 15:00 UTC, LEC 16:00 to
21:00, LCS/CBLOL 20:00 to 03:00) and OE refreshes around 07:05 UTC, which
drives the schedule. GitHub often starts scheduled runs late; that is fine
here.

| Workflow | Trigger (UTC) | Runs | Commits |
|---|---|---|---|
| `daily_evaluate.yml` | cron `45 7 * * *`, `workflow_dispatch` | `daily_run.py --evaluate` | state files |
| `daily_predict.yml` | `workflow_run` after Daily Evaluate (any conclusion), crons `0 3 * * *` and `0 15 * * *`, `workflow_dispatch` | `daily_run.py --predict` | nothing |
| `keepalive.yml` | cron `17 6 * * *`, `workflow_dispatch` | Supabase ping + `gh workflow enable` | nothing |
| `check_sources.yml` | `workflow_dispatch` only | `oe.py --check`, `lolesports.py --check` | nothing |

### `daily_evaluate.yml`: Daily Evaluate

Runs at 07:45 UTC, after the OE refresh. `daily_run.py --evaluate` runs two
independent steps, each in its own try/except:

1. `evaluate.py`: resolve started predictions without a result via lolesports
   `getEventDetails`, then recompute `running_record`.
2. `daily_state.run()`: download the current OE year file, ingest new games
   into `team_games_{year}.csv` and the ELO state.

A failure in the first does not skip the second; either failing makes the job
exit non-zero so the run shows red.

The commit step runs **even if the previous step failed** (if only evaluate
failed, the state update still succeeded and must be committed). It does
`git add src/data/`, which can only stage the whitelisted state files:
`team_games_*.csv`, `player_elo_current.parquet`, `player_elo_recent.parquet`
and `role_norms.json` (frozen, so it normally does not change). If something
is staged it commits `chore: daily state update [skip ci]` as
`riftside-oracle[bot]`, runs `git pull --rebase origin <branch>`, and pushes.
Needs `permissions: contents: write`, which the workflow declares.

### `daily_predict.yml`: Daily Predictions

- `workflow_run` on Daily Evaluate completing with **any** conclusion, so a
  failed evaluate (for example a blocked OE download) still predicts, on the
  committed state. It checks out `main` explicitly, because a `workflow_run`
  checkout can otherwise be the SHA from before evaluate's state commit.
- Cron `0 3 * * *` is a safety net for early-UTC matches if evaluate runs late
  (it uses yesterday's state). Cron `0 15 * * *` picks up bracket matches
  whose `TBD` teams resolved during the day.
- `predict_upcoming()` covers every unstarted `PREDICT_LEAGUES` match in the
  next 36 hours. It upserts on `match_id`, refreshes rows of unstarted matches,
  and never touches a row that already has `actual_winner`. Commits nothing;
  only writes to Supabase (`predictions`, `shap_values`).
- LCK/LPL matches between about 05:00 and 09:00 UTC are predicted on state
  that is missing the previous day, because OE refreshes around 07:05 UTC.
  This is by design.

### `keepalive.yml`: Keepalive

- **Purpose 1:** Supabase free-tier projects auto-pause after about 7 days
  without activity. A `curl -f` authenticated read of `running_record` keeps
  the project awake and fails the job if Supabase is unreachable or paused.
- **Purpose 2:** GitHub disables scheduled workflows in public repos after 60
  days without repository activity. The second step runs
  `gh workflow enable` for `daily_evaluate.yml`, `daily_predict.yml` and
  `keepalive.yml` on every run (`if: always()`, so it still runs when the ping
  failed). Needs `permissions: actions: write`.

### `check_sources.yml`: Check Data Sources

Manual. Runs `oe.py --check` and `lolesports.py --check` from a GitHub runner
(the second even if the first fails). It passed once on a runner, so neither
source needs a proxy. Run it again when a daily run suddenly cannot reach
either source.

## Required GitHub secrets

| Secret | Used by | Notes |
|---|---|---|
| `SUPABASE_URL` | `daily_predict.yml`, `daily_evaluate.yml`, `keepalive.yml` | Public project URL (`https://pdblqlbgrelyqjgkpdte.supabase.co`). |
| `SUPABASE_KEY` | same | **Service-role key**: bypasses RLS, full read/write. Never put it in the web app. The web app uses the separate, RLS-restricted anon key via `web/.env.local` (not a GitHub secret; see [Deploying the web app](#deploying-the-web-app)). |
| `GOOGLE_DRIVE_API_KEY` | `daily_evaluate.yml`, `check_sources.yml` | Optional. The owner's Google API key, restricted to the Drive API; `oe.py` then lists and downloads OE through the Drive API v3 (key in the `X-Goog-Api-Key` header, never logged). Unset = the public Drive links. See [Creating the Drive API key](#creating-the-drive-api-key). |

There is no proxy secret: neither OE nor lolesports blocks GitHub runners.

### Creating the Drive API key

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a
   new project (for example `riftside-oracle`).
2. Select the new project in the project picker at the top of the console.
3. APIs & Services → Library → enable **Google Drive API**.
4. APIs & Services → Credentials → **Create credentials → API key**. Copy the
   key from the dialog that shows it.
5. Edit the key → API restrictions → **Restrict key** → select only
   **Google Drive API** → Save. (No application restriction: GitHub runners
   have no fixed IP.)
6. In the GitHub repo: Settings → Secrets and variables → Actions → New
   repository secret `GOOGLE_DRIVE_API_KEY` with the key.
7. Actions tab → **Check Data Sources** → **Run workflow**. The OE step should
   print `source    Drive API (GOOGLE_DRIVE_API_KEY set)` and end with
   `OK - OE folder and <year> download work (via the Drive API).`

An API key only reads public files, which the OE folder is. Google's Drive API
usage-limits page says standard use is free; it does not say whether the
project needs a billing account. The per-file download quota may still apply
to API downloads; the public-link fallback and the 14-day lookback remain.

## When a job fails

Check the failed run's logs in the Actions tab first; every pipeline script
logs at `INFO` level with timestamps. Common causes:

- **Drive quota block on the OE download.** Log: `OE download failed ...`
  from `daily_state.py` (a body that is not the OE CSV, titled for example
  "Google Drive - Quota exceeded"). With `GOOGLE_DRIVE_API_KEY` set, a warning
  `OE 2026: Drive API quota exceeded (HTTP 403: ...) - retrying via the public link`
  comes first (for the listing:
  `OE folder: Drive API quota exceeded (HTTP 429: ...) - retrying via the public folder view`);
  the error is only raised when the public link is blocked too.
  Consequences: the state is not updated, the
  Daily Evaluate run shows as failed, `evaluate` and `predict` still ran
  (predict on the committed state). **Do nothing**: the next day's run catches
  up, because every run rescans from 14 days (`LOOKBACK_DAYS`) before the last
  logged game. Do not work around the block with chunked Range requests. If it
  persists for days, download the current-year file by hand in a browser into
  `src/data/raw/oe/` (original name; it is reused while its modification time
  is under 20 hours old), run `python src/pipeline/daily_state.py` and commit
  the state files. `python src/pipeline/oe.py --check` tells whether the block
  is still active.
- **Drive API key problem.** Log: `Drive API key in GOOGLE_DRIVE_API_KEY is
  invalid (HTTP 400: ...)` or `Drive API error (HTTP 403: ...)` (for example
  the Drive API not enabled in the key's project, or the key restricted to
  another API). There is no fallback for these: fix or recreate the key
  ([Creating the Drive API key](#creating-the-drive-api-key)), or delete the
  secret to go back to the public links. `Drive API: file not found or not
  public` means the OE file id or its sharing changed. `Drive API: unexpected
  HTTP 3xx redirect to <host> (not followed)` means Drive redirected to a host
  other than https `*.googleusercontent.com` / `*.googleapis.com`, without a
  `Location`, or a second time; the key was not sent there. There is no
  fallback: check whether Drive changed its download flow, or delete the
  secret to go back to the public links.
- **Unmapped team.** Log: `unmapped team <code>/<name>` (and `unmapped team,
  skipped` for the match); the match gets no prediction. Add a row to
  `src/pipeline/team_aliases.csv`:
  `lolesports_code,lolesports_name,oe_teamid`, with the code and name exactly
  as lolesports spells them. Find the OE id from the committed log, for
  example `grep -h "Team Name" src/data/raw/team_games_2026.csv | head -1 | cut -d, -f10`
  (column 10 is `teamid`). A team that has no OE game yet cannot be mapped;
  its matches stay skipped until OE has one. After the fix, re-run
  **Daily Predictions** (`workflow_dispatch`) or check locally with
  `python src/pipeline/predict.py --dry-run`. A name that matches several OE
  teams logs `(ambiguous: n OE teamids)` and also needs an alias.
- **Failed evaluate.** Each match is isolated: one failing lookup is logged
  (`evaluation failed`), the others still resolve, `running_record` is still
  recomputed, and `evaluate()` raises once at the end so the run is red. The
  state update and its commit are unaffected. Re-running the workflow retries
  the unresolved matches. A match that is still undecided 7 days after its
  start (postponed, forfeited with 0-0 `gameWins`) only gets a warning;
  nothing auto-voids, so fix such a row by hand in Supabase if it matters.
- **Supabase paused or unreachable.** `predict.py` / `evaluate.py` raise
  `EnvironmentError` or a `supabase` client exception, and the keepalive ping
  fails too. Unpause the project in the Supabase dashboard (manual on the free
  tier) and re-run the workflow.
- **`42501 permission denied for table predictions`.** The GRANTs at the end of
  `schema.sql` are missing (newer Supabase projects grant nothing by
  default). Run them in the SQL editor, see
  [Applying schema.sql](#applying-a-supabase-migration).
- **lolesports API changed.** Calls raise on HTTP or shape errors rather than
  returning an empty result, so a changed API shows up as a red run. Run
  `python src/pipeline/lolesports.py --check`.
- **Incomplete game in OE.** Not a failure: `daily_state.py` logs
  `incomplete game (rows/roles/result), skipped for now` and a later run picks
  it up. A game OE adds a day or more late (LPL especially) is applied in
  arrival order; a full rebuild re-sorts.
- **Push conflict on the state commit.** The commit step already does
  `git pull --rebase` before `git push`, which handles another commit landing
  in between. A genuine conflict (two runs touching the same state file) needs
  manual resolution: re-run `daily_state.py` locally against current `main`
  and push.
- **Rare double write failure.** If `player_elo_current.parquet` is written but
  `player_elo_recent.parquet` is not, the next run can double-count those
  games. If a run logs a failure between the two writes, revert its state
  commit (or restore both files from the previous commit) before the next run.

### Re-enabling disabled workflows manually

If `keepalive.yml` itself is disabled and the daily workflows stop firing:

```bash
gh workflow enable daily_evaluate.yml --repo IcaruzSoftware/riftside-oracle
gh workflow enable daily_predict.yml  --repo IcaruzSoftware/riftside-oracle
gh workflow enable keepalive.yml      --repo IcaruzSoftware/riftside-oracle
```

## Checking the sources

```bash
python src/pipeline/oe.py --check          # says Drive API vs public link, lists the Drive folder, starts a real download of the latest year, prints status / size / latency
python src/pipeline/lolesports.py --check  # getLeagues; requires lck, lpl, lec, worlds
```

Both exit 0 on success, 1 on failure. `oe.py --check` deliberately starts a
full (non-Range) download and reads only the first 20 KB, because Drive's quota
block only affects full downloads. The same checks run from a GitHub runner
via `check_sources.yml`.

## Full historical bootstrap / ELO rebuild

Only needed if the raw data or the state is lost, or the ELO or feature logic
changes. The raw files must be in `src/data/raw/oe/` (download by hand, or
`python src/scripts/bootstrap_data.py`, about 856 MB; files younger than 20
hours are skipped and `--force` re-downloads them).

```bash
python src/pipeline/elo.py --force                      # ~21 s, full 2014-2026 replay; also recomputes role_norms.json
python src/pipeline/daily_state.py --init               # team_games_*.csv from the same OE files (delete the existing ones first)
python src/pipeline/build_dataset.py --force            # ~35-65 s, 93,753 games since 2017
python src/pipeline/train.py --trials 100 --force       # ~25-40 min
```

- Always rebuild the ELO before `build_dataset.py`: it reads the uncommitted
  `player_elo.parquet`, and a stale one silently falls back to the current ELO
  for later games (look-ahead).
- `daily_state.py --init` refuses to overwrite: delete the existing
  `src/data/raw/team_games_*.csv` first. Run it right after `elo.py --force`
  so the CSVs and the ELO state are built from the same OE files and match.
  Commit the ELO state only together with a fresh `--init`: otherwise the
  games missing from `team_games` are lost for good, because `daily_state`
  only passes gameids that are new to `team_games`.
- `elo.py --force` prints the top 10 players by `outcome_elo`; well-known
  LCK/LPL names should dominate.
- `train.py` prints the test-slice metrics, both baselines and a `GATE` line
  (PASS only if the calibrated model beats the outcome-ELO baseline on log loss
  **and** accuracy on the major-league slice). Do not ship a FAIL without
  looking at it. Details in [model.md](model.md).
- Commit only the state files (`team_games_*.csv`, the two `player_elo_*`
  parquets, `role_norms.json`) and `src/models/xgb_model.pkl`. Never the raw OE
  files, `player_elo.parquet` or `feature_matrix.parquet`.
- Retraining is not part of the daily pipeline. Re-pin `scikit-learn` and
  `xgboost` in `requirements.txt` together if you change the training
  versions.

<a id="applying-a-supabase-migration"></a>

## Applying schema.sql on a new Supabase project

There is no `migrations/` directory: a fresh project is set up by running
`src/supabase/schema.sql` once in the Supabase SQL editor. It creates
`predictions`, `shap_values` and `running_record` (seeded with one zero row),
the indexes, the RLS policies (public `SELECT` only), and the table GRANTs.
Everything in it is idempotent (`IF NOT EXISTS`, `DROP POLICY IF EXISTS`,
`ON CONFLICT DO NOTHING`), so it is safe to re-run; it does not alter a table
that already exists in a different shape, so a column change on a live
database needs a hand-written `ALTER TABLE`.

**The GRANTs are required.** Newer Supabase projects do not grant table
privileges by default, and without them the first predict run fails with
`42501 permission denied for table predictions`, even with the service-role
key. The last statements of `schema.sql` (schema `USAGE`; full CRUD for
`service_role`; `SELECT` for `anon` and `authenticated`) fix that. If you
created the tables some other way, run just those statements.

After creating the project:

1. Put the project URL and the **service-role** key into the GitHub secrets
   `SUPABASE_URL` / `SUPABASE_KEY`.
2. Put the URL and the **anon** key into `web/.env.local`.
3. Enable the workflows.

## Deploying the web app

**Not automatic: pushing to `main` will not update the site.** The site is a
static export (`output: "export"` in `web/next.config.ts`), to be served by the
owner's Plesk/nginx server (not deployed yet). Because it is static, all data fetching happens in
the browser directly against Supabase (anon key), so new predictions and
results show up without a redeploy. Only code or layout changes need the steps
below.

1. Create `web/.env.local` (gitignored, never commit it) from
   `web/.env.local.example`. Both values are public and get baked into the
   static JS bundle:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://pdblqlbgrelyqjgkpdte.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
   ```
   **Anon key only, never the service-role key.** Without these variables the
   build still succeeds, but `next.config.ts` prints a warning and the pages
   show empty states.
2. Build locally:
   ```bash
   cd web && npm install && npm run build
   ```
   This produces the static export in `web/out/`.
3. Manually upload the contents of `web/out/` to the document root of the
   Riftside Oracle subdomain on the Plesk server. The site metadata assumes
   `https://riftside-oracle.gerritvisser.de`.

**For `/card` to be iframed on gerritvisser.de, a CSP `frame-ancestors` header
must be set in the server/nginx config when deploying, not by the Next.js app**:
a static export cannot set response headers itself. Re-check it after any
vhost change.

The `/card` route (420×260) is the standalone widget meant to be embedded as an
iframe on the portfolio site (`IcaruzSoftware/portfolio`, https://gerritvisser.de);
the embed is an open owner step (see [HANDOFF.md](../HANDOFF.md)). Once embedded,
the iframe points at the subdomain's `/card`, so uploading a new build updates
the embedded card too. A breaking change to `/card`'s markup or query must be
avoided and needs a check in that repo.

Local development: `cd web && npm install && npm run dev`; `npm run lint`
runs `next lint`. If `next dev` crashes with "segment-explorer-node", it is
sharing `web/.next` with `npm run build`: delete `web/.next`.

### Previewing the web app without data

- `/demo` renders `GameCard` with mock rows (upcoming with SHAP, resolved
  correct and wrong, no logos or a long team name), labelled "demo / not real
  data". It exists in the static export too.
- `?demo=1` on any page (`/?demo=1`, `/card?demo=1`) serves the mock rows
  from `web/lib/mock.ts` instead of Supabase, **only under `npm run dev`**
  (gated on `NODE_ENV`, so the production build ignores it). The mock times
  are relative to the viewer's today.

## Season rollover

Nothing to run manually. OE publishes a new year file with a new Drive id each
January; `oe.list_files()` resolves ids from the Drive API listing (with a
key) or the folder view on every run, and
`daily_state.py` reads both the previous and the current year file in early
January.

- The public record only counts predictions with `match_date >= RECORD_START`
  (`"2026-10-15"` in `evaluate.py`). Bump the constant to reset the ticker.
- League renames between seasons (OE and lolesports codes) need the sets
  updated: `predict.PREDICT_LEAGUES` / `OE_LEAGUE`, `elo.LEAGUE_PRIOR` /
  `elo.INTERNATIONAL`, `train.MAJOR`. See
  [data.md](data.md#shape-and-quirks) for how codes changed in the past.
- Rosters change in the off-season, so early-season predictions are the
  weakest; see [model.md](model.md).
