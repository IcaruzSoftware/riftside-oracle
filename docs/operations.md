# Operations runbook

## Background

The original build PC was reset and `src/data/raw/` / the ELO parquet state
were never committed — the pre-fix `.gitignore` ignored the `data/` dirs
wholesale, and git cannot re-include a file whose parent directory is itself
ignored, so the `!` negations for the state files silently did nothing. The
historical data was re-downloaded from scratch with `src/scripts/bootstrap_data.py`
(2015-16..2025-26, 14,128 games, traditional/advanced/tracking box scores) and
the ELO rebuilt. The fixed `.gitignore` now ignores `src/data/raw/*` and
`src/data/processed/*` but whitelists the state files listed in
[data.md](data.md), so this shouldn't recur — but if `git status` ever shows
none of the state files tracked after a fresh clone, that's the symptom to
check for.

## The three GitHub Actions workflows

All three live in `.github/workflows/` and run on `ubuntu-latest` with Python
3.12 (matches the local `.venv`).

### `daily_predict.yml` — Daily Predictions

- **Trigger:** `workflow_run` on `daily_evaluate.yml` completing (any
  conclusion), not its own cron — so it always runs right after evaluate
  commits fresh state, regardless of scheduler delay. Also `workflow_dispatch`
  (manual trigger from the Actions tab).
- **Runs:** `python src/scripts/daily_run.py --predict`.
- **Commits nothing.** Only writes to Supabase (`predictions`, `shap_values`)
  via `SUPABASE_URL` / `SUPABASE_KEY`.

### `daily_evaluate.yml` — Daily Evaluate

- **Schedule:** `0 7 * * *` (07:00 UTC = 3:00 AM ET, 2:00 AM in winter/EST,
  after West Coast games are final). Also `workflow_dispatch`.
- **Runs:** `python src/scripts/daily_run.py --evaluate`, which runs two
  independent steps, each in its own try/except: first `evaluate.py` (resolve
  pending predictions, recompute `running_record`), then `daily_state.run()`
  (advance the committed game logs / player stats / ELO state from the CDN).
  A failure in the first does not skip the second — either failing still
  fails the job (non-zero exit) so the workflow shows red.
- **Commits:** `git add src/data/` then commits + pushes only if that stages
  a diff (`git diff --cached --quiet` gate). Since `.gitignore` only allows
  the whitelisted state files under `src/data/`, this can only ever commit
  `game_log_*.csv`, `player_season_stats_*.csv`, `player_elo_current.parquet`,
  and `player_elo_recent.parquet` — never raw box scores. Commit message:
  `chore: daily state update [skip ci]`. Needs `permissions: contents: write`
  (already set).

### `keepalive.yml` — Keepalive

- **Schedule:** `17 6 * * *` (06:17 UTC, off the hour on purpose).
  Also `workflow_dispatch`.
- **Purpose 1:** Supabase free-tier projects auto-pause after ~7 days with no
  activity. A `curl -f` authenticated read against `running_record` keeps the
  project awake and fails the job loudly if Supabase is unreachable/paused.
- **Purpose 2:** GitHub disables scheduled workflows in public repos after 60
  days with no repository activity (this already happened to both
  `daily_*.yml` once, in August 2026). The second step runs
  `gh workflow enable daily_evaluate.yml / daily_predict.yml / keepalive.yml`
  on every run (`if: always()`, so it still re-enables the workflows even if
  the Supabase ping above failed), keeping all three schedules alive
  indefinitely. Needs `permissions: actions: write`.
- **The daily CDN calls go through residential proxies** (`NBA_PROXIES`
  secret): cdn.nba.com returns 403 from GitHub-hosted runners (NBA blacklists
  cloud IP ranges), so direct access only works locally. See the "NBA CDN
  block / 403" entry under "When a job fails" below.

## Required GitHub secrets

| Secret | Used by | Notes |
|---|---|---|
| `SUPABASE_URL` | `daily_predict.yml`, `daily_evaluate.yml`, `keepalive.yml` | Public project URL, safe to also keep in local `.env` |
| `SUPABASE_KEY` | same | **Service-role key** — bypasses RLS, full read/write. Never expose this to the web app (see `web`'s `NEXT_PUBLIC_SUPABASE_ANON_KEY`, which is a different, RLS-restricted key baked in at build time via `web/.env.local` — not a GitHub secret, see "Deploying the web app" below) |
| `NBA_PROXIES` | `daily_predict.yml`, `daily_evaluate.yml`, `check_nba_proxies.yml` | 1–3 static residential proxies, comma-separated `http(s)://user:pass@host:port`. URL-encode special characters in the password (e.g. `@`→`%40`) or urllib3 fails to parse it. Unset/empty → direct connection (local dev). Required in CI (see below). |

`PROXIES` (singular, no `NBA_` prefix) is **obsolete** — it was only ever read
by `collect.py`'s proxy pool, which nothing in the current daily pipeline uses.
Safe to delete from repo secrets. The pipeline's proxy support is the separate
`NBA_PROXIES` secret above (CDN failover), not `PROXIES`.

## When a job fails

Check the failed run's logs in the Actions tab first — every pipeline script
logs at `INFO` level with timestamps. Common causes:

- **Supabase paused / unreachable.** `predict.py` / `evaluate.py` raise
  `EnvironmentError` or a `supabase` client exception; `keepalive.yml`'s ping
  step will also be failing. Reopen the project in the Supabase dashboard
  (unpausing is manual on the free tier) and re-run the workflow
  (`workflow_dispatch`).
- **NBA CDN block / 403.** **Confirmed on the first live runs:** cdn.nba.com
  returns 403 from GitHub-hosted runners even with the correct `CDN_HEADERS`
  (NBA blacklists cloud IP ranges); stats.nba.com times out from runners too;
  ESPN's endpoints are reachable. Because the owner wants to keep NBA data, the
  daily jobs route the CDN calls through the residential proxies in the
  `NBA_PROXIES` secret. `nba_cdn` tries the proxies in order, sticks to the last
  one that worked, and treats a 403 (or a connection error/timeout) from a proxy
  as "this IP is blocked → fail over to the next". A 404 is still normal (game
  not played yet → `fetch_boxscore()` returns `None`). If **every** proxy fails,
  the fetch raises a summary naming each `proxy #i (host:port): <reason>` —
  host:port only, never credentials — and the job fails loudly (silently skipping
  a day's state update would be worse). **Fix:** rotate or replace the blocked
  proxy in the `NBA_PROXIES` secret and re-run the workflow. **Verify proxies
  first** with the manual **Check NBA Proxies** workflow (`check_nba_proxies.yml`,
  `workflow_dispatch`) or locally:
  `NBA_PROXIES='http://user:pass@host:port' python src/pipeline/nba_cdn.py --check`
  — it fetches the schedule + a box score through each proxy individually and
  prints status / size / latency, exiting 0 if at least one proxy fully works.
- **State conflict / merge conflict on push.** `daily_evaluate.yml`'s commit
  step already does `git pull --rebase origin <branch>` before `git push`, so
  the common case — another commit landed on `main` between the checkout and
  the push (e.g. a manual `workflow_dispatch` overlapping the schedule) — is
  handled automatically. A genuine conflict (two runs racing on the *same*
  state file, so the rebase itself can't apply cleanly) still needs manual
  resolution — re-run `daily_state.py` locally against the current `main` and
  push the result.
- **Missing box score yet.** Not a failure — a 404 from the CDN is the normal
  case for a game that hasn't been played yet. `daily_state.py` logs
  `"box score not available yet"` and simply skips that game until a later
  run picks it up (it re-scans from `LOOKBACK_DAYS` before the last logged
  game every time, so nothing is lost).

## Re-enabling disabled workflows manually

If `keepalive.yml` itself is disabled (or hasn't run yet) and the daily
workflows stop firing:

```bash
gh workflow enable daily_evaluate.yml --repo <owner>/courtside-oracle
gh workflow enable daily_predict.yml  --repo <owner>/courtside-oracle
gh workflow enable keepalive.yml      --repo <owner>/courtside-oracle
```

## Full historical bootstrap / ELO rebuild

Only needed if the raw data or ELO state is lost again, or to extend the
historical window to a new season from scratch. Expect this to run for
several hours unattended — the script is written for that (see
`bootstrap_data.py`'s module docstring for the sleep-prevention and backoff
behavior).

```bash
python src/scripts/bootstrap_data.py                 # phase 1 then phase 2, ~7h total
python src/scripts/bootstrap_data.py --phase 1        # game logs + player season stats only (fast)
python src/scripts/bootstrap_data.py --phase 2 --limit 3   # box scores, first 3 pending games (smoke test)
```

Exit code is `2` if any game is still missing a box score at the end of a
phase-2 run (see `src/data/raw/missing_boxscores.txt` for which ones) — safe
to just re-run, it resumes from wherever it left off and never re-downloads a
complete game.

Then rebuild ELO from the downloaded box scores:

```bash
python src/pipeline/elo.py --force
```

This writes `player_elo.parquet` (full history, not committed) and the two
committed state files (`player_elo_current.parquet`,
`player_elo_recent.parquet`). **Commit only the state files** — never the
contents of `src/data/raw/` or `player_elo.parquet` (see [data.md](data.md)
for exactly what `.gitignore` allows through).

If you also need to retrain the model against the new data, continue with
`python src/pipeline/build_dataset.py` then `python src/pipeline/train.py`
(see [model.md](model.md)).

## Applying a Supabase migration

New setups: run `src/supabase/schema.sql` once in the Supabase SQL editor —
it already includes everything, including `game_time_utc` and the RLS
policies.

Existing deployments created before `game_time_utc` / RLS existed: apply
`src/supabase/migrations/001_game_time_and_rls.sql` in the SQL editor. It's
idempotent (`ADD COLUMN IF NOT EXISTS`, `DROP POLICY IF EXISTS` before each
`CREATE POLICY`) — safe to re-run.

**Apply migration 001 before deploying any web build that queries
`game_time_utc`** (the `/card` page's live/upcoming/tipoff-label logic reads
this column via `getCardPrediction()` in `web/lib/supabase.ts`) — querying a
column that doesn't exist yet fails that request outright on an
unmigrated database.

## Deploying the web app

**Not automatic — pushing to `main` does not update the live site.** The
site is a static export (`output: "export"` in `next.config.ts`) served from
the owner's own Plesk/nginx server (confirmed: `courtside-oracle.gerritvisser.de`
and `gerritvisser.de` both resolve to the same server, 185.45.149.138; the
live `/card` files on it date from 2026-06-19). There's a Vercel project
somewhere in this repo's history too, but what's actually serving the live
site is that Plesk server — treat Vercel as not relevant to the deploy
procedure below.

Because it's a static export, all data fetching happens client-side in the
browser directly against Supabase (anon key) — so changes to the *data*
(predictions, running record) show up immediately with no redeploy. Only
code/layout changes need the steps below.

Deploy procedure:

1. Create `web/.env.local` (gitignored, never commit it) with the two
   build-time env vars — both are public values that get baked into the
   static JS bundle, not secrets:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://pxvimjflsishbfrublwp.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
   ```
2. **If this build is the first one to query `game_time_utc`**, confirm
   [migration 001 has been applied](#applying-a-supabase-migration) to the
   target Supabase project first — otherwise the `/card` page's
   `getCardPrediction()` query fails outright.
3. Build locally:
   ```bash
   cd web && npm run build
   ```
   This produces the static export in `web/out/`.
4. Manually upload the contents of `web/out/` to the
   `courtside-oracle.gerritvisser.de` subdomain's document root on the Plesk
   server.

**The CSP `frame-ancestors` header that allows `/card` to be iframed on
gerritvisser.de is set by the server/nginx config, not by the Next.js app** —
a static export can't set response headers itself (`next.config.ts`'s
`headers()` doesn't run in `output: "export"`). That header must stay
configured server-side independently of anything in this repo; it won't
survive a server/vhost change unless someone re-adds it there.

Local dev: `cd web && npm install && npm run dev`. `npm run lint` runs
`next lint`.

The `/card` route (420×260) is the standalone widget embedded as an iframe on
the portfolio site (`IcaruzSoftware/portfolio`, https://gerritvisser.de) —
the iframe points at `https://courtside-oracle.gerritvisser.de/card`, so
uploading a new build updates the embedded card too; there is nothing to
change in the portfolio repo itself unless the iframe URL changes.

## Season rollover

Nothing to run manually — season boundaries are derived from dates
everywhere (`_season_of()` / `_date_to_season()`: October or later = new
season). 2026-27: preseason from 2026-10-03, regular season opens
2026-10-20. The public running record only counts predictions with
`game_date >= RECORD_START` (currently `"2026-10-01"`, in `evaluate.py`) —
bump this constant if you want the ticker to reset for a new season instead
of accumulating across seasons. The 5 rows inserted by `seed_finals.py` (the
2026 Finals, predicted retroactively) have earlier `game_date`s and are
excluded by this cutoff regardless.
