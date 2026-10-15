# Courtside Oracle

An NBA game-outcome predictor: a custom per-player ELO system feeds an
XGBoost classifier, predictions run daily via GitHub Actions, and results are
tracked publicly in Supabase and shown on a small Next.js site.

**Live site:** https://courtside-oracle.gerritvisser.de
**Embedded widget:** the `/card` page is embedded as a 420×260 iframe on
https://gerritvisser.de

Every prediction is logged with a confidence score before tip-off and marked
correct/incorrect the next morning once the game is final — the running
record on the site is the real, live track record, not a backtest number.

## Architecture

```
                     One-time bootstrap
  stats.nba.com → bootstrap_data.py → src/data/raw/
                                          │
                                          ▼
                              elo.py --force → player_elo*.parquet
                                          │
                                          ▼
                  build_dataset.py → feature_matrix.parquet → train.py → models/xgb_model.pkl

                     Daily automation (GitHub Actions)
  cdn.nba.com → daily_state.py  → src/data/raw/*.csv (appended) → elo.py (incremental update)
  cdn.nba.com → evaluate.py     → Supabase: predictions.correct, running_record
  cdn.nba.com → predict.py      → features.py (live) → xgb_model.pkl → Supabase: predictions, shap_values
                                          │
                                          ▼
      Supabase (Postgres, RLS)  →  web/ (Next.js, static export)  →  manual build+upload  →  Plesk/nginx
```

Full breakdown of each stage: [docs/architecture.md](docs/architecture.md).
Every file the pipeline reads/writes and every Supabase table:
[docs/data.md](docs/data.md).

## Quick start (local)

Requires Python 3.12 (matches CI). From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file or export directly (never commit these):

```bash
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-service-role-key
```

Set up the Supabase schema once — paste `src/supabase/schema.sql` into the
Supabase SQL editor and run it. (Upgrading an existing, older database?
See [docs/operations.md](docs/operations.md#applying-a-supabase-migration).)

Run the tests:

```bash
python -m pytest
```

Run the daily pipeline manually:

```bash
python src/scripts/daily_run.py                          # evaluate + predict, both, for today (ET)
python src/scripts/daily_run.py --predict --dry-run       # print instead of writing to Supabase
python src/scripts/daily_run.py --evaluate --date 2026-11-05
```

Predict a single matchup from the CLI:

```bash
python src/pipeline/predict.py --game-id 0042500237                       # a game already in the data
python src/pipeline/predict.py --home NYK --away OKC --date 2026-06-05    # a hypothetical matchup
```

## Daily automation

Three scheduled GitHub Actions workflows keep the site current:

- **Daily Evaluate** (`daily_evaluate.yml`) — 3:00 AM ET, resolves yesterday's
  predictions against final box scores, recomputes the running record, and
  advances the committed ELO/state files from the NBA CDN.
- **Daily Predictions** (`daily_predict.yml`) — triggered by Daily Evaluate's
  completion (not its own schedule), predicts every
  regular-season/playoff/play-in/Cup-final game tipping off that day.
- **Keepalive** (`keepalive.yml`) — pings Supabase daily so the free-tier
  project doesn't auto-pause, and re-enables the other two workflows (GitHub
  auto-disables scheduled workflows in public repos after 60 days of
  inactivity).

Full schedules, required secrets, and what to do when one of these fails:
[docs/operations.md](docs/operations.md).

## The website / iframe

`web/` is a Next.js 15 app, statically exported (`output: "export"`) and
served from the owner's own Plesk/nginx server — there's no server runtime,
so all Supabase reads happen client-side with the public anon key.
`app/(main)/` is the full site; `app/card/` is the compact widget embedded
elsewhere. Env vars (`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`)
go in `web/.env.local` at build time, not in GitHub secrets. Deploying is a
manual build + upload, not a push-to-deploy — see
[docs/operations.md](docs/operations.md#deploying-the-web-app). Local dev:
`cd web && npm install && npm run dev`.

## Bootstrap / full rebuild

The historical dataset (2015-16 through 2025-26, ~14,100 games) and the ELO
state were rebuilt once from scratch with `src/scripts/bootstrap_data.py`
(expect several hours, unattended) followed by `python src/pipeline/elo.py --force`.
You shouldn't need to do this unless the raw data or ELO state is lost again
— procedure and background: [docs/operations.md](docs/operations.md).

## The model

An XGBoost binary classifier (home-team win probability), Platt-calibrated,
trained on a custom 7-skill-per-player ELO system plus rolling team form,
rest, head-to-head, and efficiency features. On a chronological, held-out
test set of 2,116 games: **67.5% accuracy, 0.729 AUC-ROC, 0.209 Brier score,
0.606 log loss**.

**Honest note:** that backtest number is optimistic relative to what the
live pipeline actually delivers, mainly because training sees each
historical game's real box-score lineup and full-season stats, while live
prediction has to approximate both from the committed ELO state and
season-to-date stats. Expect live accuracy below 67.5% — the site's running
record is the number to trust. Full explanation of the gap, feature list,
and training procedure: [docs/model.md](docs/model.md). Player ELO design in
detail: [docs/elo.md](docs/elo.md).

## Docs

- [docs/architecture.md](docs/architecture.md) — components, data flow, module responsibilities
- [docs/operations.md](docs/operations.md) — runbook: workflows, secrets, failure modes, bootstrap, migrations, deploys
- [docs/data.md](docs/data.md) — every data file and Supabase table, columns the code relies on
- [docs/elo.md](docs/elo.md) — the player ELO system
- [docs/model.md](docs/model.md) — features, training, metrics, caveats

For an AI coding session working in this repo, start with
[CLAUDE.md](CLAUDE.md) instead of this file.

## License

MIT — see [LICENSE](LICENSE).
