# Riftside Oracle

A pre-match predictor for professional League of Legends. For every upcoming
series in the major leagues and the international events (Bo1, Bo3, Bo5) it
publishes a win probability before the first game, and marks the pick correct
or wrong once the series is over. The running record on the site is the live
track record, not a backtest number.

Forked from courtside-oracle (NBA).

**Site:** https://riftside-oracle.gerritvisser.de (subdomain not confirmed yet)
**Embedded widget:** the `/card` page is a 420×260 iframe, meant to be
embedded on https://gerritvisser.de (embed not set up yet)

> Riftside Oracle isn't endorsed by Riot Games and doesn't reflect the views or
> opinions of Riot Games or anyone officially involved in producing or managing
> Riot Games properties. Riot Games, and all associated properties are
> trademarks or registered trademarks of Riot Games, Inc.

Match data: Oracle's Elixir and lolesports.

## How it works

1. **Data.** Oracle's Elixir publishes one CSV per year of pro match stats
   (2014 onward); the lolesports API provides the schedule and results.
2. **Player ELO.** Every player carries five performance ratings (combat,
   damage, economy, laning, vision, each normalised by role) plus an outcome
   ELO that carries strength across regions.
3. **Model.** An XGBoost classifier (Platt-calibrated) turns the two teams'
   ELOs, recent form, side splits, rest, head-to-head and roster features into
   the probability that the blue side wins one game. Averaged over both side
   assignments and compounded over the best-of length, that gives the series
   probability.
4. **Automation.** GitHub Actions resolve finished series, advance the ELO
   state and predict the next 36 hours of matches every day. Results go to
   Supabase and are shown on a static Next.js site.

The model sees the teams before the draft, so picks and bans are not part of
the prediction.

## Results

On a held-out test period (2025-05-20 to 2026-10-14), counted per series:

- **70.2%** accuracy across all leagues (6,260 series)
- **69.9%** on the major leagues (1,453 series; 72.7% in Bo3s)

On the major leagues almost all of the signal sits in the outcome ELO; the
model is level with a plain "higher outcome ELO wins" baseline there. Across
all leagues it is significantly better in log loss; series accuracy is 70.2% vs
69.8% for that baseline. Details and caveats: [docs/model.md](docs/model.md).

## Quick start (local)

Requires Python 3.12 (matches CI). From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt pytest
python -m pytest              # no Supabase or raw data needed
```

Predict without touching Supabase:

```bash
python src/scripts/daily_run.py --predict --dry-run                         # next 36h of matches
python src/pipeline/predict.py --team1 "Gen.G Esports" --team2 T1 --bo 5    # ad-hoc series
```

Writing predictions needs `SUPABASE_URL` and `SUPABASE_KEY` (service-role key)
in the environment, and the schema from `src/supabase/schema.sql` run once in
the Supabase SQL editor. Rebuilding the ELO or retraining needs the raw
Oracle's Elixir files; see [docs/operations.md](docs/operations.md).

The site lives in `web/` (Next.js 15, static export):
`cd web && npm install && npm run dev`. The anon key and project URL go in
`web/.env.local`; deploying is a manual build and upload, see
[docs/operations.md](docs/operations.md#deploying-the-web-app).

## Docs

- [docs/architecture.md](docs/architecture.md) — components, data flow, module responsibilities
- [docs/operations.md](docs/operations.md) — runbook: workflows, secrets, failure modes, bootstrap, schema changes, deploys
- [docs/data.md](docs/data.md) — every data file and Supabase table, columns the code relies on
- [docs/elo.md](docs/elo.md) — the player ELO system
- [docs/model.md](docs/model.md) — features, training, metrics, caveats
- [HANDOFF.md](HANDOFF.md) — current status, decisions and open points

For an AI coding session working in this repo, start with
[CLAUDE.md](CLAUDE.md) instead of this file.

## License

MIT — see [LICENSE](LICENSE).
