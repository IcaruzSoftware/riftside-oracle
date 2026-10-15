# Architecture

## Components

```
                     ┌─────────────────────────────────────────────┐
                     │             One-time bootstrap               │
                     │  bootstrap_data.py → src/data/raw/            │
                     │  elo.py --force    → player_elo*.parquet      │
                     │  build_dataset.py  → feature_matrix.parquet   │
                     │  train.py          → models/xgb_model.pkl     │
                     └─────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────── Daily automation (GitHub Actions) ────────────────────────────────┐
│                                                                                                      │
│  daily_evaluate.yml (via daily_run.py --evaluate) — two independent steps, both always attempted:  │
│    step 1  cdn.nba.com (box scores) ─▶ evaluate.py ─▶ Supabase: predictions.correct, running_record│
│    step 2  cdn.nba.com (schedule+box scores) ─▶ daily_state.py ─▶ src/data/raw/*.csv (game logs,   │
│            player stats) ─▶ elo.py update_elo/update_team_assignments ─▶ player_elo_current /      │
│            player_elo_recent .parquet (committed)   [step 2 still runs even if step 1 raised]      │
│                                                                                                      │
│  daily_predict.yml (via daily_run.py --predict):                                                   │
│    cdn.nba.com (schedule) ─▶ predict.py ─▶ features.py (live mode) ─▶ xgb_model.pkl ─▶ Supabase:   │
│    predictions, shap_values                                                                        │
│                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
         Supabase (Postgres, RLS)  ──▶  web/ (Next.js static export)  ──▶  manual build+upload  ──▶  Plesk/nginx
```

Two independent NBA data sources feed this (details in
[data.md](data.md)): **stats.nba.com** via `nba_api`, used only for the
historical bootstrap, and **cdn.nba.com**, used by everything that runs
daily. They never mix within one run.

## Data flow, end to end

1. **Bootstrap** (`bootstrap_data.py`, one-time / rare) downloads game logs,
   player season stats, and 3 box-score types per game from stats.nba.com for
   2015-16..2025-26 into `src/data/raw/`.
2. **ELO** (`elo.py`, `build_elo`) processes every game chronologically,
   producing the full pre-game ELO history (`player_elo.parquet`, not
   committed) and the two committed live-state files
   (`player_elo_current.parquet`, `player_elo_recent.parquet`).
3. **Feature matrix** (`build_dataset.py`) walks every historical game and
   calls `features.py`'s `build_feature_matrix()` in **training mode**,
   producing `feature_matrix.parquet`.
4. **Train** (`train.py`) fits and calibrates the XGBoost model on that
   matrix, producing the committed `models/xgb_model.pkl`.
5. **Evaluate** (`evaluate.py`, run as part of `daily_run.py --evaluate`,
   first of its two independent steps) resolves any `predictions` rows still
   missing a result against the CDN box score, then recomputes
   `running_record` from `RECORD_START` onward.
6. **Daily state** (`daily_state.py`, the second, independent step of
   `daily_run.py --evaluate` — it still runs even if step 5 raised) fetches
   newly-completed games from the cdn.nba.com schedule + box scores, appends
   to the game-log / player-stats CSVs, and calls `elo.py`'s `update_elo`
   (stateful games) / `update_team_assignments` (preseason) to incrementally
   advance the two committed ELO state files.
7. **Predict** (`predict.py`, run as `daily_run.py --predict`) fetches
   today's schedule, calls `features.py`'s `build_feature_matrix()` in
   **live mode** for each stateful game, runs the calibrated model, and
   upserts `predictions` + `shap_values` to Supabase.
8. **Web** (`web/`, Next.js static export) reads `predictions`,
   `shap_values`, and `running_record` directly from Supabase client-side
   (anon key, RLS-restricted to `SELECT`) and renders the site and the
   `/card` iframe embed.

## Module-by-module responsibilities

**`src/pipeline/`**

- `collect.py` — the original full nba_api collector (proxy-pool aware,
  season + per-game + per-player endpoints). Superseded for the actual
  rebuild by `bootstrap_data.py`, which reuses its low-level helpers
  (`_api_call`, `_to_dict`, `_save_json`, `ProxyPool`, `RAW_DIR`) but only
  fetches what `elo.py` needs. Several of `collect.py`'s endpoints
  (matchups, hustle, synergy, tracking-stat breakdowns, lineups, player bio,
  and 4 of its 6 box-score types) aren't read by any other module today —
  see [data.md](data.md).
- `elo.py` — the player ELO system: skill scoring, ranking, zero-sum deltas,
  full rebuild (`build_elo`) and incremental update (`update_elo`,
  `update_team_assignments`), plus the CDN box-score adapter
  (`load_cdn_game_players`) that lets the daily pipeline reuse the exact same
  algorithm with a thinner stat set. Full detail in [elo.md](elo.md).
- `nba_cdn.py` — thin cdn.nba.com client (schedule + box score), game-ID
  prefix helpers (`is_preseason`/`is_all_star`/`is_stateful`), team
  tricode↔ID lookup. No proxies; relies on a browser-like header set (see
  [operations.md](operations.md)).
- `daily_state.py` — brings the committed state (game logs, player stats,
  ELO) up to date from the CDN. Replaces the older, removed
  `update_season.py`. Entry point: `run(date=None, dry_run=False)`.
- `features.py` — feature engineering, both training and live paths (see
  below). Entry point: `build_feature_matrix(...)`.
- `build_dataset.py` — walks historical games and calls `features.py` in
  training mode to build `feature_matrix.parquet`.
- `train.py` — Optuna + XGBoost + Platt calibration training pipeline.
- `predict.py` — prediction for a single matchup (CLI) or all of today's
  games (`predict_todays_games`, called by `daily_run.py`).
- `evaluate.py` — resolves pending predictions against CDN results and
  recomputes `running_record`.
- `seed_finals.py` — one-time historical seed script (already run; don't
  re-run — see [operations.md](operations.md)).
- `shap_export.py` — global SHAP beeswarm plot + summary JSON. Standalone
  utility, not called by the daily pipeline (see [data.md](data.md)).

**`src/scripts/`**

- `bootstrap_data.py` — resumable stats.nba.com downloader for the ELO
  rebuild (phase 1: game logs + player stats; phase 2: per-game box scores).
- `daily_run.py` — the GitHub Actions entry point. `--predict`, `--evaluate`,
  or both (default). The evaluate path also runs `daily_state.run()` so the
  committed state is current before the next prediction run.

**`web/`** — Next.js 15 (App Router, static export, `output: "export"`).
`app/(main)/` is the public site (today's predictions, running record, recent
predictions, About-the-model section); `app/card/` is the standalone 420×260
iframe widget embedded on the portfolio site; `lib/supabase.ts` is the only
place that talks to Supabase (anon key, browser-side, since static export has
no server runtime for secrets or ISR).

## Live vs. training feature paths (why they differ)

`build_feature_matrix()` takes a `use_current_elo` flag that changes exactly
one thing structurally — how `get_elo_features()` and
`get_player_form_features()` source their data — but the *consequence*
ripples through the other ELO-adjacent inputs too:

| | Training (`use_current_elo=False`) | Live (`use_current_elo=True`) |
|---|---|---|
| ELO source | `player_elo.parquet`, this exact `game_id`'s pre-game snapshot | `player_elo_current.parquet`, post-game state |
| Roster | Players who actually appeared in *this* game's box score, minus inactives (real subtraction if `boxscore_summary` exists for that game — true for the data the committed model was trained on; the re-downloaded bootstrap dataset lacks it, so it's a no-op there — see [data.md](data.md)) | Players whose most recent ELO-state appearance was for that team (current season, falling back to previous season if fewer than 8 found, never older than the previous season) |
| Form trajectory | `player_elo.parquet` full history per player | `player_elo_recent.parquet`, last 11 snapshots per player |
| Season stats (PPG weights, star-out) | Full final-season file (`player_stats_df` covers the whole season, including games after the one being featured) | Season-to-date file (only games played so far), falling back to the previous season for early-season players |

The reason live mode exists at all: at prediction time there is no future
box score to read a real lineup from, and only a fraction of the full
historical dataset (the two small committed state files) is available to the
daily-pipeline runner — `player_elo.parquet` and the full per-season raw box
scores are not shipped to GitHub Actions. This is also *why* the reported
backtest accuracy is optimistic relative to live performance — see
[model.md](model.md) for the full breakdown of what training sees that live
prediction doesn't.

Everything else in `build_feature_matrix()` (rolling stats, home/away
splits, rest, head-to-head, availability, efficiency differential, streaks)
reads the same `game_log_df` / `player_stats_df` shape in both modes and
doesn't branch on `use_current_elo`.
