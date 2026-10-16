# Handoff: Riftside Oracle (status 2026-10-16)

Read this first in a new session, then [CLAUDE.md](CLAUDE.md) and the spec
[RIFTSIDE_HANDOVER.md](RIFTSIDE_HANDOVER.md). Where this file and the spec
disagree, this file records a later decision and wins.

## TL;DR

- **Phases 0–5 of the handover are done, plus one model-improvement round.** Each one is committed and tested; every phase was reviewed by two independent reviewers.
- **Next step: Phase 6**, the docs: rewrite CLAUDE.md, README.md and docs/*.md for Riftside and remove the NBA references (RIFTSIDE_HANDOVER.md §9 Phase 6).
- **The daily pipeline works and is live** (since 2026-10-16, see "Live since 2026-10-16"): `daily_state.py`, `predict.py`, `evaluate.py` and `daily_run.py` are LoL code now, no longer broken NBA code. 91 tests, all passing.
- **The web app is built** (Phase 5), but not yet deployed and not yet checked against real Supabase data (see open point 4).
- **Model quality, at the level that matters to the owner (who wins the match):**
  - about **70% series accuracy** on the test period 2025-05-20..2026-10-14: 6,260 series across all leagues;
  - **69.9% on the major leagues**: 1,453 series; 72.7% in Bo3s.
- **Verdict after the improvement round: it is good as it is.** The remaining headroom is small and not reachable with the data we have (see "Model", below).

## Resume on another PC

```bash
git clone https://github.com/IcaruzSoftware/riftside-oracle.git && cd riftside-oracle
uv venv --python 3.12 .venv          # Python 3.12 is required (pins don't install on 3.13+)
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest           # 91 tests, all must pass
```

- **Raw data is not in git.** The Oracle's Elixir CSVs (13 files, ~856 MB) are gitignored.
  - Download them **by hand** from the Drive folder <https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH> into `src/data/raw/oe/`.
  - Keep the original names, e.g. `2026_LoL_esports_match_data_from_OraclesElixir.csv`.
  - `bootstrap_data.py` can do the same, but Drive often blocks scripted full downloads (see "Data access", below).
- **Committed state:** `player_elo_current.parquet`, `player_elo_recent.parquet`, `role_norms.json`, `src/models/xgb_model.pkl` and `team_games_{2014..2026}.csv`.
  - The `team_games` CSVs are committed state too (102,870 games, ~79 MB in total, largest file ~10 MB). `daily_state` writes them; do not hand-edit.
  - With these, the tests and live predictions work without the raw data.
  - Rebuilding the history or retraining needs the raw data.
- **Rebuild commands** (only when needed):

  ```bash
  .venv/bin/python src/pipeline/elo.py --force            # ~21 s, full 2014–2026 ELO replay
  .venv/bin/python src/pipeline/build_dataset.py --force  # ~35–65 s, 93,753 games since 2017
  .venv/bin/python src/pipeline/train.py --trials 100 --force   # ~25–40 min
  .venv/bin/python src/pipeline/daily_state.py --init     # regenerate team_games_*.csv from the local OE files
  ```

- **Local Windows note (2026-10-15 PC):** `uv` was not installed, so the venv was made with `py -3.12 -m venv`. pyarrow 26 is blocked by Windows Application Control on this PC, so `.venv` has `pyarrow==21.0.0` installed by hand. `requirements.txt` is unchanged and CI is unaffected.
- **GitHub state:**
  - The repo is pushed.
  - `check_sources.yml` passed on a GitHub runner, so OE and lolesports need no proxies.
  - `Daily Evaluate`, `Daily Predictions` and `Keepalive` are **enabled** and the secrets are set (since 2026-10-16).

## Commits so far

| Commit | What |
|---|---|
| 1882948 | courtside-oracle history merged in (so NBA code like `nba_cdn.py` is recoverable via `git show`) |
| d68d5bf | Phase 0: NBA code/state removed; `oe.py`/`lolesports.py` `--check`; `check_sources.yml` |
| c9008de | Phase 1: data layer: OE download/load/split, lolesports schedule/results/team resolution |
| 47b92b0 | Phase 2: player ELO (5 role-normalised skills + outcome ELO) |
| ee312df | Phase 3: features, dataset, training, baseline gate |
| 0435478, 3ee0afc, bebc2b6 | Improvement round: refined outcome ELO, Optuna fix, committed state + model, this file |
| 3cc93f2 | Phase 4: daily pipeline (`daily_state`, `predict`, `evaluate`, `daily_run`), workflows, `team_games_*.csv`, `schema.sql` |
| d00edc0 | `schema.sql`: explicit GRANTs |
| (this commit) | Phase 5: web app (Next.js static export, `/card` widget, `/demo`) |

## Live since 2026-10-16

- The owner created the Supabase project (`https://pdblqlbgrelyqjgkpdte.supabase.co`), ran `schema.sql`, set the GitHub secrets and enabled the workflows.
- The first predict run failed with 42501 "permission denied for table predictions". Newer Supabase projects don't grant table privileges by default.
  - Fix: explicit GRANTs at the end of `schema.sql` (service_role full CRUD; anon/authenticated SELECT). The owner ran them in the SQL editor.
- First Worlds day (2026-10-15, predicted ad hoc from the committed state): KC over C9 correct, FURIA over MVK wrong.

## Decisions made (by the owner or with the owner's OK) and deviations from the spec

1. **Players with a missing `playerid` are skipped** in the ELO, with no rating and no snapshot. The rest of their game is processed.
   - This only happens in minor leagues: about 2% of rows in LJL, LPLOL, LIT, LAS and similar.
2. **`LEAGUE_PRIOR` maps the historical OE codes too:**
   - 1600: LCK, OGN, LPL
   - 1500: LEC, EU LCS, LCS, NA LCS, LTA N, LTA
   - 1450: LCP, PCS, LMS, CBLOL, LTA S
   - default: 1350
3. **Roster feature** (deviates from spec §6.2): `roster_churn` = number of starters shared between the team's **last two lineups before the day**. It is identical in training and live.
   - The spec version ("starters who also started the previous game") would always be 5 in live mode.
   - `rookie_count` (< 20 games) is as specified.
4. **Point-in-time is day-level.** A game on day d sees only team_games rows from before day d and ELO "as of the start of day d".
   - Training uses each player's first pre-game snapshot on or after d. Live uses `player_elo_current`.
   - So game 2 of a series never sees game 1 in training, just as a pre-series live prediction can't.
5. **Outcome ELO refinements** from the improvement round. These deviate from spec §5.2, which says "zero-sum" and "no seasonal regression in v1":
   - `REGRESS = 0.2`: at a player's first domestic game of a new calendar year, the outcome ELO moves 20% toward that league's mean. The pull is applied after the pre-game snapshot, so training and live are identical; this is tested.
   - Margin of victory: K × `mov_mult` = sqrt of team |gold diff @15| relative to the mean. Partial games without @15 data use game length instead.
   - `NEW_K_MULT = 2` for a player's first 20 games. Games between new and established players are therefore **not zero-sum**; this is documented in `elo.py`.
6. **`train.py`:** the Optuna CV now runs on the **train slice only** (Courtside used train+calibration) and minimises log loss instead of maximising AUC.
7. **Git history:** this repo contains courtside-oracle's full history, merged with `--allow-unrelated-histories`.
8. **Drive quota: option A (Phase 4).** There is no Drive API path. A blocked download makes the state update fail loudly.
   - Evaluate and predict still run; predict uses the committed state; the run exits 1.
   - The next day's run catches up through `LOOKBACK_DAYS = 14`.
9. **`player_elo_recent.parquet` is committed in full every day**, with no pruning.
10. **`academy` was removed from `lolesports._SUFFIXES`**, because academy teams are separate teams.
11. **`RECORD_START = "2026-10-15"`** in `evaluate.py`.
12. **`schema.sql` was updated in Phase 4**, ahead of the spec's Phase 5, so the owner could create the Supabase project now.
    - §8.1 `predictions` table; `model_metadata` dropped; `shap_values` and `running_record` unchanged.

## Phase 4: implementation choices

- **`team_games_{2014..2026}.csv`** were generated once with `daily_state.py --init` from the local OE files.
  - 102,870 games, ~79 MB in total, largest file ~10 MB.
  - They match the committed ELO state exactly.
  - 14 minor-league games are in team_games but not in the ELO, because none of their players has a `playerid`.
- **`daily_state` ingestion:**
  - It only ingests games dated strictly before the target day (UTC); same-day games wait for the next run.
  - Only gameids that are new to team_games reach `update_elo`, and the ELO update runs before the CSV append.
- **Live `playoffs` context:** 1 when the league is domestic and the lolesports `blockName` matches `playoff|final|knockout|play.?in`; internationals are always 0.
- **`evaluate`:**
  - The winner comes from `gameWins` reaching ⌈bo/2⌉, because live `getEventDetails` has no `outcome`. `score` is in team1-team2 order.
  - Each match is isolated; an unknown id (event null) counts as unresolved.
  - `running_record` is counted server-side (avoids the 1000-row cap).
- **`predict`:**
  - Each match is isolated.
  - SHAP uses XGBoost's own `pred_contribs` (same values as the `shap` package, much faster to import; `shap` is still in `requirements.txt`).
  - The ad-hoc CLI takes team **names**.
  - 11 aliases were added to `team_aliases.csv`. All 56 teams in the current `PREDICT_LEAGUES` schedule resolve.
- **Workflows:**
  - evaluate runs on cron `45 7 * * *` and commits state even if evaluate itself failed;
  - predict runs on `workflow_run` with any conclusion, plus crons `0 3` and `0 15`;
  - bot `riftside-oracle[bot]`; `NBA_PROXIES` removed.

## Phase 5: implementation choices

- **No team table:** `lib/teams.ts` was deleted; names and logos come from the row. `TeamLogo` rewrites http→https for static.lolesports.com and falls back to a grey circle with the team code.
- **Home page** groups by the viewer's local date:
  - Today (ascending);
  - Upcoming: unresolved, later days, ascending (needed because predict runs up to 36h ahead);
  - Recent: earlier days, newest first.
- **GameCard:** `league_name · Bo{n}`, a series-win split bar coloured relative to the pick, and the score (team1-team2) once resolved.
  - SHAP shows the top 5 of the stored 10 by |value|. Feature names are mapped to readable labels (all 136 model features covered); blue_ = team1.
- **`/card`** (420×260, public interface unchanged, iframed from the portfolio repo):
  - the live window is best_of × 1h;
  - a resolved match shows the actual winner plus "✓ Correct · Pick X (nn%)";
  - long names use line-clamp-2 and a smaller size.
- **"About the model"** hardcodes 70.2% series (6,260), 69.9% major (1,453), 65.3% game accuracy and log loss 0.618, plus the 69.8% ELO baseline in the Limits paragraph.
- **Texts:** Riot disclaimer verbatim, and "Match data: Oracle's Elixir and lolesports". No betting language.
- **Without env vars** the build still succeeds: the Supabase client is null, the pages show empty states, and `next.config.ts` prints a warning.
- **Previewing:** `/demo` (public, labelled as demo data) and `?demo=1` on any page, the latter **only under `npm run dev`** (gated by NODE_ENV). Mock rows are in `lib/mock.ts`, placed relative to the viewer's today.
- Package renamed to riftside-oracle; no new dependencies.
- **Dev gotcha:** `next dev` can crash ("segment-explorer-node") if it shares `web/.next` with `npm run build`; delete `web/.next`.
- **Not verified against real Supabase data yet** (no anon key locally).

## Model: results and what was tried

### Final numbers (test slice 2025-05-20..2026-10-14, never used for any choice)

**Series accuracy** (the owner's metric):
- Side-neutral per-game probability: `p = (P(t1 blue) + 1 − P(t2 blue)) / 2`.
- Series probability: `series_win_prob(p, best_of)`.
- Series grouped by day + team pair; Bo inferred from the winner's game count.

| | Model | Phase 3 model | Baseline: higher outcome ELO |
|---|---|---|---|
| All leagues (6,260) | **70.2%** | 70.4% | 69.8% |
| All: Bo1 / Bo3 / Bo5 | 67.4 / 72.4 / 68.7 | 67.5 / 72.7 / 68.7 | 67.2 / 71.9 / 68.6 |
| Major leagues (1,453) | **69.9%** | 69.3% | 70.0% |
| Major: Bo1 (146) / Bo3 (904) / Bo5 (403) | 61.6 / 72.7 / 66.5 | 59.6 / 72.8 / 65.0 | 61.6 / 72.2 / 68.0 |

The major leagues are LCK, LPL, LEC, LCS, CBLOL, LCP, LTA N, LTA S, LTA, WLDs (Worlds), MSI and FST (First Stand).

**Game level:**
- All leagues: model 65.3% / log loss 0.618; outcome-ELO baseline 64.9% / 0.626; always-blue 53.4%.
- Major leagues: 63.8% / 0.639 vs 63.2% / 0.641.
- The GATE in `train.py` says PASS.

**Bootstrap 95% CIs on the major leagues:**
- The new model's series log loss is significantly better than Phase 3's: −0.006, CI [−0.011, −0.001].
- Series accuracy vs Phase 3: +0.55pp, CI [−0.55, +1.65], not significant.
- Against the plain outcome-ELO baseline the model is **level on major-league series**. Across all leagues it is significantly better in log loss.

**Two test looks:** the test slice was evaluated twice. The second look came after a correctness fix (a look-ahead in the new-year pull, found in review), with no tuning on test. The validation gain was unchanged after the fix.

### Honest assessment

- Per single game, pre-draft, about 64–65% is close to the ceiling with this data. The draft (unknown before the match) and many coin-flip matchups cap it. 70% per game is not realistic pre-draft.
- At series level, ~70% is where we are, and it is what the site's public record will count.
- On the major leagues, almost all of the predictive power is in the outcome ELO. The XGBoost layer adds little there, though it does help across all leagues.
- In major-league Bo5s the model's series probabilities are slightly worse calibrated than the plain ELO's: series log loss 0.637 vs 0.620. Compounding a per-game p over 5 games amplifies small calibration errors. This is an optional future fix, for example shrinking p before compounding, chosen on validation. It only affects the displayed probability, not who is picked.

### Tried in the improvement round, with no measurable gain on the major leagues

None of these were adopted:
- K values (16/24/32/48)
- an international K multiplier
- a team-level Elo
- strength-of-schedule-adjusted form
- residual form
- patch form
- lineup activity
- a player margin rating
- champion meta-share and champion pool breadth (would also need extra committed state)
- objective-control rates
- MAJOR or recency sample weights
- top-k feature pruning (worse)
- logistic regression
- an XGB+LR ensemble
- separate calibration for the major leagues

Method: walk-forward validation over 3 windows (2021-07..2025-05, all before the test slice), fixed hyperparameters, 3 seeds, paired bootstrap.

## Data facts (full audit of all 13 OE files)

- **Game shape:** every game has exactly 12 rows: 10 players (5 roles per side) plus 2 team rows. The `playerid` format `oe:player:<hex>` is stable 2014–2026.
- **Missing ids:** 0.3–1.9% of rows per year have no `playerid`, and 0.3–3% have no `teamid`. **Both are 0% in the major leagues.**
- **`partial` games** (no @10/15/20/25 stats): 17–20% in 2016–18 and 1.5% in 2026. On the major leagues: 30% in 2025 (LPL) and 4.5% in 2026. "LPL is partial" was true until about Jan 2026.
- **League codes over time:**
  - `OGN` (2015) became `LCK`.
  - `NA LCS` became `LCS` (2019), then `LTA N` (2025), then `LCS` again (2026).
  - `EU LCS` became `LEC`.
  - `CBLOL` became `LTA S` (2025), then `CBLOL` again.
  - `LMS` became `PCS` (2020); `LCP` exists since 2025.
  - Internationals: `WLDs`, `MSI`, `FST`, `EWC`, `MSC`.
- **Coverage:** 2014 has only NA/EU and Worlds. Korea starts in 2015 and China in 2016.
- **OE quirks:**
  - Dates are not sorted within a file; always sort.
  - The `year` column can say 2027 inside the 2026 file; use `date`.
  - 3 games have both sides marked as losers, and 12 have the same teamid on both sides. Both are handled.
  - OE refreshes around 07:05 UTC.

## Data access

- **Google Drive "Quota exceeded":** on 2026-10-15 every scripted full OE download got Drive's quota page, even for the 2014 file. Small Range requests still worked.
  - We do **not** chunk around the quota with Range requests. That would get around Google's limit.
  - `oe.py --check` now starts a real download, so it detects the block.
  - OE's old S3 mirror is gone (404).
- **Daily job:** it downloads the current year's file (~78 MB) every morning. Decision 8 covers a blocked download: fail loudly, still evaluate and predict on the committed state, catch up the next day.
- **lolesports API:** works with the public site key and needs no proxy.

## Open points for Phase 6 and later

1. **Known small issues (nits):**
   - `league_tier` is 1350 for internationals (`is_international` distinguishes them).
   - `mov_mult` treats comebacks as dominant wins.
   - A side with a missing playerid gets a partial gold-diff sum.
   - The pull is skipped for a year if a player's first game of that year is international (rare).
   - An empty starter slot is excluded from team means in training but filled with the init value live.
2. **Known limitations of Phase 4, accepted:**
   - The playoffs `blockName` rule disagrees with OE in some leagues: "Regional Qualifier" is 1 in OE for LCK and LCP, and "Play-Ins" is 0 in OE for CBLOL and LCP. The effect on the side-neutral p is small.
   - Forfeits (0-0 `gameWins`) never resolve automatically; they get only the 7-day warning.
   - A rare double write failure (`player_elo_current` written but `recent` not) could double-count games on the next run.
   - The evaluate `pending` select is unpaged; this only matters with more than 1000 unresolved predictions at once.
   - LCK/LPL matches between about 05:00 and 09:00 UTC are predicted on state that is missing the previous day, because OE refreshes around 07:05 UTC. This is by design.
   - On days with the Drive quota block, the evaluate run will show as failed.
3. **For Phase 6 docs** (`docs/model.md`; still NBA text, which also contradicts the new Optuna setup):
   - About 1.1% of games belong to series that cross UTC midnight. In training, their later games see the earlier ones; live prediction doesn't.
   - The roster train/live difference (training uses actual lineups, live the last-known lineup).
   - `role_norms` are computed over 2022–2025.
   - `team_games_*.csv` are committed state written by `daily_state`.
4. **Owner-only steps** (spec §10):
   - Done: the Supabase project, `schema.sql` (incl. GRANTs), the GitHub secrets and the enabled workflows.
   - Still to do:
     - create `web/.env.local` with `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` (anon key only);
     - set up the Plesk subdomain (the OG URL assumes `https://riftside-oracle.gerritvisser.de`, still to be confirmed);
     - build and upload `web/out/`, as described in docs/operations.md (still the NBA text until Phase 6);
     - set the CSP `frame-ancestors` for `/card`.
5. **Odd but plausible ELO observations:**
   - The mean outcome ELO of players whose last game was in LPL (≈1615) is below LEC (≈1690). One reason: many stars' last game is currently Worlds.
   - Two EMEA Masters players reach the overall outcome top 10 (~1920).
   - Ratings drift up slowly over the years (≈1450 → 1520).

## How this was worked (for the next session)

- **Orchestration:** the owner's global CLAUDE.md asks for implementation to go to subagents (coder/writer/chore). Each step gets a read-only review. Each phase is accepted only when two independent reviewers both report "IN ORDNUNG". This was followed for every phase.
- **Model protocol:** choose on validation only (walk-forward windows before the test slice). Look at the test slice once, at the end. Report series accuracy first (all + major leagues, by Bo1/Bo3/Bo5), then game-level log loss.
- **Owner preferences:**
  - Answers are short and concrete, with numbers.
  - The owner is happy with ~70% series accuracy and wants optimisations without overfitting.
  - The owner pushes and deploys; agents commit per phase.
