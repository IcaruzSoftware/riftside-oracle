# Handoff: Riftside Oracle (status 2026-10-16)

Read [CLAUDE.md](CLAUDE.md) first in a new session, then this file. This file is
the status and decision log; CLAUDE.md is the working guide; `docs/` has the depth.

## TL;DR

- **Phases 0–6 are done** (data layer, ELO, features and training, daily pipeline, web app, docs), plus one model-improvement round. Each phase is committed and was reviewed by two independent reviewers.
- **The daily pipeline is live** since 2026-10-16 (see "Live setup"). 118 tests, all passing.
- **The web app is built but not deployed** and not yet checked against real Supabase data (no anon key locally).
- **Next steps are owner-only** (see "Next steps"): `web/.env.local`, the Plesk subdomain, the build and upload, the CSP `frame-ancestors` header.
- **Model quality, at the level that matters (who wins the series):**
  - about **70% series accuracy** on the test period 2025-05-20..2026-10-14: 6,260 series across all leagues;
  - **69.9% on the major leagues**: 1,453 series; 72.7% in Bo3s.
- **Verdict after the improvement round: good as it is.** The remaining headroom is small and not reachable with the data we have (see "Model").

## Resume on another PC

```bash
git clone https://github.com/IcaruzSoftware/riftside-oracle.git && cd riftside-oracle
uv venv --python 3.12 .venv          # Python 3.12 is required (pins don't install on 3.13+)
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest           # 118 tests, all must pass
```

- **Raw data is not in git.** The Oracle's Elixir CSVs (13 files, ~856 MB) are gitignored.
  - Download them **by hand** from the Drive folder <https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH> into `src/data/raw/oe/`, keeping the original names, e.g. `2026_LoL_esports_match_data_from_OraclesElixir.csv`.
  - `bootstrap_data.py` can do the same, but Drive often blocks scripted full downloads (see "Data access").
- **Committed state:** `player_elo_current.parquet`, `player_elo_recent.parquet`, `role_norms.json`, `src/models/xgb_model.pkl` and `team_games_{2014..2026}.csv` (102,870 games, ~79 MB in total, largest file ~10 MB; `daily_state` writes them, don't hand-edit).
  - With these, the tests and live predictions work without the raw data. Rebuilding the history or retraining needs it.
- **Rebuild commands** (only when needed):

  ```bash
  .venv/bin/python src/pipeline/elo.py --force            # ~21 s, full 2014–2026 ELO replay
  .venv/bin/python src/pipeline/daily_state.py --init     # team_games_*.csv from the same OE files; delete the existing team_games_*.csv first (it refuses to overwrite)
  .venv/bin/python src/pipeline/build_dataset.py --force  # ~35–65 s, 93,753 games since 2017
  .venv/bin/python src/pipeline/train.py --trials 100 --force   # ~25–40 min
  ```

  - Always rebuild the ELO before `build_dataset` (it reads the uncommitted `player_elo.parquet`). Commit the ELO state only together with a fresh `--init`.

- **Local Windows note (2026-10-15 PC):** `uv` was not installed, so the venv was made with `py -3.12 -m venv`. pyarrow 26 is blocked by Windows Application Control on this PC, so `.venv` has `pyarrow==21.0.0` installed by hand. `requirements.txt` is unchanged and CI is unaffected.

## Commits so far

| Commit | What |
|---|---|
| 1882948 | Merge: the repo contains the original upstream history (merged with `--allow-unrelated-histories`) |
| d68d5bf | Phase 0: old code/state removed; `oe.py`/`lolesports.py` `--check`; `check_sources.yml` |
| c9008de | Phase 1: data layer: OE download/load/split, lolesports schedule/results/team resolution |
| 47b92b0 | Phase 2: player ELO (5 role-normalised skills + outcome ELO) |
| ee312df | Phase 3: features, dataset, training, baseline gate |
| 0435478, 3ee0afc, bebc2b6 | Improvement round: refined outcome ELO, Optuna fix, committed state + model, this file |
| 3cc93f2 | Phase 4: daily pipeline (`daily_state`, `predict`, `evaluate`, `daily_run`), workflows, `team_games_*.csv`, `schema.sql` |
| d00edc0 | `schema.sql`: explicit GRANTs |
| 04bd766 | Phase 5: web app (Next.js static export, `/card` widget, `/demo`) |
| (this commit) | Phase 6: docs (CLAUDE.md, README.md, HANDOFF.md, `docs/`); the spec file is removed (it stays in git history) |

## Live setup

- Supabase project `pdblqlbgrelyqjgkpdte` (`https://pdblqlbgrelyqjgkpdte.supabase.co`), `schema.sql` applied, GitHub secrets `SUPABASE_URL` / `SUPABASE_KEY` set, workflows `Daily Evaluate`, `Daily Predictions` and `Keepalive` enabled (since 2026-10-16).
- `check_sources.yml` passed on a GitHub runner, so OE and lolesports need no proxies.
- The first predict run failed with 42501 "permission denied for table predictions": newer Supabase projects don't grant table privileges by default. Fix: explicit GRANTs at the end of `schema.sql` (service_role full CRUD; anon/authenticated SELECT), run by the owner in the SQL editor.
- First Worlds day (2026-10-15, predicted ad hoc from the committed state): KC over C9 correct, FURIA over MVK wrong.

## Decisions (by the owner or with the owner's OK)

1. **Players with a missing `playerid` are skipped** in the ELO, with no rating and no snapshot. The rest of their game is processed. This only happens in minor leagues: about 2% of rows in LJL, LPLOL, LIT, LAS and similar.
2. **`LEAGUE_PRIOR` maps the historical OE codes too:**
   - 1600: LCK, OGN, LPL
   - 1500: LEC, EU LCS, LCS, NA LCS, LTA N, LTA
   - 1450: LCP, PCS, LMS, CBLOL, LTA S
   - default: 1350
3. **Roster feature:** `roster_churn` = number of starters shared between the team's **last two lineups before the day**, identical in training and live. (The simpler "starters who also started the previous game" would always be 5 live.) `rookie_count` = starters with < 20 games.
4. **Point-in-time is day-level.** A game on day d sees only team_games rows from before day d and ELO "as of the start of day d". Training uses each player's first pre-game snapshot on or after d; live uses `player_elo_current`. So game 2 of a series never sees game 1 in training, just as a pre-series live prediction can't.
5. **Outcome ELO refinements** (the improvement round):
   - `REGRESS = 0.2`: at a player's first domestic game of a new calendar year, the outcome ELO moves 20% toward that league's mean. Applied after the pre-game snapshot, so training and live are identical (tested).
   - Margin of victory: K × `mov_mult` = sqrt of team |gold diff @15| relative to the mean. Partial games without @15 data use game length instead.
   - `NEW_K_MULT = 2` for a player's first 20 games. Games between new and established players are therefore **not zero-sum**; documented in `elo.py`.
6. **`train.py`:** the Optuna CV runs on the **train slice only** and minimises log loss.
7. **Git history:** the repo contains the original upstream history (merged with `--allow-unrelated-histories`).
8. **Drive quota: Drive API with the owner's key (option B, since 2026-10-17; was option A, no Drive API path).** With the GitHub secret `GOOGLE_DRIVE_API_KEY` set, `oe.py` lists and downloads through the Drive API v3 (key in the `X-Goog-Api-Key` header, never logged); without it, the public links as before. An API quota error falls back to the public link once, for the folder listing and for the download separately. The per-file download quota may still apply to API downloads, so a block stays possible: then the state update fails loudly, evaluate and predict still run on the committed state, the run exits 1, and the next day's run catches up through `LOOKBACK_DAYS = 14`.
9. **`player_elo_recent.parquet` is committed in full every day**, with no pruning.
10. **`academy` was removed from `lolesports._SUFFIXES`**, because academy teams are separate teams.
11. **`RECORD_START = "2026-10-15"`** in `evaluate.py`.
12. **`schema.sql`:** `predictions` table with the lolesports match shape; `model_metadata` dropped (nothing wrote to it); `shap_values` and `running_record` unchanged. It was done in Phase 4, ahead of Phase 5, so the owner could create the Supabase project early.

## Implementation choices

### Phase 4: daily pipeline

- **`team_games_{2014..2026}.csv`** were generated once with `daily_state.py --init` from the local OE files and match the committed ELO state exactly. 14 minor-league games are in team_games but not in the ELO, because none of their players has a `playerid`.
- **`daily_state`** only ingests games dated strictly before the target day (UTC); same-day games wait for the next run. Only gameids new to team_games reach `update_elo`, and the ELO update runs before the CSV append.
- **Live `playoffs` context:** 1 when the league is domestic and the lolesports `blockName` matches `playoff|final|knockout|play.?in`; internationals are always 0.
- **`evaluate`:** the winner comes from `gameWins` reaching ⌈bo/2⌉, because live `getEventDetails` has no `outcome`. `score` is in team1-team2 order. Each match is isolated; an unknown id (event null) counts as unresolved. `running_record` is counted server-side (avoids the 1000-row cap).
- **`predict`:** each match is isolated. SHAP uses XGBoost's own `pred_contribs` (same values as the `shap` package, much faster to import; `shap` is still in `requirements.txt`). The ad-hoc CLI takes team **names**. 11 aliases were added to `team_aliases.csv`; all 56 teams in the current `PREDICT_LEAGUES` schedule resolve.
- **Workflows:** evaluate runs on cron `45 7 * * *` and commits state even if evaluate itself failed; predict runs on `workflow_run` with any conclusion, plus crons `0 3` and `0 15`; bot `riftside-oracle[bot]`.

### Phase 5: web

- **No team table:** names and logos come from the row. `TeamLogo` rewrites http→https for static.lolesports.com and falls back to a grey circle with the team code.
- **Home page** groups by the viewer's local date: Today (ascending); Upcoming (unresolved, later days; needed because predict runs up to 36h ahead); Recent (earlier days, newest first).
- **GameCard:** `league_name · Bo{n}`, a series-win split bar coloured relative to the pick, and the score (team1-team2) once resolved. SHAP shows the top 5 of the stored 10 by |value|, with readable labels for all 136 model features; blue_ = team1.
- **`/card`** (420×260, public interface unchanged): the live window is best_of × 1h; a resolved match shows the actual winner plus "✓ Correct · Pick X (nn%)"; long names use line-clamp-2 and a smaller size.
- **"About the model"** hardcodes 70.2% series (6,260), 69.9% major (1,453), 65.3% game accuracy and log loss 0.618, plus the 69.8% ELO baseline in the Limits paragraph.
- **Texts:** Riot disclaimer verbatim, and "Match data: Oracle's Elixir and lolesports". No betting language.
- **Without env vars** the build still succeeds: the Supabase client is null, the pages show empty states, and `next.config.ts` prints a warning.
- **Previewing:** `/demo` (public, labelled as demo data) and `?demo=1` on any page, the latter **only under `npm run dev`**. Mock rows are in `lib/mock.ts`.
- **Dev gotcha:** `next dev` can crash ("segment-explorer-node") if it shares `web/.next` with `npm run build`; delete `web/.next`.

## Model: results and what was tried

### Final numbers (test slice 2025-05-20..2026-10-14, never used for any choice)

**Series accuracy** (the owner's metric): side-neutral per-game probability `p = (P(t1 blue) + 1 − P(t2 blue)) / 2`, series probability `series_win_prob(p, best_of)`, series grouped by day + team pair, Bo inferred from the winner's game count.

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

- Per single game, pre-draft, about 64–65% is close to the ceiling with this data. The draft (unknown before the match) and many coin-flip matchups cap it.
- At series level, ~70% is where we are, and it is what the site's public record counts.
- On the major leagues, almost all of the predictive power is in the outcome ELO. The XGBoost layer adds little there, though it does help across all leagues.
- In major-league Bo5s the model's series probabilities are slightly worse calibrated than the plain ELO's: series log loss 0.637 vs 0.620. Compounding a per-game p over 5 games amplifies small calibration errors (see follow-ups).

### Tried in the improvement round, no measurable gain on the major leagues (none adopted)

K values (16/24/32/48); an international K multiplier; a team-level Elo; strength-of-schedule-adjusted form; residual form; patch form; lineup activity; a player margin rating; champion meta-share and pool breadth (would also need extra committed state); objective-control rates; MAJOR or recency sample weights; top-k feature pruning (worse); logistic regression; an XGB+LR ensemble; separate calibration for the major leagues.

Method: walk-forward validation over 3 windows (2021-07..2025-05, all before the test slice), fixed hyperparameters, 3 seeds, paired bootstrap.

## Data facts (audit of all 13 OE files)

- **Game shape:** every game has exactly 12 rows: 10 players (5 roles per side) plus 2 team rows. The `playerid` format `oe:player:<hex>` is stable 2014–2026.
- **Missing ids:** 0.3–1.9% of rows per year have no `playerid`, and 0.3–3% have no `teamid`. **Both are 0% in the major leagues.**
- **`partial` games** (no @10/15/20/25 stats): 17–20% in 2016–18 and 1.5% in 2026. On the major leagues: 30% in 2025 (LPL) and 4.5% in 2026.
- **League codes over time:** `OGN` (2015) became `LCK`; `NA LCS` became `LCS` (2019), then `LTA N` (2025), then `LCS` again (2026); `EU LCS` became `LEC`; `CBLOL` became `LTA S` (2025), then `CBLOL` again; `LMS` became `PCS` (2020); `LCP` exists since 2025. Internationals: `WLDs`, `MSI`, `FST`, `EWC`, `MSC`.
- **Coverage:** 2014 has only NA/EU and Worlds. Korea starts in 2015 and China in 2016.
- **OE quirks:** dates are not sorted within a file (always sort); the `year` column can say 2027 inside the 2026 file (use `date`); 3 games have both sides marked as losers and 12 have the same teamid on both sides (both handled); OE refreshes around 07:05 UTC.

## Data access

- **Google Drive "Quota exceeded":** on 2026-10-15 every scripted full OE download got Drive's quota page, even for the 2014 file. Small Range requests still worked. We do **not** chunk around the quota with Range requests, because that would get around Google's limit. `oe.py --check` says which path it uses (Drive API or public link) and starts a real download, so it detects the block. OE's old S3 mirror is gone (404).
- **Drive API key (decision 8):** not yet verified with a real key; a bogus key gets the expected `API_KEY_INVALID` error, which confirms Drive reads the key from the `X-Goog-Api-Key` header.
- **Daily job:** it downloads the current year's file (~78 MB) every morning. Decision 8 covers a blocked download.
- **lolesports API:** works with the public site key and needs no proxy.

## Next steps

Owner-only (the agents can't do these):

1. Create `web/.env.local` from `web/.env.local.example`, with the anon key only.
2. Set up the Plesk subdomain (the OG URL in `web/app/layout.tsx` assumes `https://riftside-oracle.gerritvisser.de`, still to be confirmed).
3. Build and upload `web/out/`, as described in [docs/operations.md](docs/operations.md#deploying-the-web-app).
4. Set the CSP `frame-ancestors` header for `/card` in the server config, so it can be iframed on `gerritvisser.de`.
5. Optional: add the card to the portfolio repo (`IcaruzSoftware/portfolio`).
6. Check the web app against real Supabase data once it is deployed.
7. Create a Google Drive API key and add it as the repo secret `GOOGLE_DRIVE_API_KEY` ([docs/operations.md](docs/operations.md#creating-the-drive-api-key)), then run `Check Data Sources` by hand and confirm the OE step reports `Drive API` and `OK`.

Optional follow-ups (none is needed for a good record):

- **Bo5 calibration:** shrink the per-game p before compounding to the series probability, chosen on validation. It only affects the displayed probability, not who is picked.
- **Playoffs per-league rule:** the live `blockName` rule disagrees with OE in some leagues (see limitations); a per-league rule would remove that.
- **Leaguepedia as a possible source for official pre-match lineups** (would address the live-roster limitation). Anonymous queries were rate-limited on 2026-10-15.
- A live in-game win-probability feature could be built on the lolesports live-stats feed (`feed.lolesports.com`). Not needed for the pre-match model.

## Known limitations and nits (accepted)

- **Small issues:**
  - `league_tier` is 1350 for internationals (`is_international` distinguishes them).
  - `mov_mult` treats comebacks as dominant wins.
  - A side with a missing playerid gets a partial gold-diff sum.
  - The new-year pull is skipped for a year if a player's first game of that year is international (rare).
  - An empty starter slot is excluded from team means in training but filled with the init value live.
- **Phase 4:**
  - The playoffs `blockName` rule disagrees with OE in some leagues: "Regional Qualifier" is 1 in OE for LCK and LCP, and "Play-Ins" is 0 in OE for CBLOL and LCP. The effect on the side-neutral p is small.
  - Forfeits (0-0 `gameWins`) never resolve automatically; they get only the 7-day warning.
  - A rare double write failure (`player_elo_current` written but `recent` not) could double-count games on the next run.
  - The evaluate `pending` select is unpaged; this only matters with more than 1000 unresolved predictions at once.
  - LCK/LPL matches between about 05:00 and 09:00 UTC are predicted on state that is missing the previous day, because OE refreshes around 07:05 UTC. This is by design.
  - On days with the Drive quota block (API and public link both blocked), the evaluate run shows as failed.
- **Train vs live:**
  - About 1.1% of games belong to series that cross UTC midnight. In training, their later games see the earlier ones; live prediction doesn't.
  - Training uses each game's actual lineups, live the last-known lineup.
  - `role_norms` are computed over 2022–2025 and frozen.
- **Odd but plausible ELO observations:** the mean outcome ELO of players whose last game was in LPL (≈1615) is below LEC (≈1690), partly because many stars' last game is Worlds (as of 2026-10-15). Two EMEA Masters players reach the overall outcome top 10 (~1920). Ratings drift up slowly over the years (≈1450 → 1520).

## How this was worked (for the next session)

- **Orchestration:** the owner's global CLAUDE.md asks for implementation to go to subagents (coder/writer/chore). Each step gets a read-only review. Each phase is accepted only when two independent reviewers both report "IN ORDNUNG".
- **Model protocol:** choose on validation only (walk-forward windows before the test slice). Look at the test slice once, at the end. Report series accuracy first (all + major leagues, by Bo1/Bo3/Bo5), then game-level log loss.
- **Owner preferences:**
  - Answers are short and concrete, with numbers.
  - The owner is happy with ~70% series accuracy and wants optimisations without overfitting.
  - The owner pushes and deploys; agents commit per phase.
