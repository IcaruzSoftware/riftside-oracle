# Handoff: Riftside Oracle (status 2026-10-15, end of day)

Read this first in a new session, then [CLAUDE.md](CLAUDE.md) and the spec
[RIFTSIDE_HANDOVER.md](RIFTSIDE_HANDOVER.md). Where this file and the spec
disagree, this file records a later decision and wins.

## TL;DR

- **Phases 0–3 of the handover are done, plus one model-improvement round.** Each one is committed, tested and was reviewed by two independent reviewers.
- **Next step: Phase 4**, the daily pipeline (`daily_state.py`, `predict.py`, `evaluate.py`, `daily_run.py`, the workflows).
- **Model quality, at the level that matters to the owner (who wins the match):**
  - about **70% series accuracy** on the test period 2025-05-20..2026-10-14: 6,260 series across all leagues;
  - **69.9% on the major leagues**: 1,453 series; 72.7% in Bo3s.
- **Verdict after the improvement round: it is good as it is.** The remaining headroom is small and not reachable with the data we have (see "Model", below).
- **Still broken:** `daily_state.py`, `predict.py`, `evaluate.py` and `daily_run.py` are NBA code with broken imports. That is expected; Phase 4 rewrites them.

## Resume on another PC

```bash
git clone https://github.com/IcaruzSoftware/riftside-oracle.git && cd riftside-oracle
uv venv --python 3.12 .venv          # Python 3.12 is required (pins don't install on 3.13+)
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest           # 57 tests, all must pass
```

- **Raw data is not in git.** The Oracle's Elixir CSVs (13 files, ~856 MB) are gitignored.
  - Download them **by hand** from the Drive folder <https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH> into `src/data/raw/oe/`.
  - Keep the original names, e.g. `2026_LoL_esports_match_data_from_OraclesElixir.csv`.
  - `bootstrap_data.py` can do the same, but Drive often blocks scripted full downloads (see "Data access", below).
- **Committed state:** `player_elo_current.parquet`, `player_elo_recent.parquet`, `role_norms.json` and `src/models/xgb_model.pkl`.
  - With these, the tests and future live predictions work without the raw data.
  - Rebuilding the history or retraining needs the raw data.
- **Rebuild commands** (only when needed):

  ```bash
  .venv/bin/python src/pipeline/elo.py --force            # ~21 s, full 2014–2026 ELO replay
  .venv/bin/python src/pipeline/build_dataset.py --force  # ~35–65 s, 93,753 games since 2017
  .venv/bin/python src/pipeline/train.py --trials 100 --force   # ~25–40 min
  ```

- **GitHub state:**
  - The repo is pushed.
  - `check_sources.yml` passed on a GitHub runner, so OE and lolesports need no proxies.
  - `Daily Evaluate`, `Daily Predictions` and `Keepalive` are **disabled** on GitHub until Phase 4 is done. Keepalive re-enables the others, which is why it is off too.

## Commits so far

| Commit | What |
|---|---|
| 1882948 | courtside-oracle history merged in (so NBA code like `nba_cdn.py` is recoverable via `git show`) |
| d68d5bf | Phase 0: NBA code/state removed; `oe.py`/`lolesports.py` `--check`; `check_sources.yml` |
| c9008de | Phase 1: data layer: OE download/load/split, lolesports schedule/results/team resolution |
| 47b92b0 | Phase 2: player ELO (5 role-normalised skills + outcome ELO) |
| ee312df | Phase 3: features, dataset, training, baseline gate |
| (this push) | Improvement round: refined outcome ELO, Optuna fix, committed state + model, this file |

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

## Data access

- **Google Drive "Quota exceeded":** on 2026-10-15 every scripted full OE download got Drive's quota page, even for the 2014 file. Small Range requests still worked.
  - We do **not** chunk around the quota with Range requests. That would get around Google's limit.
  - `oe.py --check` now starts a real download, so it detects the block.
  - OE's old S3 mirror is gone (404).
- **Risk for Phase 4:** the daily job downloads the current year's file (~78 MB) every morning. If the quota block is common, the daily job fails.
  - Fix A: retry later, since the block lifts within 24 h.
  - Fix B, recommended: a Google Drive API v3 download path with the owner's own API key, stored as a GitHub secret.
  - Decide this early in Phase 4. The owner hasn't decided yet.
- **lolesports API:** works with the public site key and needs no proxy.

## Open points for Phase 4 and later

1. **Initial `team_games_*.csv`:** these files don't exist yet. Phase 4 must write them once from OE with `oe.team_rows` (one file per year), then let `daily_state` append. Features in live mode read them.
2. **Idempotency:** `daily_state` must pass **only new games** to `elo.update_elo` (new = `gameid` not in `team_games_*.csv`). `update_elo`'s own check uses the recent snapshots, which miss games older than a player's last 11.
3. **Live context mapping:** `build_feature_matrix` needs `context={"league": <OE code>, "playoffs": 0/1}`. Map lolesports slugs to OE codes, e.g. lck→LCK, worlds→WLDs, first_stand→FST, lcs→LCS, cblol-brazil→CBLOL, lcp→LCP. `playoffs` can come from the lolesports `blockName`.
4. **Timing:** features in live mode use ELO from `player_elo_current` but cut team_games at day < d. Phase 4 must not ingest same-day results before predicting that day, or the ELO would include day d while the rolling stats don't.
5. **Daily commit size:** `player_elo_recent.parquet` is 4.2 MB and is committed daily, about 1.5 GB of git history per year.
   - Option: keep only players active in the last 365 days, which gives 1.1 MB.
   - Ask the owner. I suggested it but it isn't decided.
6. **Open question from Phase 1:** the team-name suffix list in `lolesports.normalize_team_name` includes `academy` (from the spec). So "T1 Esports Academy" and "T1" normalise the same, and T1 can come out as *ambiguous* and get skipped. It fails safe, nothing is guessed wrong.
   - My recommendation: drop `academy` from the list. The owner hasn't answered yet.
7. **Known small issues (nits):**
   - `league_tier` is 1350 for internationals (`is_international` distinguishes them).
   - `mov_mult` treats comebacks as dominant wins.
   - A side with a missing playerid gets a partial gold-diff sum.
   - The pull is skipped for a year if a player's first game of that year is international (rare).
   - An empty starter slot is excluded from team means in training but filled with the init value live.
8. **For Phase 6 docs** (`docs/model.md`; still NBA text, which also contradicts the new Optuna setup):
   - About 1.1% of games belong to series that cross UTC midnight. In training, their later games see the earlier ones; live prediction doesn't.
   - The roster train/live difference (training uses actual lineups, live the last-known lineup).
   - `role_norms` are computed over 2022–2025.
9. **Odd but plausible ELO observations:**
   - The mean outcome ELO of players whose last game was in LPL (≈1615) is below LEC (≈1690). One reason: many stars' last game is currently Worlds.
   - Two EMEA Masters players reach the overall outcome top 10 (~1920).
   - Ratings drift up slowly over the years (≈1450 → 1520).
10. **`RECORD_START`** in `evaluate.py` is still the NBA value. Set it to the launch date (the plan is the first 2027 major-league match day).
11. **Owner-only steps** (spec §10): create a new Supabase project and run `schema.sql`; set the secrets; set up the Plesk subdomain; enable the workflows after Phase 4.

## How this was worked (for the next session)

- **Orchestration:** the owner's global CLAUDE.md asks for implementation to go to subagents (coder/writer/chore). Each step gets a read-only review. Each phase is accepted only when two independent reviewers both report "IN ORDNUNG". This was followed for every phase.
- **Model protocol:** choose on validation only (walk-forward windows before the test slice). Look at the test slice once, at the end. Report series accuracy first (all + major leagues, by Bo1/Bo3/Bo5), then game-level log loss.
- **Owner preferences:**
  - Answers are short and concrete, with numbers.
  - The owner is happy with ~70% series accuracy and wants optimisations without overfitting.
  - The owner pushes and deploys; agents commit per phase.
