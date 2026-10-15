# Riftside Oracle — handover spec (courtside-oracle → League of Legends)

You are working in a **fork of `courtside-oracle`** (NBA predictor). Your job:
translate it into **Riftside Oracle**, a pre-match predictor for **professional
League of Legends** series, keeping the architecture and replacing everything
NBA-specific. Read this whole file first, then the repo's `CLAUDE.md` and
`docs/` to understand what exists. This file overrides the NBA docs wherever
they conflict.

Written 2026-10-15. Every source and field name below was checked live on that
date (see §3).

---

## 0. TL;DR

| | Courtside (NBA) | Riftside (LoL) |
|---|---|---|
| Unit predicted | one game, home vs away | one **series** (Bo1/Bo3/Bo5), team1 vs team2. The model predicts **one game, blue vs red**. Series probability is derived from it |
| History source | stats.nba.com (bootstrap only, ~7h crawl) | **Oracle's Elixir** yearly CSVs (one download per year) |
| Daily stats source | cdn.nba.com box scores (thinner than bootstrap) | **The same Oracle's Elixir CSVs**, so training and live use one source with no degradation |
| Schedule + results | cdn.nba.com | **lolesports API** (`esports-api.lolesports.com`) |
| Proxies | required (NBA blocks cloud IPs) | probably not needed. Verify in Phase 0 |
| ELO | per-player, 7 skills, ranked among 14–22 players | per-player, **5 role-normalised skills** ranked among 10 players, **plus a new outcome ELO** per player (needed for cross-region strength) |
| Label | `home_win` | `blue_win` |
| Public record | games correct | **series** correct |

---

## 1. Ground rules for this translation

1. **Translate, don't redesign.** Keep the module boundaries, file names,
   orchestration (`daily_run.py`), committed-state pattern, Supabase + RLS
   pattern, static-export web app and test style. Change only what LoL
   requires. If something here looks wrong once you're in the code, stop and
   ask the owner. Don't guess.
2. **One code path for features.** The biggest flaw in Courtside is that
   training features differ from live ones (`docs/model.md`, "Honest caveats").
   Riftside uses one stats source, so keep it that way: training and live call
   the same functions on the same data shape. Section 6 calls out the one
   remaining difference (rosters).
3. **Delete NBA code. Don't keep it around "just in case".**
4. **Every phase ends green**: `python -m pytest` passes, and the phase's
   acceptance checks (§9) pass before the next phase starts.
5. Python 3.12. Keep the pinned scientific stack. You will retrain from
   scratch anyway, so pins may move together **once**, at the training phase.
6. Commit per phase. Don't push or deploy. The owner does that (§10).

---

## 2. Repo map: keep / adapt / rewrite / delete

| Path | Action | Notes |
|---|---|---|
| `src/pipeline/collect.py` | **delete** | nba_api collector |
| `src/pipeline/seed_finals.py` | **delete** | NBA one-off |
| `src/pipeline/shap_export.py` | **delete** | not wired into anything |
| `src/pipeline/nba_cdn.py` | **delete** → replaced by `lolesports.py` + `oe.py` | Keep the proxy-failover idea in your back pocket. Port it **only** if Phase 0 shows runners are blocked |
| `src/pipeline/oe.py` | **new** | Oracle's Elixir download + load + normalise (§3.1) |
| `src/pipeline/lolesports.py` | **new** | schedule, event details, league list, team mapping (§3.2, §3.3) |
| `src/pipeline/elo.py` | **rewrite**, same public API shape | `build_elo`, `update_elo`, `rank_to_delta` (reuse as is), `compute_skill_scores` (new skills), new outcome ELO (§5) |
| `src/pipeline/daily_state.py` | **rewrite** | OE delta → team game log + ELO update (§7.1) |
| `src/pipeline/features.py` | **rewrite**, same structure | `build_feature_matrix()` plus `get_*_features()` groups, each in its own try/except (§6) |
| `src/pipeline/build_dataset.py` | **adapt** | walk OE games chronologically → `feature_matrix.parquet` |
| `src/pipeline/train.py` | **adapt, lightly** | label `blue_win`, `META_COLS`, baselines in metrics (§6.4) |
| `src/pipeline/predict.py` | **rewrite** | lolesports schedule → side-averaged game prob → series prob (§7.2) |
| `src/pipeline/evaluate.py` | **adapt** | resolve via lolesports `getEventDetails`, record = series (§7.3) |
| `src/scripts/bootstrap_data.py` | **rewrite**, much smaller | download all OE year files (§3.1) |
| `src/scripts/daily_run.py` | **keep** | update docstrings and env vars only |
| `src/supabase/schema.sql` | **rewrite** | new `predictions` shape (§8.1) |
| `src/supabase/migrations/` | **delete** | fresh project, so there is nothing to migrate |
| `.github/workflows/daily_*.yml` | **adapt** | crons, secrets, bot name (§7.4) |
| `.github/workflows/keepalive.yml` | **keep** | |
| `.github/workflows/check_nba_proxies.yml` | **replace** with `check_sources.yml` | Phase 0 reachability check from a runner |
| `tests/` | **rewrite** fixtures and tests, **keep** `conftest.py`'s FakeSupabase pattern | §9 |
| `tests/fixtures/boxscore_*.json` | **delete** → add small OE CSV and lolesports JSON fixtures | |
| `web/` | **adapt** | §8.2 |
| `retry_games.txt`, `50` (stray root files) | **delete** | NBA leftovers |
| `src/data/raw/*`, `src/data/processed/*`, `src/models/xgb_model.pkl` | **delete contents** | NBA state. Regenerate everything |
| `.gitignore` negations | **rewrite** | §4 |
| `CLAUDE.md`, `README.md`, `docs/*.md` | **rewrite last** (Phase 6) | same structure, LoL content |
| `requirements.txt` | drop `nba_api`. Add nothing unless it's unavoidable (`requests` arrives transitively via `supabase`. Add it explicitly if you import it) | |

---

## 3. Data sources (verified 2026-10-15)

### 3.1 Oracle's Elixir (OE): history *and* daily stats

- Free, by Tim Sevenhuysen. **One CSV per year, 2014–2026**, in a public Google
  Drive folder `1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH`. Updated **once per day**.
  The 2026 file's `last-modified` was `Thu, 15 Oct 2026 07:05:01 GMT`. It
  already contained games from 2026-10-14, so the lag is about one day.
- File IDs (from `https://drive.google.com/embeddedfolderview?id=1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH`):

  | Year | Drive file id |
  |---|---|
  | 2021 | `1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA` |
  | 2022 | `1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy` |
  | 2023 | `1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2` |
  | 2024 | `1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN` |
  | 2025 | `1v6LRphp2kYciU4SXp0PCjEMuev1bDejc` |
  | 2026 | `1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm` |

  Older years are in the same folder. `oe.py` should **resolve IDs by parsing
  the embedded folder view** (regex on `/file/d/<id>/` and
  `flip-entry-title">YYYY_LoL_esports_match_data_from_OraclesElixir.csv`).
  The 2027 file will appear with a new ID, so hardcoding IDs breaks every
  January.
- Direct download, scriptable, no auth:
  `https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t`
  The 2026 file was **77,481,933 bytes** on 2026-10-15. Range requests work.
- **Shape: 12 rows per game**: 10 player rows (`participantid` 1–10,
  `position` ∈ `top, jng, mid, bot, sup`) + 2 team rows (`participantid`
  100/200, `position == "team"`). `side` ∈ `Blue, Red`. `result` ∈ `0, 1`.
  `gamelength` is in seconds. `date` looks like `2026-10-14 18:23:21`.
- **IDs are OE's own**, not lolesports': `playerid` = `oe:player:<hex>`,
  `teamid` = `oe:team:<hex>`. `gameid` is e.g. `LOLTMNT05_228413` (Riot
  platform) or `13057-13057_game_1` (LPL).
- `datacompleteness` ∈ `complete, partial`. **LPL rows are `partial`**: they
  lack some fields, notably the `*at10/15/20/25` early-game columns. They can
  also contain `playername == "MISSING"` (use `playerid`, never the name).
  Treat missing columns like Courtside treats missing tracking data: the
  affected skill doesn't update for that game (§5.1).
- Header (165 columns), verbatim:
  `gameid, datacompleteness, url, league, year, split, playoffs, date, game,
  patch, participantid, side, position, playername, playerid, teamname, teamid,
  firstPick, champion, ban1..ban5, pick1..pick5, gamelength, result, kills,
  deaths, assists, teamkills, teamdeaths, doublekills, triplekills,
  quadrakills, pentakills, firstblood, firstbloodkill, firstbloodassist,
  firstbloodvictim, team kpm, ckpm, firstdragon, dragons, opp_dragons,
  elementaldrakes, opp_elementaldrakes, infernals, mountains, clouds, oceans,
  chemtechs, hextechs, dragons (type unknown), elders, opp_elders,
  firstherald, heralds, opp_heralds, void_grubs, opp_void_grubs, firstbaron,
  barons, opp_barons, atakhans, opp_atakhans, firsttower, towers, opp_towers,
  firstmidtower, firsttothreetowers, turretplates, opp_turretplates,
  inhibitors, opp_inhibitors, damagetochampions, dpm, damageshare,
  damagetakenperminute, damagemitigatedperminute, damagetotowers,
  wardsplaced, wpm, wardskilled, wcpm, controlwardsbought, visionscore, vspm,
  totalgold, earnedgold, earned gpm, earnedgoldshare, goldspent, gspd, gpr,
  total cs, minionkills, monsterkills, monsterkillsownjungle,
  monsterkillsenemyjungle, cspm, goldat10, xpat10, csat10, opp_goldat10,
  opp_xpat10, opp_csat10, golddiffat10, xpdiffat10, csdiffat10, killsat10,
  assistsat10, deathsat10, opp_killsat10, opp_assistsat10, opp_deathsat10,`
  then the same 15-column block for `at15`, `at20` and `at25`.
  Columns change across years (e.g. `void_grubs`, `atakhans` are recent).
  `oe.load_year()` must tolerate missing columns (reindex to the needed set,
  NaN-fill).
- `league` uses OE's short codes (`LCK`, `LPL`, `LEC`, `EM` = EMEA Masters,
  `DCup`, `WLDs`, `MSI`, …). Build the league table from the data. Don't
  assume the codes.
- Terms: OE's page states that the game statistics are Riot's property and
  that any use must follow Riot Games' terms and policies. Credit OE on the
  site (§8.2).

### 3.2 lolesports API: schedule and results

Unofficial but stable and widely used. It is undocumented, so wrap every call
and fail loudly (same philosophy as `nba_cdn.py`).

- Base `https://esports-api.lolesports.com/persisted/gw/`, header
  `x-api-key: 0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z` (the public key the
  lolesports.com site itself uses). Always pass `hl=en-US`.
- `getLeagues` → `data.leagues[]`: `id, slug, name, region`. Verified slugs
  (2026): `worlds, msi, first_stand, lck, lpl, lec, lcs, cblol-brazil, lcp,
  emea_masters, lck_challengers_league, nacl, ljl-japan, vcs, pcs, lfl,
  primeleague, les, …`. Note that 2026 has **LCS and CBLOL again** (not LTA).
- `getSchedule[&leagueId=<id>[,<id>…]][&pageToken=…]` →
  `data.schedule.{pages:{older,newer}, events[]}`. Event:
  ```json
  {"startTime":"2026-10-14T16:00:00Z","state":"completed","type":"match",
   "blockName":"Playoffs","league":{"name":"EMEA Masters","slug":"emea_masters"},
   "match":{"id":"117173205319911429","flags":["hasVod","isSpoiler"],
     "teams":[{"name":"Anubis Gaming","code":"ANB","image":"http://static.lolesports.com/teams/….png",
               "result":{"outcome":"loss","gameWins":1},"record":{"wins":2,"losses":1}},
              {"name":"Solary","code":"SLY", "...": "...", "result":{"outcome":"win","gameWins":3}}],
     "strategy":{"type":"bestOf","count":5}}}
  ```
  `state` ∈ `unstarted, inProgress, completed`. Upcoming bracket matches
  have teams named **`TBD`**. Skip those, and a later run picks them up.
  Only `type == "match"` events matter (`show` events exist too).
- `getEventDetails?id=<matchId>` → `data.event.{id, type, tournament.id,
  league{id,slug,name,image}, match{strategy, teams, games[]}}`. Each game has
  `{number, id, state, teams[{id, side}]}`. Game `state` can be `completed`,
  `unneeded` (series already decided), and so on. **Use this to resolve
  results** (§7.3).
- Live feed (not needed for v1, documented for later):
  `https://feed.lolesports.com/livestats/v1/window/<gameId>?startingTime=<ISO, multiple of 10s>`
  and `/details/<gameId>` give 10-second frames (gold, kills, towers, dragons,
  per-participant KDA, CS, `championDamageShare`, wards), with
  `gameMetadata.{blue,red}TeamMetadata.participantMetadata[]` →
  `esportsPlayerId, summonerName, championId, role`. Without
  `startingTime` you only get the first frames. This would be the source for
  a live in-game win-probability feature (v2). Ignore it for v1.

### 3.3 The join between the two sources (the one fiddly part)

lolesports identifies teams by `name`/`code`/`image`. OE identifies them by
`teamid`/`teamname`. There is no shared ID.

- `lolesports.resolve_team(name, code) -> oe_teamid | None`:
  1. Manual override: `src/pipeline/team_aliases.csv`
     (`lolesports_code,lolesports_name,oe_teamid`). Tracked as normal source,
     not under `src/data/`.
  2. Otherwise, an exact match on a normalised name against the OE teams seen in
     the last 365 days (from the committed team game log). Normalise by
     casefolding, stripping non-alphanumerics, and dropping a trailing
     `esports|gaming|team|club|academy`… Keep the list short and tested.
  3. Unresolved → log `WARNING unmapped team <code>/<name>` and **skip the
     match** (no prediction). Never guess fuzzily.
- Results never need the join: evaluation reads winners straight from
  lolesports (§7.3).
- Expect a handful of manual aliases per season (sponsor renames). Phase 4
  acceptance includes "every major-league team in the current schedule
  resolves".

### 3.4 Not used (deliberately)

- **Riot developer API** (`developer.riotgames.com`): solo queue only, with
  no pro games.
- **Leaguepedia Cargo API**: anonymous queries were rate-limited immediately
  on 2026-10-15. OE already covers what we need. Possible later use: official
  pre-match lineups.
- **gol.gg**: scraping. Not needed.

---

## 4. Committed live state (replaces the NBA four-file contract)

`.gitignore`: keep the pattern (ignore `src/data/raw/*` and
`src/data/processed/*`, then re-include exactly these):

```
!src/data/raw/team_games_*.csv            # 2 rows per game (one per team), per year
!src/data/processed/player_elo_current.parquet
!src/data/processed/player_elo_recent.parquet
!src/data/processed/role_norms.json
src/models/*.pkl
!src/models/xgb_model.pkl
```

- `team_games_{year}.csv`: the team-row subset of OE. It replaces the NBA
  `game_log_*.csv` and plays the same role: rolling, side-split, H2H, streak
  and rest features, plus "which games are already in state". Columns:
  `gameid, date, league, year, split, playoffs, game, patch, side, teamid,
  teamname, result, gamelength, teamkills, teamdeaths, golddiffat15,
  xpdiffat15, csdiffat15, firstblood, firstdragon, firstherald, firsttower,
  firstbaron, dragons, barons, towers, opp_towers, starters`. `starters` is the
  5 `playerid`s in role order `top|jng|mid|bot|sup`, `|`-joined. This is how
  live mode knows a team's last lineup without the raw files.
- `player_elo_current.parquet`: one row per `playerid`: `player_id,
  player_name, team_id, position, league, last_game_date, games_played,
  outcome_elo, general_elo, {skill}_elo×5`.
- `player_elo_recent.parquet`: last `RECENT_N = 11` pre-game snapshots per
  player (same columns as history).
- `role_norms.json`: per `position × stat` mean/std used for role
  normalisation (§5.1). It is computed **once** by `build_elo --force` from
  the historical data and then frozen. Freezing it keeps incremental updates
  equal to a full rebuild.
- **Not committed:** OE raw CSVs (about 77 MB/year), `player_elo.parquet`
  (full history), `feature_matrix.parquet`, `train_metrics.json`.
- `player_season_stats_*.csv` has **no equivalent**. NBA needed it for PPG
  weights. In LoL all 5 starters play the whole game, so team composites are
  plain means.

---

## 5. ELO (`elo.py`)

Two independent rating families per player, both updated per game.

### 5.1 Performance skill ELOs (translated from Courtside)

Same mechanics as `docs/elo.md`: per-game composite → rank among all
qualifying players in that game → `rank_to_delta(rank, n)` (reuse as is;
`n = 10` gives +5…+1, −1…−5) → zero-sum. Differences:

- **No minutes weighting, no qualification threshold.** All 10 play the full
  game. Drop `MIN_MINUTES`, `_per36` and the sqrt-minutes scaling. Keep the
  mean-subtraction (harmless, and it keeps zero-sum when a skill has fewer
  than 10 valid players).
- **Role normalisation before ranking.** A support's damage must not be
  ranked against an ADC's raw damage. For every component stat:
  `z = (x − mean[position, stat]) / std[position, stat]` using the frozen
  `role_norms.json` (computed over 2022–2025 `complete` rows). Then composite
  = mean of the component z's (inverted where noted). Then rank all 10.
- **Rates use `gamelength`** (minutes = `gamelength/60`) where OE gives raw
  counts.
- **Skills** (OE column names exact):

  | Skill | Components |
  |---|---|
  | `combat` | `kills/min`, `assists/min`, `deaths/min` (inverted) |
  | `damage` | `dpm`, `damageshare` |
  | `economy` | `earned gpm`, `earnedgoldshare`, `cspm` |
  | `laning` | `golddiffat15`, `xpdiffat15`, `csdiffat15` (**NaN in `partial`/LPL rows → skill skipped for that game**, exactly like NBA hustle) |
  | `vision` | `vspm`, `wcpm`, `controlwardsbought/min` |

  `general_elo = sum of the 5 skills`, `INITIAL_ELO = 1000` per skill.
- Skill-to-skill weighting is left to XGBoost, as in Courtside.

### 5.2 Outcome ELO (new, and why)

Within-game ranking is zero-sum **inside one game**. Regions almost never play
each other, so performance ELO can't tell an LCK player at 1100 from an LEC
player at 1100. Outcome ELO fixes that, and carries strength across regions
via international events and player transfers.

- Classic Elo on game results, **per player**: team rating `R = mean(outcome_elo
  of its 5 starters)`. Expected score `E = 1 / (1 + 10^((R_opp − R)/400))`.
  Each starter: `outcome_elo += K · (result − E)`, with `K = 24` as a module
  constant and tuning knob.
- **New-player init by league prior.** A player without a rating starts at
  the current mean outcome ELO of players whose last game was in the same
  OE `league`, if that league has ≥ 10 rated players. Otherwise they start at
  `LEAGUE_PRIOR.get(league, 1350)`. Seed `LEAGUE_PRIOR` with tiers:
  `LCK/LPL 1600`, `LEC/LCS 1500`, `LCP/CBLOL 1450`, everything else
  `1350`. The keys are OE `league` codes. Confirm the exact spellings against the data, since only `LCK`, `LPL`, `EM`, `DCup` were seen verbatim. Mark it with `# ponytail: hand-set tiers, fit from international
  results if it matters`.
- Games in international leagues (`WLDs`, `MSI`, and whatever OE codes First
  Stand/EWC use. `FST` is a guess. Build the set from the data) are where the cross-region
  signal flows. Leave K unchanged in v1.
- No seasonal regression in v1. Revisit if the backtest shows stale ratings
  across off-seasons.

### 5.3 Entry points (same names as Courtside)

- `build_elo(force=False, games=None)` replays every OE game since 2014 in
  `(date, gameid)` order. It writes `player_elo.parquet` (pre-game snapshot per
  player per game, including `outcome_elo`), the two state files, and (first
  run only, or `--force`) `role_norms.json`. ELO **warm-up** uses all years.
  Training starts later (§6.4).
- `update_elo(games)` is incremental and idempotent (it skips `gameid`s
  already in the recent snapshots). Its input is OE player rows.
- Drop `update_team_assignments` (no preseason concept). A player's `team_id`
  and `position` simply follow their latest game.
- **Late games.** OE sometimes adds a game a day or more late (LPL especially).
  `update_elo` applies late games in arrival order, not date order. Mark it
  `# ponytail: arrival-order updates for late OE rows; full rebuild re-sorts`.
  The parity test (§9) uses in-order input.

---

## 6. Features (`features.py`) and training

### 6.1 Orientation

The model predicts **P(blue side wins one game)**. Every feature is
`blue_*`, `red_*` and `*_diff` (blue − red). Side asymmetry (blue wins
roughly 52–55% in pro play) is learned from the label, so no explicit side
feature is needed.

`build_feature_matrix(blue_team_id, red_team_id, game_date, context, team_games_df,
use_current_elo, game_id=None, blue_starters=None, red_starters=None) -> dict[str, float]`.

### 6.2 Feature groups (one `get_*_features()` each, each in its own try/except)

| Group | Replaces (NBA) | Features |
|---|---|---|
| `get_elo_features` | same | per team: mean of 5 starters for 5 skills + `general` + `outcome` (14 per team, 7 diffs). **Plus per-role lane diffs**: `diff_{top,jng,mid,bot,sup}_general_elo` and `diff_{role}_outcome_elo` |
| `get_team_rolling_stats` | same | last-5 / last-10 games: win%, mean `golddiffat15` (NaN-safe), mean `gamelength`, kills/min, deaths/min, first-blood/dragon/herald/tower/baron rates, dragons/game |
| `get_side_splits` | home/away splits | team's win% on blue and on red in the last 365 days, and the side-matching one per orientation |
| `get_rest_features` | same | days since last game, games in last 7 days (drop back-to-back) |
| `get_head_to_head_features` | same | H2H games/wins/win% in the last 365 days (OE `teamid`s) |
| `get_roster_features` | player availability | starters who also started the team's previous game (0–5); starters with < 20 pro games ("rookie count") |
| `get_player_form_features` | same | mean change in `general_elo` and `outcome_elo` over the 5 starters' last 10 games |
| `get_streak_features` | same | current streak, last-5 win% |
| `get_context_features` | new | `is_international`, `playoffs` (OE), `league_tier` (numeric, from `LEAGUE_PRIOR` tier), `blue_league_mean_outcome_elo` / `red_…` (mean outcome ELO of each team's home league) |

**Not in v1:** draft (picks/bans/champions). The prediction is made pre-draft.
Patch features: `patch` is stored, but skip it until the backtest says
otherwise.

### 6.3 Rosters (the one remaining train/live difference)

- **Training:** the starters are the 5 players who actually played the game
  (from OE). Lineups are usually announced before the match, so this leaks
  little. Document it in `docs/model.md` anyway.
- **Live:** the team's `starters` from its most recent row in `team_games_*.csv`.
  Look up ELO from `player_elo_current.parquet`. A missing player falls back to
  the §5.2 init value.
- Everything else reads `team_games_df` + ELO state identically in both modes.

### 6.4 Dataset and training (`build_dataset.py`, `train.py`)

- `build_dataset.py`: one row per OE game from **2017-01-01** onward (ELO has
  warmed up since 2014), across **all leagues** (more data, and ERL players
  feed the majors). Label `blue_win`. Meta columns: `game_id, game_date,
  league, blue_team_id, red_team_id, blue_win`. Use pre-game ELO snapshots
  from `player_elo.parquet`. Rolling and other features use only rows with
  `date <` the game's date (strictly point-in-time. Assert it in a test).
- `train.py`: keep the chronological 70/15/15 split, Optuna +
  `TimeSeriesSplit`, and Platt calibration. Change: `META_COLS`, the label
  name. Add **two baselines** to `train_metrics.json`, computed on the test
  slice: (a) always pick blue, (b) pick the higher team `outcome_elo`. Also
  report metrics on the test slice **restricted to major leagues +
  internationals** (OE codes for LCK, LPL, LEC, LCS, CBLOL, LCP, Worlds, MSI, First Stand. Verify the spellings), since
  that is what the site predicts.
- **Gate:** if the calibrated model doesn't beat baseline (b) on log loss
  and accuracy on the major-league slice, **stop and report to the owner**
  before Phase 4.
- Re-pin `scikit-learn`/`xgboost` to the versions you train with, and note it
  in `requirements.txt`'s header comment, as Courtside does.

---

## 7. Daily pipeline

### 7.1 `daily_state.py`: `run(date=None, dry_run=False)`

1. Download the OE file(s) covering `last_state_date − LOOKBACK_DAYS(14)`
   through today. In early January that means **both** the previous and
   current year.
2. New games = `gameid`s not already in `team_games_*.csv`. Require exactly 12
   rows, 10 players with 5 roles per side, and a `result` present. Skip and
   log incomplete games. A later run picks them up.
3. Append team rows to `team_games_{year}.csv`. Call `elo.update_elo` on the
   player rows in `(date, gameid)` order.
4. Idempotent: running twice is a no-op. `--dry-run` writes nothing.

### 7.2 `predict.py`: `predict_upcoming(window_hours=36, date=None, dry_run=False)`

1. `lolesports.getSchedule` for `PREDICT_LEAGUES` (a single module
   constant of slugs: `lck, lpl, lec, lcs, cblol-brazil, lcp, worlds, msi,
   first_stand`) → `type == "match"`, `state == "unstarted"`, and
   `startTime` within `[now, now + window_hours]`.
2. Skip any match with a `TBD` team or an unresolved team (§3.3).
3. Per match (team1 = `teams[0]`, team2 = `teams[1]`):
   - `p1 = P(team1 blue wins)` from features (team1 blue, team2 red).
   - `p2 = 1 − P(team2 blue wins)` from features (team2 blue, team1 red).
   - `p_game = (p1 + p2) / 2`, the side-neutral per-game probability. Side
     selection isn't reliably known pre-match.
   - Series probability with `n = strategy.count`, assuming independent games
     (`# ponytail: iid games, ignores momentum and side alternation`):
     - Bo1: `p`
     - Bo3: `p²(3 − 2p)`
     - Bo5: `p³(10 − 15p + 6p²)`
     - In general, P(first to ⌈n/2⌉ wins). Implement the general form and
       test it against these closed forms.
   - SHAP: top 10 from the team1-blue orientation, signed so positive pushes
     toward team1.
4. Upsert keyed on `match_id`. **Never overwrite a row that has
   `actual_winner`**. Rows for still-unstarted matches may be refreshed by
   later runs (same rule as Courtside).
5. CLI, mirroring Courtside: `--match-id <id>`, and
   `--team1 GEN --team2 T1 --bo 5 [--league lck]` for ad-hoc predictions.

### 7.3 `evaluate.py`

- For every `predictions` row with `actual_winner IS NULL` and
  `start_time_utc < now`: call `getEventDetails(match_id)`. If the match's
  teams have `result.outcome`, set `actual_winner` (the winning team's code),
  `score` (`"3-1"` from `gameWins`), and `correct`.
- Still unresolved 7 days after `start_time_utc` (postponed or forfeited):
  log a warning and leave it alone. Nothing auto-voids.
- `RECORD_START = "<launch date>"`. Recompute `running_record` on every run
  (series-level), as Courtside does.

### 7.4 Workflows and timing

Matches run around the clock: LCK and LPL around 08:00–15:00 UTC, LEC
16:00–21:00, LCS and CBLOL 20:00–03:00. OE refreshes around 07:05 UTC.

- `daily_evaluate.yml`: cron `45 7 * * *` (07:45 UTC, after the OE refresh)
  → `daily_run.py --evaluate`, then commit state (keep the
  `git pull --rebase` + `[skip ci]` logic). Bot name `riftside-oracle[bot]`.
- `daily_predict.yml`: `workflow_run` after evaluate (keep), **plus** cron
  `0 3 * * *` (a safety net for early-UTC matches if evaluate runs late; it
  uses yesterday's state, which is fine) **plus** `0 15 * * *` (picks up
  bracket matches whose `TBD` resolved during the day).
- `keepalive.yml`: unchanged apart from workflow names.
- `check_sources.yml` (manual): runs `python src/pipeline/oe.py --check` and
  `python src/pipeline/lolesports.py --check` from a GitHub runner. Each one
  fetches a tiny piece (an OE Range request for the first 20 KB, plus
  `getLeagues`) and prints status and latency.
- Secrets: `SUPABASE_URL`, `SUPABASE_KEY`. Only add a proxy secret if Phase 0
  fails.

---

## 8. Storage and web

### 8.1 Supabase (`schema.sql`, new project)

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id           TEXT NOT NULL UNIQUE,      -- lolesports match id
    league_slug        TEXT NOT NULL,             -- e.g. "lck"
    league_name        TEXT NOT NULL,
    block_name         TEXT,                      -- e.g. "Playoffs", "Week 3"
    start_time_utc     TIMESTAMPTZ NOT NULL,
    match_date         DATE NOT NULL,             -- UTC date of start_time_utc
    best_of            SMALLINT NOT NULL CHECK (best_of IN (1, 3, 5)),
    team1_code TEXT NOT NULL, team1_name TEXT NOT NULL, team1_image TEXT,
    team2_code TEXT NOT NULL, team2_name TEXT NOT NULL, team2_image TEXT,
    team1_game_prob    NUMERIC(5, 4) NOT NULL,    -- side-neutral per-game p
    team1_series_prob  NUMERIC(5, 4) NOT NULL,
    team2_series_prob  NUMERIC(5, 4) NOT NULL,
    predicted_team     TEXT NOT NULL,             -- team code
    confidence         NUMERIC(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    actual_winner      TEXT,                      -- team code, NULL until resolved
    score              TEXT,                      -- e.g. "3-1"
    correct            BOOLEAN,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Plus indexes on `match_date DESC` and `correct`. `shap_values` and
`running_record` stay **unchanged**. **Drop `model_metadata`**, since nothing
ever wrote to it. Keep RLS exactly as Courtside does: public `SELECT` only,
and the pipeline uses the service-role key.

### 8.2 Web (`web/`)

- Rename the site, metadata and copy to Riftside Oracle. Keep the layout and
  components.
- `lib/types.ts`: mirror the new `predictions` row.
- `lib/teams.ts`: **delete**. Names and logos come from the row
  (`team*_name`, `team*_image`). `TeamLogo` takes a URL. Rewrite
  lolesports' `http://static.lolesports.com/...` to `https://`.
- `GameCard` / `card/page.tsx`: show `league_name · Bo{best_of}`, team1 vs
  team2, series win %, the score once resolved. Replace "tip-off" with
  "start". The live window becomes `best_of × 1h`.
- Dates: group by the **viewer's local date** of `start_time_utc` (NBA used
  ET). Adapt `lib/date.ts`.
- "About the model" copy: rewrite for LoL. Keep metrics as **placeholders**
  until Phase 3 produces real numbers, then hardcode them as Courtside does.
- Footer, required for Riot fan projects: *"Riftside Oracle isn't endorsed by
  Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
  officially involved in producing or managing Riot Games properties. Riot
  Games, and all associated properties are trademarks or registered trademarks
  of Riot Games, Inc."* Add: "Match data: Oracle's Elixir and lolesports".
- No betting language, odds or affiliate links anywhere (Riot policy, plus
  the scope of a fan project).

---

## 9. Phases and acceptance checks

Do these in order. Each phase is its own commit.

**Phase 0: clean slate and reachability**
- Delete everything marked delete in §2. Rename in `README`/`CLAUDE.md` minimally
  (full docs come in Phase 6). Add `check_sources.yml`, `oe.py --check` and
  `lolesports.py --check`.
- ✅ `pytest` runs (an empty or near-empty suite is fine). Both `--check`
  commands pass locally. **Ask the owner to run `check_sources.yml` once on
  GitHub.** If either source is blocked from runners, port the proxy failover
  from git history (`nba_cdn.py`) into a shared helper before Phase 4.

**Phase 1: data layer**
- `oe.py`: `list_files()`, `download_year(year, dest)` (skips the download if
  the local file is < 20h old), `load_year(year) -> DataFrame` (normalised
  dtypes, needed columns, NaN-tolerant), `player_rows(df)`, `team_rows(df)`,
  `--check`.
- `lolesports.py`: `get_leagues()`, `get_schedule(league_ids, page_token=None)`,
  `upcoming_matches(slugs, start, end)`, `get_event_details(match_id)`,
  `resolve_team(name, code)`, `--check`.
- `bootstrap_data.py`: downloads every year into `src/data/raw/oe/`.
- Fixtures: 3 OE games (36 rows: one `complete`, one LPL `partial` with
  `MISSING` name, one international), plus one `getSchedule` page and one
  `getEventDetails` JSON. **Capture these from the live API, then trim them.**
- ✅ Tests for loading, row splitting, the partial-row NaNs, team resolution
  (alias hit, normalised hit, miss → None), and TBD filtering.

**Phase 2: ELO**
- §5 in full.
- ✅ Tests: `rank_to_delta` sums to 0. Role normalisation is applied. A
  `partial` game skips `laning` only. Outcome ELO is zero-sum per game
  between the two teams' starter means. A new player gets the league-prior
  init. **Incremental `update_elo` over in-order games equals a full
  `build_elo`** (port `test_elo_update.py`). Re-applying the same games is a
  no-op.
- ✅ A full local `elo.py --force` run over 2014–2026 finishes. Print the top
  10 players by `outcome_elo` for the owner to sanity-check (well-known LCK/LPL
  names should dominate).

**Phase 3: features, dataset, training**
- §6 in full.
- ✅ Tests: point-in-time (no row with `date >=` the game's date feeds its
  features). Live and training produce the same keys. The series-probability
  closed forms. A feature group's failure leaves the other groups intact.
- ✅ `build_dataset.py --force` → `train.py --trials 100 --force` completes.
  The metrics and baselines (§6.4) go into a short report to the owner. **Gate
  on beating baseline (b).**

**Phase 4: daily pipeline**
- §7 in full: `daily_state.py`, `predict.py`, `evaluate.py`, `daily_run.py`,
  workflows.
- ✅ Tests (FakeSupabase): predict skips started/TBD/unmapped/resolved
  matches, evaluate resolves the winner and score, `daily_state` is
  idempotent.
- ✅ `daily_run.py --dry-run` against the live APIs prints sensible
  predictions for the next 36h, and every major-league team resolves (add
  aliases as needed).

**Phase 5: storage and web**
- §8 in full.
- ✅ `cd web && npm run build` succeeds. `npm run dev` with mock or real rows
  renders `/` and `/card` (420×260). Screenshot both for the owner.

**Phase 6: docs**
- Rewrite `CLAUDE.md` (same sections: what this is, data flow, repo map,
  running locally, committed-state contract, data access rules, invariants,
  external things, known limitations, common changes), `README.md`, and
  `docs/{architecture,data,elo,model,operations}.md` for Riftside. Remove
  every NBA reference (`grep -ri "nba\|courtside" .` comes back clean, apart
  from a one-line "forked from courtside-oracle" credit).

---

## 10. Owner-only steps (the agent can't do these)

1. Create the GitHub repo and push. Set Actions secrets `SUPABASE_URL` and
   `SUPABASE_KEY`.
2. Run `check_sources.yml` once (Phase 0).
3. Create a **new** Supabase project (free tier allows 2 active projects) and
   run `schema.sql` in the SQL editor. Put the anon key in `web/.env.local`.
4. Run the full ELO build and training locally (Phases 2–3; minutes, not
   hours). Commit the state files and `xgb_model.pkl`.
5. Set `RECORD_START` to the launch date. Enable the workflows.
6. Set up the Plesk subdomain `riftside-oracle.gerritvisser.de`. Add a CSP
   `frame-ancestors` entry if `/card` will be embedded on `gerritvisser.de`.
   Build and upload `web/out/` manually, as for Courtside.
7. Optional: add the card to the portfolio repo (`IcaruzSoftware/portfolio`).

---

## 11. Known limitations to document (not to solve in v1)

- **Pre-draft only.** The draft swings games, and the model can't see it.
  The ceiling is well below what draft-aware or live models reach.
- **Cross-region strength** rests on international games (a few hundred per
  year), player transfers and hand-set league priors.
- **Small samples and roster churn.** 100–200 games per split per major
  league. Rosters change every off-season (Nov–Jan), so early-season
  predictions are the weakest.
- **Live roster = last-known starters.** Same-day substitutions are invisible.
- **OE lag about 1 day.** A team playing on consecutive days is predicted
  without yesterday's game. LPL rows can lag more and lack early-game stats.
- **Unofficial lolesports API.** It can change without notice. Calls fail
  loudly and never silently skip.
- **Series math assumes iid games.**

---

## 12. Timing context

On 2026-10-15, Worlds 2026 is in progress (finals in early November). The
2027 season starts around January (LCK Cup, LPL Split 1, LEC). Target: Phases
0–5 done during the off-season, launched with `RECORD_START` at the first 2027
major-league match day. Worlds 2026 games already in OE are good backtest
material for the international/cross-region logic.
