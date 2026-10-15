"""
Resumable, single-threaded bootstrap downloader for the player-ELO rebuild.

Re-downloads exactly the raw data the player-centric ELO needs, after the
historical src/data/raw/ dump was lost. Built for an unattended overnight run
on a single Windows PC — no proxies, no threads.

Phases
------
  1  Season-level : game logs (regular + playoffs) and player season stats
                    for every season in SEASONS.
  2  Per-game     : the three box scores elo.load_game_players() reads —
                    traditional, advanced and tracking — for every game,
                    in chronological order (~14.1k games x 3 calls).

It reuses src/pipeline/collect.py unchanged: its endpoint classes, rate-limited
ProxyPool, retrying _api_call, _to_dict serialiser, _save_json and RAW_DIR. The
module-level pool is initialised with a single no-proxy slot.

Robustness
----------
  * Skips any file already present with size > 50 bytes (collect.py's rule).
  * Never deletes existing files.
  * Atomic writes (temp file + os.replace) — Ctrl+C never leaves a valid-looking
    partial file behind.
  * Backs off on sustained failure: 10 consecutive failed games -> 5 min pause;
    every third consecutive pause -> 30 min instead. A completed game resets both
    counters.
  * Keeps the PC awake on Windows via SetThreadExecutionState (no power settings
    are changed); no-op on other platforms.
  * Logs to stdout and src/data/raw/bootstrap.log (append); prints a progress
    line every 50 games with done/total, failures, calls/min and ETA.

CLI
---
  python src/scripts/bootstrap_data.py                 # phases 1 then 2
  python src/scripts/bootstrap_data.py --phase 1
  python src/scripts/bootstrap_data.py --phase 2 --limit 3

Exit code: 0 if no box scores are missing at the end of a phase-2 run, else 2.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipeline import collect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model training window — historical, so hardcoding is correct (2015-16 .. 2025-26).
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2026)]
SEASON_TYPES = ["Regular Season", "Playoffs"]

# The three box scores elo.load_game_players() reads — reuse collect.py's exact
# classes so file names and formats always match.
_WANTED = ("traditional", "advanced", "tracking")
BOXSCORE_TYPES = [(n, c) for n, c in collect.BOXSCORE_ENDPOINTS_CORE if n in _WANTED]

MIN_FILE_BYTES = 50            # collect.py's "file is real" threshold

FAIL_STREAK_PAUSE   = 10       # consecutive failed games before a pause
SHORT_PAUSE_SECONDS = 5 * 60
LONG_PAUSE_SECONDS  = 30 * 60
PAUSES_BEFORE_LONG  = 3        # every 3rd consecutive pause is the long one

PROGRESS_EVERY = 50            # games between progress lines

LOG_PATH     = collect.RAW_DIR / "bootstrap.log"
MISSING_PATH = collect.RAW_DIR / "missing_boxscores.txt"

# Windows SetThreadExecutionState flags
ES_CONTINUOUS      = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

logger = logging.getLogger("bootstrap")


# ---------------------------------------------------------------------------
# Logging and power management
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Log to stdout AND to bootstrap.log (append). Also route collect.py's own
    warnings (retries, per-file failures) into the same file."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file.setFormatter(fmt)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.handlers = [stream, file]
    logger.propagate = False

    logging.getLogger().addHandler(file)  # capture pipeline.collect logs in the file


def _prevent_sleep() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        logger.info("Sleep prevention enabled (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).")


def _allow_sleep() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention released.")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_FILE_BYTES


def _bs_path(game_id: str, name: str) -> Path:
    return collect.RAW_DIR / f"boxscore_{name}_{game_id}.json"


def _game_complete(game_id: str) -> bool:
    return all(_have(_bs_path(game_id, name)) for name, _ in BOXSCORE_TYPES)


def _atomic_save(data: dict, path: Path) -> None:
    """Serialise via collect._save_json to a temp file, then atomically replace.
    Ctrl+C never leaves a partial file at the real path."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        collect._save_json(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------------------
# Phase 1 — season-level data
# ---------------------------------------------------------------------------

def run_phase1() -> None:
    logger.info("=== Phase 1: game logs + player season stats ===")
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            try:
                collect.collect_game_log(season, season_type)  # skips existing
            except Exception as exc:
                logger.error("game_log %s %s failed: %s", season, season_type, exc)
        try:
            collect.collect_player_season_stats(season)  # skips existing
        except Exception as exc:
            logger.error("player_season_stats %s failed: %s", season, exc)

    n_logs = len(list(collect.RAW_DIR.glob("game_log_*.csv")))
    n_stats = len(list(collect.RAW_DIR.glob("player_season_stats_*.csv")))
    logger.info("Phase 1 complete: %d game-log CSVs, %d player-season-stats CSVs.",
                n_logs, n_stats)


# ---------------------------------------------------------------------------
# Game index
# ---------------------------------------------------------------------------

def _game_index() -> list[str]:
    """Unique game IDs (zfilled to 10, matching elo.py) from every game-log CSV,
    sorted chronologically by game date."""
    frames = []
    for csv in sorted(collect.RAW_DIR.glob("game_log_*.csv")):
        try:
            frames.append(pd.read_csv(csv, usecols=["GAME_ID", "GAME_DATE"]))
        except Exception as exc:
            logger.warning("could not read %s: %s", csv.name, exc)
    if not frames:
        raise FileNotFoundError(
            f"No game_log_*.csv in {collect.RAW_DIR} — run --phase 1 first."
        )
    idx = pd.concat(frames, ignore_index=True)
    idx["GAME_ID"] = idx["GAME_ID"].astype(str).str.zfill(10)
    idx["GAME_DATE"] = pd.to_datetime(idx["GAME_DATE"], errors="coerce")
    idx = (idx.drop_duplicates("GAME_ID")
              .sort_values(["GAME_DATE", "GAME_ID"])
              .reset_index(drop=True))
    return idx["GAME_ID"].tolist()


# ---------------------------------------------------------------------------
# Phase 2 — per-game box scores
# ---------------------------------------------------------------------------

def _process_game(game_id: str) -> tuple[int, bool]:
    """Download whichever of the 3 box scores are missing for one game.
    Returns (api_calls_made, complete_afterwards)."""
    calls = 0
    for name, endpoint_cls in BOXSCORE_TYPES:
        path = _bs_path(game_id, name)
        if _have(path):
            continue
        calls += 1
        try:
            result = collect._api_call(endpoint_cls, game_id=game_id)
            _atomic_save(collect._to_dict(result), path)
        except Exception as exc:
            logger.warning("boxscore_%s %s failed: %s", name, game_id, exc)
    return calls, _game_complete(game_id)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _log_progress(done: int, total: int, failed: int, n_calls: int, start: float) -> None:
    elapsed = time.time() - start
    cpm = n_calls / elapsed * 60 if elapsed > 0 else 0.0
    rate = done / elapsed if elapsed > 0 else 0.0        # games / sec
    eta = (total - done) / rate if rate > 0 else 0.0
    logger.info("Progress %d/%d games | %d failed | %.1f calls/min | ETA %s",
                done, total, failed, cpm, _fmt_duration(eta))


def run_phase2(limit: int | None) -> int:
    """Download the 3 box scores for every not-yet-complete game.
    Returns the number of games still missing any of the 3 files afterwards."""
    all_games = _game_index()
    pending = [g for g in all_games if not _game_complete(g)]
    logger.info("=== Phase 2: box scores === %d games, %d complete, %d pending.",
                len(all_games), len(all_games) - len(pending), len(pending))

    work = pending[:limit] if limit else pending
    if limit:
        logger.info("Limit active: processing first %d pending game(s).", len(work))

    total = len(work)
    start = time.time()
    n_calls = failed = consec_fail = consec_pause = 0
    processed = 0
    interrupted = False

    for i, game_id in enumerate(work, 1):
        try:
            calls, ok = _process_game(game_id)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Interrupted by user at game %d/%d (%s).", i, total, game_id)
            break

        processed = i
        n_calls += calls

        if ok:
            consec_fail = consec_pause = 0
        else:
            failed += 1
            consec_fail += 1
            if consec_fail >= FAIL_STREAK_PAUSE:
                consec_pause += 1
                secs = (LONG_PAUSE_SECONDS
                        if consec_pause % PAUSES_BEFORE_LONG == 0
                        else SHORT_PAUSE_SECONDS)
                logger.warning("%d consecutive failed games (pause #%d) — sleeping %d min.",
                               consec_fail, consec_pause, secs // 60)
                try:
                    time.sleep(secs)
                except KeyboardInterrupt:
                    interrupted = True
                    logger.warning("Interrupted during backoff pause.")
                    break
                consec_fail = 0

        if i % PROGRESS_EVERY == 0 or i == total:
            _log_progress(i, total, failed, n_calls, start)

    # Report what is still missing across ALL games, not just this run's slice.
    missing = [g for g in all_games if not _game_complete(g)]
    MISSING_PATH.write_text("\n".join(missing) + ("\n" if missing else ""),
                            encoding="utf-8")
    logger.info("Phase 2 %s: %d processed, %d failed this run, %d still missing "
                "(-> %s).",
                "interrupted" if interrupted else "complete",
                processed, failed, len(missing), MISSING_PATH.name)
    return len(missing)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap raw NBA data for the player-ELO rebuild."
    )
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N not-yet-complete games (phase 2).")
    args = parser.parse_args()

    _setup_logging()
    collect._pool = collect.ProxyPool([None])   # single slot, no proxies

    logger.info("Bootstrap start | phase=%s | limit=%s | seasons=%s..%s",
                args.phase, args.limit, SEASONS[0], SEASONS[-1])

    exit_code = 0
    _prevent_sleep()
    try:
        if args.phase in ("1", "all"):
            run_phase1()
        if args.phase in ("2", "all"):
            missing = run_phase2(args.limit)
            exit_code = 0 if missing == 0 else 2
        elif args.phase == "1":
            try:
                logger.info("Unique game IDs available: %d", len(_game_index()))
            except FileNotFoundError as exc:
                logger.warning("%s", exc)
    finally:
        _allow_sleep()

    logger.info("Bootstrap finished | exit=%d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
