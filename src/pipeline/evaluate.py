"""
Check prediction results and update Supabase.

Run the morning after games complete:
  SUPABASE_URL=... SUPABASE_KEY=... python src/pipeline/evaluate.py
  python src/pipeline/evaluate.py --dry-run     # print, don't write

What it does:
  1. Fetches predictions from Supabase where actual_winner is NULL
  2. Pulls final results from the NBA CDN box score (never stats.nba.com game logs)
  3. Marks each prediction correct/wrong + stores actual_winner
  4. Recomputes the running_record over predictions from RECORD_START onward, so the
     retroactively-seeded 2026 Finals rows (seed_finals.py) are excluded.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The running record counts only predictions from the current season onward. The
# 2026 Finals rows seeded by seed_finals.py have earlier game_dates and are excluded.
RECORD_START = "2026-10-01"


def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def _fetch_results_cdn(game_ids: list[str]) -> dict[str, str]:
    """Return {game_id: winning_team_tricode} for games that are final on the CDN."""
    from pipeline.nba_cdn import fetch_boxscore

    results: dict[str, str] = {}
    for gid in game_ids:
        game = fetch_boxscore(gid)
        if not game or game.get("gameStatus") != 3:
            continue
        home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
        hs, as_ = home.get("score"), away.get("score")
        if hs is None or as_ is None:
            continue
        results[gid] = home["teamTricode"] if hs > as_ else away["teamTricode"]
    return results


def evaluate(dry_run: bool = False, db=None) -> None:
    if db is None:
        if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
            if dry_run:
                logger.info("Supabase not configured — dry run has nothing to evaluate.")
                return
            raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
        db = _get_supabase()

    # All predictions without a result yet
    pending = (db.table("predictions").select("*").is_("actual_winner", "null").execute().data) or []
    if not pending:
        logger.info("No pending predictions to evaluate.")
    else:
        logger.info("Evaluating %d pending prediction(s)%s...",
                    len(pending), " [dry-run]" if dry_run else "")
        results = _fetch_results_cdn([p["game_id"] for p in pending])
        updated = 0
        for pred in pending:
            actual = results.get(pred["game_id"])
            if actual is None:
                logger.info("  %s — result not available yet", pred["game_id"])
                continue
            correct = (pred["predicted_team"] == actual)
            logger.info(
                "  %s @ %s  predicted: %s  actual: %s  %s%s",
                pred.get("away_team"), pred.get("home_team"),
                pred["predicted_team"], actual,
                "CORRECT" if correct else "WRONG",
                " [dry-run]" if dry_run else "",
            )
            if not dry_run:
                db.table("predictions").update({
                    "actual_winner": actual,
                    "correct":       correct,
                }).eq("game_id", pred["game_id"]).execute()
            updated += 1
        if updated == 0:
            logger.info("No completed games found yet.")

    # Recompute the running record on every run, so the site reflects the correct
    # (RECORD_START-onward) tally even on days with nothing to grade.
    _recompute_running_record(db, dry_run)


def _recompute_running_record(db, dry_run: bool) -> None:
    """Recompute + upsert running_record over resolved predictions from RECORD_START."""
    rows = (db.table("predictions").select("correct,game_date")
              .not_.is_("correct", "null").execute().data) or []
    counted          = [r for r in rows if str(r["game_date"]) >= RECORD_START]
    total_correct    = sum(1 for r in counted if r["correct"])
    total_incorrect  = sum(1 for r in counted if not r["correct"])
    total            = total_correct + total_incorrect
    accuracy         = round(total_correct / total, 4) if total > 0 else None

    logger.info(
        "Running record (from %s): %d correct / %d total (%.1f%%)%s",
        RECORD_START, total_correct, total, (accuracy or 0) * 100,
        " [dry-run]" if dry_run else "",
    )
    if not dry_run:
        db.table("running_record").upsert({
            "id":              1,
            "total_correct":   total_correct,
            "total_incorrect": total_incorrect,
            "accuracy":        accuracy,
            "last_updated":    datetime.now(timezone.utc).isoformat(),
        }, on_conflict="id").execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against CDN results")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing to Supabase")
    args = parser.parse_args()
    evaluate(dry_run=args.dry_run)
