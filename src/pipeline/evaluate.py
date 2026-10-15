"""
Resolve predictions against lolesports results and update Supabase.

  SUPABASE_URL=... SUPABASE_KEY=... python src/pipeline/evaluate.py
  python src/pipeline/evaluate.py --dry-run     # print, don't write

What it does:
  1. Fetches predictions with actual_winner NULL whose start_time_utc has passed.
  2. getEventDetails per match: the winner is the team with result.outcome "win",
     else the team whose gameWins reached ceil(best_of / 2) (getEventDetails only
     carries gameWins). Sets actual_winner (team code), score ("team1-team2" game
     wins, e.g. "3-1") and correct.
  3. Still unresolved UNRESOLVED_WARN_DAYS after the start (postponed, forfeited):
     warning only; nothing auto-voids.
  4. Recomputes the series-level running_record over match_date >= RECORD_START on
     every run.
A match whose lookup raises is logged and skipped; the others still resolve, the
record is still recomputed, and evaluate() raises once at the end.
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import lolesports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The public running record counts only predictions from the launch date onward.
RECORD_START = "2026-10-15"
UNRESOLVED_WARN_DAYS = 7


def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def match_result(event: dict | None) -> tuple[str, str] | None:
    """
    (winning team code, "team1-team2" game wins) from getEventDetails, or None if
    undecided. A None event (unknown/removed match id: data.event is null) is undecided.
    """
    if event is None:
        return None
    teams = event["match"]["teams"]
    results = [t.get("result") or {} for t in teams]
    wins = [r.get("gameWins") or 0 for r in results]
    need = math.ceil(event["match"]["strategy"]["count"] / 2)
    for t, r, w in zip(teams, results, wins):
        if r.get("outcome") == "win" or w >= need:
            return t["code"], f"{wins[0]}-{wins[1]}"
    return None


def evaluate(dry_run: bool = False, db=None, now: datetime | None = None) -> None:
    if db is None:
        if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
            if dry_run:
                logger.info("Supabase not configured - dry run has nothing to evaluate.")
                return
            raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
        db = _get_supabase()
    now = now or datetime.now(timezone.utc)

    pending = (db.table("predictions").select("*").is_("actual_winner", "null").execute().data) or []
    started = [p for p in pending if datetime.fromisoformat(p["start_time_utc"]) < now]
    logger.info("Evaluating %d started, unresolved prediction(s)%s...", len(started),
                " [dry-run]" if dry_run else "")
    failed = []
    for pred in started:
        try:
            _resolve(db, pred, now, dry_run)
        except Exception as exc:        # one bad match must not block the others
            logger.warning("  %s: evaluation failed: %s", pred.get("match_id"), exc)
            failed.append(pred.get("match_id"))

    # Recompute the running record on every run, so the site reflects the correct
    # (RECORD_START-onward) tally even on days with nothing to resolve.
    _recompute_running_record(db, dry_run)
    if failed:
        raise RuntimeError(f"evaluation failed for {len(failed)} match(es): {', '.join(map(str, failed))}")


def _resolve(db, pred: dict, now: datetime, dry_run: bool) -> None:
    """Resolve one started prediction from getEventDetails, or log why it stays open."""
    result = match_result(lolesports.get_event_details(pred["match_id"]))
    label = f"{pred['team1_code']} vs {pred['team2_code']} ({pred['match_id']})"
    if result is None:
        if now - datetime.fromisoformat(pred["start_time_utc"]) > timedelta(days=UNRESOLVED_WARN_DAYS):
            logger.warning("  %s still unresolved %d+ days after its start", label,
                           UNRESOLVED_WARN_DAYS)
        else:
            logger.info("  %s - result not available yet", label)
        return
    winner, score = result
    correct = pred["predicted_team"] == winner
    logger.info("  %s  predicted %s  actual %s %s  %s%s", label, pred["predicted_team"],
                winner, score, "CORRECT" if correct else "WRONG", " [dry-run]" if dry_run else "")
    if not dry_run:
        db.table("predictions").update({"actual_winner": winner, "score": score,
                                        "correct": correct}).eq("match_id", pred["match_id"]).execute()


def _recompute_running_record(db, dry_run: bool) -> None:
    """
    Recompute + upsert running_record over resolved predictions from RECORD_START.
    Counted server-side (count="exact", no rows returned): a plain select would stop
    at Supabase's 1000-row default and undercount.
    """
    def count(correct: bool) -> int:
        return (db.table("predictions").select("id", count="exact", head=True)
                .gte("match_date", RECORD_START).eq("correct", correct).execute().count) or 0

    total_correct    = count(True)
    total_incorrect  = count(False)
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
    parser = argparse.ArgumentParser(description="Resolve predictions against lolesports results")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing to Supabase")
    args = parser.parse_args()
    evaluate(dry_run=args.dry_run)
