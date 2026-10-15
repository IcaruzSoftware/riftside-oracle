"""
Daily orchestration entry point for GitHub Actions.

Usage:
    python src/scripts/daily_run.py --predict            # predict today's games
    python src/scripts/daily_run.py --evaluate           # evaluate + update state
    python src/scripts/daily_run.py                       # both
    python src/scripts/daily_run.py --predict --dry-run --date 2026-06-13

The evaluate step also runs the daily state update (CDN box scores -> game logs,
player stats, incremental ELO) so the committed state is current before the next
day's predictions.

Required environment variables (unless --dry-run):
    SUPABASE_URL
    SUPABASE_KEY
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("daily_run")


def run_predict(date: str | None, dry_run: bool) -> None:
    from pipeline.predict import predict_todays_games
    predictions = predict_todays_games(date=date, dry_run=dry_run)
    logger.info("Predictions complete: %d game(s).", len(predictions))


def run_evaluate(dry_run: bool) -> None:
    from pipeline.evaluate import evaluate
    evaluate(dry_run=dry_run)


def run_state_update(date: str | None, dry_run: bool) -> None:
    from pipeline.daily_state import run as update_state
    update_state(date=date, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run",  action="store_true", help="Print instead of writing")
    parser.add_argument("--date",     help="ET date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    run_both = not args.predict and not args.evaluate
    errors = []

    if args.evaluate or run_both:
        # evaluate and the state update are independent: a Supabase hiccup must not
        # skip the state update, and either failing must fail the job.
        logger.info("=== Evaluating results ===")
        try:
            run_evaluate(args.dry_run)
        except Exception as exc:
            logger.error("evaluate failed: %s", exc, exc_info=True)
            errors.append(str(exc))

        logger.info("=== Updating state ===")
        try:
            run_state_update(args.date, args.dry_run)
        except Exception as exc:
            logger.error("state update failed: %s", exc, exc_info=True)
            errors.append(str(exc))

    if args.predict or run_both:
        logger.info("=== Generating predictions ===")
        try:
            run_predict(args.date, args.dry_run)
        except Exception as exc:
            logger.error("predict failed: %s", exc, exc_info=True)
            errors.append(str(exc))

    if errors:
        sys.exit(1)

    logger.info("=== Done. ===")


if __name__ == "__main__":
    main()
