"""
Daily orchestration entry point for GitHub Actions.

Usage:
    python src/scripts/daily_run.py --predict            # predict the next 36h of matches
    python src/scripts/daily_run.py --evaluate           # evaluate + update state
    python src/scripts/daily_run.py                       # both
    python src/scripts/daily_run.py --predict --dry-run --date 2026-10-16

The evaluate step also runs the daily state update (OE -> team_games + incremental
ELO) so the committed state is current before the next predictions. Every part is
attempted even if an earlier one failed (a blocked OE download must not stop
evaluate or predict; predict then uses the committed state); the exit code is
non-zero if any part failed.

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
    from pipeline.predict import predict_upcoming
    predictions = predict_upcoming(date=date, dry_run=dry_run)
    logger.info("Predictions complete: %d match(es).", len(predictions))


def run_evaluate(dry_run: bool) -> None:
    from pipeline.evaluate import evaluate
    evaluate(dry_run=dry_run)


def run_state_update(date: str | None, dry_run: bool) -> None:
    from pipeline.daily_state import run as update_state
    update_state(date=date, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run",  action="store_true", help="Print instead of writing")
    parser.add_argument("--date",     help="UTC date YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    run_both = not args.predict and not args.evaluate
    steps = []
    if args.evaluate or run_both:
        # Independent: a Supabase hiccup must not skip the state update.
        steps += [("evaluate", lambda: run_evaluate(args.dry_run)),
                  ("state update", lambda: run_state_update(args.date, args.dry_run))]
    if args.predict or run_both:
        steps.append(("predict", lambda: run_predict(args.date, args.dry_run)))

    failed = []
    for name, step in steps:
        logger.info("=== %s ===", name)
        try:
            step()
        except Exception as exc:
            logger.error("%s failed: %s", name, exc, exc_info=True)
            failed.append(name)

    if failed:
        logger.error("=== Done with failures: %s ===", ", ".join(failed))
        return 1
    logger.info("=== Done. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
