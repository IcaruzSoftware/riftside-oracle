"""
Predict pro LoL series from the lolesports schedule.

Per match (team1 = teams[0], team2 = teams[1]):
  p1     = P(team1 wins a game on blue)            features: team1 blue, team2 red
  p2     = 1 - P(team2 wins a game on blue)        features: team2 blue, team1 red
  p_game = (p1 + p2) / 2                           side-neutral (sides aren't known pre-match)
  series = features.series_win_prob(p_game, best_of)
  SHAP   = top 10 of the team1-blue orientation (positive pushes toward team1)

Live context: the lolesports league slug maps to the OE league code (OE_LEAGUE);
playoffs = 1 for a domestic match whose blockName names a knockout stage
(Playoffs, Finals, Knockouts, Play-Ins / Play In Knockouts; OE flags those as
playoffs), always 0 for internationals (OE never flags them).

Modes
-----
  python src/pipeline/predict.py --match-id 115660588583171786
  python src/pipeline/predict.py --team1 "Gen.G Esports" --team2 T1 --bo 5 [--league lck]
  (daily pipeline) predict_upcoming(): every unstarted PREDICT_LEAGUES match in the
  next window_hours -> Supabase predictions + shap_values. A row that already has
  actual_winner is never touched; unstarted rows are refreshed.
"""

import argparse
import logging
import os
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import elo, lolesports
from pipeline.daily_state import load_team_games
from pipeline.features import build_feature_matrix, series_win_prob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.pkl"

PREDICT_LEAGUES = ["lck", "lpl", "lec", "lcs", "cblol-brazil", "lcp", "worlds", "msi", "first_stand"]
OE_LEAGUE = {"lck": "LCK", "lpl": "LPL", "lec": "LEC", "lcs": "LCS", "cblol-brazil": "CBLOL",
             "lcp": "LCP", "worlds": "WLDs", "msi": "MSI", "first_stand": "FST"}
_PLAYOFF_BLOCK = re.compile(r"playoff|final|knockout|play.?in", re.IGNORECASE)
SHAP_TOP_N = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _matrix(bundle: dict, feats: list[dict]) -> np.ndarray:
    return np.array([[f.get(n, np.nan) for n in bundle["feature_names"]] for f in feats],
                    dtype=np.float32)


def _predict_proba(bundle: dict, feats: list[dict]) -> np.ndarray:
    """Platt-calibrated P(blue wins) per feature dict."""
    raw = bundle["base_model"].predict_proba(_matrix(bundle, feats))[:, 1].reshape(-1, 1)
    return bundle["platt"].predict_proba(raw)[:, 1]


def _shap_top(bundle: dict, feats: dict, n: int = SHAP_TOP_N) -> list[tuple[str, float]]:
    """Top-n (feature, SHAP value) by |impact| on the base model's log-odds (XGBoost TreeSHAP)."""
    contribs = bundle["base_model"].get_booster().predict(
        xgb.DMatrix(_matrix(bundle, [feats])), pred_contribs=True)[0, :-1]   # last = bias
    pairs = sorted(zip(bundle["feature_names"], contribs), key=lambda x: abs(x[1]), reverse=True)
    return [(f, float(v)) for f, v in pairs[:n]]


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def match_context(league_slug: str, block_name: str | None) -> dict:
    """{"league": OE code, "playoffs": 0/1} for build_feature_matrix (see module docstring)."""
    league = OE_LEAGUE[league_slug]
    playoffs = league not in elo.INTERNATIONAL and bool(_PLAYOFF_BLOCK.search(block_name or ""))
    return {"league": league, "playoffs": int(playoffs)}


def predict_match(team1_id: str, team2_id: str, best_of: int, day, context: dict,
                  team_games: pd.DataFrame, bundle: dict) -> dict:
    """Side-neutral game probability, series probability and SHAP top list for team1."""
    f1 = build_feature_matrix(team1_id, team2_id, day, context, team_games, True)
    f2 = build_feature_matrix(team2_id, team1_id, day, context, team_games, True)
    p1_blue, p2_blue = _predict_proba(bundle, [f1, f2])
    p_game = float((p1_blue + 1.0 - p2_blue) / 2.0)
    return {"game_prob": p_game, "series_prob": series_win_prob(p_game, best_of),
            "shap": _shap_top(bundle, f1), "features": f1}


# ---------------------------------------------------------------------------
# Daily pipeline entry point
# ---------------------------------------------------------------------------

def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def _row(event: dict, start: datetime, pred: dict) -> dict:
    t1, t2 = event["match"]["teams"]
    p, s = round(pred["game_prob"], 4), round(pred["series_prob"], 4)
    pick = t1 if s >= 0.5 else t2
    return {
        "match_id":          event["match"]["id"],
        "league_slug":       event["league"]["slug"],
        "league_name":       event["league"]["name"],
        "block_name":        event.get("blockName"),
        "start_time_utc":    start.isoformat(),
        "match_date":        start.date().isoformat(),
        "best_of":           event["match"]["strategy"]["count"],
        "team1_code":        t1["code"], "team1_name": t1["name"], "team1_image": t1.get("image"),
        "team2_code":        t2["code"], "team2_name": t2["name"], "team2_image": t2.get("image"),
        "team1_game_prob":   p,
        "team1_series_prob": s,
        "team2_series_prob": round(1.0 - s, 4),
        "predicted_team":    pick["code"],
        "confidence":        max(s, round(1.0 - s, 4)),
    }


def predict_upcoming(window_hours: float = 36, date: str | None = None, dry_run: bool = False,
                     db=None) -> list[dict]:
    """
    Predict every unstarted PREDICT_LEAGUES match starting within window_hours of now
    (or of 00:00 UTC on `date`) and upsert it on match_id with its SHAP rows.

    Skipped: TBD teams and started matches (lolesports.upcoming_matches), teams that
    don't resolve to an OE teamid (logged), best_of outside {1, 3, 5}, and rows that
    already have actual_winner. dry_run logs instead of writing (db is then optional).
    A match that raises (malformed event, failed write) is logged and skipped; the
    rest still run, then this raises once. Returns the prediction rows.
    """
    start = (datetime.fromisoformat(date).replace(tzinfo=timezone.utc) if date
             else datetime.now(timezone.utc))
    events = lolesports.upcoming_matches(PREDICT_LEAGUES, start, start + timedelta(hours=window_hours))
    if not events:
        logger.info("No unstarted matches in the next %sh.", window_hours)
        return []
    logger.info("Predicting %d match(es) starting %s .. +%sh", len(events),
                start.isoformat(timespec="minutes"), window_hours)

    team_games = load_team_games()
    bundle = _load_model()
    if not dry_run and db is None:
        db = _get_supabase()

    results, failed = [], []
    for e in events:
        try:
            row = _predict_event(e, start, team_games, bundle, db, dry_run)
        except Exception as exc:        # one bad match must not block the others
            match_id = (e.get("match") or {}).get("id")
            logger.warning("  %s: prediction failed: %s", match_id, exc)
            failed.append(str(match_id))
            continue
        if row is not None:
            results.append(row)

    logger.info("Predicted %d of %d match(es)%s.", len(results), len(events),
                " [dry-run]" if dry_run else "")
    if failed:
        raise RuntimeError(f"prediction failed for {len(failed)} match(es): {', '.join(failed)}")
    return results


def _predict_event(e: dict, start: datetime, team_games: pd.DataFrame, bundle: dict, db,
                   dry_run: bool) -> dict | None:
    """Predict + upsert one schedule event; None if it is skipped (logged)."""
    m = e["match"]
    t1, t2 = m["teams"]
    ids = [lolesports.resolve_team(t["name"], t["code"], team_games, as_of=start) for t in (t1, t2)]
    if None in ids:
        logger.warning("  %s %s vs %s: unmapped team, skipped", m["id"], t1["code"], t2["code"])
        return None
    if m["strategy"]["count"] not in (1, 3, 5):
        logger.warning("  %s: best of %s not supported, skipped", m["id"], m["strategy"]["count"])
        return None
    if db is not None:
        existing = (db.table("predictions").select("actual_winner")
                    .eq("match_id", m["id"]).execute().data)
        if existing and existing[0].get("actual_winner") is not None:
            logger.info("  %s already resolved - leaving it untouched", m["id"])
            return None

    when = lolesports._start(e)
    pred = predict_match(ids[0], ids[1], m["strategy"]["count"], when,
                         match_context(e["league"]["slug"], e.get("blockName")), team_games, bundle)
    row = _row(e, when, pred)
    logger.info("  %s %-9s %-14s Bo%d  %s vs %s  game %.3f  series %.3f -> %s (%.1f%%)",
                when.strftime("%m-%d %H:%M"), row["league_slug"], row["block_name"] or "",
                row["best_of"], row["team1_code"], row["team2_code"], row["team1_game_prob"],
                row["team1_series_prob"], row["predicted_team"], row["confidence"] * 100)

    if not dry_run:
        pred_id = db.table("predictions").upsert(row, on_conflict="match_id").execute().data[0]["id"]
        db.table("shap_values").delete().eq("prediction_id", pred_id).execute()
        db.table("shap_values").insert([
            {"prediction_id": pred_id, "feature_name": f, "shap_value": round(v, 6),
             "feature_value": None if np.isnan(pred["features"].get(f, np.nan))
             else round(pred["features"][f], 4)}
            for f, v in pred["shap"]]).execute()
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_prediction(title: str, t1: str, t2: str, best_of: int, pred: dict) -> None:
    s = pred["series_prob"]
    print()
    print("=" * 60)
    print(f"  {title}")
    print(f"  {t1}  vs  {t2}   (Bo{best_of})")
    print("=" * 60)
    print(f"  {t1} game win prob (side-neutral): {pred['game_prob'] * 100:.1f}%")
    print(f"  {t1} series win prob             : {s * 100:.1f}%")
    print(f"  Prediction: {t1 if s >= 0.5 else t2} wins ({max(s, 1 - s) * 100:.1f}%)")
    print()
    print(f"  Top factors (SHAP, {t1} on blue):")
    for f, v in pred["shap"]:
        print(f"    {f:<38} {v:+.4f}  -> {t1 if v > 0 else t2}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Predict a pro LoL series")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--match-id", help="lolesports match id")
    group.add_argument("--team1", help="Team name as on lolesports/OE, e.g. 'Gen.G Esports'")
    parser.add_argument("--team2", help="Second team name (with --team1)")
    parser.add_argument("--bo", type=int, choices=(1, 3, 5), help="Best of (with --team1)")
    parser.add_argument("--league", choices=sorted(OE_LEAGUE),
                        help="League slug (with --team1; default: team1's latest OE league)")
    parser.add_argument("--date", help="UTC date YYYY-MM-DD (default: today)")
    parser.add_argument("--window", type=float, default=36, help="Hours ahead (default mode)")
    parser.add_argument("--dry-run", action="store_true", help="Default mode: print, don't write")
    args = parser.parse_args()

    if not args.match_id and not args.team1:
        predict_upcoming(window_hours=args.window, date=args.date, dry_run=args.dry_run)
        return

    team_games = load_team_games()
    day = pd.Timestamp(args.date or datetime.now(timezone.utc).date())
    if args.match_id:
        # getEventDetails has no blockName: playoffs = 0.
        e = lolesports.get_event_details(args.match_id)
        if e is None:
            sys.exit(f"Unknown lolesports match id: {args.match_id}")
        if e["league"]["slug"] not in OE_LEAGUE:
            sys.exit(f"League {e['league']['slug']!r} is not predicted (known: {', '.join(sorted(OE_LEAGUE))})")
        t1, t2 = e["match"]["teams"]
        names, best_of = (t1["name"], t2["name"]), e["match"]["strategy"]["count"]
        ids = [lolesports.resolve_team(t["name"], t["code"], team_games) for t in (t1, t2)]
        context, title = match_context(e["league"]["slug"], None), f"{e['league']['name']} {args.match_id}"
    else:
        if not args.team2 or not args.bo:
            parser.error("--team2 and --bo are required with --team1")
        names, best_of = (args.team1, args.team2), args.bo
        ids = [lolesports.resolve_team(n, n, team_games) for n in names]
    if None in ids:
        sys.exit(f"Unmapped team: {names[ids.index(None)]} (add it to team_aliases.csv)")
    if not args.match_id:
        if args.league:
            league = OE_LEAGUE[args.league]
        else:       # team1's latest domestic league
            rows = team_games[(team_games["teamid"] == ids[0])
                              & ~team_games["league"].isin(elo.INTERNATIONAL)]
            if rows.empty:
                sys.exit(f"No domestic games for {names[0]}: pass --league")
            league = rows.loc[rows["date"].idxmax(), "league"]
        context, title = {"league": league, "playoffs": 0}, f"ad hoc, league {league}"
    pred = predict_match(ids[0], ids[1], best_of, day, context, team_games, _load_model())
    _print_prediction(f"{title}  {day.date()}", names[0], names[1], best_of, pred)


if __name__ == "__main__":
    main()
