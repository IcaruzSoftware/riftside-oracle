"""
Train XGBoost classifier on the feature matrix to predict blue-side game wins.

Approach
--------
  - Chronological train / calibration / test split, 70/15/15 (test = most recent games)
  - Optuna hyperparameter search minimising log loss in TimeSeriesSplit CV on the
    train slice only (70%), so the calibration slice stays clean for Platt and for
    model selection
  - Final model retrained on full training set with best params
  - Platt calibration applied so output probabilities are well-calibrated
  - Metrics, feature importance, and model saved to disk
  - Test-slice baselines: (a) always blue, (b) higher team outcome ELO; everything
    also on the MAJOR-league slice, and a GATE line: PASS only if the calibrated
    model beats baseline (b) on both log loss and accuracy there

Outputs
-------
  models/xgb_model.pkl            calibrated model ready for predict.py
  data/processed/train_metrics.json
  data/processed/feature_importance.csv

Run
---
  python src/pipeline/train.py                   # 50 Optuna trials (default)
  python src/pipeline/train.py --trials 100      # more search
  python src/pipeline/train.py --force           # retrain even if model exists
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH            = MODELS_DIR / "xgb_model.pkl"
METRICS_PATH          = PROCESSED_DIR / "train_metrics.json"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "feature_importance.csv"

OPTUNA_N_TRIALS = 50
TEST_FRACTION   = 0.15   # hold out most recent 15% as test set
CV_FOLDS        = 5
# ponytail: half the logical CPUs; all SMT threads made fits ~6x slower on the dev box
N_JOBS          = max(1, (os.cpu_count() or 2) // 2)

META_COLS = {"game_id", "game_date", "league", "blue_team_id", "red_team_id", "blue_win"}
LABEL     = "blue_win"
# OE codes of the leagues the site predicts (incl. their 2025 names) + internationals.
MAJOR = {"LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP", "LTA N", "LTA S", "LTA",
         "WLDs", "MSI", "FST"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data() -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Load feature_matrix.parquet, sort chronologically, return arrays.

    Returns:
        X:             float array (n_samples, n_features)
        y:             int array   (n_samples,)  1=blue win
        feature_names: list of column names matching X columns
        df:            the sorted frame (meta columns for the test-slice reports)
    """
    path = PROCESSED_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run build_dataset.py first."
        )

    df = pd.read_parquet(path)
    df = df.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL].values.astype(np.int32)

    logger.info(
        "Loaded %d games | %d features | blue win rate %.1f%%",
        len(df), len(feature_cols), y.mean() * 100,
    )
    return X, y, feature_cols, df


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_train: np.ndarray, y_train: np.ndarray) -> callable:
    """Return an Optuna objective that runs TimeSeriesSplit CV and returns mean log loss."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int  ("n_estimators",      300,  1500),
            "max_depth":         trial.suggest_int  ("max_depth",         3,    8),
            "learning_rate":     trial.suggest_float("learning_rate",     0.01, 0.2,  log=True),
            "subsample":         trial.suggest_float("subsample",         0.6,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.5,  1.0),
            "min_child_weight":  trial.suggest_int  ("min_child_weight",  1,    10),
            "gamma":             trial.suggest_float("gamma",             0.0,  2.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-4, 10.0, log=True),
            "objective":         "binary:logistic",
            "eval_metric":       "logloss",
            "tree_method":       "hist",
            "random_state":      42,
            "n_jobs":            N_JOBS,
            "verbosity":         0,
        }

        tscv   = TimeSeriesSplit(n_splits=CV_FOLDS)
        losses = []

        for train_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
            probs = model.predict_proba(X_train[val_idx])[:, 1]
            losses.append(log_loss(y_train[val_idx], probs))

        return float(np.mean(losses))

    return objective


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """accuracy (pick blue if p >= 0.5) / log loss / Brier / AUC of probabilities p."""
    return {
        "accuracy":    float(((p >= 0.5) == y).mean()),
        "log_loss":    float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "auc_roc":     float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
    }


def _report(y: np.ndarray, probs: dict[str, np.ndarray]) -> dict:
    return {"rows": int(len(y)), **{name: _metrics(y, p) for name, p in probs.items()}}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_trials: int = OPTUNA_N_TRIALS, force: bool = False) -> object:
    """
    Full training pipeline:
      1. Load feature matrix
      2. Chronological train/test split
      3. Optuna search on train (TimeSeriesSplit CV, log loss)
      4. Retrain best params on full training set
      5. Platt calibration on a held-out calibration slice
      6. Evaluate calibrated model on held-out test set
      7. Save model, metrics, feature importance

    Args:
        n_trials: Number of Optuna hyperparameter trials.
        force:    Retrain even if model already exists.

    Returns:
        Fitted calibrated model.
    """
    if MODEL_PATH.exists() and not force:
        logger.info("Model already exists — loading from %s (use --force to retrain)", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    X, y, feature_names, df = load_training_data()
    n = len(X)

    # ── Chronological splits ─────────────────────────────────────────────────
    # |<------- train+cal (85%) ------->|<---- test (15%) --->|
    # |<-- train (70%) -->|<-- cal (15%) -->|
    test_split  = int(n * (1 - TEST_FRACTION))
    cal_split   = int(n * (1 - TEST_FRACTION * 2))   # half of 30% → 15% cal

    X_test, y_test = X[test_split:], y[test_split:]
    X_train,    X_cal  = X[:cal_split],  X[cal_split:test_split]
    y_train,    y_cal  = y[:cal_split],  y[cal_split:test_split]

    logger.info(
        "Split → train: %d | calibration: %d | test: %d",
        len(X_train), len(X_cal), len(X_test),
    )

    # ── Optuna hyperparameter search ─────────────────────────────────────────
    logger.info("Optuna search: %d trials × %d CV folds...", n_trials, CV_FOLDS)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        _make_objective(X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, "objective": "binary:logistic",
                   "tree_method": "hist", "random_state": 42, "n_jobs": N_JOBS, "verbosity": 0}
    logger.info("Best CV log loss: %.4f", study.best_value)
    logger.info("Best params: %s", best_params)

    # ── Train base model on X_train ──────────────────────────────────────────
    logger.info("Training base model on %d samples...", len(X_train))
    base_model = xgb.XGBClassifier(**best_params)
    base_model.fit(X_train, y_train, verbose=False)

    # ── Platt calibration on held-out cal set ────────────────────────────────
    # Fit a logistic regression on top of the base model's raw probabilities.
    logger.info("Calibrating on %d samples...", len(X_cal))
    raw_cal = base_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt   = LogisticRegression(C=1.0)
    platt.fit(raw_cal, y_cal)

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        raw = base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return platt.predict_proba(raw)[:, 1]

    # ── Evaluate on test set ─────────────────────────────────────────────────
    test_probs  = _predict_proba(X_test)
    test_preds  = (test_probs >= 0.5).astype(int)

    accuracy    = float((test_preds == y_test).mean())
    auc_roc     = float(roc_auc_score(y_test, test_probs))
    brier       = float(brier_score_loss(y_test, test_probs))
    logloss     = float(log_loss(y_test, test_probs))

    # ── Baselines + MAJOR slice ──────────────────────────────────────────────
    # (a) always blue: p = train-slice blue rate (> 0.5, so every pick is blue).
    # (b) higher team outcome ELO: p = 1 / (1 + 10^(-diff/400)); a NaN diff -> 0.5.
    test_df   = df.iloc[test_split:]
    blue_rate = float(y_train.mean())
    elo_diff  = test_df["diff_outcome_elo"].fillna(0.0).to_numpy()
    probs = {
        "model":                test_probs,
        "baseline_always_blue": np.full(len(y_test), blue_rate),
        "baseline_outcome_elo": 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)),
    }
    major = test_df["league"].isin(MAJOR).to_numpy()
    report = {
        "all":   _report(y_test, probs),
        "major": _report(y_test[major], {k: v[major] for k, v in probs.items()}),
    }
    m, b = report["major"]["model"], report["major"]["baseline_outcome_elo"]
    gate = "PASS" if m["log_loss"] < b["log_loss"] and m["accuracy"] > b["accuracy"] else "FAIL"

    # Calibration curve (10 bins)
    frac_pos, mean_pred = calibration_curve(y_test, test_probs, n_bins=10)

    logger.info(
        "Test → Accuracy: %.4f | AUC-ROC: %.4f | Brier: %.4f | LogLoss: %.4f",
        accuracy, auc_roc, brier, logloss,
    )

    # ── Feature importance ───────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": base_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    top_n = 15
    logger.info("Top %d features by importance:", top_n)
    for _, row in importance_df.head(top_n).iterrows():
        logger.info("  %-40s %.4f", row["feature"], row["importance"])

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "accuracy":              accuracy,
        "auc_roc":               auc_roc,
        "brier_score":           brier,
        "log_loss":              logloss,
        "optuna_best_cv_logloss": study.best_value,
        "train_samples":         len(X_train),
        "calibration_samples":   len(X_cal),
        "test_samples":          len(X_test),
        "n_features":            len(feature_names),
        "train_blue_rate":       blue_rate,
        "test_from":             str(test_df["game_date"].min().date()),
        "test_to":               str(test_df["game_date"].max().date()),
        "test_report":           report,
        "gate":                  gate,
        "best_params":           best_params,
        "calibration_curve": {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value":  mean_pred.tolist(),
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", METRICS_PATH)

    # ── Save model ───────────────────────────────────────────────────────────
    saved = {"base_model": base_model, "platt": platt, "feature_names": feature_names}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(saved, f)
    logger.info("Model saved → %s", MODEL_PATH)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LoL blue-side win XGBoost model")
    parser.add_argument("--trials", type=int, default=OPTUNA_N_TRIALS,
                        help=f"Number of Optuna trials (default: {OPTUNA_N_TRIALS})")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if model already exists")
    args = parser.parse_args()

    model = train(n_trials=args.trials, force=args.force)

    # Print summary
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            m = json.load(f)
        print(f"\nTraining complete.")
        print(f"  Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
        print(f"  AUC-ROC   : {m['auc_roc']:.4f}")
        print(f"  Brier     : {m['brier_score']:.4f}  (lower = better, 0.25 = coin flip)")
        print(f"  Log Loss  : {m['log_loss']:.4f}")
        print(f"\nTest slice {m['test_from']} .. {m['test_to']}")
        for name, rep in m["test_report"].items():
            print(f"\n  {name} ({rep['rows']:,} games)   accuracy  log_loss  brier   auc")
            for k in ("model", "baseline_always_blue", "baseline_outcome_elo"):
                r = rep[k]
                print(f"  {k:<22} {r['accuracy']:.4f}    {r['log_loss']:.4f}   "
                      f"{r['brier_score']:.4f}  {r['auc_roc']:.4f}")
        print(f"\nGATE: {m['gate']} - calibrated model vs baseline_outcome_elo on the MAJOR "
              "slice (needs lower log loss AND higher accuracy)")
        print(f"\nModel → {MODEL_PATH}")
        print(f"Metrics → {METRICS_PATH}")
