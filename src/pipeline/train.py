"""
Train XGBoost classifier on the feature matrix to predict home team wins.

Approach
--------
  - Chronological train/test split (no shuffle — test = most recent games)
  - Optuna hyperparameter search using TimeSeriesSplit CV on training portion
  - Final model retrained on full training set with best params
  - Platt calibration applied so output probabilities are well-calibrated
  - Metrics, feature importance, and model saved to disk

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

META_COLS = {"game_id", "game_date", "season", "home_team_id", "away_team_id", "home_win"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load feature_matrix.parquet, sort chronologically, return arrays.

    Returns:
        X:             float array (n_samples, n_features)
        y:             int array   (n_samples,)  1=home win
        feature_names: list of column names matching X columns
    """
    path = PROCESSED_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run build_dataset.py first."
        )

    df = pd.read_parquet(path)
    df = df.sort_values("game_date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df["home_win"].values.astype(np.int32)

    logger.info(
        "Loaded %d games | %d features | home win rate %.1f%%",
        len(df), len(feature_cols), y.mean() * 100,
    )
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_train: np.ndarray, y_train: np.ndarray) -> callable:
    """Return an Optuna objective that runs TimeSeriesSplit CV and returns mean AUC."""

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
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }

        tscv      = TimeSeriesSplit(n_splits=CV_FOLDS)
        auc_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
            probs = model.predict_proba(X_train[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y_train[val_idx], probs))

        return float(np.mean(auc_scores))

    return objective


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_trials: int = OPTUNA_N_TRIALS, force: bool = False) -> object:
    """
    Full training pipeline:
      1. Load feature matrix
      2. Chronological train/test split
      3. Optuna search on training portion (TimeSeriesSplit CV)
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

    X, y, feature_names = load_training_data()
    n = len(X)

    # ── Chronological splits ─────────────────────────────────────────────────
    # |<------- train+cal (85%) ------->|<---- test (15%) --->|
    # |<-- train (70%) -->|<-- cal (15%) -->|
    test_split  = int(n * (1 - TEST_FRACTION))
    cal_split   = int(n * (1 - TEST_FRACTION * 2))   # half of 30% → 15% cal

    X_trainval, X_test = X[:test_split], X[test_split:]
    y_trainval, y_test = y[:test_split], y[test_split:]
    X_train,    X_cal  = X[:cal_split],  X[cal_split:test_split]
    y_train,    y_cal  = y[:cal_split],  y[cal_split:test_split]

    logger.info(
        "Split → train: %d | calibration: %d | test: %d",
        len(X_train), len(X_cal), len(X_test),
    )

    # ── Optuna hyperparameter search ─────────────────────────────────────────
    logger.info("Optuna search: %d trials × %d CV folds...", n_trials, CV_FOLDS)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _make_objective(X_trainval, y_trainval),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, "objective": "binary:logistic",
                   "random_state": 42, "n_jobs": -1, "verbosity": 0}
    logger.info("Best CV AUC: %.4f", study.best_value)
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
        "optuna_best_cv_auc":    study.best_value,
        "train_samples":         len(X_train),
        "calibration_samples":   len(X_cal),
        "test_samples":          len(X_test),
        "n_features":            len(feature_names),
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
    parser = argparse.ArgumentParser(description="Train NBA game outcome XGBoost model")
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
        print(f"\nModel → {MODEL_PATH}")
        print(f"Metrics → {METRICS_PATH}")
