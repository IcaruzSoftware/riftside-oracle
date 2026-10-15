"""
SHAP value generation and export.

Provides:
  - Per-prediction SHAP values written to Supabase (for the public subpage)
  - Global feature importance beeswarm plot saved as an image
  - Summary stats JSON for the website subpage

All feature names must be human-readable — they are displayed publicly.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SHAP_PLOT_PATH = PROCESSED_DIR / "shap_beeswarm.png"
SHAP_SUMMARY_PATH = PROCESSED_DIR / "shap_summary.json"


# ---------------------------------------------------------------------------
# Per-prediction SHAP (called from predict.py)
# ---------------------------------------------------------------------------

def export_shap_for_prediction(
    prediction_id: str,
    model,
    feature_row: pd.DataFrame,
    supabase,
) -> None:
    """
    Compute SHAP values for a single prediction and write them to Supabase.

    Each feature gets its own row in the shap_values table so the frontend
    can render a ranked waterfall chart per game.

    Args:
        prediction_id:  UUID of the stored prediction row
        model:          trained XGBClassifier
        feature_row:    single-row DataFrame with the game's feature vector
        supabase:       authenticated Supabase client
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_row)

        # shap_values shape: (1, n_features) for binary XGBoost
        values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        records = [
            {
                "prediction_id": prediction_id,
                "feature_name": feature_name,
                "shap_value": round(float(shap_val), 6),
                "feature_value": round(float(feature_val), 6)
                if not np.isnan(float(feature_val))
                else None,
            }
            for feature_name, shap_val, feature_val in zip(
                feature_row.columns, values, feature_row.iloc[0].values
            )
        ]

        supabase.table("shap_values").insert(records).execute()
        logger.info("Stored %d SHAP values for prediction %s", len(records), prediction_id)

    except Exception as exc:
        logger.error("SHAP export failed for prediction %s: %s", prediction_id, exc)


# ---------------------------------------------------------------------------
# Global feature importance (run after training or on demand)
# ---------------------------------------------------------------------------

def generate_global_shap_plots(model, X_sample: pd.DataFrame) -> None:
    """
    Generate and save a global SHAP beeswarm plot for model explainability.

    Uses a sample of the training data to compute SHAP values across the dataset.
    The resulting plot is saved to data/processed/shap_beeswarm.png for use
    on the website's model explainability subpage.

    Args:
        model:    trained XGBClassifier
        X_sample: representative sample of the feature matrix (e.g. last 2 seasons)
    """
    logger.info("Computing global SHAP values on %d samples...", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("Feature Impact on NBA Game Outcome Predictions", fontsize=14, pad=16)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Beeswarm plot saved → %s", SHAP_PLOT_PATH)


def generate_shap_summary_stats(model, X_sample: pd.DataFrame) -> dict:
    """
    Compute and save a JSON summary of global SHAP statistics.

    Includes mean absolute SHAP value per feature, ranked by importance.
    This JSON is intended to be served directly to the website subpage.

    Args:
        model:    trained XGBClassifier
        X_sample: representative sample of the feature matrix

    Returns:
        dict with feature importance summary, also written to data/processed/shap_summary.json
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        [
            {"feature_name": name, "mean_abs_shap": round(float(val), 6)}
            for name, val in zip(X_sample.columns, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    summary = {
        "n_samples_used": len(X_sample),
        "n_features": len(X_sample.columns),
        "feature_importance": feature_importance,
    }

    with open(SHAP_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("SHAP summary saved → %s", SHAP_SUMMARY_PATH)

    return summary
