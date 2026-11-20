"""Train the XGBoost risk model and save SHAP-based explanations."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from generate_data import DATA_DIR, FEATURE_COLUMNS
except ImportError:
    from src.generate_data import DATA_DIR, FEATURE_COLUMNS

REASON_TEXT = {
    "value_trend": lambda value: (
        f"Order value {'down' if value < 0 else 'up'} {abs(value) * 100:.0f}% "
        "vs prior 3 months"
    ),
    "count_trend": lambda value: (
        f"Order count {'down' if value < 0 else 'up'} {abs(value) * 100:.0f}% "
        "vs prior 3 months"
    ),
    "visit_trend": lambda value: (
        f"Clinic visits {'down' if value < 0 else 'up'} {abs(value) * 100:.0f}% "
        "vs prior 3 months"
    ),
    "days_since_last_order": lambda value: (
        f"Last order was about {value:.0f} days ago"
    ),
    "vaccines_share_change": lambda value: (
        f"Vaccines share {'down' if value < 0 else 'up'} "
        f"{abs(value) * 100:.0f} percentage points"
    ),
    "diagnostics_share_change": lambda value: (
        f"Diagnostics share {'down' if value < 0 else 'up'} "
        f"{abs(value) * 100:.0f} percentage points"
    ),
    "nutrition_share_change": lambda value: (
        f"Nutrition share {'down' if value < 0 else 'up'} "
        f"{abs(value) * 100:.0f} percentage points"
    ),
    "size_ordinal": lambda value: (
        f"{['Small', 'Medium', 'Large'][min(max(int(value), 0), 2)]} clinic size"
    ),
}


def classification_metrics(actual: pd.Series, predicted: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    """Return precision, recall and optional ROC-AUC."""
    result = {
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 3),
    }
    if scores is not None:
        result["roc_auc"] = round(float(roc_auc_score(actual, scores)), 3)
    return result


def build_reasons(latest: pd.DataFrame, shap_values: np.ndarray) -> dict[str, list[str]]:
    """Turn the top three SHAP contributions per clinic into plain English."""
    reasons: dict[str, list[str]] = {}
    for row_index, clinic_id in enumerate(latest["clinic_id"]):
        ranked = np.argsort(np.abs(shap_values[row_index]))[::-1][:3]
        clinic_reasons = []
        for feature_index in ranked:
            name = FEATURE_COLUMNS[feature_index]
            value = float(latest.iloc[row_index][name])
            clinic_reasons.append(REASON_TEXT[name](value))
        reasons[str(clinic_id)] = clinic_reasons
    return reasons


def main() -> None:
    """Run training, evaluation, persistence and explanation generation."""
    snapshots = pd.read_csv(DATA_DIR / "snapshots.csv")
    train = snapshots[snapshots["as_of_month"] == 8]
    test = snapshots[snapshots["as_of_month"] == 9]
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[FEATURE_COLUMNS], train["reduced_orders_next_90d"])

    scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (scores >= 0.5).astype(int)
    baseline = (test["value_trend"] <= -0.15).astype(int).to_numpy()
    metrics = {
        "xgboost": classification_metrics(
            test["reduced_orders_next_90d"], predictions, scores
        ),
        "rule_baseline": classification_metrics(
            test["reduced_orders_next_90d"], baseline
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positive_rate": round(
            float(test["reduced_orders_next_90d"].mean()), 3
        ),
    }

    latest = snapshots[snapshots["as_of_month"] == 9][
        ["clinic_id"] + FEATURE_COLUMNS
    ].copy()
    latest_scores = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["risk_score"] = np.round(latest_scores, 4)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest[FEATURE_COLUMNS])
    reasons = build_reasons(latest, shap_values)

    joblib.dump(model, DATA_DIR / "model.joblib")
    latest.to_csv(DATA_DIR / "latest_scores.csv", index=False)
    (DATA_DIR / "risk_reasons.json").write_text(
        json.dumps(reasons, indent=2), encoding="utf-8"
    )
    (DATA_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Model metrics (time-aware split):")
    print(json.dumps(metrics["xgboost"], indent=2))
    print("Rule baseline (order value down >15% over 3 months):")
    print(json.dumps(metrics["rule_baseline"], indent=2))
    print(f"Train rows: {metrics['train_rows']} | Test rows: {metrics['test_rows']}")
    print(f"Reasons saved for {len(reasons)} clinics")
    print(f"Model: {DATA_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
