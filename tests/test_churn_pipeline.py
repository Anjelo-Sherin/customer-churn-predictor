"""
Automated pytest suite for Day 03: Customer Churn Predictor with Explainable AI Dashboard.

Verifies:
1. Dataset schema, 11 whitespace `TotalCharges` records (`tenure == 0`), and missingness indicator
2. Strict leak-free train/val/test splitting and `ColumnTransformer` preprocessing
3. Model training, PR-AUC > 0.60, ROC-AUC > 0.80, and cost-sensitive threshold optimization
4. Per-customer TreeSHAP local explanation + plain-English driver generation
5. Live What-If counterfactual simulation and ROI-ranked retention action engine
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.action_engine import recommend_retention_actions, simulate_what_if
from src.data_pipeline import (
    build_preprocessor,
    clean_and_engineer_features,
    ensure_telco_dataset,
    inspect_data_quality,
    split_train_val_test,
)
from src.explainer import ChurnExplainer
from src.model_trainer import (
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_METRICS_PATH,
    optimize_cost_threshold,
    train_and_evaluate_all,
)


def test_data_quality_and_whitespace_total_charges() -> None:
    csv_path = ensure_telco_dataset()
    raw_df = pd.read_csv(csv_path)
    audit = inspect_data_quality(raw_df)

    assert audit["n_rows"] == 7043
    assert audit["n_columns"] == 21
    assert audit["blank_total_charges_count"] == 11
    assert audit["blank_total_charges_all_tenure_zero"] is True
    assert audit["missing_target_count"] == 0

    cleaned = clean_and_engineer_features(raw_df)
    assert int(cleaned["TotalCharges_WasMissing"].sum()) == 11
    assert not cleaned["TotalCharges"].isna().any()
    assert set(cleaned["ChurnBinary"].unique()) == {0, 1}


def test_strict_split_and_preprocessor_no_leakage() -> None:
    raw_df = pd.read_csv(ensure_telco_dataset())
    cleaned = clean_and_engineer_features(raw_df)
    train_df, val_df, test_df = split_train_val_test(cleaned, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)
    # Stratified split preserves churn rate within 0.5% across splits
    assert abs(train_df["ChurnBinary"].mean() - test_df["ChurnBinary"].mean()) < 0.005

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    assert X_train.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_end_to_end_training_shap_and_what_if() -> None:
    report = train_and_evaluate_all()
    assert DEFAULT_ARTIFACTS_PATH.exists()
    assert DEFAULT_METRICS_PATH.exists()

    xgb_metrics = report["models"]["XGBoost (Cost-Aware Regularized)"]["test_metrics_optimal"]
    lr_metrics = report["models"]["Logistic Regression (L2 Balanced)"]["test_metrics_optimal"]

    assert xgb_metrics["pr_auc"] >= 0.62
    assert xgb_metrics["roc_auc"] >= 0.82
    assert lr_metrics["pr_auc"] >= 0.60
    assert xgb_metrics["net_savings_vs_do_nothing"] > 0

    import joblib

    bundle = joblib.load(DEFAULT_ARTIFACTS_PATH)
    catalog = bundle["scored_catalog"]
    model = bundle["models"][bundle["champion_name"]]
    preprocessor = bundle["preprocessor"]
    explainer = ChurnExplainer(
        model=model,
        preprocessor=preprocessor,
        feature_names=bundle["feature_names"],
        background_matrix=bundle["background_matrix"],
    )

    # Pick a high-risk month-to-month customer and test SHAP + What-If + Action Engine
    high_risk_cust = catalog[
        (catalog["Contract"] == "Month-to-month") & (catalog["TechSupport"] == "No")
    ].sort_values("xgb_churn_prob", ascending=False).iloc[[0]]

    exp = explainer.explain_customer(high_risk_cust)
    assert exp["churn_probability"] > 0.60
    assert len(exp["plain_summary"]) > 10
    assert len(exp["top_risk_drivers"]) >= 1

    # Simulate upgrading to a Two-year contract + Tech Support + Online Security
    sim = simulate_what_if(
        original_row=high_risk_cust.iloc[0],
        overrides={
            "Contract": "Two year",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
        },
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
    )
    assert sim["new_probability"] < sim["original_probability"]
    assert sim["relative_reduction_pct"] > 20.0

    actions = recommend_retention_actions(
        customer_row=high_risk_cust.iloc[0],
        model=model,
        preprocessor=preprocessor,
    )
    assert len(actions["all_recommendations"]) >= 2
    assert actions["primary_recommendation"]["post_action_probability"] < actions["baseline_probability"]
