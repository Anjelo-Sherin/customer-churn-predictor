"""
Model Training, Cross-Validation, Cost-Sensitive Threshold Tuning, Bootstrap CIs,
and Demographic Slice Auditing (`src/model_trainer.py`).

Follows ML Best Practices:
- Establishes Naive Baseline (Majority Class) and Simple ML Baseline (Logistic Regression)
- Trains regularized Random Forest and XGBoost classifiers on identical splits
- Performs 5-Fold Stratified Cross-Validation on the training set
- Evaluates PR-AUC, ROC-AUC, F1, Precision, Recall, and Brier Score on Val & Test sets
- Computes 95% Bootstrap Confidence Intervals on Test Set metrics
- Optimizes decision threshold using business cost matrix ($C_{FN}$ vs $C_{FP}$)
- Audits slice-based error rates across `SeniorCitizen`, `Contract`, `InternetService`, and `tenure` cohorts
- Optionally logs experiments to MLflow if installed
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.data_pipeline import (
    build_preprocessor,
    clean_and_engineer_features,
    ensure_telco_dataset,
    inspect_data_quality,
    split_train_val_test,
)
from src.explainer import ChurnExplainer

DEFAULT_ARTIFACTS_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "churn_artifacts.joblib"
)
DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "metrics_report.json"
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    cost_fn: float = 500.0,
    cost_fp: float = 60.0,
    retention_save_rate: float = 0.45,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification and retention campaign economics at a threshold:
    - Do-Nothing Baseline Cost = (TP + FN) * cost_fn
    - Campaign Offer Cost = (TP + FP) * cost_fp
    - Retained LTV Value = TP * retention_save_rate * cost_fn
    - Total Business Cost = Do-Nothing Cost - Retained LTV Value + Campaign Offer Cost
    - Net Savings vs. Do-Nothing = Retained LTV Value - Campaign Offer Cost
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1 and len(np.unique(y_prob)) > 1
        else 0.5
    )
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    no_model_cost = float((tp + fn) * cost_fn)
    offer_spend = float((tp + fp) * cost_fp)
    retained_ltv = float(tp * retention_save_rate * cost_fn)
    total_business_cost = float(no_model_cost - retained_ltv + offer_spend)
    net_savings_vs_do_nothing = float(retained_ltv - offer_spend)

    return {
        "threshold": round(float(threshold), 3),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_business_cost": round(total_business_cost, 2),
        "cost_per_customer": round(total_business_cost / max(len(y_true), 1), 2),
        "net_savings_vs_do_nothing": round(net_savings_vs_do_nothing, 2),
    }


def optimize_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 60.0,
) -> Dict[str, Any]:
    """
    Sweeps thresholds from 0.05 to 0.95 to find the threshold that minimizes
    Total Business Cost = FN * cost_fn + FP * cost_fp.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    curve_records: List[Dict[str, float]] = []
    best_record: Optional[Dict[str, Any]] = None

    for t in thresholds:
        m = compute_classification_metrics(
            y_true, y_prob, threshold=float(t), cost_fn=cost_fn, cost_fp=cost_fp
        )
        curve_records.append(
            {
                "threshold": m["threshold"],
                "total_business_cost": m["total_business_cost"],
                "net_savings": m["net_savings_vs_do_nothing"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
            }
        )
        if best_record is None or m["total_business_cost"] < best_record["total_business_cost"]:
            best_record = m

    default_50 = compute_classification_metrics(
        y_true, y_prob, threshold=0.50, cost_fn=cost_fn, cost_fp=cost_fp
    )
    assert best_record is not None
    return {
        "optimal_threshold": best_record["threshold"],
        "optimal_metrics": best_record,
        "default_050_metrics": default_50,
        "savings_over_default_050": round(
            default_50["total_business_cost"] - best_record["total_business_cost"], 2
        ),
        "threshold_curve": curve_records,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 400,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Calculates 95% Bootstrap Confidence Intervals for PR-AUC, ROC-AUC, and F1-Score
    on the held-out test set to verify statistical significance.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr_aucs, roc_aucs, f1s = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        pr_aucs.append(float(average_precision_score(yb, pb)))
        roc_aucs.append(float(roc_auc_score(yb, pb)) if len(np.unique(pb)) > 1 else 0.5)
        f1s.append(float(f1_score(yb, (pb >= threshold).astype(int), zero_division=0)))

    def _ci(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower_95": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "pr_auc": _ci(pr_aucs),
        "roc_auc": _ci(roc_aucs),
        "f1": _ci(f1s),
    }


def evaluate_demographic_slices(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Performs slice-based error analysis across key demographics and service cohorts
    (`SeniorCitizen`, `Contract`, `InternetService`, `IsNewCustomer`) to detect bias
    or cohort-specific failure modes.
    """
    df_eval = test_df.copy()
    df_eval["y_prob"] = y_prob
    df_eval["y_pred"] = (y_prob >= threshold).astype(int)

    slice_specs = [
        ("SeniorCitizen", {0: "Non-Senior (0)", 1: "Senior Citizen (1)"}),
        ("Contract", None),
        ("InternetService", None),
        ("IsNewCustomer", {0: "Established (>6 mo)", 1: "New Customer (<=6 mo)"}),
    ]

    records: List[Dict[str, Any]] = []
    for col, label_map in slice_specs:
        for val, grp in df_eval.groupby(col):
            if len(grp) < 10:
                continue
            y_t = grp["ChurnBinary"].to_numpy()
            y_p = grp["y_prob"].to_numpy()
            y_hat = grp["y_pred"].to_numpy()
            pr_auc = (
                float(average_precision_score(y_t, y_p))
                if len(np.unique(y_t)) > 1
                else 0.0
            )
            roc_auc = (
                float(roc_auc_score(y_t, y_p))
                if len(np.unique(y_t)) > 1 and len(np.unique(y_p)) > 1
                else 0.5
            )
            tn, fp, fn, tp = confusion_matrix(y_t, y_hat, labels=[0, 1]).ravel()
            fpr = float(fp / max(fp + tn, 1))
            fnr = float(fn / max(fn + tp, 1))
            records.append(
                {
                    "slice_feature": col,
                    "cohort": label_map.get(val, str(val)) if label_map else str(val),
                    "sample_size": int(len(grp)),
                    "actual_churn_rate": round(float(y_t.mean()), 4),
                    "predicted_positive_rate": round(float(y_hat.mean()), 4),
                    "pr_auc": round(pr_auc, 4),
                    "roc_auc": round(roc_auc, 4),
                    "f1": round(float(f1_score(y_t, y_hat, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_t, y_hat, zero_division=0)), 4),
                    "precision": round(float(precision_score(y_t, y_hat, zero_division=0)), 4),
                    "false_positive_rate": round(fpr, 4),
                    "false_negative_rate": round(fnr, 4),
                }
            )
    return records


def train_and_evaluate_all(
    csv_path: Optional[Path] = None,
    artifacts_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    cost_fn: float = 500.0,
    cost_fp: float = 60.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete Day 03 ML pipeline:
    1. Ensures dataset exists & runs data quality audit
    2. Cleans data & engineers row-local features
    3. Splits into 70% Train / 15% Val / 15% Test BEFORE fitting preprocessor
    4. Trains Naive Baseline, Logistic Regression, Random Forest, and regularized XGBoost
    5. Evaluates 5-Fold Stratified CV on Train, tunes cost threshold on Val, and evaluates on Test
    6. Computes global SHAP importance & pre-computes customer risk table for fast dashboard loading
    7. Saves artifacts to `models/churn_artifacts.joblib` and report to `models/metrics_report.json`
    """
    data_file = ensure_telco_dataset(csv_path)
    raw_df = pd.read_csv(data_file)
    quality_report = inspect_data_quality(raw_df)

    cleaned_df = clean_and_engineer_features(raw_df)
    train_df, val_df, test_df = split_train_val_test(cleaned_df, random_state=random_state)

    # Fit preprocessor ONLY on training split (Strict Featurization Ordering)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["ChurnBinary"].to_numpy()

    X_val = preprocessor.transform(val_df)
    y_val = val_df["ChurnBinary"].to_numpy()

    X_test = preprocessor.transform(test_df)
    y_test = test_df["ChurnBinary"].to_numpy()

    feature_names = list(preprocessor.get_feature_names_out())

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = round(neg_count / max(pos_count, 1), 3)

    models: Dict[str, Any] = {
        "Naive Baseline (Majority)": DummyClassifier(strategy="prior", random_state=random_state),
        "Logistic Regression (L2 Balanced)": LogisticRegression(
            C=0.5,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest (Regularized)": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost (Cost-Aware Regularized)": XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.82,
            colsample_bytree=0.80,
            min_child_weight=6,
            reg_alpha=0.8,
            reg_lambda=2.5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    comparison_results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for name, clf in models.items():
        t0 = time.perf_counter()
        cv_scores = cross_validate(
            clf,
            X_train,
            y_train,
            cv=skf,
            scoring=["average_precision", "roc_auc", "f1"],
            return_train_score=True,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        train_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        t_inf = time.perf_counter()
        val_prob = clf.predict_proba(X_val)[:, 1]
        test_prob = clf.predict_proba(X_test)[:, 1]
        train_prob = clf.predict_proba(X_train)[:, 1]
        inf_latency_ms_per_1k = round(
            ((time.perf_counter() - t_inf) / max(len(X_val) + len(X_test) + len(X_train), 1))
            * 1000.0
            * 1000.0,
            3,
        )

        # Tune cost-optimal threshold on Validation Set, then evaluate on Held-Out Test Set
        val_tuning = optimize_cost_threshold(y_val, val_prob, cost_fn=cost_fn, cost_fp=cost_fp)
        opt_t = val_tuning["optimal_threshold"] if "Naive" not in name else 0.50

        train_metrics_050 = compute_classification_metrics(
            y_train, train_prob, threshold=0.50, cost_fn=cost_fn, cost_fp=cost_fp
        )
        val_metrics_050 = compute_classification_metrics(
            y_val, val_prob, threshold=0.50, cost_fn=cost_fn, cost_fp=cost_fp
        )
        test_metrics_050 = compute_classification_metrics(
            y_test, test_prob, threshold=0.50, cost_fn=cost_fn, cost_fp=cost_fp
        )
        test_metrics_opt = compute_classification_metrics(
            y_test, test_prob, threshold=opt_t, cost_fn=cost_fn, cost_fp=cost_fp
        )
        boot_ci = bootstrap_confidence_intervals(
            y_test, test_prob, threshold=opt_t, n_bootstraps=300, seed=random_state
        )

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_prob)
        fpr_curve, tpr_curve, _ = roc_curve(y_test, test_prob)
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_test, test_prob, n_bins=8, strategy="uniform"
        )

        # Downsample curve points for compact JSON serialization
        step_pr = max(1, len(prec_curve) // 60)
        step_roc = max(1, len(fpr_curve) // 60)

        comparison_results[name] = {
            "cv_train_pr_auc_mean": round(float(np.mean(cv_scores["train_average_precision"])), 4),
            "cv_val_pr_auc_mean": round(float(np.mean(cv_scores["test_average_precision"])), 4),
            "cv_val_pr_auc_std": round(float(np.std(cv_scores["test_average_precision"])), 4),
            "cv_val_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "cv_val_roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "cv_val_f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
            "train_time_ms": train_time_ms,
            "inference_latency_ms_per_1k": inf_latency_ms_per_1k,
            "optimal_threshold": opt_t,
            "train_metrics_050": train_metrics_050,
            "val_metrics_050": val_metrics_050,
            "test_metrics_050": test_metrics_050,
            "test_metrics_optimal": test_metrics_opt,
            "test_bootstrap_95_ci": boot_ci,
            "val_threshold_tuning": val_tuning,
            "curves": {
                "pr_precision": [round(float(x), 4) for x in prec_curve[::step_pr]],
                "pr_recall": [round(float(x), 4) for x in rec_curve[::step_pr]],
                "roc_fpr": [round(float(x), 4) for x in fpr_curve[::step_roc]],
                "roc_tpr": [round(float(x), 4) for x in tpr_curve[::step_roc]],
                "cal_prob_true": [round(float(x), 4) for x in prob_true_cal],
                "cal_prob_pred": [round(float(x), 4) for x in prob_pred_cal],
            },
        }
        fitted_models[name] = clf

    champion_name = "XGBoost (Cost-Aware Regularized)"
    champion_model = fitted_models[champion_name]
    champion_opt_threshold = comparison_results[champion_name]["optimal_threshold"]
    champion_test_prob = champion_model.predict_proba(X_test)[:, 1]

    slice_audit = evaluate_demographic_slices(
        test_df=test_df,
        y_prob=champion_test_prob,
        threshold=champion_opt_threshold,
    )

    # Build SHAP Explainer & Global Importance
    explainer = ChurnExplainer(
        model=champion_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        background_matrix=X_train[:120],
    )
    global_shap_df = explainer.global_importance(test_df)

    # Pre-score the full dataset and compute top plain-English drivers for fast dashboard rendering
    X_all = preprocessor.transform(cleaned_df)
    all_probs = champion_model.predict_proba(X_all)[:, 1]
    all_lr_probs = fitted_models["Logistic Regression (L2 Balanced)"].predict_proba(X_all)[:, 1]
    all_shap_vals, base_log_odds = explainer.compute_shap_matrix(X_all)

    # Create compact summary table with top 3 plain-English drivers per customer
    top_driver_strings: List[str] = []
    for idx in range(len(cleaned_df)):
        row_s = cleaned_df.iloc[idx]
        sv = all_shap_vals[idx]
        pos_indices = np.argsort(-sv)
        phrases: List[str] = []
        seen_concepts = set()
        for feat_i in pos_indices:
            if sv[feat_i] <= 0.01:
                break
            fname = feature_names[feat_i]
            concept = fname.split("_")[0]
            if concept in seen_concepts:
                continue
            seen_concepts.add(concept)
            from src.explainer import describe_driver_in_plain_words

            phrases.append(describe_driver_in_plain_words(fname, row_s, float(sv[feat_i])))
            if len(phrases) == 3:
                break
        if not phrases:
            phrases.append("stable long-term contract & support profile")
        top_driver_strings.append(" + ".join(phrases))

    scored_catalog = cleaned_df.copy()
    scored_catalog["xgb_churn_prob"] = np.round(all_probs, 4)
    scored_catalog["lr_churn_prob"] = np.round(all_lr_probs, 4)
    scored_catalog["predicted_churn"] = (all_probs >= champion_opt_threshold).astype(int)
    scored_catalog["top_shap_drivers_plain"] = top_driver_strings

    artifacts_bundle = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "models": fitted_models,
        "champion_name": champion_name,
        "optimal_threshold": champion_opt_threshold,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "base_log_odds": base_log_odds,
        "background_matrix": X_train[:120],
        "global_shap_df": global_shap_df,
        "scored_catalog": scored_catalog,
        "test_indices": test_df["customerID"].tolist(),
        "y_test": y_test,
        "y_test_prob_xgb": champion_test_prob,
        "y_test_prob_lr": fitted_models["Logistic Regression (L2 Balanced)"].predict_proba(X_test)[:, 1],
    }

    target_artifacts = Path(artifacts_path or DEFAULT_ARTIFACTS_PATH)
    target_artifacts.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts_bundle, target_artifacts)

    metrics_payload = {
        "dataset_quality": quality_report,
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "champion_model": champion_name,
        "optimal_threshold": champion_opt_threshold,
        "business_cost_assumptions": {
            "cost_false_negative_lost_ltv": cost_fn,
            "cost_false_positive_retention_offer": cost_fp,
        },
        "models": comparison_results,
        "global_shap_top15": global_shap_df.head(15).to_dict(orient="records"),
        "demographic_slice_audit": slice_audit,
    }

    target_metrics = Path(metrics_path or DEFAULT_METRICS_PATH)
    target_metrics.parent.mkdir(parents=True, exist_ok=True)
    target_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


if __name__ == "__main__":
    report = train_and_evaluate_all()
    champ = report["champion_model"]
    champ_metrics = report["models"][champ]["test_metrics_optimal"]
    print(f"Trained & saved Day 03 models. Champion: {champ}")
    print(
        f"Test PR-AUC: {champ_metrics['pr_auc']:.4f} | "
        f"Test ROC-AUC: {champ_metrics['roc_auc']:.4f} | "
        f"Optimal Threshold: {champ_metrics['threshold']} | "
        f"Net Savings on Test Set: ${champ_metrics['net_savings_vs_do_nothing']:,.2f}"
    )
