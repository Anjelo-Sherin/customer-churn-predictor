"""
Retention Action & What-If Counterfactual Engine (`src/action_engine.py`).

Given any customer's profile, current predicted churn probability, and SHAP
drivers, this engine:
1. Simulates counterfactual interventions (e.g. migrating Month-to-month to
   a 1-Year contract with a loyalty discount, adding a free Tech Support +
   Online Security bundle, enrolling in Auto-Pay with $5 bill credit).
2. Re-scores the modified customer profile through the live cleaning, feature
   engineering, and ML pipeline to compute exact post-intervention churn risk.
3. Ranks interventions by Expected Net Retention Value ($):
   Expected Net Value = (Risk_Reduction * Customer_LTV) - Intervention_Cost
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import clean_and_engineer_features


def classify_risk_tier(probability: float, optimal_threshold: float = 0.35) -> Dict[str, str]:
    """Classifies a churn probability into an actionable business tier."""
    if probability >= max(0.65, optimal_threshold + 0.20):
        return {
            "tier": "Critical Risk",
            "badge_color": "#D81B60",
            "urgency": "Immediate 24h Outbound Call + Contract Lock-In Offer",
        }
    if probability >= optimal_threshold:
        return {
            "tier": "High Risk",
            "badge_color": "#F57C00",
            "urgency": "Proactive Retention Campaign within 7 Days",
        }
    if probability >= optimal_threshold * 0.65:
        return {
            "tier": "Moderate Watch",
            "badge_color": "#FBC02D",
            "urgency": "Automated In-App Nudge & Value Check-In",
        }
    return {
        "tier": "Low Risk (Healthy)",
        "badge_color": "#2E7D32",
        "urgency": "No Discount Needed — Eligible for Upsell",
    }


def score_single_profile(
    raw_customer_dict: Dict[str, Any],
    model: Any,
    preprocessor: Any,
) -> Tuple[float, pd.DataFrame]:
    """
    Takes a raw dictionary of customer attributes, runs `clean_and_engineer_features`
    so derived features (`TotalServicesCount`, `HasSupportBundle`, `ChargeToTenureRatio`, etc.)
    update dynamically, and returns `(churn_probability, cleaned_single_row_df)`.
    """
    single_df = pd.DataFrame([raw_customer_dict])
    if "Churn" not in single_df.columns:
        single_df["Churn"] = "No"
    # Recompute TotalCharges proportionally if tenure/monthly changed in what-if simulation
    cleaned = clean_and_engineer_features(single_df)
    X_trans = preprocessor.transform(cleaned)
    prob = float(model.predict_proba(X_trans)[0, 1])
    return prob, cleaned


def simulate_what_if(
    original_row: pd.Series,
    overrides: Dict[str, Any],
    model: Any,
    preprocessor: Any,
    explainer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Applies user-selected What-If overrides (`Contract`, `MonthlyCharges`,
    `TechSupport`, `OnlineSecurity`, `PaymentMethod`, `InternetService`, `tenure`)
    and computes before/after churn risk and SHAP drivers.
    """
    base_dict = original_row.to_dict()
    base_prob, base_cleaned = score_single_profile(base_dict, model, preprocessor)

    modified_dict = dict(base_dict)
    modified_dict.update(overrides)

    # Keep TotalCharges consistent when MonthlyCharges or tenure is adjusted in What-If
    new_tenure = int(modified_dict.get("tenure", base_dict.get("tenure", 1)))
    new_monthly = float(modified_dict.get("MonthlyCharges", base_dict.get("MonthlyCharges", 50.0)))
    if (
        "MonthlyCharges" in overrides
        or "tenure" in overrides
    ) and "TotalCharges" not in overrides:
        modified_dict["TotalCharges"] = f"{new_tenure * new_monthly:.2f}" if new_tenure > 0 else " "

    new_prob, new_cleaned = score_single_profile(modified_dict, model, preprocessor)

    result: Dict[str, Any] = {
        "original_probability": base_prob,
        "new_probability": new_prob,
        "absolute_delta": new_prob - base_prob,
        "relative_reduction_pct": (
            ((base_prob - new_prob) / base_prob) * 100.0 if base_prob > 0 else 0.0
        ),
        "modified_profile": new_cleaned.iloc[0].to_dict(),
    }

    if explainer is not None:
        result["new_explanation"] = explainer.explain_customer(new_cleaned)

    return result


def recommend_retention_actions(
    customer_row: pd.Series,
    model: Any,
    preprocessor: Any,
    customer_ltv: float = 500.0,
    optimal_threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Evaluates a catalog of targeted retention playbooks against the customer's
    live profile using counterfactual model scoring, and returns the top actions
    ranked by Expected Net Savings ($).
    """
    base_dict = customer_row.to_dict()
    base_prob, _ = score_single_profile(base_dict, model, preprocessor)
    risk_meta = classify_risk_tier(base_prob, optimal_threshold=optimal_threshold)

    candidate_playbooks: List[Dict[str, Any]] = []

    # Playbook 1: Contract Upgrade (1-Year Loyalty Lock-In + 10% Bill Credit)
    if base_dict.get("Contract") == "Month-to-month":
        discounted_monthly = round(max(18.25, float(base_dict.get("MonthlyCharges", 65.0)) * 0.90), 2)
        candidate_playbooks.append(
            {
                "action_id": "contract_1yr_discount",
                "title": "1-Year Contract Lock-In + 10% Loyalty Credit",
                "description": (
                    f"Migrate from Month-to-month to a 1-Year agreement and lower monthly bill "
                    f"from ${float(base_dict.get('MonthlyCharges', 0)):.2f} to ${discounted_monthly:.2f}/mo."
                ),
                "overrides": {
                    "Contract": "One year",
                    "MonthlyCharges": discounted_monthly,
                },
                "offer_cost": 65.0,
                "target_driver": "Month-to-month contract & price sensitivity",
            }
        )

    # Playbook 2: Complimentary Tech Support + Online Security Bundle (6 Months)
    if (
        base_dict.get("InternetService") in ("Fiber optic", "DSL")
        and (
            base_dict.get("TechSupport") == "No"
            or base_dict.get("OnlineSecurity") == "No"
        )
    ):
        candidate_playbooks.append(
            {
                "action_id": "free_support_security_bundle",
                "title": "Complimentary Tech Support & Online Security Bundle",
                "description": (
                    "Activate Priority Tech Support and Online Security at $0 add-on cost for 6 months "
                    "to eliminate service friction and increase ecosystem lock-in."
                ),
                "overrides": {
                    "TechSupport": "Yes",
                    "OnlineSecurity": "Yes",
                },
                "offer_cost": 35.0,
                "target_driver": "No tech support / no online security",
            }
        )

    # Playbook 3: Auto-Pay Enrollment Incentive ($5/mo Bill Reduction)
    if "automatic" not in str(base_dict.get("PaymentMethod", "")).lower():
        autopay_monthly = round(max(18.25, float(base_dict.get("MonthlyCharges", 65.0)) - 5.0), 2)
        candidate_playbooks.append(
            {
                "action_id": "autopay_switch_credit",
                "title": "Auto-Pay Enrollment + $5/mo Bill Credit",
                "description": (
                    "Switch from manual check/electronic check to Credit Card (automatic) "
                    "with a $5/month autopay discount to remove monthly payment friction."
                ),
                "overrides": {
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": autopay_monthly,
                },
                "offer_cost": 30.0,
                "target_driver": "Electronic/manual check friction",
            }
        )

    # Playbook 4: Comprehensive Executive Save Package (1-Year + Support Bundle + Auto-Pay + 12% Credit)
    if base_prob >= optimal_threshold:
        save_monthly = round(max(18.25, float(base_dict.get("MonthlyCharges", 65.0)) * 0.88), 2)
        save_overrides: Dict[str, Any] = {
            "Contract": "One year" if base_dict.get("Contract") == "Month-to-month" else base_dict.get("Contract"),
            "MonthlyCharges": save_monthly,
            "PaymentMethod": "Credit card (automatic)",
        }
        if base_dict.get("InternetService") != "No":
            save_overrides["TechSupport"] = "Yes"
            save_overrides["OnlineSecurity"] = "Yes"
        candidate_playbooks.append(
            {
                "action_id": "executive_save_bundle",
                "title": "VIP Retention Save Package (1-Yr + Support Bundle + Auto-Pay)",
                "description": (
                    f"Full-stack retention bundle combining 1-Year contract stability, complimentary "
                    f"Tech Support + Online Security, Auto-Pay, and a 12% bill reduction (${save_monthly:.2f}/mo)."
                ),
                "overrides": save_overrides,
                "offer_cost": 85.0,
                "target_driver": "Multi-factor high churn risk",
            }
        )

    evaluated_actions: List[Dict[str, Any]] = []
    for pb in candidate_playbooks:
        sim = simulate_what_if(
            original_row=customer_row,
            overrides=pb["overrides"],
            model=model,
            preprocessor=preprocessor,
        )
        risk_drop = max(0.0, base_prob - sim["new_probability"])
        expected_saved_ltv = risk_drop * customer_ltv
        net_roi = expected_saved_ltv - float(pb["offer_cost"])
        evaluated_actions.append(
            {
                **pb,
                "post_action_probability": sim["new_probability"],
                "absolute_risk_reduction": risk_drop,
                "relative_risk_reduction_pct": sim["relative_reduction_pct"],
                "expected_gross_savings": round(expected_saved_ltv, 2),
                "expected_net_roi": round(net_roi, 2),
            }
        )

    evaluated_actions.sort(
        key=lambda x: (x["expected_net_roi"], x["absolute_risk_reduction"]),
        reverse=True,
    )

    if evaluated_actions and base_prob >= optimal_threshold:
        primary = evaluated_actions[0]
    elif evaluated_actions:
        primary = evaluated_actions[0]
    else:
        primary = {
            "action_id": "maintain_relationship",
            "title": "No Discount Needed — Standard Loyalty Check-In",
            "description": "Customer already has strong contract and support protections. Avoid unnecessary margin dilution.",
            "overrides": {},
            "offer_cost": 0.0,
            "target_driver": "Low baseline churn risk",
            "post_action_probability": base_prob,
            "absolute_risk_reduction": 0.0,
            "relative_risk_reduction_pct": 0.0,
            "expected_gross_savings": 0.0,
            "expected_net_roi": 0.0,
        }

    return {
        "customerID": str(customer_row.get("customerID", "UNKNOWN")),
        "baseline_probability": base_prob,
        "risk_tier": risk_meta["tier"],
        "badge_color": risk_meta["badge_color"],
        "urgency": risk_meta["urgency"],
        "primary_recommendation": primary,
        "all_recommendations": evaluated_actions,
    }
