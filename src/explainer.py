"""
SHAP Explainability Engine (`src/explainer.py`).

Computes exact TreeSHAP / LinearSHAP values for global and per-customer local
explanations, aggregates one-hot encoded columns back to intuitive business
concepts when needed, and translates top SHAP drivers into plain-English
reasons (e.g. "month-to-month contract + high monthly charge ($98.50) + no tech support").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap


HUMAN_FEATURE_LABELS: Dict[str, str] = {
    "tenure": "Customer Tenure",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Historical Charges",
    "TotalServicesCount": "Active Services Count",
    "ChargeToTenureRatio": "Charge-to-Tenure Ratio",
    "MonthlyOverHistoricalDelta": "Recent Bill Increase vs Avg",
    "Contract": "Contract Term",
    "SeniorCitizen": "Senior Citizen Status",
    "TotalCharges_WasMissing": "New First-Cycle Account",
    "HasSupportBundle": "Security + Tech Support Bundle",
    "IsAutoPay": "Automatic Payment Setup",
    "IsNewCustomer": "Early Tenure (<6 Months)",
}


def format_feature_name(raw_name: str) -> str:
    """Converts encoded feature names into readable business labels."""
    if raw_name in HUMAN_FEATURE_LABELS:
        return HUMAN_FEATURE_LABELS[raw_name]
    if "_" in raw_name:
        prefix, val = raw_name.split("_", 1)
        return f"{prefix}: {val}"
    return raw_name


def describe_driver_in_plain_words(
    raw_feature: str,
    customer_row: pd.Series,
    shap_val: float,
) -> str:
    """
    Translates a single encoded feature and its SHAP value for a specific customer
    into a clear, plain-English phrase suitable for business stakeholders.
    """
    direction = "increases" if shap_val > 0 else "reduces"
    if raw_feature == "Contract":
        c_val = customer_row.get("Contract", "Month-to-month")
        return f"{c_val.lower()} contract"
    if raw_feature == "tenure":
        t_val = int(customer_row.get("tenure", 0))
        if t_val <= 6:
            return f"short tenure ({t_val} mo)"
        if t_val >= 36:
            return f"loyal tenure ({t_val} mo)"
        return f"tenure of {t_val} months"
    if raw_feature == "MonthlyCharges":
        m_val = float(customer_row.get("MonthlyCharges", 0.0))
        level = "high" if m_val >= 75.0 else ("low" if m_val <= 35.0 else "moderate")
        return f"{level} monthly charge (${m_val:.2f}/mo)"
    if raw_feature == "TotalCharges":
        tot_val = float(customer_row.get("TotalCharges", 0.0))
        return f"total spend (${tot_val:,.0f})"
    if raw_feature == "ChargeToTenureRatio":
        m_val = float(customer_row.get("MonthlyCharges", 0.0))
        t_val = int(customer_row.get("tenure", 0))
        return f"high spend relative to tenure (${m_val:.0f}/mo at {t_val} mo)"
    if raw_feature == "IsNewCustomer":
        t_val = int(customer_row.get("tenure", 0))
        return f"onboarding risk window ({t_val} mo tenure)" if t_val <= 6 else f"past onboarding window ({t_val} mo)"
    if raw_feature == "HasSupportBundle":
        has_bundle = int(customer_row.get("HasSupportBundle", 0)) == 1
        return "full security + tech support bundle" if has_bundle else "missing security/tech support bundle"
    if raw_feature == "IsAutoPay":
        is_auto = int(customer_row.get("IsAutoPay", 0)) == 1
        return "automatic payment enrolled" if is_auto else "manual payment method"
    if raw_feature == "SeniorCitizen":
        return "senior citizen demographic" if int(customer_row.get("SeniorCitizen", 0)) == 1 else "non-senior demographic"
    if raw_feature == "TotalServicesCount":
        cnt = int(customer_row.get("TotalServicesCount", 0))
        return f"{cnt} active service add-ons"

    if "_" in raw_feature:
        col, category = raw_feature.split("_", 1)
        actual_val = str(customer_row.get(col, ""))
        if col == "InternetService":
            if actual_val == "Fiber optic":
                return "fiber optic internet plan"
            if actual_val == "DSL":
                return "DSL internet plan"
            return "no internet service"
        if col == "OnlineSecurity":
            return "no online security" if actual_val == "No" else f"online security ({actual_val.lower()})"
        if col == "TechSupport":
            return "no tech support" if actual_val == "No" else f"tech support ({actual_val.lower()})"
        if col == "PaymentMethod":
            return f"{actual_val.lower()} payment"
        if col == "PaperlessBilling":
            return "paperless billing" if actual_val == "Yes" else "paper billing"
        if col == "OnlineBackup":
            return "no online backup" if actual_val == "No" else "online backup active"
        if col == "DeviceProtection":
            return "no device protection" if actual_val == "No" else "device protection active"
        if actual_val == category:
            return f"{col}: {actual_val}"
        return f"{col} is {actual_val} (not {category})"

    return f"{format_feature_name(raw_feature)} ({direction} risk)"


class ChurnExplainer:
    """
    Wraps a fitted tree or linear classifier + preprocessor to provide fast
    global and per-customer SHAP explanations and plain-English summaries.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        feature_names: List[str],
        background_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = list(feature_names)
        if background_matrix is not None and len(background_matrix) > 150:
            bg = background_matrix[:150]
        else:
            bg = background_matrix

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.LinearExplainer(self.model, bg)

    def compute_shap_matrix(self, X_transformed: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns `(shap_values_array, base_value)` in log-odds / margin space."""
        raw_shap = self.explainer.shap_values(X_transformed)
        if isinstance(raw_shap, list):
            shap_vals = np.asarray(raw_shap[1], dtype=float)
        elif hasattr(raw_shap, "values"):
            shap_vals = np.asarray(raw_shap.values, dtype=float)
        else:
            shap_vals = np.asarray(raw_shap, dtype=float)

        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        expected = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(expected, (list, np.ndarray)):
            base_val = float(expected[-1])
        else:
            base_val = float(expected)

        return shap_vals, base_val

    def global_importance(self, df_cleaned: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP value across a dataframe and returns a sorted
        table of global feature importances.
        """
        X_trans = self.preprocessor.transform(df_cleaned)
        shap_vals, _ = self.compute_shap_matrix(X_trans)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)

        out = pd.DataFrame(
            {
                "feature": self.feature_names,
                "display_name": [format_feature_name(f) for f in self.feature_names],
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return out.reset_index(drop=True)

    def explain_customer(
        self,
        customer_df: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for a single customer (1-row DataFrame)
        and produces both structured waterfall drivers and a plain-English summary.
        """
        if len(customer_df) != 1:
            customer_df = customer_df.iloc[[0]]

        row = customer_df.iloc[0]
        X_trans = self.preprocessor.transform(customer_df)
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        shap_vals, base_val = self.compute_shap_matrix(X_trans)
        sv = shap_vals[0]

        drivers_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "display_name": [format_feature_name(f) for f in self.feature_names],
                "shap_value": sv,
                "abs_shap": np.abs(sv),
                "plain_phrase": [
                    describe_driver_in_plain_words(f, row, float(s))
                    for f, s in zip(self.feature_names, sv)
                ],
            }
        ).sort_values("abs_shap", ascending=False)

        # Deduplicate overlapping phrases (e.g. OnlineSecurity_No and OnlineSecurity_Yes)
        seen_phrases = set()
        unique_rows = []
        for _, d_row in drivers_df.iterrows():
            phrase = d_row["plain_phrase"]
            concept_key = d_row["feature"].split("_")[0]
            if concept_key not in seen_phrases:
                seen_phrases.add(concept_key)
                unique_rows.append(d_row)

        dedup_df = pd.DataFrame(unique_rows)

        positive_drivers = dedup_df[dedup_df["shap_value"] > 0.01].head(top_k)
        protective_drivers = dedup_df[dedup_df["shap_value"] < -0.01].head(top_k)

        if not positive_drivers.empty:
            top_phrases = positive_drivers["plain_phrase"].head(3).tolist()
            plain_summary = " + ".join(top_phrases)
        else:
            top_protective = protective_drivers["plain_phrase"].head(2).tolist()
            plain_summary = "Protected by " + " & ".join(top_protective)

        return {
            "customerID": str(row.get("customerID", "UNKNOWN")),
            "churn_probability": prob,
            "base_log_odds": base_val,
            "plain_summary": plain_summary,
            "top_risk_drivers": positive_drivers.to_dict(orient="records"),
            "top_protective_drivers": protective_drivers.to_dict(orient="records"),
            "all_drivers": dedup_df.head(12).to_dict(orient="records"),
        }
