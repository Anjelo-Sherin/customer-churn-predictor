"""
ChurnSight — Day 03: Customer Churn Predictor with Explainable AI Dashboard.

Interactive Streamlit application featuring:
1. Executive Risk Watchlist & Cohort Triage (with plain-English SHAP drivers + recommended action badges)
2. Per-Customer SHAP Deep-Dive (calibrated risk gauge, plain-words narrative, SHAP waterfall/bar chart)
3. Live What-If Simulator & ROI Action Recommender (interactive contract/price/support sliders that re-score live)
4. Model Arena, Cost-Sensitive Threshold Optimizer & Demographic Fairness/Slice Audit
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.action_engine import (
    classify_risk_tier,
    recommend_retention_actions,
    simulate_what_if,
)
from src.explainer import ChurnExplainer
from src.model_trainer import (
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_METRICS_PATH,
    compute_classification_metrics,
    optimize_cost_threshold,
    train_and_evaluate_all,
)

st.set_page_config(
    page_title="ChurnSight — Explainable AI Churn & Retention Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styled after the 30 Projects in 30 Days crimson/pink accent palette (#D81B60)
st.markdown(
    """
    <style>
    .main-hero {
        background: linear-gradient(135deg, #880E4F 0%, #D81B60 65%, #F06292 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 20px rgba(216, 27, 96, 0.18);
    }
    .main-hero h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .main-hero p {
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
        opacity: 0.94;
        color: #FCE4EC;
    }
    .driver-pill {
        background: #FCE4EC;
        border-left: 5px solid #D81B60;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        color: #4A0E2E;
        font-size: 0.98rem;
        margin: 0.5rem 0 1rem 0;
    }
    .action-card {
        background: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-left: 5px solid #2E7D32;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained XGBoost + Logistic Regression models & SHAP explainer...")
def load_artifacts_and_metrics() -> Tuple[Dict[str, Any], Dict[str, Any], ChurnExplainer]:
    if not DEFAULT_ARTIFACTS_PATH.exists() or not DEFAULT_METRICS_PATH.exists():
        train_and_evaluate_all()
    bundle = joblib.load(DEFAULT_ARTIFACTS_PATH)
    metrics = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
    explainer = ChurnExplainer(
        model=bundle["models"][bundle["champion_name"]],
        preprocessor=bundle["preprocessor"],
        feature_names=bundle["feature_names"],
        background_matrix=bundle["background_matrix"],
    )
    return bundle, metrics, explainer


bundle, metrics_report, shap_explainer = load_artifacts_and_metrics()
catalog_df: pd.DataFrame = bundle["scored_catalog"]
champion_model = bundle["models"][bundle["champion_name"]]
preprocessor = bundle["preprocessor"]
default_opt_threshold = float(bundle["optimal_threshold"])

# Sidebar Controls — Global Cost Matrix & Customer Quick-Select
with st.sidebar:
    st.markdown("### ⚙️ Business Cost & Threshold Controls")
    st.caption(
        "Tune the decision threshold dynamically based on your unit economics."
    )
    cost_fn = st.slider(
        "Lost Customer LTV Cost ($C_{FN}$)",
        min_value=150,
        max_value=1500,
        value=int(bundle.get("cost_fn", 500)),
        step=50,
        help="Cost of a False Negative: losing a churning customer without intervening.",
    )
    cost_fp = st.slider(
        "Retention Offer Cost ($C_{FP}$)",
        min_value=15,
        max_value=250,
        value=int(bundle.get("cost_fp", 60)),
        step=5,
        help="Cost of a False Positive: giving a retention discount/perk to a customer who would have stayed.",
    )

    live_tuning = optimize_cost_threshold(
        y_true=bundle["y_test"],
        y_prob=bundle["y_test_prob_xgb"],
        cost_fn=float(cost_fn),
        cost_fp=float(cost_fp),
    )
    active_threshold = float(live_tuning["optimal_threshold"])

    st.metric(
        "Cost-Optimal Churn Threshold",
        f"{active_threshold:.2f}",
        delta=f"{active_threshold - 0.50:+.2f} vs naive 0.50",
        delta_color="off",
    )
    st.metric(
        "Test Set Net Savings vs. No Model",
        f"${live_tuning['optimal_metrics']['net_savings_vs_do_nothing']:,.0f}",
        delta=f"+${live_tuning['savings_over_default_050']:,.0f} vs 0.50 cutoff",
    )

    st.divider()
    st.markdown("### 🔍 Customer Inspector Selection")
    risk_filter = st.selectbox(
        "Filter Customer Pool",
        [
            "High / Critical Risk First",
            "Borderline / Persuadable (35% - 65%)",
            "Low Risk / Loyal",
            "All Customers",
        ],
    )
    if risk_filter == "High / Critical Risk First":
        pool_df = catalog_df.sort_values("xgb_churn_prob", ascending=False)
    elif risk_filter == "Borderline / Persuadable (35% - 65%)":
        pool_df = catalog_df[
            (catalog_df["xgb_churn_prob"] >= 0.35)
            & (catalog_df["xgb_churn_prob"] <= 0.65)
        ].sort_values("MonthlyCharges", ascending=False)
    elif risk_filter == "Low Risk / Loyal":
        pool_df = catalog_df.sort_values("xgb_churn_prob", ascending=True)
    else:
        pool_df = catalog_df

    customer_options = [
        f"{row.customerID} — {row.xgb_churn_prob*100:.1f}% risk (${row.MonthlyCharges:.0f}/mo, {row.Contract})"
        for _, row in pool_df.head(250).iterrows()
    ]
    selected_option = st.selectbox("Select Customer to Inspect & Simulate", customer_options)
    selected_customer_id = selected_option.split(" — ")[0]

# Hero Header
st.markdown(
    """
    <div class="main-hero">
        <h1>ChurnSight — Customer Churn Predictor with Explainable AI Dashboard</h1>
        <p>Day 03 of 30 Projects in 30 Days · Not just who will leave — but why in plain words (SHAP), how what-if changes affect risk, and which retention offer maximizes ROI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top-level KPI Summary Row
flagged_mask = catalog_df["xgb_churn_prob"] >= active_threshold
flagged_df = catalog_df[flagged_mask]
at_risk_monthly_rev = float(flagged_df["MonthlyCharges"].sum())
at_risk_annual_rev = at_risk_monthly_rev * 12.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Total Active Accounts",
    f"{len(catalog_df):,}",
    f"Baseline Churn Rate: {catalog_df['ChurnBinary'].mean()*100:.1f}%",
    delta_color="off",
)
kpi2.metric(
    f"Flagged At-Risk (p ≥ {active_threshold:.2f})",
    f"{len(flagged_df):,}",
    f"{len(flagged_df)/len(catalog_df)*100:.1f}% of customer base",
    delta_color="inverse",
)
kpi3.metric(
    "Annual Recurring Revenue at Risk",
    f"${at_risk_annual_rev:,.0f}/yr",
    f"${at_risk_monthly_rev:,.0f}/mo billed",
    delta_color="off",
)
kpi4.metric(
    "Champion Model (XGBoost) PR-AUC",
    f"{metrics_report['models']['XGBoost (Cost-Aware Regularized)']['test_metrics_optimal']['pr_auc']:.3f}",
    f"ROC-AUC: {metrics_report['models']['XGBoost (Cost-Aware Regularized)']['test_metrics_optimal']['roc_auc']:.3f} | Recall: {live_tuning['optimal_metrics']['recall']*100:.1f}%",
)

tab_watchlist, tab_customer, tab_whatif, tab_models = st.tabs(
    [
        "📋 1. Risk Watchlist & Global SHAP",
        "🔎 2. Per-Customer SHAP Explainer",
        "🎛️ 3. Live What-If Simulator & Action Engine",
        "⚖️ 4. Model Comparison, Cost Tuning & Fairness Audit",
    ]
)

# ==============================================================================
# TAB 1: EXECUTIVE RISK WATCHLIST & GLOBAL SHAP DRIVERS
# ==============================================================================
with tab_watchlist:
    col_left, col_right = st.columns([1.25, 1.0])

    with col_left:
        st.subheader("🚨 Priority Retention Watchlist (Plain-English SHAP Drivers)")
        c_f1, c_f2, c_f3 = st.columns(3)
        contract_filter = c_f1.multiselect(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            default=["Month-to-month", "One year", "Two year"],
        )
        internet_filter = c_f2.multiselect(
            "Internet Service",
            options=["Fiber optic", "DSL", "No"],
            default=["Fiber optic", "DSL", "No"],
        )
        min_monthly = c_f3.slider("Min Monthly Charge ($)", 18, 115, 18)

        filtered_watch = catalog_df[
            (catalog_df["Contract"].isin(contract_filter))
            & (catalog_df["InternetService"].isin(internet_filter))
            & (catalog_df["MonthlyCharges"] >= min_monthly)
        ].sort_values("xgb_churn_prob", ascending=False)

        display_watch = filtered_watch.head(100).copy()
        display_watch["Churn Risk (%)"] = (display_watch["xgb_churn_prob"] * 100).round(1)
        display_watch["Risk Tier"] = display_watch["xgb_churn_prob"].apply(
            lambda p: classify_risk_tier(float(p), active_threshold)["tier"]
        )
        display_watch["Recommended Action"] = display_watch.apply(
            lambda r: (
                "1-Yr Lock-In + 10% Credit"
                if r["Contract"] == "Month-to-month" and r["xgb_churn_prob"] >= active_threshold
                else (
                    "Free Security + Tech Support Bundle"
                    if (r["TechSupport"] == "No" or r["OnlineSecurity"] == "No")
                    and r["xgb_churn_prob"] >= active_threshold
                    else (
                        "Auto-Pay $5/mo Credit"
                        if r["IsAutoPay"] == 0 and r["xgb_churn_prob"] >= active_threshold
                        else "Standard Loyalty Check-In"
                    )
                )
            ),
            axis=1,
        )

        st.dataframe(
            display_watch[
                [
                    "customerID",
                    "Churn Risk (%)",
                    "Risk Tier",
                    "Contract",
                    "tenure",
                    "MonthlyCharges",
                    "top_shap_drivers_plain",
                    "Recommended Action",
                ]
            ].rename(
                columns={
                    "customerID": "Customer ID",
                    "tenure": "Tenure (Mo)",
                    "MonthlyCharges": "Monthly ($)",
                    "top_shap_drivers_plain": "Top SHAP Drivers (Plain Words)",
                }
            ),
            use_container_width=True,
            height=430,
            hide_index=True,
        )

        csv_bytes = display_watch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Priority Retention Watchlist (CSV)",
            data=csv_bytes,
            file_name="churnsight_priority_watchlist.csv",
            mime="text/csv",
        )

    with col_right:
        st.subheader("🌍 Global SHAP Feature Importance (Top 12 Drivers)")
        global_shap = bundle["global_shap_df"].head(12).iloc[::-1]
        fig_global = go.Figure(
            go.Bar(
                x=global_shap["mean_abs_shap"],
                y=global_shap["display_name"],
                orientation="h",
                marker=dict(
                    color=global_shap["mean_signed_shap"],
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="Signed Impact"),
                ),
                text=[f"{v:.3f}" for v in global_shap["mean_abs_shap"]],
                textposition="outside",
            )
        )
        fig_global.update_layout(
            height=450,
            margin=dict(l=10, r=35, t=20, b=20),
            xaxis_title="Mean |SHAP Value| (Impact on Churn Log-Odds)",
            yaxis_title="",
        )
        st.plotly_chart(fig_global, use_container_width=True)

# ==============================================================================
# TAB 2: PER-CUSTOMER SHAP EXPLAINER
# ==============================================================================
with tab_customer:
    cust_df = catalog_df[catalog_df["customerID"] == selected_customer_id].iloc[[0]]
    cust_row = cust_df.iloc[0]
    local_exp = shap_explainer.explain_customer(cust_df, top_k=6)
    action_plan = recommend_retention_actions(
        customer_row=cust_row,
        model=champion_model,
        preprocessor=preprocessor,
        customer_ltv=float(cost_fn),
        optimal_threshold=active_threshold,
    )

    c_top1, c_top2 = st.columns([1.0, 1.35])
    with c_top1:
        st.subheader(f"👤 Customer `{selected_customer_id}` Risk Profile")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=local_exp["churn_probability"] * 100.0,
                number={"suffix": "%", "font": {"size": 42}},
                title={"text": f"Predicted Churn Probability ({action_plan['risk_tier']})"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": action_plan["badge_color"]},
                    "steps": [
                        {"range": [0, active_threshold * 100], "color": "#E8F5E9"},
                        {"range": [active_threshold * 100, 65], "color": "#FFF3E0"},
                        {"range": [65, 100], "color": "#FCE4EC"},
                    ],
                    "threshold": {
                        "line": {"color": "#111111", "width": 4},
                        "thickness": 0.8,
                        "value": active_threshold * 100.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=270, margin=dict(l=20, r=20, t=45, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"""
            <div class="driver-pill">
                <strong>💬 Plain-English SHAP Explanation:</strong><br/>
                Customer <code>{selected_customer_id}</code> has a <strong>{local_exp['churn_probability']*100:.1f}%</strong>
                predicted churn probability driven primarily by:
                <strong>{local_exp['plain_summary']}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("Contract", str(cust_row["Contract"]))
        p2.metric("Monthly Bill", f"${float(cust_row['MonthlyCharges']):.2f}")
        p3.metric("Tenure", f"{int(cust_row['tenure'])} months")

        p4, p5, p6 = st.columns(3)
        p4.metric("Internet", str(cust_row["InternetService"]))
        p5.metric("Tech Support", str(cust_row["TechSupport"]))
        p6.metric("Online Security", str(cust_row["OnlineSecurity"]))

    with c_top2:
        st.subheader("📊 Per-Customer SHAP Feature Contributions")
        all_drv = pd.DataFrame(local_exp["all_drivers"]).iloc[::-1]
        colors = ["#D81B60" if v > 0 else "#1E88E5" for v in all_drv["shap_value"]]
        fig_local = go.Figure(
            go.Bar(
                x=all_drv["shap_value"],
                y=all_drv["plain_phrase"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in all_drv["shap_value"]],
                textposition="outside",
            )
        )
        fig_local.update_layout(
            height=440,
            margin=dict(l=10, r=45, t=25, b=20),
            xaxis_title="SHAP Contribution to Churn Log-Odds (Red = Pushes toward Churn, Blue = Protects)",
            yaxis_title="",
        )
        st.plotly_chart(fig_local, use_container_width=True)

        primary_rec = action_plan["primary_recommendation"]
        st.markdown(
            f"""
            <div class="action-card">
                <strong>🎯 Top Recommended Retention Action: {primary_rec['title']}</strong><br/>
                <span>{primary_rec['description']}</span><br/>
                <small>
                    <strong>Post-Action Churn Risk:</strong> {primary_rec['post_action_probability']*100:.1f}%
                    (Drops by <strong>{primary_rec['absolute_risk_reduction']*100:.1f} percentage points</strong>) ·
                    <strong>Offer Cost:</strong> ${primary_rec['offer_cost']:.0f} ·
                    <strong>Expected Net Savings:</strong> <strong>${primary_rec['expected_net_roi']:+.2f}</strong>
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 3: LIVE WHAT-IF SIMULATOR & ACTION RECOMMENDER
# ==============================================================================
with tab_whatif:
    st.subheader(f"🎛️ Live Counterfactual What-If Simulator — Customer `{selected_customer_id}`")
    st.caption(
        "Adjust contract term, monthly bill, add-on bundles, or payment method below to see how the XGBoost model and SHAP explanations respond in real time."
    )

    sim_col_controls, sim_col_results = st.columns([1.05, 1.35])

    with sim_col_controls:
        contract_choices = ["Month-to-month", "One year", "Two year"]
        sim_contract = st.selectbox(
            "Simulated Contract Type",
            options=contract_choices,
            index=contract_choices.index(str(cust_row["Contract"])),
        )
        sim_monthly = st.slider(
            "Simulated Monthly Charge ($/mo)",
            min_value=18.25,
            max_value=118.75,
            value=float(cust_row["MonthlyCharges"]),
            step=1.0,
        )
        sim_tenure = st.slider(
            "Simulated Tenure (Months)",
            min_value=0,
            max_value=72,
            value=int(cust_row["tenure"]),
            step=1,
        )

        internet_choices = ["Fiber optic", "DSL", "No"]
        sim_internet = st.selectbox(
            "Simulated Internet Service",
            options=internet_choices,
            index=internet_choices.index(str(cust_row["InternetService"])),
        )

        addon_choices = ["No", "Yes", "No internet service"]
        c_s1, c_s2 = st.columns(2)
        sim_tech = c_s1.selectbox(
            "Simulated Tech Support",
            options=["No", "Yes"] if sim_internet != "No" else ["No internet service"],
            index=(
                1
                if str(cust_row["TechSupport"]) == "Yes" and sim_internet != "No"
                else 0
            ),
        )
        sim_sec = c_s2.selectbox(
            "Simulated Online Security",
            options=["No", "Yes"] if sim_internet != "No" else ["No internet service"],
            index=(
                1
                if str(cust_row["OnlineSecurity"]) == "Yes" and sim_internet != "No"
                else 0
            ),
        )

        pay_choices = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        sim_payment = st.selectbox(
            "Simulated Payment Method",
            options=pay_choices,
            index=pay_choices.index(str(cust_row["PaymentMethod"])),
        )

    sim_overrides = {
        "Contract": sim_contract,
        "MonthlyCharges": sim_monthly,
        "tenure": sim_tenure,
        "InternetService": sim_internet,
        "TechSupport": sim_tech,
        "OnlineSecurity": sim_sec,
        "PaymentMethod": sim_payment,
    }

    sim_result = simulate_what_if(
        original_row=cust_row,
        overrides=sim_overrides,
        model=champion_model,
        preprocessor=preprocessor,
        explainer=shap_explainer,
    )

    with sim_col_results:
        m1, m2, m3 = st.columns(3)
        orig_p = sim_result["original_probability"] * 100.0
        new_p = sim_result["new_probability"] * 100.0
        delta_p = sim_result["absolute_delta"] * 100.0
        expected_ltv_saved = max(0.0, (orig_p - new_p) / 100.0 * float(cost_fn))

        m1.metric("Original Churn Risk", f"{orig_p:.1f}%")
        m2.metric(
            "Simulated What-If Risk",
            f"{new_p:.1f}%",
            delta=f"{delta_p:+.1f}% pts",
            delta_color="inverse",
        )
        m3.metric(
            "Expected Retained LTV Value",
            f"${expected_ltv_saved:,.2f}",
            delta=f"At ${cost_fn} LTV per saved customer",
            delta_color="off",
        )

        fig_compare = go.Figure(
            data=[
                go.Bar(
                    name="Current Baseline Profile",
                    x=["Baseline Churn Risk"],
                    y=[orig_p],
                    marker_color="#D81B60",
                    text=[f"{orig_p:.1f}%"],
                    textposition="outside",
                ),
                go.Bar(
                    name="Simulated What-If Profile",
                    x=["Simulated What-If Risk"],
                    y=[new_p],
                    marker_color="#2E7D32" if new_p < orig_p else "#F57C00",
                    text=[f"{new_p:.1f}%"],
                    textposition="outside",
                ),
            ]
        )
        fig_compare.add_hline(
            y=active_threshold * 100.0,
            line_dash="dash",
            line_color="#111111",
            annotation_text=f"Action Threshold ({active_threshold*100:.0f}%)",
        )
        fig_compare.update_layout(
            height=260,
            yaxis=dict(range=[0, 105], title="Predicted Churn Probability (%)"),
            margin=dict(l=20, r=20, t=25, b=20),
            showlegend=True,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        new_exp = sim_result["new_explanation"]
        st.markdown(
            f"**Updated Plain-English SHAP Drivers after What-If:** `{new_exp['plain_summary']}`"
        )

        st.markdown("#### 💡 Automated Retention Playbook Comparison for This Customer")
        if action_plan["all_recommendations"]:
            rec_table = pd.DataFrame(action_plan["all_recommendations"])[
                [
                    "title",
                    "target_driver",
                    "post_action_probability",
                    "absolute_risk_reduction",
                    "offer_cost",
                    "expected_net_roi",
                ]
            ].copy()
            rec_table["post_action_probability"] = (
                rec_table["post_action_probability"] * 100.0
            ).round(1)
            rec_table["absolute_risk_reduction"] = (
                rec_table["absolute_risk_reduction"] * 100.0
            ).round(1)
            st.dataframe(
                rec_table.rename(
                    columns={
                        "title": "Playbook Intervention",
                        "target_driver": "Targeted SHAP Driver",
                        "post_action_probability": "Post-Offer Risk (%)",
                        "absolute_risk_reduction": "Risk Drop (% pts)",
                        "offer_cost": "Offer Cost ($)",
                        "expected_net_roi": "Expected Net ROI ($)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

# ==============================================================================
# TAB 4: MODEL COMPARISON, COST-THRESHOLD TUNER & FAIRNESS SLICE AUDIT
# ==============================================================================
with tab_models:
    st.subheader("⚖️ Rigorous Model Comparison (5-Fold Stratified CV + 95% Bootstrap CIs)")
    st.caption(
        "Why Accuracy is misleading: A naive majority-class predictor achieves ~73.5% accuracy while catching 0% of churning customers. We select models using PR-AUC and Expected Business Cost."
    )

    model_rows = []
    for m_name, m_data in metrics_report["models"].items():
        opt_m = m_data["test_metrics_optimal"]
        ci_pr = m_data["test_bootstrap_95_ci"]["pr_auc"]
        ci_roc = m_data["test_bootstrap_95_ci"]["roc_auc"]
        model_rows.append(
            {
                "Model": m_name,
                "5-Fold CV PR-AUC": f"{m_data['cv_val_pr_auc_mean']:.4f} ± {m_data['cv_val_pr_auc_std']:.4f}",
                "Test PR-AUC (95% CI)": f"{opt_m['pr_auc']:.4f} [{ci_pr['ci_lower_95']:.3f}, {ci_pr['ci_upper_95']:.3f}]",
                "Test ROC-AUC (95% CI)": f"{opt_m['roc_auc']:.4f} [{ci_roc['ci_lower_95']:.3f}, {ci_roc['ci_upper_95']:.3f}]",
                "Optimal Threshold": opt_m["threshold"],
                "Test F1": opt_m["f1"],
                "Test Recall": opt_m["recall"],
                "Test Precision": opt_m["precision"],
                "Test Accuracy": opt_m["accuracy"],
                "Brier Score (Calibration)": opt_m["brier_score"],
                "Net Savings ($)": f"${opt_m['net_savings_vs_do_nothing']:,.0f}",
                "Inference (ms/1k rows)": m_data["inference_latency_ms_per_1k"],
            }
        )
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 💰 Cost-Sensitive Threshold Optimization Curve (XGBoost)")
        curve_df = pd.DataFrame(live_tuning["threshold_curve"])
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=curve_df["threshold"],
                y=curve_df["total_business_cost"],
                mode="lines",
                name="Total Business Cost ($)",
                line=dict(color="#D81B60", width=3),
            )
        )
        fig_cost.add_vline(
            x=active_threshold,
            line_dash="dash",
            line_color="#2E7D32",
            annotation_text=f"Optimal t = {active_threshold:.2f} (${live_tuning['optimal_metrics']['total_business_cost']:,.0f})",
        )
        fig_cost.add_vline(
            x=0.50,
            line_dash="dot",
            line_color="#757575",
            annotation_text=f"Default t = 0.50 (${live_tuning['default_050_metrics']['total_business_cost']:,.0f})",
        )
        fig_cost.update_layout(
            height=340,
            xaxis_title="Decision Probability Threshold (t)",
            yaxis_title="Total Expected Cost ($ = FN·C_FN + FP·C_FP)",
            margin=dict(l=15, r=15, t=20, b=20),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with c_chart2:
        st.markdown("#### 📈 Precision-Recall Curves Across Models (Held-Out Test Set)")
        fig_pr = go.Figure()
        palette = {
            "XGBoost (Cost-Aware Regularized)": "#D81B60",
            "Logistic Regression (L2 Balanced)": "#1E88E5",
            "Random Forest (Regularized)": "#43A047",
            "Naive Baseline (Majority)": "#9E9E9E",
        }
        for m_name, m_data in metrics_report["models"].items():
            curves = m_data["curves"]
            fig_pr.add_trace(
                go.Scatter(
                    x=curves["pr_recall"],
                    y=curves["pr_precision"],
                    mode="lines",
                    name=f"{m_name} (PR-AUC={m_data['test_metrics_optimal']['pr_auc']:.3f})",
                    line=dict(color=palette.get(m_name, "#333333"), width=2.5),
                )
            )
        fig_pr.update_layout(
            height=340,
            xaxis_title="Recall (Fraction of Actual Churners Caught)",
            yaxis_title="Precision (Fraction of Flagged Customers Who Churn)",
            margin=dict(l=15, r=15, t=20, b=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("#### 🔍 Slice-Based Error & Fairness Analysis (`SeniorCitizen`, `Contract`, `InternetService`, `Tenure`)")
    slice_df = pd.DataFrame(metrics_report["demographic_slice_audit"])
    st.dataframe(
        slice_df.rename(
            columns={
                "slice_feature": "Dimension",
                "cohort": "Subgroup Cohort",
                "sample_size": "Test N",
                "actual_churn_rate": "Actual Churn Rate",
                "predicted_positive_rate": "Flagged Rate",
                "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
                "f1": "F1-Score",
                "recall": "Recall",
                "precision": "Precision",
                "false_positive_rate": "FPR",
                "false_negative_rate": "FNR",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
