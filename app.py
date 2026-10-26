"""
ChurnSight — Day 03: Customer Churn Predictor with Explainable AI Dashboard.
Redesigned in high-voltage Neo-Brutalism aesthetic:
- Heavy solid black borders (3px - 4px)
- Hard, unblurred offset drop shadows (4px - 6px)
- Vibrant retro palette (Canary Yellow, Hot Coral, Electric Cyan, Lavender, Mint)
- High-contrast chunky sticker tags and interactive cards
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

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
    page_title="ChurnSight — Neo-Brutalist Explainable AI Churn Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Neo-Brutalism Design System CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800;900&family=JetBrains+Mono:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #000000;
        background-color: #FFFDF7;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Neo-Brutalism Hero Header */
    .nb-hero {
        background: #FFE600;
        border: 3.5px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        border-radius: 8px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .nb-hero-badge {
        display: inline-block;
        background: #000000;
        color: #FFE600;
        font-weight: 900;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .nb-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 900;
        color: #000000;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .nb-hero p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #111111;
    }
    .nb-sticker-bar {
        display: flex;
        gap: 0.6rem;
        margin-top: 0.9rem;
        flex-wrap: wrap;
    }
    .nb-sticker {
        background: #FFFFFF;
        border: 2px solid #000000;
        box-shadow: 2.5px 2.5px 0px #000000;
        font-size: 0.8rem;
        font-weight: 800;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        color: #000000;
    }

    /* Neo-Brutalist st.metric Cards */
    div[data-testid="stMetric"] {
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 8px !important;
        padding: 1rem 1.15rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 7px 7px 0px #000000 !important;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #000000 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.95rem !important;
        font-weight: 900 !important;
        color: #000000 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 800 !important;
        color: #111111 !important;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] {
        background-color: #E9D5FF !important; /* Lavender */
    }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] {
        background-color: #FECDD3 !important; /* Coral */
    }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] {
        background-color: #BAE6FD !important; /* Cyan */
    }
    div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {
        background-color: #FEF08A !important; /* Canary Yellow */
    }

    /* Neo-Brutalist Box Cards */

    .nb-box {
        background: #FFFFFF;
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px #000000;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.2rem;
    }
    .nb-box-header {
        font-weight: 900;
        font-size: 1.15rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Explanation Banner */
    .nb-driver-pill {
        background: #FFF1F2;
        border: 3px solid #000000;
        box-shadow: 4px 4px 0px #000000;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #000000;
        font-size: 0.98rem;
        font-weight: 600;
        margin: 0.8rem 0 1rem 0;
    }

    /* Retention Action Card */
    .nb-action-card {
        background: #FFFFFF;
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px #000000;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 1rem;
    }
    .nb-action-header {
        background: #A7F3D0;
        border-bottom: 3px solid #000000;
        padding: 0.75rem 1.1rem;
        font-weight: 900;
        font-size: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .nb-action-body {
        padding: 1rem 1.1rem;
        font-size: 0.95rem;
        color: #111111;
        font-weight: 500;
    }
    .nb-action-footer {
        background: #F4F4F5;
        border-top: 2px solid #000000;
        padding: 0.65rem 1.1rem;
        font-size: 0.84rem;
        font-weight: 700;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    /* Custom Neo-Brutalist Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFDF7 !important;
        border-right: 3.5px solid #000000 !important;
    }
    section[data-testid="stSidebar"] .stSlider, 
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.8rem;
    }
    .nb-sidebar-header {
        background: #000000;
        color: #FFFFFF;
        padding: 0.75rem 1rem;
        font-weight: 900;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-radius: 4px;
        box-shadow: 3px 3px 0px #FFE600;
        margin-bottom: 1rem;
    }

    /* Neo-Brutalism Tab Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        border-bottom: 3px solid #000000 !important;
        padding-bottom: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 3.5px 3.5px 0px #000000 !important;
        border-radius: 6px !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
        color: #000000 !important;
        padding: 0.5rem 1rem !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translate(-1px, -1px) !important;
        box-shadow: 5px 5px 0px #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        transform: translate(-2px, -2px) !important;
    }

    /* Neo-Brutalism Buttons */
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 6px !important;
        font-weight: 900 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.2rem !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        background-color: #FFDE00 !important;
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px #000000 !important;
        color: #000000 !important;
    }
    .stButton > button:active, div[data-testid="stDownloadButton"] > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 1px 1px 0px #000000 !important;
    }

    /* Neo-Brutalist Table Container */
    div[data-testid="stDataFrame"] {
        border: 3px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 6px !important;
        background: #FFFFFF !important;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_neo_brutalism_plotly(fig: go.Figure) -> go.Figure:
    """Applies high-contrast Neo-Brutalist canvas formatting to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#000000", size=12),
        margin=dict(l=15, r=20, t=35, b=25),
        xaxis=dict(
            showline=True,
            linewidth=2.5,
            linecolor="#000000",
            gridcolor="#F0F0F0",
            gridwidth=1.5,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#000000",
            tickfont=dict(color="#000000", family="JetBrains Mono", size=11),
        ),
        yaxis=dict(
            showline=True,
            linewidth=2.5,
            linecolor="#000000",
            gridcolor="#F0F0F0",
            gridwidth=1.5,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#000000",
            tickfont=dict(color="#000000", family="JetBrains Mono", size=11),
        ),
    )
    return fig


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

# Neo-Brutalist Sidebar Controls
with st.sidebar:
    st.markdown(
        """
        <div class="nb-sidebar-header">
            ⚙️ BUSINESS COST ENGINE
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Tune the decision cutoff dynamically based on your company's asymmetric unit economics:")

    cost_fn = st.slider(
        "Lost Customer LTV ($C_FN)",
        min_value=150,
        max_value=1500,
        value=int(bundle.get("cost_fn", 500)),
        step=50,
        help="Cost of False Negative: losing a churner without contacting them.",
    )
    cost_fp = st.slider(
        "Retention Offer Cost ($C_FP)",
        min_value=15,
        max_value=250,
        value=int(bundle.get("cost_fp", 60)),
        step=5,
        help="Cost of False Positive: giving retention discounts to customers who would stay.",
    )

    live_tuning = optimize_cost_threshold(
        y_true=bundle["y_test"],
        y_prob=bundle["y_test_prob_xgb"],
        cost_fn=float(cost_fn),
        cost_fp=float(cost_fp),
    )
    active_threshold = float(live_tuning["optimal_threshold"])

    st.markdown(
        f"""
        <div class="nb-box" style="padding: 0.9rem; margin-top: 0.7rem; background: #FEF08A;">
            <div style="font-size: 0.72rem; font-weight: 900; text-transform: uppercase;">OPTIMAL CHURN CUTOFF</div>
            <div style="font-size: 1.7rem; font-weight: 900; font-family: 'JetBrains Mono';">t* = {active_threshold:.2f}</div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #333;">{active_threshold - 0.50:+.2f} vs naive 0.50 cutoff</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="nb-box" style="padding: 0.9rem; background: #BAF7D0;">
            <div style="font-size: 0.72rem; font-weight: 900; text-transform: uppercase;">NET CAMPAIGN SAVINGS</div>
            <div style="font-size: 1.55rem; font-weight: 900; font-family: 'JetBrains Mono';">+${live_tuning['optimal_metrics']['net_savings_vs_do_nothing']:,.0f}</div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #333;">+${live_tuning['savings_over_default_050']:,.0f} over standard 0.50</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        """
        <div class="nb-sidebar-header" style="background: #2563EB;">
            🔍 CUSTOMER TRIAGE
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    selected_option = st.selectbox("Select Customer to Deep-Dive", customer_options)
    selected_customer_id = selected_option.split(" — ")[0]

# Neo-Brutalist Main Hero Header
st.markdown(
    """
    <div class="nb-hero">
        <div class="nb-hero-badge">⚡ CHURNSIGHT // DAY 03 OF 30</div>
        <h1>⚡ ChurnSight: Explainable AI Retention Studio</h1>
        <p>Not just who will churn — but why in plain words (SHAP), how what-if adjustments reduce risk, and which retention offer maximizes Net ROI.</p>
        <div class="nb-sticker-bar">
            <span class="nb-sticker">🎯 0% BLACK BOX</span>
            <span class="nb-sticker">🌲 TREESHAP EXPLAINER</span>
            <span class="nb-sticker">🎛️ WHAT-IF SIMULATOR</span>
            <span class="nb-sticker">💰 COST MATRIX OPTIMIZER</span>
            <span class="nb-sticker">⚖️ FAIRNESS AUDITED</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Neo-Brutalist Top-Row KPI Metrics
flagged_mask = catalog_df["xgb_churn_prob"] >= active_threshold
flagged_df = catalog_df[flagged_mask]
at_risk_monthly_rev = float(flagged_df["MonthlyCharges"].sum())
at_risk_annual_rev = at_risk_monthly_rev * 12.0
xgb_opt_metrics = metrics_report["models"]["XGBoost (Cost-Aware Regularized)"]["test_metrics_optimal"]

st.markdown(
    f"""
    <div class="nb-kpi-grid">
        <div class="nb-kpi-card kpi-lavender">
            <div class="nb-kpi-title">👥 TOTAL ACTIVE ACCOUNTS</div>
            <div class="nb-kpi-val">{len(catalog_df):,}</div>
            <div class="nb-kpi-sub">Baseline Churn: <strong>{catalog_df['ChurnBinary'].mean()*100:.1f}%</strong></div>
        </div>
        <div class="nb-kpi-card kpi-coral">
            <div class="nb-kpi-title">🚨 AT-RISK FLAG (p ≥ {active_threshold:.2f})</div>
            <div class="nb-kpi-val">{len(flagged_df):,}</div>
            <div class="nb-kpi-sub"><strong>{len(flagged_df)/len(catalog_df)*100:.1f}%</strong> of customer base</div>
        </div>
        <div class="nb-kpi-card kpi-cyan">
            <div class="nb-kpi-title">💵 AT-RISK ANNUAL ARR</div>
            <div class="nb-kpi-val">${at_risk_annual_rev/1000:,.0f}K</div>
            <div class="nb-kpi-sub">${at_risk_monthly_rev:,.0f}/mo billable ARR</div>
        </div>
        <div class="nb-kpi-card kpi-yellow">
            <div class="nb-kpi-title">🏆 CHAMPION PR-AUC</div>
            <div class="nb-kpi-val">{xgb_opt_metrics['pr_auc']:.3f}</div>
            <div class="nb-kpi-sub">ROC-AUC: <strong>{xgb_opt_metrics['roc_auc']:.3f}</strong> · Recall: <strong>{live_tuning['optimal_metrics']['recall']*100:.1f}%</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)




tab_watchlist, tab_customer, tab_whatif, tab_models = st.tabs(
    [
        "📋 1. EXECUTIVE WATCHLIST & GLOBAL SHAP",
        "🔎 2. CUSTOMER DEEP-DIVE & SHAP EXPLAINER",
        "🎛️ 3. LIVE WHAT-IF SIMULATOR & ACTION ENGINE",
        "⚖️ 4. MODEL ARENA, COST TUNER & FAIRNESS AUDIT",
    ]
)

# ==============================================================================
# TAB 1: EXECUTIVE WATCHLIST & GLOBAL SHAP
# ==============================================================================
with tab_watchlist:
    col_left, col_right = st.columns([1.25, 1.0])

    with col_left:
        st.markdown(
            """
            <div class="nb-box">
                <div class="nb-box-header">🚨 High-Risk Retention Watchlist</div>
                <div style="font-size: 0.9rem; font-weight: 500; margin-bottom: 0.8rem;">
                    Customers flagged above the cost-optimal decision threshold with plain-English TreeSHAP drivers:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_f1, c_f2, c_f3 = st.columns(3)
        contract_filter = c_f1.multiselect(
            "Contract Term",
            options=["Month-to-month", "One year", "Two year"],
            default=["Month-to-month", "One year", "Two year"],
        )
        internet_filter = c_f2.multiselect(
            "Internet Tier",
            options=["Fiber optic", "DSL", "No"],
            default=["Fiber optic", "DSL", "No"],
        )
        min_monthly = c_f3.slider("Min Monthly Bill ($)", 18, 115, 18)

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
                    "Free Security + Support Bundle"
                    if (r["TechSupport"] == "No" or r["OnlineSecurity"] == "No")
                    and r["xgb_churn_prob"] >= active_threshold
                    else (
                        "Auto-Pay $5/mo Credit"
                        if r["IsAutoPay"] == 0 and r["xgb_churn_prob"] >= active_threshold
                        else "Loyalty Check-In"
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
                    "top_shap_drivers_plain": "Top TreeSHAP Drivers (Plain Words)",
                }
            ),
            width="stretch",
            height=440,
            hide_index=True,
        )

        csv_bytes = display_watch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ EXPORT PRIORITY WATCHLIST (CSV)",
            data=csv_bytes,
            file_name="churnsight_priority_watchlist.csv",
            mime="text/csv",
        )

    with col_right:
        st.markdown(
            """
            <div class="nb-box">
                <div class="nb-box-header">🌍 Global TreeSHAP Feature Drivers</div>
                <div style="font-size: 0.88rem; font-weight: 500;">
                    Mean absolute SHAP value impact on model log-odds across all 7,043 customers:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        global_shap = bundle["global_shap_df"].head(12).iloc[::-1]
        fig_global = go.Figure(
            go.Bar(
                x=global_shap["mean_abs_shap"],
                y=global_shap["display_name"],
                orientation="h",
                marker=dict(
                    color=["#FF4B72" if s > 0 else "#00D2FF" for s in global_shap["mean_signed_shap"]],
                    line=dict(color="#000000", width=2.5),
                ),
                text=[f"{v:.3f}" for v in global_shap["mean_abs_shap"]],
                textposition="outside",
            )
        )
        apply_neo_brutalism_plotly(fig_global)
        fig_global.update_layout(
            height=460,
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
        st.markdown(
            f"""
            <div class="nb-box" style="border-left: 6px solid #FFE600;">
                <div class="nb-box-header">👤 Customer <code>{selected_customer_id}</code> Profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=local_exp["churn_probability"] * 100.0,
                number={"suffix": "%", "font": {"size": 44, "family": "JetBrains Mono", "color": "#000000"}},
                title={"text": f"PREDICTED CHURN RISK ({action_plan['risk_tier'].upper()})", "font": {"family": "Plus Jakarta Sans", "size": 13, "color": "#000000"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#000000"},
                    "bar": {"color": "#000000", "thickness": 0.28},
                    "bgcolor": "#FFFFFF",
                    "borderwidth": 3,
                    "bordercolor": "#000000",
                    "steps": [
                        {"range": [0, active_threshold * 100], "color": "#BAF7D0"},
                        {"range": [active_threshold * 100, 65], "color": "#FEF08A"},
                        {"range": [65, 100], "color": "#FECDD3"},
                    ],
                    "threshold": {
                        "line": {"color": "#FF0033", "width": 5},
                        "thickness": 0.85,
                        "value": active_threshold * 100.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=250,
            margin=dict(l=25, r=25, t=35, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"""
            <div class="nb-driver-pill">
                <strong>💬 PLAIN-ENGLISH SHAP VERDICT:</strong><br/>
                Customer <code>{selected_customer_id}</code> has a <strong>{local_exp['churn_probability']*100:.1f}%</strong>
                predicted churn probability driven primarily by:
                <strong>{local_exp['plain_summary']}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)
        p1.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>CONTRACT</small><br><strong>{cust_row['Contract']}</strong></div>", unsafe_allow_html=True)
        p2.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>MONTHLY BILL</small><br><strong>${float(cust_row['MonthlyCharges']):.2f}</strong></div>", unsafe_allow_html=True)
        p3.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>TENURE</small><br><strong>{int(cust_row['tenure'])} Mo</strong></div>", unsafe_allow_html=True)

        p4, p5, p6 = st.columns(3)
        p4.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>INTERNET</small><br><strong>{cust_row['InternetService']}</strong></div>", unsafe_allow_html=True)
        p5.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>TECH SUPPORT</small><br><strong>{cust_row['TechSupport']}</strong></div>", unsafe_allow_html=True)
        p6.markdown(f"<div class='nb-box' style='padding:0.6rem;text-align:center;'><small>ONLINE SECURITY</small><br><strong>{cust_row['OnlineSecurity']}</strong></div>", unsafe_allow_html=True)

    with c_top2:
        st.markdown(
            """
            <div class="nb-box">
                <div class="nb-box-header">📊 Individual TreeSHAP Feature Attributions</div>
                <div style="font-size: 0.88rem; font-weight: 500;">
                    Red bars push customer toward churn; Cyan bars protect against churn:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_drv = pd.DataFrame(local_exp["all_drivers"]).iloc[::-1]
        colors = ["#FF4B72" if v > 0 else "#00D2FF" for v in all_drv["shap_value"]]
        fig_local = go.Figure(
            go.Bar(
                x=all_drv["shap_value"],
                y=all_drv["plain_phrase"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#000000", width=2.5)),
                text=[f"{v:+.3f}" for v in all_drv["shap_value"]],
                textposition="outside",
            )
        )
        apply_neo_brutalism_plotly(fig_local)
        fig_local.update_layout(
            height=430,
            xaxis_title="SHAP Log-Odds Contribution (Red = Risk Driver, Cyan = Protective)",
            yaxis_title="",
        )
        st.plotly_chart(fig_local, use_container_width=True)

        primary_rec = action_plan["primary_recommendation"]
        st.markdown(
            f"""
            <div class="nb-action-card">
                <div class="nb-action-header">
                    <span>🎯 TOP RETENTION PLAYBOOK: {primary_rec['title']}</span>
                    <span style="background:#000000;color:#A7F3D0;padding:2px 8px;border-radius:4px;font-size:0.75rem;">RECOMMENDED</span>
                </div>
                <div class="nb-action-body">
                    {primary_rec['description']}
                </div>
                <div class="nb-action-footer">
                    <span>📉 New Risk: <strong>{primary_rec['post_action_probability']*100:.1f}%</strong> (-{primary_rec['absolute_risk_reduction']*100:.1f}% pts)</span>
                    <span>💸 Offer Cost: <strong>${primary_rec['offer_cost']:.0f}</strong></span>
                    <span>💰 Expected Net Savings: <strong style="color:#059669;">${primary_rec['expected_net_roi']:+.2f}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 3: LIVE WHAT-IF SIMULATOR & ACTION RECOMMENDER
# ==============================================================================
with tab_whatif:
    st.markdown(
        f"""
        <div class="nb-box" style="border-left: 6px solid #00D2FF;">
            <div class="nb-box-header">🎛️ Counterfactual What-If Simulator — Customer <code>{selected_customer_id}</code></div>
            <div style="font-size: 0.9rem; font-weight: 500;">
                Test business interventions in real time. The modified customer profile re-runs through the live XGBoost pipeline and TreeSHAP explainer:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_col_controls, sim_col_results = st.columns([1.05, 1.35])

    with sim_col_controls:
        st.markdown("<div class='nb-box'>", unsafe_allow_html=True)
        contract_choices = ["Month-to-month", "One year", "Two year"]
        sim_contract = st.selectbox(
            "Simulated Contract Term",
            options=contract_choices,
            index=contract_choices.index(str(cust_row["Contract"])),
        )
        sim_monthly = st.slider(
            "Simulated Monthly Bill ($/mo)",
            min_value=18.25,
            max_value=118.75,
            value=float(cust_row["MonthlyCharges"]),
            step=1.0,
        )
        sim_tenure = st.slider(
            "Simulated Account Tenure (Months)",
            min_value=0,
            max_value=72,
            value=int(cust_row["tenure"]),
            step=1,
        )
        sim_internet = st.selectbox(
            "Simulated Internet Service",
            options=["Fiber optic", "DSL", "No"],
            index=["Fiber optic", "DSL", "No"].index(str(cust_row["InternetService"])),
        )
        c_s1, c_s2 = st.columns(2)
        sim_tech = c_s1.selectbox(
            "Tech Support",
            options=["No", "Yes"] if sim_internet != "No" else ["No internet service"],
            index=(1 if str(cust_row["TechSupport"]) == "Yes" and sim_internet != "No" else 0),
        )
        sim_sec = c_s2.selectbox(
            "Online Security",
            options=["No", "Yes"] if sim_internet != "No" else ["No internet service"],
            index=(1 if str(cust_row["OnlineSecurity"]) == "Yes" and sim_internet != "No" else 0),
        )
        sim_payment = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(str(cust_row["PaymentMethod"])),
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
        orig_p = sim_result["original_probability"] * 100.0
        new_p = sim_result["new_probability"] * 100.0
        delta_p = sim_result["absolute_delta"] * 100.0
        expected_ltv_saved = max(0.0, (orig_p - new_p) / 100.0 * float(cost_fn))

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.75rem;margin-bottom:1rem;">
                <div class="nb-box" style="padding:0.75rem;text-align:center;background:#FECDD3;">
                    <div style="font-size:0.72rem;font-weight:900;">BASELINE RISK</div>
                    <div style="font-size:1.6rem;font-weight:900;font-family:'JetBrains Mono';">{orig_p:.1f}%</div>
                </div>
                <div class="nb-box" style="padding:0.75rem;text-align:center;background:{'#BAF7D0' if new_p < orig_p else '#FEF08A'};">
                    <div style="font-size:0.72rem;font-weight:900;">SIMULATED RISK</div>
                    <div style="font-size:1.6rem;font-weight:900;font-family:'JetBrains Mono';">{new_p:.1f}%</div>
                    <div style="font-size:0.75rem;font-weight:800;">{delta_p:+.1f}% pts</div>
                </div>
                <div class="nb-box" style="padding:0.75rem;text-align:center;background:#BAE6FD;">
                    <div style="font-size:0.72rem;font-weight:900;">RETAINED LTV</div>
                    <div style="font-size:1.6rem;font-weight:900;font-family:'JetBrains Mono';">${expected_ltv_saved:,.0f}</div>
                    <div style="font-size:0.75rem;font-weight:800;">At ${cost_fn} LTV</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_compare = go.Figure(
            data=[
                go.Bar(
                    name="Current Baseline Profile",
                    x=["Baseline Churn Risk"],
                    y=[orig_p],
                    marker=dict(color="#FF4B72", line=dict(color="#000000", width=2.5)),
                    text=[f"{orig_p:.1f}%"],
                    textposition="outside",
                ),
                go.Bar(
                    name="Simulated What-If Profile",
                    x=["Simulated What-If Risk"],
                    y=[new_p],
                    marker=dict(color="#00E599" if new_p < orig_p else "#FFDE59", line=dict(color="#000000", width=2.5)),
                    text=[f"{new_p:.1f}%"],
                    textposition="outside",
                ),
            ]
        )
        fig_compare.add_hline(
            y=active_threshold * 100.0,
            line_dash="dash",
            line_color="#000000",
            line_width=2.5,
            annotation_text=f"Cutoff ({active_threshold*100:.0f}%)",
        )
        apply_neo_brutalism_plotly(fig_compare)
        fig_compare.update_layout(
            height=260,
            yaxis=dict(range=[0, 110], title="Predicted Churn Probability (%)"),
            showlegend=True,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        new_exp = sim_result["new_explanation"]
        st.markdown(
            f"""
            <div class="nb-box" style="padding:0.8rem 1rem;background:#F4F4F5;">
                <strong>⚡ UPDATED SHAP DRIVERS AFTER WHAT-IF:</strong><br>
                <code>{new_exp['plain_summary']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 💡 Candidate Retention Playbooks Ranked by Net ROI")
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
            rec_table["post_action_probability"] = (rec_table["post_action_probability"] * 100.0).round(1)
            rec_table["absolute_risk_reduction"] = (rec_table["absolute_risk_reduction"] * 100.0).round(1)
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
                width="stretch",
                hide_index=True,
            )

# ==============================================================================
# TAB 4: MODEL ARENA, COST-THRESHOLD TUNER & FAIRNESS SLICE AUDIT
# ==============================================================================
with tab_models:
    st.markdown(
        """
        <div class="nb-box" style="border-left: 6px solid #C084FC;">
            <div class="nb-box-header">⚖️ Multi-Model Arena (5-Fold Stratified CV + 95% Bootstrap CIs)</div>
            <div style="font-size: 0.9rem; font-weight: 500;">
                Why Accuracy is misleading: Naive majority-class prediction achieves ~71.5% accuracy while catching 0% of churners. We evaluate models using PR-AUC and Net Financial ROI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
                "Optimal Cutoff": opt_m["threshold"],
                "Test F1": opt_m["f1"],
                "Test Recall": opt_m["recall"],
                "Test Precision": opt_m["precision"],
                "Calibration Brier": opt_m["brier_score"],
                "Net Savings ($)": f"${opt_m['net_savings_vs_do_nothing']:,.0f}",
                "Inference (ms/1k)": m_data["inference_latency_ms_per_1k"],
            }
        )
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 💰 Cost-Sensitive Threshold Curve (XGBoost)")
        curve_df = pd.DataFrame(live_tuning["threshold_curve"])
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=curve_df["threshold"],
                y=curve_df["total_business_cost"],
                mode="lines",
                name="Total Business Cost ($)",
                line=dict(color="#FF4B72", width=3.5),
            )
        )
        fig_cost.add_vline(
            x=active_threshold,
            line_dash="dash",
            line_color="#000000",
            line_width=2.5,
            annotation_text=f"Optimal t* = {active_threshold:.2f}",
        )
        fig_cost.add_vline(
            x=0.50,
            line_dash="dot",
            line_color="#888888",
            line_width=2,
            annotation_text="Naive t=0.50",
        )
        apply_neo_brutalism_plotly(fig_cost)
        fig_cost.update_layout(
            height=340,
            xaxis_title="Decision Cutoff Threshold (t)",
            yaxis_title="Total Cost ($ = FN·LTV + FP·Offer)",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with c_chart2:
        st.markdown("#### 📈 Precision-Recall Curves (Held-Out Test Set)")
        fig_pr = go.Figure()
        palette = {
            "XGBoost (Cost-Aware Regularized)": "#FF4B72",
            "Logistic Regression (L2 Balanced)": "#00D2FF",
            "Random Forest (Regularized)": "#00E599",
            "Naive Baseline (Majority)": "#000000",
        }
        for m_name, m_data in metrics_report["models"].items():
            curves = m_data["curves"]
            fig_pr.add_trace(
                go.Scatter(
                    x=curves["pr_recall"],
                    y=curves["pr_precision"],
                    mode="lines",
                    name=f"{m_name.split(' (')[0]} ({m_data['test_metrics_optimal']['pr_auc']:.3f})",
                    line=dict(color=palette.get(m_name, "#333333"), width=3.0),
                )
            )
        apply_neo_brutalism_plotly(fig_pr)
        fig_pr.update_layout(
            height=340,
            xaxis_title="Recall (% of Churners Caught)",
            yaxis_title="Precision (% Flagged Who Churn)",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("#### 🔍 Subgroup Demographic Slice & Fairness Audit")
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
        width="stretch",
        hide_index=True,
    )
