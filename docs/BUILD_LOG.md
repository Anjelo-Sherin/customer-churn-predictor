# ChurnSight — Day 03 Build Log

## Completed Milestones (2026-10-19)

1. **Phase 1 — Data Quality & Leak-Free Feature Engineering (`src/data_pipeline.py`):**
   - Audited the 7,043-row IBM Telco Customer Churn dataset (`21` columns).
   - Identified and handled the `11` disguised missing-value rows where `TotalCharges == " "` (whitespace strings occurring exclusively when `tenure == 0` and `Churn == "No"`), creating `TotalCharges_WasMissing = 1` and imputing `TotalCharges = 0.0`.
   - Engineered row-local domain features (`TotalServicesCount`, `HasSupportBundle`, `IsAutoPay`, `IsNewCustomer`, `ChargeToTenureRatio`, `MonthlyOverHistoricalDelta`).
   - Enforced strict featurization ordering: Stratified `70% Train / 15% Validation / 15% Test` split **before** fitting `ColumnTransformer` (`OrdinalEncoder` for `Contract`, `OneHotEncoder` for nominal features, `StandardScaler` for numerical features, and passthrough for pre-encoded binary indicators).

2. **Phase 2 — Multi-Model Arena, Cost-Sensitive Thresholding & Fairness Audit (`src/model_trainer.py`):**
   - Compared `Naive Baseline (Majority)`, `Logistic Regression (L2 Balanced)`, `Random Forest (Regularized)`, and `XGBoost (Cost-Aware Regularized)` using 5-Fold Stratified Cross-Validation and 95% Bootstrap Confidence Intervals on held-out test metrics (`PR-AUC`, `ROC-AUC`, `F1`).
   - Built a business cost optimizer ($\text{Cost}(t) = FN(t)\cdot \$500 + FP(t)\cdot \$60$) that tunes the decision threshold on the validation split and quantifies net retained LTV savings vs. a naive `0.50` threshold.
   - Conducted slice-based error and fairness analysis across `SeniorCitizen`, `Contract`, `InternetService`, and `IsNewCustomer` cohorts.

3. **Phase 3 — Plain-English TreeSHAP Explainer & Counterfactual Action Engine (`src/explainer.py`, `src/action_engine.py`):**
   - Implemented `ChurnExplainer` using `shap.TreeExplainer` to compute global feature importances and per-customer local attributions translated into natural business phrases.
   - Built `simulate_what_if` and `recommend_retention_actions` to evaluate candidate retention playbooks via live counterfactual re-scoring and rank them by Expected Net ROI ($(\Delta \text{Risk} \times \text{LTV}) - \text{Offer Cost}$).

4. **Phase 4 — Interactive Streamlit Dashboard, Story Notebook & Automated Tests (`app.py`, `notebooks/01_eda_and_model_comparison.ipynb`, `tests/test_churn_pipeline.py`):**
   - Built the 4-tab Streamlit application (`app.py`), publication charts (`assets/`), story-driven Jupyter notebook, and `pytest` verification suite.
