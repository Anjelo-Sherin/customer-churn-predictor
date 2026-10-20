# API & Programmatic Usage Reference — ChurnSight

ChurnSight provides modular Python modules that can be imported and integrated directly into backend microservices, batch scoring jobs, or REST APIs.

---

## 1. Batch Customer Scoring & Preprocessing

```python
import pandas as pd
import joblib
from src.data_pipeline import clean_and_engineer_features

# Load trained artifact bundle
bundle = joblib.load("models/churn_artifacts.joblib")
model = bundle["models"][bundle["champion_name"]]
preprocessor = bundle["preprocessor"]
optimal_threshold = bundle["optimal_threshold"]

# Load raw incoming customer records
raw_data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean, validate, and compute row-local engineered features
cleaned = clean_and_engineer_features(raw_data)

# Transform using training-fitted ColumnTransformer
X_trans = preprocessor.transform(cleaned)

# Compute calibrated churn probabilities
churn_probabilities = model.predict_proba(X_trans)[:, 1]

# Apply cost-optimal threshold
predicted_churn = (churn_probabilities >= optimal_threshold).astype(int)
```

---

## 2. Generating Plain-English SHAP Explanations

```python
from src.explainer import ChurnExplainer

explainer = ChurnExplainer(
    model=model,
    preprocessor=preprocessor,
    feature_names=bundle["feature_names"],
    background_matrix=bundle["background_matrix"],
)

# Explain an individual customer
customer_row = cleaned.iloc[[0]]
explanation = explainer.explain_customer(customer_row, top_k=5)

print("Customer ID:", explanation["customerID"])
print("Churn Probability:", f"{explanation['churn_probability'] * 100:.1f}%")
print("Plain-English Drivers:", explanation["plain_summary"])
# Output: month-to-month contract + high monthly charge ($94.37/mo) + no tech support
```

---

## 3. Counterfactual What-If Simulation & Action Recommendations

```python
from src.action_engine import recommend_retention_actions, simulate_what_if

# Recommend and rank retention playbooks by Expected Net ROI
recommendations = recommend_retention_actions(
    customer_row=customer_row.iloc[0],
    model=model,
    preprocessor=preprocessor,
    customer_ltv=500.0,
    optimal_threshold=optimal_threshold,
)

primary = recommendations["primary_recommendation"]
print("Recommended Action:", primary["title"])
print("Post-Action Churn Risk:", f"{primary['post_action_probability'] * 100:.1f}%")
print("Expected Net ROI ($):", f"${primary['expected_net_roi']:+.2f}")

# Simulate custom user overrides
whatif_sim = simulate_what_if(
    original_row=customer_row.iloc[0],
    overrides={
        "Contract": "Two year",
        "MonthlyCharges": 75.0,
        "TechSupport": "Yes",
    },
    model=model,
    preprocessor=preprocessor,
    explainer=explainer,
)

print("Original Risk:", f"{whatif_sim['original_probability'] * 100:.1f}%")
print("Simulated Risk:", f"{whatif_sim['new_probability'] * 100:.1f}%")
print("Risk Reduction:", f"{whatif_sim['relative_reduction_pct']:.1f}%")
```
