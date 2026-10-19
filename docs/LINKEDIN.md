# LinkedIn Showcase Kit & Interview Prep — Day 03: ChurnSight

## Ready-to-Post LinkedIn Draft

**Day 03/30 — ChurnSight: Customer Churn Predictor with Explainable AI Dashboard** 📊🔍

Most churn notebooks stop at a single ROC-AUC score. In a real subscription business, knowing a customer has a "78% churn probability" is useless unless the retention team knows **why** they are leaving and **which offer** will actually save them without wasting margin.

Today for Day 03 of my #30ProjectsIn30Days challenge, I built **ChurnSight** — an end-to-end Explainable AI & Counterfactual Retention Studio on the IBM Telco Customer Churn dataset (7,043 accounts):

🔹 **Plain-English TreeSHAP Explanations:** Translates raw log-odds SHAP values into human-readable risk narratives per customer (*e.g., "month-to-month contract + high monthly charge (\$99.65/mo) + no tech support"*).
🔹 **Live Counterfactual What-If Simulator & ROI Action Engine:** Business users drag interactive sliders (`Contract`, `MonthlyCharges`, `TechSupport`, `OnlineSecurity`, `PaymentMethod`) to re-score customers live and rank retention playbooks by **Expected Net ROI** ($(\Delta \text{Risk} \times \text{LTV}) - \text{Offer Cost}$).
🔹 **Imbalance-Aware PR-AUC & Cost-Sensitive Thresholding:** Compares a Naive Baseline, L2-Balanced Logistic Regression, Regularized Random Forest, and Cost-Aware XGBoost with 5-fold Stratified CV, 95% Bootstrap Confidence Intervals, and a business cost matrix ($C_{FN}=\$500$ lost LTV vs. $C_{FP}=\$60$ retention offer).

⚠️ **Key Lesson:** During data auditing, I found 11 whitespace strings (`" "`) in `TotalCharges`—all corresponding to brand-new customers with `tenure == 0` who hadn't completed their first billing cycle (`Churn == "No"`). Preserving those rows with a `TotalCharges_WasMissing` indicator and `$0.00` imputation avoided silent row-dropping bias.

#30ProjectsIn30Days #DataScience #ExplainableAI #SHAP #XGBoost #MachineLearning #Streamlit #CustomerChurn #BuildInPublic

---

## Interview Talking Points (From Challenge Playbook)

1. **Why is Accuracy misleading on Customer Churn?**
   - In the Telco dataset, 73.46% of customers do not churn (`Churn == "No"`). A trivial majority-class classifier achieves **73.46% accuracy** while catching **0% of churning customers** (`Recall = 0.00`, `F1 = 0.00`).
2. **Why optimize PR-AUC instead of only ROC-AUC?**
   - Because true negatives ($TN$) outnumber positives roughly 3-to-1, the False Positive Rate ($FPR = FP / (FP + TN)$) in ROC-AUC remains artificially small even when false alarms spike. **Precision-Recall AUC (PR-AUC)** conditions directly on positive predictions and true positives, exposing precision degradation on the minority churn class.
3. **How do you tune the decision threshold by business cost instead of using `0.50`?**
   - Missing a churner (False Negative, $C_{FN} \approx \$500$ lost LTV) is ~8.3× more expensive than offering a loyalty perk to a customer who would have stayed (False Positive, $C_{FP} \approx \$60$). Sweeping thresholds $t \in [0.05, 0.95]$ to minimize $\text{Cost}(t) = FN(t)\cdot C_{FN} + FP(t)\cdot C_{FP}$ shifts the optimal threshold down to $t^* \approx 0.32 - 0.38$, increasing recall on at-risk accounts and saving thousands in net retained LTV over a naive `0.50` threshold.
4. **How do you interpret SHAP values for a business stakeholder?**
   - TreeSHAP decomposes the model's prediction for customer $i$ into an exact additive sum: $f(x_i) = \mathbb{E}[f(X)] + \sum_{j=1}^M \phi_{ij}$. Positive $\phi_{ij}$ pushes log-odds of churn above the baseline average; negative $\phi_{ij}$ acts as a protective buffer (e.g., a Two-Year contract or active Tech Support).
