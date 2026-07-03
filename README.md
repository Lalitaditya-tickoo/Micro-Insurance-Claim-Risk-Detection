# Micro Insurance Claim Risk and Anomaly Detection System

A decision-support system that scores insurance claims for fraud risk, flags
unusual (anomalous) claims, explains *why* each claim looks suspicious, and
recommends whether a human should review it — so claims teams can focus their
limited review time on the claims that matter most.

---

## 1. Problem Statement

Insurance teams cannot manually inspect every claim in detail. Most claims are
legitimate, but a minority are fraudulent, and reviewing everything by hand is
slow and expensive. This project builds a working system that automatically
flags suspicious or abnormal claims for human review, with a clear reason and a
recommended action for each claim.

## 2. Dataset / Reference Source

- **Dataset:** Insurance Fraud Detection (auto-insurance claims)
- **Source:** https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection
- **Size:** 1,000 claims, 39 columns
- **Target field:** `fraud_reported` (Y / N)
- **Fraud rate:** 24.7% (imbalanced — most claims are legitimate)

The `?` character is the dataset's missing-value token and is cleaned during
preprocessing.

## 3. Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (Random Forest,
Isolation Forest), Streamlit, Joblib.

## 4. Project Workflow

```
Raw claims  ->  Clean (? -> missing)  ->  Feature engineering  ->
Supervised classifier  +  Anomaly detector  ->  Reason codes  ->
Human-review recommendation  ->  Streamlit app / scored CSV
```

## 5. AI / ML Component

Two complementary models plus a decision layer:

1. **Supervised fraud classifier** — a `RandomForestClassifier` with
   `class_weight="balanced"` to handle the 25/75 imbalance. Outputs a fraud
   probability (0–1) for each claim.
2. **Anomaly detector** — an `IsolationForest` that flags claims that are
   statistically unusual, catching odd cases even if they don't match known
   fraud patterns.
3. **Reason codes** — transparent rules (e.g. "claim over 8x premium",
   "no police report") that explain each flag in plain English.
4. **Human-review recommendation** — combines the fraud probability and the
   anomaly flag into one action band: Auto-approve / Manual review /
   High-priority review.

The AI is not decorative: the classifier ranks claims by risk, the anomaly
detector adds a safety net, and the decision layer turns scores into an action
a claims officer can take.

## 6. How to Run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train the models (creates models/*.joblib + metadata.json)
python src/main.py

# 3. launch the demo app
streamlit run app/app.py
```

The EDA + modelling walkthrough is in `notebooks/exploration_or_modeling.ipynb`
(runs on Kaggle — attach the dataset and set `DATA_PATH`).

## 7. Demo Screenshots

See `docs/figures/` for EDA charts and add app screenshots after running
Streamlit. Key figures:
- `01_class_balance.png` — fraud vs legit split
- `02_fraud_by_severity.png` — fraud rate is far higher for "Major Damage"
- `03_claim_amount.png` — claim-amount distribution by class
- `05_correlation.png` — numeric feature correlations

## 8. Results and Insights

Held-out test set (250 claims):

| Metric | Value |
|---|---|
| ROC-AUC | 0.769 |
| Fraud precision | 0.64 |
| Fraud recall | 0.53 |
| Fraud F1 | 0.58 |

**Top fraud signals:** incident severity "Major Damage" (by far the strongest),
vehicle/property/injury claim amounts, and the claim-to-premium ratio.

**Insight:** most fraud risk concentrates in a small set of features, which is
why the reason codes are readable and useful to a human reviewer.

## 9. Limitations

- Small dataset (1,000 rows) and auto-insurance only — results may not transfer
  to other insurance lines.
- Historical labels can carry human bias; the model learns from past decisions.
- ~53% fraud recall means some fraud is still missed — this is a triage aid, not
  a replacement for investigators.
- Reason codes are rule-based and illustrative, not a full causal explanation.

## 10. Responsible Use

This system is **decision-support only**. It must not auto-reject or auto-deny
any claim. Every flagged claim is routed to a human, and the final decision
always rests with a trained claims officer. Scores should never be the sole
basis for denying a customer's claim.

## 11. Future Improvements

- Gradient boosting (XGBoost/LightGBM) and probability calibration.
- SHAP values for per-claim explanations instead of rule-based reason codes.
- Threshold tuning to trade off recall vs. reviewer workload.
- Larger, multi-line dataset and periodic retraining with drift monitoring.

## 12. Team Members

- Lalitaditya Tickoo
- Aditya Rai Chauhan
- Yugal Aggarwal
- Aagam Jain

---

### Repository Structure

```
micro_insurance_claim_risk_and_anomaly_detection_system/
├── data/insurance_claims.csv
├── notebooks/exploration_or_modeling.ipynb
├── src/
│   ├── preprocessing.py     # cleaning + feature engineering
│   ├── main.py              # training pipeline (both models)
│   └── utils.py             # reason codes + recommendation
├── app/app.py               # Streamlit demo
├── models/                  # saved models (created by main.py)
├── docs/
│   ├── project_report.pdf
│   └── figures/
├── requirements.txt
└── README.md
```
