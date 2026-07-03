# Project Report — Micro Insurance Claim Risk & Anomaly Detection System

## 1. Problem & Stakeholders

Insurance teams cannot manually inspect every claim in detail. Most claims are
legitimate; a minority are fraudulent. Reviewing everything by hand is slow and
expensive. The stakeholders are **claims officers and fraud investigators**, who
need to know which claims to look at first. The system takes claim data and
produces, for each claim, a fraud risk score, an anomaly flag, plain-English
reasons, and a recommended action.

## 2. Data

- **Source:** Insurance Fraud Detection, Kaggle (auto-insurance claims).
- **Shape:** 1,000 rows × 39 columns.
- **Target:** `fraud_reported` (Y/N). Fraud rate 24.7%.
- **Cleaning:** the `?` token is converted to missing; categorical gaps filled
  with `unknown`; numeric gaps filled with the median. Identifier and free-text
  columns (policy number, zip, location, raw dates, high-cardinality make/model)
  are dropped to avoid noise and leakage.

## 3. Method

**Feature engineering.** Added interpretable signals: claim-to-premium ratio,
injury share of the claim, new-customer flag, no-witness flag, no-police-report
flag.

**Model 1 — supervised classifier.** `RandomForestClassifier` with
`class_weight="balanced"` inside a scikit-learn `Pipeline` that scales numeric
features and one-hot encodes categoricals. Outputs a fraud probability.

**Model 2 — anomaly detector.** `IsolationForest` with contamination set to the
fraud rate, giving an unsupervised flag for unusual claims.

**Decision layer.** Rule-based reason codes plus a recommendation function that
maps probability + anomaly into Auto-approve / Manual review / High-priority
review.

## 4. Results

Held-out test set (250 claims):

| Metric | Value |
|---|---|
| ROC-AUC | 0.769 |
| Fraud precision | 0.64 |
| Fraud recall | 0.53 |
| Fraud F1 | 0.58 |

Confusion matrix (rows = true, cols = predicted): [[169, 19], [29, 33]].

**Top fraud signals:** incident severity "Major Damage" (dominant), vehicle /
property / injury claim amounts, and claim-to-premium ratio.

## 5. Validation

- Stratified train/test split preserves the fraud ratio.
- Metrics chosen for imbalance (precision, recall, F1, ROC-AUC) rather than
  accuracy alone.
- Manual scenario checks: a large "Major Damage" claim with no police report
  scores high and returns High-priority review; a small trivial claim returns
  Auto-approve. Both behave as expected.

## 6. Limitations

Small, single-line (auto only) dataset; historical labels may embed bias;
recall of ~53% means some fraud is missed; reason codes are illustrative rules,
not full causal explanations.

## 7. Responsible Use

Decision-support only. The system never auto-rejects a claim. Every flag is
routed to a human, and final decisions rest with a trained claims officer.

## 8. Future Work

Gradient boosting with calibration, SHAP per-claim explanations, threshold
tuning for reviewer workload, and a larger multi-line dataset with retraining
and drift monitoring.
