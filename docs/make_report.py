"""Generate a full 2-page project report PDF (Himshikhar AIML Capstone)."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "project_report.pdf")

NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#0083b0")

styles = getSampleStyleSheet()
H = ParagraphStyle("H", parent=styles["Heading2"], spaceBefore=7, spaceAfter=2,
                   textColor=NAVY, fontSize=11.5, fontName="Helvetica-Bold")
body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.3, leading=13,
                      alignment=TA_JUSTIFY, spaceAfter=2)
small = ParagraphStyle("S", parent=styles["Normal"], fontSize=8.5, leading=11,
                       textColor=colors.HexColor("#444444"))
center = ParagraphStyle("C", parent=styles["Normal"], fontSize=9, leading=12,
                        alignment=TA_CENTER, textColor=colors.HexColor("#444444"))

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=13*mm, bottomMargin=12*mm,
                        leftMargin=16*mm, rightMargin=16*mm)
S = []

# ---- Header block ----
kicker = ParagraphStyle("K", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
                        textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=1)
title = ParagraphStyle("T", parent=styles["Title"], fontSize=17, textColor=NAVY,
                       spaceAfter=2, alignment=TA_CENTER)
S.append(Paragraph("HIMSHIKHAR CAPSTONE PROJECT — AIML TRACK — JULY 2026", kicker))
S.append(Paragraph("Project 10: Micro Insurance Claim Risk &amp; Anomaly Detection System", title))
S.append(Paragraph("Domain: Insurance Operations &amp; Fraud-Risk Review  |  Difficulty: Intermediate", center))
S.append(Paragraph("Team Members: Lalitaditya Tickoo, Aditya Rai Chauhan, Yugal Aggarwal, Aagam Jain", center))
S.append(Spacer(1, 3))
S.append(HRFlowable(width="100%", thickness=1.2, color=NAVY))
S.append(Spacer(1, 3))

S.append(Paragraph("1. Problem Statement &amp; Real-World Impact", H))
S.append(Paragraph(
    "Insurance companies receive a large volume of claims every day and cannot manually inspect each one in "
    "detail. The vast majority of claims are genuine, but a small fraction are fraudulent or abnormal. "
    "Reviewing every claim by hand is slow, expensive, and inconsistent, while missing fraud leads to direct "
    "financial loss. The goal of this project is to build a working system that automatically scores each claim "
    "for fraud risk, flags statistically unusual claims, explains in plain language why a claim looks suspicious, "
    "and recommends whether a human reviewer should examine it. The real-world impact is twofold: investigators "
    "spend their limited time on the claims most likely to be fraudulent, and genuine low-risk claims are cleared "
    "quickly through a fast track — reducing operational cost and processing delay while ensuring suspicious "
    "claims are still reviewed responsibly by a person.", body))

S.append(Paragraph("2. Dataset", H))
S.append(Paragraph(
    "We used the <b>Insurance Fraud Detection</b> dataset from Kaggle, containing 1,000 auto-insurance claims "
    "described by 39 columns covering policy, customer, incident, and claim-amount information. The target "
    "field is <b>fraud_reported</b> (Y/N). The dataset is imbalanced, with a fraud rate of 24.7% — most claims "
    "are legitimate, which is realistic for insurance. Missing categorical values are encoded with the '?' "
    "token; these are converted to proper missing values during preprocessing and filled with an explicit "
    "'unknown' label, while numeric gaps are filled with the column median. Identifier and free-text fields "
    "(policy number, ZIP code, incident location, raw dates, and very high-cardinality vehicle make/model) are "
    "removed before modelling to avoid noise and data leakage.", body))

S.append(Paragraph("3. System Workflow", H))
S.append(Paragraph(
    "The pipeline moves from raw data to an actionable decision: <b>raw claims &rarr; data cleaning &rarr; "
    "feature engineering &rarr; supervised classifier + anomaly detector &rarr; reason-code generation &rarr; "
    "human-review recommendation &rarr; Streamlit application or scored CSV</b>. Engineered features include "
    "the claim-to-premium ratio, the injury share of the total claim, and flags for new customers, missing "
    "witnesses, and missing police reports — all chosen because they are interpretable and directly support "
    "the reason codes shown to reviewers.", body))

S.append(Paragraph("4. AI / ML Innovation", H))
S.append(Paragraph(
    "The system combines two complementary models with a transparent decision layer. First, a <b>Random Forest "
    "classifier</b> with balanced class weights outputs a fraud probability for each claim; balancing the class "
    "weights addresses the 25/75 imbalance without discarding data. Second, an <b>Isolation Forest</b> anomaly "
    "detector provides an unsupervised safety net that flags unusual claims even when they do not match known "
    "fraud patterns. A rule-based <b>reason-code generator</b> then explains each flag in plain English (for "
    "example, 'total claim is over eight times the annual premium' or 'no police report available'), and a "
    "<b>recommendation engine</b> merges the fraud probability and the anomaly flag into a single action band: "
    "Auto-approve, Manual review, or High-priority review. The AI is meaningful rather than decorative — it "
    "ranks claims by risk, adds an independent anomaly check, and converts scores into a decision a claims "
    "officer can act on immediately.", body))

S.append(Paragraph("5. Results (held-out test set, 250 claims)", H))
data = [["Metric", "Value", "Top fraud signals (by importance)"],
        ["ROC-AUC", "0.769", "1. Incident severity = Major Damage"],
        ["Fraud precision", "0.64", "2. Vehicle / property / injury claim amounts"],
        ["Fraud recall", "0.53", "3. Claim-to-premium ratio"],
        ["Fraud F1", "0.58", "4. Total claim amount"]]
tbl = Table(data, colWidths=[30*mm, 20*mm, 82*mm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NAVY),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
S.append(tbl)
S.append(Spacer(1, 3))
S.append(Paragraph(
    "On the held-out test set the classifier achieves a ROC-AUC of 0.769 with balanced precision and recall on "
    "the minority fraud class. The confusion matrix (169 true-legit, 33 true-fraud correctly identified) shows "
    "the model catches a meaningful share of fraud while keeping false alarms manageable. Incident severity is "
    "by far the strongest signal, followed by claim-amount features and the claim-to-premium ratio, which makes "
    "the model's behaviour intuitive and easy to explain to non-technical reviewers.", body))
S.append(Spacer(1, 2))

row = []
for f in ["01_class_balance.png", "02_fraud_by_severity.png"]:
    p = os.path.join(FIG, f)
    if os.path.exists(p):
        row.append(Image(p, width=82*mm, height=60*mm))
if row:
    ft = Table([row], colWidths=[86*mm, 86*mm])
    ft.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
    S.append(ft)
    S.append(Paragraph("Figure 1 (left): class balance — fraud is the minority class. "
                       "Figure 2 (right): fraud rate is far higher for 'Major Damage' claims.", small))

S.append(Paragraph("6. Validation", H))
S.append(Paragraph(
    "The data was split with stratification to preserve the fraud ratio in both training and test sets. Because "
    "accuracy alone is misleading on imbalanced data, evaluation focuses on precision, recall, F1, and ROC-AUC "
    "for the fraud class. Beyond metrics, scenario checks confirm sensible behaviour: a large 'Major Damage' "
    "claim with no police report scores high and returns High-priority review, whereas a small trivial-damage "
    "claim with full documentation returns Auto-approve. This end-to-end check — from raw input to recommended "
    "action — demonstrates that the workflow produces reliable, explainable output.", body))

S.append(Paragraph("7. Limitations &amp; Responsible Use", H))
S.append(Paragraph(
    "The dataset is small (1,000 rows) and covers auto-insurance only, so results may not transfer directly to "
    "other insurance lines. Historical labels can embed past human bias, and a fraud recall near 53% means some "
    "fraud is still missed. The reason codes are illustrative rules rather than a full causal explanation. "
    "Accordingly, the system is strictly <b>decision-support</b>: it never auto-rejects or denies a claim. Every "
    "flagged claim is routed to a human, and the final decision always rests with a trained claims officer. "
    "Fraud scores must never be the sole basis for denying a customer's claim.", body))

S.append(Paragraph("8. Future Improvements &amp; Conclusion", H))
S.append(Paragraph(
    "Future work includes gradient-boosting models (XGBoost/LightGBM) with probability calibration, SHAP values "
    "for precise per-claim explanations, threshold tuning to balance fraud recall against reviewer workload, and "
    "a larger multi-line dataset with periodic retraining and drift monitoring. In conclusion, this project "
    "delivers a complete, working system — backend models plus an interactive Streamlit frontend, full "
    "documentation, and a public GitHub repository — that scores claims for fraud risk, flags anomalies, "
    "explains its reasoning, and recommends a clear next action, all built responsibly as an aid to human "
    "reviewers rather than a replacement for them.", body))

doc.build(S)
print("wrote", OUT)
