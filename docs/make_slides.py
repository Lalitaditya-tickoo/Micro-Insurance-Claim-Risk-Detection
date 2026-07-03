"""Generate an 8-10 slide presentation PDF (landscape)."""
import os
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "presentation.pdf")

W, H = landscape(A4)
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#0083b0")
ACCENT = colors.HexColor("#00b4db")
LIGHT = colors.HexColor("#eaf2f8")
GREY = colors.HexColor("#555555")

c = canvas.Canvas(OUT, pagesize=landscape(A4))


def bg():
    c.setFillColor(colors.HexColor("#f4f7fb"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, H - 12*mm, W, 12*mm, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, 0, W, 5*mm, fill=1, stroke=0)


def header(title, n):
    bg()
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(15*mm, H - 9*mm, title)
    c.setFont("Helvetica", 8)
    c.drawRightString(W - 12*mm, H - 9*mm, f"Slide {n}/10")


def bullets(items, x=20*mm, y=H-30*mm, lh=9*mm, size=11):
    c.setFont("Helvetica", size)
    c.setFillColor(colors.HexColor("#222222"))
    for it in items:
        c.setFillColor(BLUE)
        c.drawString(x, y, "•")
        c.setFillColor(colors.HexColor("#222222"))
        c.drawString(x + 6*mm, y, it)
        y -= lh
    return y


# ---- Slide 1: Title ----
bg()
c.setFillColor(NAVY)
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(ACCENT)
c.rect(0, 0, W, 6*mm, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 30)
c.drawCentredString(W/2, H/2 + 22*mm, "Micro Insurance Claim Risk")
c.drawCentredString(W/2, H/2 + 8*mm, "& Anomaly Detection System")
c.setFillColor(ACCENT)
c.setFont("Helvetica", 13)
c.drawCentredString(W/2, H/2 - 8*mm, "Capstone Project — Himshikhar AIML Track — Project 10")
c.setFillColor(LIGHT)
c.setFont("Helvetica", 11)
c.drawCentredString(W/2, H/2 - 24*mm,
                    "Team: Lalitaditya Tickoo  •  Aditya Rai Chauhan  •  Yugal Aggarwal  •  Aagam Jain")
c.showPage()

# ---- Slide 2: Problem & impact ----
header("Problem & Real-World Impact", 2)
bullets([
    "Insurance teams cannot manually inspect every claim in detail.",
    "Most claims are genuine; a minority are fraudulent.",
    "Manual review of everything is slow and expensive.",
    "Our system flags suspicious / abnormal claims for human review.",
    "Impact: reviewers focus time on the riskiest claims; genuine claims",
    "      move through a fast track — reducing cost and operational risk.",
])
c.showPage()

# ---- Slide 3: Dataset ----
header("Dataset", 3)
bullets([
    "Insurance Fraud Detection (Kaggle) — auto-insurance claims.",
    "1,000 claims × 39 columns.",
    "Target: fraud_reported (Y / N).",
    "Fraud rate: 24.7% — imbalanced (most claims legitimate).",
    "'?' is the dataset's missing-value token — cleaned in preprocessing.",
])
img = os.path.join(FIG, "01_class_balance.png")
if os.path.exists(img):
    c.drawImage(ImageReader(img), W-105*mm, 15*mm, width=90*mm, height=72*mm,
                preserveAspectRatio=True, mask='auto')
c.showPage()

# ---- Slide 4: Workflow ----
header("System Workflow", 4)
steps = ["Raw claims", "Clean data", "Feature\nengineering",
         "ML models", "Reason codes", "Recommendation"]
n = len(steps)
bw, gap = 38*mm, 6*mm
total = n*bw + (n-1)*gap
x = (W - total) / 2
y = H/2 - 12*mm
for i, s in enumerate(steps):
    c.setFillColor(BLUE if i % 2 == 0 else ACCENT)
    c.roundRect(x, y, bw, 24*mm, 4*mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9.5)
    lines = s.split("\n")
    yy = y + 24*mm/2 + (len(lines)-1)*5
    for ln in lines:
        c.drawCentredString(x + bw/2, yy - 3, ln)
        yy -= 11
    if i < n-1:
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(x + bw + gap/2, y + 24*mm/2 - 4, "›")
    x += bw + gap
c.setFillColor(GREY)
c.setFont("Helvetica", 10)
c.drawCentredString(W/2, y - 14*mm,
                    "Two models run in parallel: a supervised classifier + an anomaly detector.")
c.showPage()

# ---- Slide 5: AI/ML innovation ----
header("AI / ML Innovation", 5)
bullets([
    "1. Supervised classifier — Random Forest (class-weight balanced)",
    "      outputs a fraud probability for each claim.",
    "2. Anomaly detector — Isolation Forest flags unusual claims",
    "      (unsupervised safety net, catches novel patterns).",
    "3. Reason codes — transparent rules explain each flag in plain English.",
    "4. Recommendation engine — combines score + anomaly into one action:",
    "      Auto-approve / Manual review / High-priority review.",
], lh=8.5*mm)
c.showPage()

# ---- Slide 6: Prototype / demo ----
header("Prototype / Demo (Streamlit App)", 6)
bullets([
    "Interactive app scores a single claim or a whole CSV.",
    "Shows fraud probability, anomaly flag, reason codes,",
    "and a colour-coded recommendation for the reviewer.",
], y=H-28*mm)
c.setFillColor(GREY)
c.setFont("Helvetica-Oblique", 9)
c.drawString(20*mm, 20*mm, "[ Add your app screenshots here for the final submission ]")
c.setStrokeColor(colors.HexColor("#cccccc"))
c.setLineWidth(1)
c.rect(W-120*mm, 22*mm, 105*mm, 60*mm, fill=0, stroke=1)
c.setFillColor(colors.HexColor("#999999"))
c.drawCentredString(W-67*mm, 50*mm, "App screenshot placeholder")
c.showPage()

# ---- Slide 7: Results ----
header("Results (held-out test set, 250 claims)", 7)
metrics = [("ROC-AUC", "0.769"), ("Precision (fraud)", "0.64"),
           ("Recall (fraud)", "0.53"), ("F1 (fraud)", "0.58")]
bx = 18*mm
for label, val in metrics:
    c.setFillColor(NAVY)
    c.roundRect(bx, H-70*mm, 40*mm, 34*mm, 4*mm, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(bx+20*mm, H-56*mm, val)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8.5)
    c.drawCentredString(bx+20*mm, H-66*mm, label)
    bx += 45*mm
img = os.path.join(FIG, "02_fraud_by_severity.png")
if os.path.exists(img):
    c.drawImage(ImageReader(img), 18*mm, 14*mm, width=110*mm, height=62*mm,
                preserveAspectRatio=True, mask='auto')
c.setFillColor(colors.HexColor("#222222"))
c.setFont("Helvetica", 10)
c.drawString(140*mm, H-50*mm, "Top fraud signals:")
c.setFont("Helvetica", 9.5)
for i, t in enumerate(["1. Incident severity = Major Damage",
                       "2. Vehicle / property / injury amounts",
                       "3. Claim-to-premium ratio",
                       "4. Total claim amount"]):
    c.drawString(142*mm, H-58*mm - i*7*mm, t)
c.showPage()

# ---- Slide 8: Limitations & responsible use ----
header("Limitations & Responsible Use", 8)
bullets([
    "Small, single-line dataset (1,000 auto claims only).",
    "Historical labels may carry human bias.",
    "~53% fraud recall — some fraud is still missed.",
    "Reason codes are illustrative rules, not full causal proof.",
    "",
    "Responsible use: DECISION-SUPPORT ONLY.",
    "The system never auto-rejects a claim — every flag goes to a human,",
    "and the final decision always rests with a trained claims officer.",
], lh=8*mm)
c.showPage()

# ---- Slide 9: Future improvements ----
header("Future Improvements", 9)
bullets([
    "Gradient boosting (XGBoost / LightGBM) with probability calibration.",
    "SHAP values for per-claim explanations (beyond rule-based reasons).",
    "Threshold tuning to balance recall vs. reviewer workload.",
    "Larger, multi-line dataset with periodic retraining.",
    "Drift monitoring to detect changing fraud patterns over time.",
])
c.showPage()

# ---- Slide 10: Conclusion ----
bg()
c.setFillColor(NAVY)
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(ACCENT)
c.rect(0, 0, W, 6*mm, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 24)
c.drawCentredString(W/2, H/2 + 20*mm, "Conclusion")
c.setFillColor(LIGHT)
c.setFont("Helvetica", 12)
for i, line in enumerate([
    "A working, end-to-end system that scores claims for fraud risk,",
    "flags anomalies, explains why, and recommends a clear action.",
    "Backend (models) + frontend (Streamlit app) + documentation, all delivered.",
    "Built responsibly as a decision-support tool for human reviewers.",
]):
    c.drawCentredString(W/2, H/2 + 2*mm - i*8*mm, line)
c.setFillColor(ACCENT)
c.setFont("Helvetica-Bold", 11)
c.drawCentredString(W/2, 22*mm,
                    "Lalitaditya Tickoo  •  Aditya Rai Chauhan  •  Yugal Aggarwal  •  Aagam Jain")
c.showPage()

c.save()
print("wrote", OUT)
