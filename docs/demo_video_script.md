# Demo Video Script (max 3 minutes)

The capstone letter caps the demo at **3 minutes**. Keep it tight — rehearse once.

---

**[0:00–0:20] Problem & who it's for**
> "Insurance teams get thousands of claims but can't review each one by hand.
> Most are genuine; a few are fraud. Our system flags the suspicious ones for a
> human reviewer, so investigators spend time where it matters."

**[0:20–0:45] Data**
> "We used the Insurance Fraud Detection dataset from Kaggle — 1,000 auto
> claims, 39 fields, with a fraud/legit label. About a quarter are fraud, so
> it's imbalanced." *(Show the class-balance chart.)*

**[0:45–1:15] How it works**
> "We clean the data, engineer a few signals like claim-to-premium ratio, then
> run two models: a Random Forest that scores fraud probability, and an
> Isolation Forest that flags unusual claims. A rules layer turns each score
> into plain-English reasons and a recommended action." *(Show the workflow
> diagram or the EDA fraud-by-severity chart.)*

**[1:15–2:10] Live demo (the important part)**
> Open the Streamlit app. Pick a sample claim, set severity to "Major Damage",
> a large claim amount, and no police report. Click **Assess claim**.
> "You can see the fraud probability, the anomaly flag, the reason codes —
> claim over 8x premium, no police report — and the recommendation:
> High-priority review." Then pick a small, low-severity claim and show it comes
> back **Auto-approve (low risk)**. Optionally show the batch-CSV tab ranking
> claims by risk.

**[2:10–2:35] Results**
> "On held-out data we get ROC-AUC around 0.77, with balanced precision and
> recall on the fraud class. The strongest signal is incident severity, followed
> by claim amounts and the claim-to-premium ratio."

**[2:35–3:00] Limitations & responsible use**
> "It's a small auto-only dataset, and it misses some fraud, so it's
> decision-support — it never auto-rejects a claim. Every flag goes to a human,
> who makes the final call. Future work: gradient boosting, SHAP explanations,
> and a larger dataset. Thanks for watching."

---

### Recording tips
- Train the models first (`python src/main.py`) so the app loads instantly.
- Have the app already running before you hit record.
- Record at 1080p; show the screen clearly; speak slowly.
- Upload to Zoom/Drive/YouTube as **"Anyone with the link can view."**
