"""
app.py -- Streamlit demo for the Micro Insurance Claim Risk &
Anomaly Detection System.

Run:
    python -m streamlit run app/app.py

Two modes:
  1. Score a single claim entered via the form
  2. Upload a CSV of claims and batch-score them
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import joblib

# Make src/ importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import preprocessing as pp          # noqa: E402
import utils                        # noqa: E402

MODELDIR = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data", "insurance_claims.csv")

st.set_page_config(page_title="Insurance Claim Risk Detector",
                   page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
/* page background */
.stApp {
    background: linear-gradient(160deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
}
/* main container card */
.block-container { padding-top: 2.2rem; max-width: 1150px; }

/* headings + text light on dark */
h1, h2, h3, h4, p, label, .stMarkdown { color: #eaf2f8 !important; }

/* hero */
.hero-title {
    font-size: 2.5rem; font-weight: 800; color: #ffffff;
    margin-bottom: .2rem; letter-spacing: -0.5px;
}
.hero-sub { color: #9fc0d4; font-size: 1rem; margin-bottom: 1.4rem; }

/* input cards */
.stNumberInput, .stSelectbox { margin-bottom: .3rem; }
div[data-baseweb="input"] input, div[data-baseweb="select"] > div {
    background: #ffffff !important; border-radius: 10px !important;
}

/* the assess button */
.stButton > button {
    background: linear-gradient(90deg,#00b4db,#0083b0);
    color: #fff; font-weight: 700; border: none; border-radius: 10px;
    padding: .6rem 1.6rem; font-size: 1.05rem;
    box-shadow: 0 4px 14px rgba(0,131,176,.4);
}
.stButton > button:hover { filter: brightness(1.08); }

/* result cards */
.result-card {
    border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: .4rem;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
}
.big-prob { font-size: 3rem; font-weight: 800; line-height: 1; }
.badge {
    display: inline-block; padding: .55rem 1.1rem; border-radius: 999px;
    font-weight: 800; font-size: 1.05rem; letter-spacing:.3px;
}
.badge-red   { background:#ff4757; color:#fff; }
.badge-amber { background:#ffa502; color:#1a1a1a; }
.badge-green { background:#2ed573; color:#0b2b17; }
.reason-item {
    background: rgba(255,255,255,0.08); border-left: 4px solid #00b4db;
    padding:.55rem .8rem; border-radius:8px; margin:.35rem 0; color:#eaf2f8;
}
/* metric tiles */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07); border-radius:12px; padding:.6rem .9rem;
}
/* tabs */
button[data-baseweb="tab"] { font-weight:700; font-size:1rem; }
/* expander */
details { background: rgba(255,255,255,0.05); border-radius:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(MODELDIR, "fraud_classifier.joblib"))
    iso = joblib.load(os.path.join(MODELDIR, "anomaly_detector.joblib"))
    with open(os.path.join(MODELDIR, "metadata.json")) as f:
        meta = json.load(f)
    return clf, iso, meta


@st.cache_data
def load_sample():
    return pp.load_raw(DATA)


def gauge_color(prob):
    if prob >= 0.60:
        return "#ff4757"
    if prob >= 0.30:
        return "#ffa502"
    return "#2ed573"


def render_result(result):
    prob = result["fraud_probability"]
    action = result["recommendation"]
    col = gauge_color(prob)

    badge_class = {"AUTO-APPROVE (LOW RISK)": "badge-green",
                   "MANUAL REVIEW": "badge-amber",
                   "HIGH-PRIORITY REVIEW": "badge-red"}.get(action, "badge-amber")

    st.markdown("### Assessment result")
    c1, c2, c3 = st.columns([1.1, 1, 1.3])

    with c1:
        st.markdown(
            f"<div class='result-card'>"
            f"<div style='color:#9fc0d4;font-size:.9rem;'>FRAUD PROBABILITY</div>"
            f"<div class='big-prob' style='color:{col};'>{prob*100:.0f}%</div>"
            f"<div style='height:10px;background:rgba(255,255,255,.15);border-radius:6px;margin-top:.6rem;'>"
            f"<div style='width:{prob*100:.0f}%;height:10px;background:{col};border-radius:6px;'></div>"
            f"</div></div>", unsafe_allow_html=True)

    with c2:
        anom_txt = "YES ⚠️" if result["is_anomaly"] else "No"
        anom_col = "#ffa502" if result["is_anomaly"] else "#2ed573"
        st.markdown(
            f"<div class='result-card'>"
            f"<div style='color:#9fc0d4;font-size:.9rem;'>ANOMALY FLAG</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{anom_col};'>{anom_txt}</div>"
            f"<div style='color:#9fc0d4;font-size:.8rem;margin-top:.5rem;'>Unusual vs typical claims</div>"
            f"</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"<div class='result-card'>"
            f"<div style='color:#9fc0d4;font-size:.9rem;margin-bottom:.5rem;'>RECOMMENDATION</div>"
            f"<span class='badge {badge_class}'>{action}</span>"
            f"<div style='color:#cfe0ea;font-size:.85rem;margin-top:.7rem;'>{result['note']}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("#### Why this claim was flagged")
    for r in result["reason_codes"]:
        st.markdown(f"<div class='reason-item'>• {r}</div>", unsafe_allow_html=True)


def main():
    st.markdown("<div class='hero-title'>🛡️ Micro Insurance Claim Risk &amp; Anomaly Detection</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Flags suspicious claims for human review. "
                "Decision-support only — final decisions stay with a human claims officer.</div>",
                unsafe_allow_html=True)

    try:
        clf, iso, meta = load_models()
    except FileNotFoundError:
        st.error("Models not found. Run `python src/main.py` first to train and save them.")
        st.stop()

    with st.expander("📊  Model performance (held-out test set)"):
        m = meta["metrics"]
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", m["roc_auc"])
        b.metric("Fraud recall", m["recall_fraud"])
        c.metric("Fraud precision", m["precision_fraud"])
        d.metric("Fraud F1", m["f1_fraud"])

    tab1, tab2 = st.tabs(["🔍  Score a single claim", "📁  Batch score a CSV"])
    sample = load_sample()

    # ---------------- Single claim ----------------
    with tab1:
        st.markdown("Pre-fill from a sample row, then adjust values:")
        idx = st.number_input("Sample row #", 0, len(sample) - 1, 0)
        base = sample.iloc[int(idx)].to_dict()

        col1, col2, col3 = st.columns(3)
        with col1:
            base["months_as_customer"] = st.number_input(
                "Months as customer", 0, 600, int(base.get("months_as_customer", 100)))
            base["total_claim_amount"] = st.number_input(
                "Total claim amount", 0, 200000, int(base.get("total_claim_amount", 50000)))
            base["injury_claim"] = st.number_input(
                "Injury claim", 0, 100000, int(base.get("injury_claim", 5000)))
        with col2:
            base["policy_annual_premium"] = st.number_input(
                "Annual premium", 0.0, 5000.0, float(base.get("policy_annual_premium", 1200.0)))
            base["incident_severity"] = st.selectbox(
                "Incident severity",
                ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"], index=2)
            base["witnesses"] = st.number_input("Witnesses", 0, 10, int(base.get("witnesses", 1)))
        with col3:
            base["police_report_available"] = st.selectbox(
                "Police report available", ["YES", "NO", "unknown"], index=1)
            base["property_damage"] = st.selectbox(
                "Property damage", ["YES", "NO", "unknown"], index=0)
            base["incident_type"] = st.selectbox(
                "Incident type",
                ["Single Vehicle Collision", "Multi-vehicle Collision",
                 "Vehicle Theft", "Parked Car"], index=0)

        st.write("")
        if st.button("⚡  Assess claim", type="primary"):
            row = pd.DataFrame([base])
            X, _ = pp.prepare(row, is_training=False)
            X = X.reindex(columns=meta["feature_columns"], fill_value=0)
            result = utils.full_assessment(base, X, clf, iso)
            render_result(result)

    # ---------------- Batch ----------------
    with tab2:
        st.markdown("Upload a CSV with the same columns as the training data "
                    "(e.g. `data/insurance_claims.csv`).")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            X, _ = pp.prepare(df, is_training=False)
            X = X.reindex(columns=meta["feature_columns"], fill_value=0)
            probs = clf.predict_proba(X)[:, 1]
            anom = (iso.predict(X) == -1)
            out = df.copy()
            out["fraud_probability"] = probs.round(3)
            out["is_anomaly"] = anom
            out["recommendation"] = [
                utils.recommend_action(p, a)["action"] for p, a in zip(probs, anom)]
            out = out.sort_values("fraud_probability", ascending=False)

            hi = int((out["recommendation"] == "HIGH-PRIORITY REVIEW").sum())
            mid = int((out["recommendation"] == "MANUAL REVIEW").sum())
            lo = int((out["recommendation"] == "AUTO-APPROVE (LOW RISK)").sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("🔴 High-priority", hi)
            k2.metric("🟠 Manual review", mid)
            k3.metric("🟢 Auto-approve", lo)

            st.dataframe(
                out[["fraud_probability", "is_anomaly", "recommendation"]].head(50),
                use_container_width=True)
            st.download_button("⬇️  Download scored claims",
                               out.to_csv(index=False).encode(),
                               "scored_claims.csv", "text/csv")


if __name__ == "__main__":
    main()
