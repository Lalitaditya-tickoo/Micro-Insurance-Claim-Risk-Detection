"""
utils.py
--------
Decision-support helpers that turn raw model scores into something a
claims officer can act on:

  * score_claim()         -> fraud probability + anomaly flag for one claim
  * generate_reason_codes -> plain-English reasons a claim looks risky
  * recommend_action()    -> Auto-approve / Review / High-priority review

These functions are intentionally rule-transparent so the output can be
explained to a human reviewer (responsible-use requirement).
"""

import numpy as np
import pandas as pd


# Thresholds for turning a probability into an action band.
# Tuned to be conservative: when unsure, send to a human.
LOW_RISK = 0.30
HIGH_RISK = 0.60


def score_claim(claim_df, classifier, anomaly_detector):
    """
    claim_df : single-row DataFrame already run through preprocessing.prepare(is_training=False)
    Returns dict with fraud_probability (0-1) and is_anomaly (bool).
    """
    fraud_prob = float(classifier.predict_proba(claim_df)[:, 1][0])
    # IsolationForest.predict -> -1 anomaly, 1 normal
    is_anomaly = int(anomaly_detector.predict(claim_df)[0]) == -1
    return {"fraud_probability": fraud_prob, "is_anomaly": is_anomaly}


def generate_reason_codes(raw_claim: dict):
    """
    Rule-based, human-readable reasons derived from the raw claim fields.
    These complement the model score: the model gives the 'how likely',
    these give the 'why it looks off'. Returns a list of short strings.
    """
    reasons = []

    def g(k, default=None):
        return raw_claim.get(k, default)

    # Claim size vs premium
    try:
        prem = float(g("policy_annual_premium", 0) or 0)
        claim = float(g("total_claim_amount", 0) or 0)
        if prem > 0 and claim / prem > 8:
            reasons.append(
                f"Total claim ({claim:,.0f}) is over 8x the annual premium ({prem:,.0f})."
            )
    except (TypeError, ValueError):
        pass

    # Severity
    if str(g("incident_severity", "")).lower() == "major damage":
        reasons.append("Incident marked as 'Major Damage' (higher payout category).")

    # Missing supporting evidence
    if str(g("police_report_available", "")).upper() != "YES":
        reasons.append("No police report available for the incident.")
    try:
        if int(g("witnesses", 1) or 0) == 0:
            reasons.append("No witnesses recorded.")
    except (TypeError, ValueError):
        pass

    # New customer, large claim
    try:
        if int(g("months_as_customer", 999) or 999) < 12:
            reasons.append("Policyholder is a relatively new customer (<12 months).")
    except (TypeError, ValueError):
        pass

    # High injury share (injury claims are harder to verify)
    try:
        claim = float(g("total_claim_amount", 0) or 0)
        injury = float(g("injury_claim", 0) or 0)
        if claim > 0 and injury / claim > 0.5:
            reasons.append("Injury portion is over half of the total claim amount.")
    except (TypeError, ValueError):
        pass

    # Property damage claimed but unverified
    if str(g("property_damage", "")).upper() == "YES" and \
       str(g("police_report_available", "")).upper() != "YES":
        reasons.append("Property damage claimed but not backed by a police report.")

    if not reasons:
        reasons.append("No individual red-flag rules triggered.")

    return reasons


def recommend_action(fraud_probability, is_anomaly):
    """
    Combine the classifier probability and anomaly flag into a single
    recommended action for the claims team.
    """
    # An anomaly bumps a borderline claim up a level.
    if fraud_probability >= HIGH_RISK or (is_anomaly and fraud_probability >= LOW_RISK):
        band = "HIGH-PRIORITY REVIEW"
        note = "Strong fraud signal. Route to a senior investigator before payout."
    elif fraud_probability >= LOW_RISK or is_anomaly:
        band = "MANUAL REVIEW"
        note = "Some risk indicators present. A human should verify before approval."
    else:
        band = "AUTO-APPROVE (LOW RISK)"
        note = "No significant risk indicators. Safe for standard fast-track processing."

    return {"action": band, "note": note}


def full_assessment(raw_claim: dict, prepared_row, classifier, anomaly_detector):
    """
    Convenience wrapper: score + reasons + recommendation in one object.
    raw_claim     : original field dict (for reason codes)
    prepared_row  : single-row DataFrame after preprocessing.prepare()
    """
    scored = score_claim(prepared_row, classifier, anomaly_detector)
    reasons = generate_reason_codes(raw_claim)
    rec = recommend_action(scored["fraud_probability"], scored["is_anomaly"])
    return {
        "fraud_probability": round(scored["fraud_probability"], 3),
        "is_anomaly": scored["is_anomaly"],
        "reason_codes": reasons,
        "recommendation": rec["action"],
        "note": rec["note"],
    }
