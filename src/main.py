"""
main.py
-------
Training pipeline for the Micro Insurance Claim Risk & Anomaly
Detection System.

Steps:
  1. Load + prepare data (via preprocessing.py)
  2. Build a preprocessing + model pipeline
  3. Train a supervised fraud CLASSIFIER (Random Forest, class-balanced)
  4. Train an ANOMALY detector (Isolation Forest) on the same features
  5. Evaluate the classifier (precision / recall / ROC-AUC / confusion matrix)
  6. Save both models + column metadata to /models for the app to reuse

Run:
    python src/main.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
)

import preprocessing as pp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "insurance_claims.csv")
MODELDIR = os.path.join(ROOT, "models")
os.makedirs(MODELDIR, exist_ok=True)


def build_preprocessor(X):
    numeric, categorical = pp.get_feature_lists(X)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return pre, numeric, categorical


def main():
    print("Loading data from", DATA)
    raw = pp.load_raw(DATA)
    X, y = pp.prepare(raw, is_training=True)
    print(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Fraud rate: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pre, numeric, categorical = build_preprocessor(X)

    # ---- Supervised classifier -------------------------------------------
    # class_weight='balanced' handles the 25/75 imbalance without SMOTE,
    # keeping the pipeline simple and dependency-light.
    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n===== CLASSIFIER EVALUATION =====")
    print(classification_report(y_test, pred, target_names=["legit", "fraud"]))
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    print("ROC-AUC:", round(auc, 3))
    print("Confusion matrix [rows=true, cols=pred]:\n", cm)

    metrics = {
        "roc_auc": round(float(auc), 3),
        "precision_fraud": round(float(precision_score(y_test, pred)), 3),
        "recall_fraud": round(float(recall_score(y_test, pred)), 3),
        "f1_fraud": round(float(f1_score(y_test, pred)), 3),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "fraud_rate": round(float(y.mean()), 3),
    }

    # ---- Anomaly detector -------------------------------------------------
    # Isolation Forest is unsupervised; we fit it on the transformed training
    # features and use its score to catch 'unusual' claims regardless of label.
    iso = Pipeline(steps=[
        ("pre", pre),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=float(y.mean()),  # expect ~fraud-rate anomalies
            random_state=42,
        )),
    ])
    iso.fit(X_train)

    # ---- Feature importance (for global reason-code context) --------------
    # Map one-hot feature names back for interpretability.
    ohe = clf.named_steps["pre"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical).tolist()
    feat_names = numeric + cat_names
    importances = clf.named_steps["model"].feature_importances_
    top = sorted(zip(feat_names, importances), key=lambda t: t[1], reverse=True)[:15]
    metrics["top_features"] = [{"feature": f, "importance": round(float(i), 4)} for f, i in top]

    # ---- Save artifacts ---------------------------------------------------
    joblib.dump(clf, os.path.join(MODELDIR, "fraud_classifier.joblib"))
    joblib.dump(iso, os.path.join(MODELDIR, "anomaly_detector.joblib"))
    meta = {
        "feature_columns": X.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
        "metrics": metrics,
    }
    with open(os.path.join(MODELDIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved: fraud_classifier.joblib, anomaly_detector.joblib, metadata.json")
    print("Top fraud signals:")
    for t in top[:8]:
        print(f"   {t[0]:<35} {t[1]:.4f}")

    return metrics


if __name__ == "__main__":
    main()
