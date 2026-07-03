"""
preprocessing.py
----------------
Shared data-cleaning and feature-engineering logic for the
Micro Insurance Claim Risk & Anomaly Detection System.

Both the training pipeline (src/main.py) and the Streamlit app
(app/app.py) import from here so the SAME transformations are
applied everywhere. This prevents train/serve skew.
"""

import pandas as pd
import numpy as np

# Columns that are identifiers or leak information -> dropped before modelling.
# They uniquely tag a row and carry no generalisable fraud signal.
DROP_COLS = [
    "policy_number",       # unique id
    "insured_zip",         # high-cardinality id-like
    "incident_location",   # free-text street address
    "incident_date",       # raw date (we derive from it instead)
    "policy_bind_date",    # raw date
    "auto_model",          # very high cardinality, pairs with auto_make
    "policy_csl",          # combined-single-limit string, redundant w/ deductable
    "insured_hobbies",     # 20+ noisy categories (kept optional; dropped for stability)
    "auto_make",           # high cardinality brand
    "incident_city",       # high cardinality
    "insured_occupation",  # high cardinality
]

# The '?' string is how this dataset encodes missing categorical values.
MISSING_TOKENS = ["?", "", "NA", "nan", "None"]

TARGET = "fraud_reported"


def load_raw(path):
    """Load the raw claims file (csv or xlsx)."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean(df):
    """
    Clean a raw claims dataframe:
      - replace '?' style tokens with NaN
      - fill missing categoricals with 'unknown'
      - drop a stray unnamed/empty trailing column if present
    Returns a cleaned copy (does not mutate input).
    """
    df = df.copy()

    # Drop any fully empty 'Unnamed' column Excel sometimes appends
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Normalise missing tokens to NaN across object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(MISSING_TOKENS, np.nan)

    # Fill remaining missing categoricals with an explicit 'unknown' label
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    # Numeric missing -> median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def add_features(df):
    """
    Feature engineering. Creates a few interpretable ratio/flag features
    that are useful fraud signals and that also feed the reason-code logic.
    """
    df = df.copy()

    # Guard: some engineered features need these columns
    if "total_claim_amount" in df and "policy_annual_premium" in df:
        # A very high claim relative to the premium paid is a classic red flag
        df["claim_to_premium_ratio"] = (
            df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)
        ).fillna(0)

    if {"injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"}.issubset(df.columns):
        # Share of the claim that is injury (harder to verify -> higher fraud risk)
        denom = df["total_claim_amount"].replace(0, np.nan)
        df["injury_share"] = (df["injury_claim"] / denom).fillna(0)

    if "months_as_customer" in df:
        # Newer customers filing large claims can be riskier
        df["is_new_customer"] = (df["months_as_customer"] < 12).astype(int)

    if "witnesses" in df:
        df["no_witnesses"] = (df["witnesses"] == 0).astype(int)

    if "police_report_available" in df:
        df["no_police_report"] = (df["police_report_available"] != "YES").astype(int)

    return df


def prepare(df, is_training=True):
    """
    Full pipeline: clean -> feature engineer -> drop id/leak cols -> split X, y.
    When is_training=False (serving a single claim), y is returned as None
    and the target column is ignored if absent.
    """
    df = clean(df)
    df = add_features(df)

    y = None
    if is_training and TARGET in df.columns:
        y = (df[TARGET].astype(str).str.upper() == "Y").astype(int)

    # Drop target + id/leak columns that exist
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if TARGET in df.columns:
        to_drop.append(TARGET)
    X = df.drop(columns=to_drop)

    return X, y


def get_feature_lists(X):
    """Return (numeric_cols, categorical_cols) for a prepared feature frame."""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return numeric, categorical
