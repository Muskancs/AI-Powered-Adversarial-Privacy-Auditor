import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import asyncio
import platform

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def build_preprocessor(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    # Identify numeric columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    # Identify all non-numeric columns (object or category)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    # Split by cardinality
    low_card_cols = [c for c in cat_cols if X[c].nunique() < 30]
    high_card_cols = [c for c in cat_cols if X[c].nunique() >= 30]
    # Label encode high-cardinality columns
    for col in high_card_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    # One-hot encode low-cardinality columns
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", ohe, low_card_cols)
        ],
        remainder="drop"
    )
    return pre

# --- Functions from your code (simplified for Streamlit) ---
def clean_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).fillna("missing")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def generalize_qis(df, num_qis, cat_qis):
    gen_df = df.copy()
    for col in num_qis:
        gen_df[col] = pd.qcut(gen_df[col], q=4, duplicates='drop').astype(str)
    for col in cat_qis:
        s = gen_df[col].astype(str)
        freq = s.value_counts()
        rare = freq[freq < 5].index
        gen_df[col] = s.replace(rare, "Other")
    return gen_df

def apply_k_anonymity(df, qis, k=3):
    if len(qis) == 0:
        return df.copy()
    anon = generalize_qis(df, [c for c in qis if pd.api.types.is_numeric_dtype(df[c])], [c for c in qis if not pd.api.types.is_numeric_dtype(df[c])])
    anon["QIKey"] = anon[qis].astype(str).agg("|".join, axis=1)
    sizes = anon.groupby("QIKey").size()
    keep_keys = sizes[sizes >= k].index
    return anon[anon["QIKey"].isin(keep_keys)].drop(columns=["QIKey"]).reset_index(drop=True)

def membership_inference_attack(df, target_col):
    feat_cols = [c for c in df.columns if c != target_col]
    if len(feat_cols) == 0 or len(df) < 2:
        return {"mia_accuracy": 0.0, "binary": False}, None
    X = df[feat_cols]
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    if len(np.unique(y)) < 2:
        return {"mia_accuracy": 0.0, "binary": False}, None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=7)
    # --- FIX: Use the preprocessor here ---
    pre = build_preprocessor(X_tr)
    X_tr_p = pre.fit_transform(X_tr)
    X_te_p = pre.transform(X_te)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_tr_p, y_tr)
    preds = clf.predict(X_te_p)
    acc = accuracy_score(y_te, preds)
    return {"mia_accuracy": float(acc), "binary": len(np.unique(y)) == 2}, clf

# --- Streamlit App ---
async def main():
    st.title("AI Adversarial Privacy Auditor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Dataset loaded with", raw_df.shape[0], "rows and", raw_df.shape[1], "columns.")
        st.write("Columns:", list(raw_df.columns))

        # Clean data
        raw_df = clean_dataframe(raw_df)
        
        # User input for QIs, sensitive column, and target column
        qis = st.multiselect("Select quasi-identifiers (QIs)", options=list(raw_df.columns))
        sensitive_col = st.selectbox("Select sensitive column", options=list(raw_df.columns))
        target_col = st.selectbox("Select target column for Membership Inference", options=list(raw_df.columns))

        if st.button("Run Audit"):
            with st.spinner("Running privacy audit..."):
                # Apply k-anonymity
                anon_df = apply_k_anonymity(raw_df, qis, k=3)
                
                # Run MIA on raw and anonymized data
                mia_metrics_raw, _ = membership_inference_attack(raw_df, target_col)
                mia_metrics_anon, _ = membership_inference_attack(anon_df, target_col)
                
                # Display results
                st.write("**Results:**")
                st.write(f"Raw MIA Accuracy: {mia_metrics_raw['mia_accuracy']:.3f}")
                st.write(f"Anonymized MIA Accuracy: {mia_metrics_anon['mia_accuracy']:.3f}")
                
                # Visualization
                fig, ax = plt.subplots()
                sns.histplot(data=raw_df, x=sensitive_col, kde=False, ax=ax, label="Raw")
                sns.histplot(data=anon_df, x=sensitive_col, kde=False, ax=ax, label="Anonymized")
                ax.set_title("Sensitive Attribute Distribution")
                ax.legend()
                st.pyplot(fig)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
