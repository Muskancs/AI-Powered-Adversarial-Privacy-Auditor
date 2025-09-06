import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Check for required dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    st.error("Required dependencies (matplotlib, seaborn, scikit-learn) are missing. Please install them using 'pip install -r requirements.txt' with the following content:\n\nnumpy\npandas\nstreamlit\nmatplotlib\nseaborn\nscikit-learn")
    st.stop()

# Set random seed
np.random.seed(42)

# --- Functions ---
def clean_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).fillna("missing")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)  # Convert to 0 for non-numeric
        else:
            df[col] = df[col].astype(str).fillna("missing")  # Handle other types as strings
    return df

def generalize_qis(df, num_qis, cat_qis):
    gen_df = df.copy()
    for col in num_qis:
        try:
            gen_df[col] = pd.qcut(pd.to_numeric(gen_df[col], errors="coerce"), q=4, duplicates='drop').astype(str)
        except Exception:
            gen_df[col] = gen_df[col].astype(str).fillna("Other")
    for col in cat_qis:
        s = gen_df[col].astype(str)
        freq = s.value_counts()
        rare = freq[freq < 5].index
        gen_df[col] = s.replace(rare, "Other")
    return gen_df

def apply_k_anonymity(df, qis, k=3):
    if len(qis) == 0:
        return df.copy()
    df["QIKey"] = df[qis].astype(str).agg("|".join, axis=1)
    sizes = df.groupby("QIKey").size()
    keep_keys = sizes[sizes >= k].index
    return df[df["QIKey"].isin(keep_keys)].drop(columns=["QIKey"]).reset_index(drop=True)

def membership_inference_attack(df, target_col):
    feat_cols = [c for c in df.columns if c != target_col]
    if len(feat_cols) == 0 or len(df) < 2:
        return {"mia_accuracy": 0.0, "binary": False}, None
    X = df[feat_cols]
    le = LabelEncoder()
    try:
        y = le.fit_transform(df[target_col])
    except ValueError:
        st.error(f"Target column '{target_col}' contains non-encodable values. Please select a different column.")
        return {"mia_accuracy": 0.0, "binary": False}, None
    if len(np.unique(y)) < 2:
        st.warning(f"Target column '{target_col}' has only one unique value. Skipping MIA.")
        return {"mia_accuracy": 0.0, "binary": False}, None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=7)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    try:
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
    except Exception as e:
        st.error(f"Error in MIA training: {str(e)}. Check data compatibility.")
        return {"mia_accuracy": 0.0, "binary": False}, None
    return {"mia_accuracy": float(acc), "binary": len(np.unique(y)) == 2}, clf

# --- Streamlit App ---
st.title("AI Adversarial Privacy Auditor")

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded with", raw_df.shape[0], "rows and", raw_df.shape[1], "columns.")
    st.write("Columns:", list(raw_df.columns))

    raw_df = clean_dataframe(raw_df)
    
    qis = st.multiselect("Select quasi-identifiers (QIs)", options=list(raw_df.columns))
    sensitive_col = st.selectbox("Select sensitive column", options=list(raw_df.columns))
    target_col = st.selectbox("Select target column for Membership Inference", options=list(raw_df.columns))

    if st.button("Run Audit"):
        with st.spinner("Running privacy audit..."):
            # Generalize QIs and apply k-anonymity
            cat_qis = [q for q in qis if raw_df[q].dtype == "object" or raw_df[q].nunique() < 20]
            num_qis = [q for q in qis if q not in cat_qis]
            anon_df = apply_k_anonymity(generalize_qis(raw_df, num_qis, cat_qis), qis, k=3)
            
            # Run MIA on raw and anonymized data
            mia_metrics_raw, _ = membership_inference_attack(raw_df, target_col)
            mia_metrics_anon, _ = membership_inference_attack(anon_df, target_col)
            
            st.write("**Results:**")
            st.write(f"Raw MIA Accuracy: {mia_metrics_raw['mia_accuracy']:.3f}")
            st.write(f"Anonymized MIA Accuracy: {mia_metrics_anon['mia_accuracy']:.3f}")
            
            if HAS_DEPS and not anon_df.empty:
                fig, ax = plt.subplots()
                sns.histplot(data=raw_df, x=sensitive_col, kde=False, ax=ax, label="Raw", color="blue")
                sns.histplot(data=anon_df, x=sensitive_col, kde=False, ax=ax, label="Anonymized", color="orange")
                ax.set_title("Sensitive Attribute Distribution")
                ax.legend()
                st.pyplot(fig)
