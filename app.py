import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import requests
import platform

# --------------------------
# Google API Configuration
# --------------------------
GOOGLE_API_KEY = "AIzaSyDKvWRDWJLGRa-Te0skufDsmfLAjlIlQe4"  # Replace with your key
MODEL = "models/text-bison-001"       # Replace with model from ListModels

# --------------------------
# Chat function via REST API
# --------------------------
def ask_gemini(prompt):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateText?key={GOOGLE_API_KEY}"
        payload = {
            "prompt": {"text": prompt},
            "temperature": 0.7,
            "maxOutputTokens": 256
        }
        res = requests.post(url, json=payload).json()

        # Check for legacy "candidates" key
        if "candidates" in res and len(res["candidates"]) > 0:
            return res["candidates"][0].get("outputText", "No text returned")
        # New API format
        elif "output" in res and len(res["output"]) > 0:
            content = res["output"][0].get("content", [])
            if len(content) > 0:
                return content[0].get("text", "No text returned")
        return "No text returned from API."
    except Exception as e:
        return f"Error: {e}"

# --------------------------
# Streamlit Chat UI
# --------------------------
st.markdown("### 🤖 Ask the Privacy Bot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me about k-anonymity, classifiers, or privacy...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    answer = ask_gemini(user_input)
    st.session_state.chat_history.append(("bot", answer))

for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# --------------------------
# Random seed
# --------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------
# Preprocessor
# --------------------------
def build_preprocessor(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    low_card_cols = [c for c in cat_cols if X[c].nunique() < 30]
    high_card_cols = [c for c in cat_cols if X[c].nunique() >= 30]

    for col in high_card_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

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

# --------------------------
# Helper Functions
# --------------------------
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
    anon = generalize_qis(
        df,
        [c for c in qis if pd.api.types.is_numeric_dtype(df[c])],
        [c for c in qis if not pd.api.types.is_numeric_dtype(df[c])]
    )
    anon["QIKey"] = anon[qis].astype(str).agg("|".join, axis=1)
    sizes = anon.groupby("QIKey").size()
    keep_keys = sizes[sizes >= k].index
    return anon[anon["QIKey"].isin(keep_keys)].drop(columns=["QIKey"]).reset_index(drop=True)

def membership_inference_attack(df, target_col, clf_choice="LogisticRegression"):
    feat_cols = [c for c in df.columns if c != target_col]
    if len(feat_cols) == 0 or len(df) < 2:
        return {"mia_accuracy": 0.0, "binary": False}, None

    X = df[feat_cols]
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    if len(np.unique(y)) < 2:
        return {"mia_accuracy": 0.0, "binary": False}, None

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=7)
    pre = build_preprocessor(X_tr)
    X_tr_p = pre.fit_transform(X_tr)
    X_te_p = pre.transform(X_te)

    if clf_choice == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")

    clf.fit(X_tr_p, y_tr)
    preds = clf.predict(X_te_p)
    acc = accuracy_score(y_te, preds)
    return {"mia_accuracy": float(acc), "binary": len(np.unique(y)) == 2}, clf

# --------------------------
# Streamlit App
# --------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded with", raw_df.shape[0], "rows and", raw_df.shape[1], "columns.")
    st.write("Columns:", list(raw_df.columns))

    raw_df = clean_dataframe(raw_df)

    qis = st.multiselect("Select quasi-identifiers (QIs)", options=list(raw_df.columns))
    sensitive_col = st.selectbox("Select sensitive column", options=list(raw_df.columns))
    target_col = st.selectbox("Select target column for Membership Inference", options=list(raw_df.columns))
    k_val = st.selectbox("Select k for k-anonymity", options=[2,3,5,10], index=1)
    clf_choice = st.selectbox("Classifier for MIA", options=["LogisticRegression", "RandomForest"])

    if st.button("Run Audit"):
        with st.spinner("Running privacy audit..."):
            anon_df = apply_k_anonymity(raw_df, qis, k=k_val)
            mia_metrics_raw, _ = membership_inference_attack(raw_df, target_col, clf_choice)
            mia_metrics_anon, _ = membership_inference_attack(anon_df, target_col, clf_choice)

            st.write("**Results:**")
            st.write(f"Raw MIA Accuracy: {mia_metrics_raw['mia_accuracy']:.3f}")
            st.write(f"Anonymized MIA Accuracy: {mia_metrics_anon['mia_accuracy']:.3f}")
            st.write("**Class balance in target column (raw):**")
            st.write(raw_df[target_col].value_counts())
            st.write("**Class balance in target column (anonymized):**")
            st.write(anon_df[target_col].value_counts())

            # Visualization
            fig, ax = plt.subplots()
            sns.histplot(data=raw_df, x=sensitive_col, kde=False, ax=ax, label="Raw")
            sns.histplot(data=anon_df, x=sensitive_col, kde=False, ax=ax, label="Anonymized")
            ax.set_title("Sensitive Attribute Distribution")
            ax.legend()
            st.pyplot(fig)

            if qis:
                anon_df["QIKey"] = anon_df[qis].astype(str).agg("|".join, axis=1)
                k_sizes = anon_df.groupby("QIKey").size()
                st.write(f"Number of QI groups (after anonymization): {len(k_sizes)}")
                st.write(f"Minimum group size: {k_sizes.min() if not k_sizes.empty else 0}")

            # Distribution of first QI
            if qis:
                qi_to_plot = qis[0]
                fig2, ax2 = plt.subplots()
                sns.countplot(data=anon_df, x=qi_to_plot, ax=ax2)
                ax2.set_title(f"Distribution of QI: {qi_to_plot} (Anonymized)")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

            # Download anonymized CSV
            csv = anon_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Anonymized Data as CSV",
                data=csv,
                file_name='anonymized_data.csv',
                mime='text/csv'
            )
