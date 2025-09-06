# ...existing imports...

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ...existing code...

def build_preprocessor(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
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
    pre = build_preprocessor(X_tr)
    X_tr_p = pre.fit_transform(X_tr)
    X_te_p = pre.transform(X_te)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_tr_p, y_tr)
    preds = clf.predict(X_te_p)
    acc = accuracy_score(y_te, preds)
    return {"mia_accuracy": float(acc), "binary": len(np.unique(y)) == 2}, clf

# ...rest of your Streamlit code remains unchanged...
