import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    true_positive_rate,
    false_positive_rate,
)

import altair as alt

st.set_page_config(page_title="FairHire Analytics — Prototype", layout="wide")

st.title("FairHire Analytics — Bias Detection Prototype")

st.markdown(
"""
Upload any **CSV** with a binary target (e.g. `hired` = yes/no) and pick a **sensitive attribute**
(e.g. `sex`, `race`). The app trains a simple baseline model and reports performance + fairness KPIs.
"""
)

# --------- 1) Data upload ----------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. Include a binary target column, and at least one candidate sensitive column.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty or df.shape[1] < 2:
    st.error("CSV looks empty or has too few columns.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())

# --------- 2) Column selection ----------
with st.sidebar:
    st.header("Settings")

    target_col = st.selectbox("Target (label) column", options=df.columns.tolist())
    candidates = [c for c in df.columns if c != target_col]
    if not candidates:
        st.error("No columns left after selecting the target.")
        st.stop()

    sensitive_col = st.selectbox("Sensitive attribute", options=candidates)

    # Positive class selection
    # We infer classes from the column values
    classes = df[target_col].dropna().unique().tolist()
    if len(classes) < 2:
        st.error("Target must have at least two distinct values.")
        st.stop()

    # default to the 'positive-looking' class if present
    default_pos = ">50K" if ">50K" in classes else ("yes" if "yes" in [str(x).lower() for x in classes] else classes[0])
    pos_label = st.selectbox("Positive label (the 'favorable' outcome)", options=classes, index=classes.index(default_pos) if default_pos in classes else 0)

    test_size = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

# --------- 3) Basic cleaning ----------
df = df.dropna().copy()

# Ensure target is string labels
df[target_col] = df[target_col].astype(str).str.strip()

# Split
X = df.drop(columns=[target_col])
y = df[target_col]

try:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
except ValueError:
    # Fallback if stratify fails (e.g., extreme class imbalance)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

# Column types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

# --------- 4) Pipeline (no pickles) ----------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
preprocess = ColumnTransformer([
    ("cat", ohe, categorical_cols),
    ("num", "passthrough", numeric_cols)
])

clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

pipe = Pipeline([
    ("prep", preprocess),
    ("model", clf)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)

# --------- 5) Performance ----------
acc = accuracy_score(y_valid, y_pred)
f1  = f1_score(y_valid, y_pred, pos_label=pos_label)

st.subheader("Performance")
c1, c2 = st.columns(2)
with c1:
    st.metric("Accuracy", f"{acc:.3f}")
with c2:
    st.metric(f"F1 ({pos_label})", f"{f1:.3f}")

with st.expander("Classification report"):
    st.text(classification_report(y_valid, y_pred))

# --------- 6) Fairness (Demographic Parity) ----------
if sensitive_col not in X_valid.columns:
    st.error(f"Sensitive column '{sensitive_col}' not found in features.")
    st.stop()

# Selection rates by group
mf = MetricFrame(
    metrics={"selection_rate": selection_rate},
    y_true=y_valid,
    y_pred=y_pred,
    sensitive_features=X_valid[sensitive_col]
)
dp_diff = demographic_parity_difference(
    y_valid, y_pred, sensitive_features=X_valid[sensitive_col]
)
dp_ratio = demographic_parity_ratio(
    y_valid, y_pred, sensitive_features=X_valid[sensitive_col]
)

st.subheader("Fairness — Demographic Parity")
c3, c4 = st.columns(2)
with c3:
    st.metric("DP Difference (↓ better)", f"{dp_diff:.4f}")
with c4:
    st.metric("DP Ratio (→ 1.0 better)", f"{dp_ratio:.4f}")

st.caption("DP Difference = max(group selection) − min(group selection). DP Ratio = min/max selection rate across groups.")

st.write("**Selection rate (by group)**")
by_group = mf.by_group.reset_index().rename(columns={"index": "group"})
by_group["selection_rate_pct"] = (by_group["selection_rate"] * 100).round(2)

bar = alt.Chart(by_group).mark_bar().encode(
    x=alt.X("group:N", title="Group"),
    y=alt.Y("selection_rate_pct:Q", title="Selection Rate (%)"),
    tooltip=list(by_group.columns)
).properties(height=260)

labels = bar.mark_text(dy=-6).encode(text="selection_rate_pct:Q")
st.altair_chart(bar + labels, use_container_width=True)

# --------- 7) Error by group (helpful KPI) ----------
err = (y_valid != y_pred).astype(int)
err_mf = MetricFrame(
    metrics={"error_rate": lambda yt, yp: np.mean(np.array(yt) != np.array(yp))},
    y_true=y_valid,
    y_pred=y_pred,
    sensitive_features=X_valid[sensitive_col]
)

st.write("**Error rate (by group)**")
st.dataframe(err_mf.by_group.to_frame().rename(columns={0: "error_rate"}))

# --------- 8) Feature importance (optional, model-agnostic) ----------
st.subheader("Feature influence (Permutation Importance)")
try:
    # Use balanced_accuracy which is robust for imbalanced labels
    perm = permutation_importance(
        pipe, X_valid, y_valid, n_repeats=5, random_state=42, scoring="balanced_accuracy"
    )
    # Get feature names from the ColumnTransformer
    prep = pipe.named_steps["prep"]
    ohe_names = []
    if "cat" in prep.named_transformers_:
        ohe = prep.named_transformers_["cat"]
        try:
            ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        except Exception:
            ohe_names = []
    feature_names = ohe_names + numeric_cols
    # Align lengths safely
    k = min(len(feature_names), len(perm.importances_mean))
    feat_df = pd.DataFrame({
        "feature": feature_names[:k],
        "importance": perm.importances_mean[:k]
    }).sort_values("importance", ascending=False).head(15)

    imp_chart = alt.Chart(feat_df).mark_bar().encode(
        x=alt.X("importance:Q", title="Mean importance (perm.)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")]
    ).properties(height=400)
    st.altair_chart(imp_chart, use_container_width=True)
except Exception as e:
    st.info(f"Permutation importance not shown (reason: {e})")

st.success("Done. You can now export screenshots for your report.")