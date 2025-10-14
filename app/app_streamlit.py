# app/app_streamlit.py
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

from fairlearn.metrics import MetricFrame

# ---------- Helpers ----------

def infer_column_types(df: pd.DataFrame):
    """Return (categorical_cols, numeric_cols) by dtype heuristics."""
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return cat, num

def build_pipeline(categorical_cols, numeric_cols):
    """Build a simple, portable preprocessing + LR pipeline."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", StandardScaler(with_mean=False), numeric_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=None
    )
    return Pipeline([("prep", preprocess), ("model", clf)])

def as_str_array(x):
    """Flatten and coerce to string array (avoids dtype surprises)."""
    return np.array(x).astype(str).ravel()

def safe_selection_rate(y_true, y_pred, pos_label):
    y_pred = as_str_array(y_pred)
    mask = (y_pred == str(pos_label))
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))

def safe_tpr(y_true, y_pred, pos_label):
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    pos = (yt == str(pos_label))
    if pos.sum() == 0:
        return 0.0
    return float(((pos) & (yp == str(pos_label))).sum() / pos.sum())

def safe_fpr(y_true, y_pred, pos_label):
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    neg = (yt != str(pos_label))
    if neg.sum() == 0:
        return 0.0
    return float(((neg) & (yp == str(pos_label))).sum() / neg.sum())

def _alt_selection_bars(series: pd.Series, title: str):
    import altair as alt
    dfp = series.reset_index()
    dfp.columns = ["group", "selection_rate"]
    dfp["selection_rate_pct"] = (100 * dfp["selection_rate"]).round(2)
    bars = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("group:N", title="Group"),
        y=alt.Y("selection_rate_pct:Q", title="Selection Rate (%)"),
        tooltip=["group", "selection_rate_pct"]
    ).properties(title=title, height=240)
    labels = bars.mark_text(dy=-6).encode(text="selection_rate_pct:Q")
    return bars + labels

# ---------- Page config ----------
st.set_page_config(page_title="FairHire Analytics — Bias Detection", layout="wide")
st.title("FairHire Analytics — Bias Detection")
st.caption(
    "Upload any CSV with a **binary** target (e.g., `hired` yes/no) and pick a **sensitive attribute** "
    "(e.g., `sex`, `race`). The app trains a simple baseline model (which does **not** use the sensitive "
    "attribute) and reports performance plus fairness KPIs."
)

# ---------- Sidebar: controls ----------
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Include a binary outcome column and at least one demographic column (e.g., sex, race).")

    val_split = st.slider("Validation split", 0.10, 0.40, 0.20, step=0.01)
    seed = st.number_input("Random seed", value=42, step=1)

# ---------- Body: Upload & Preview ----------
st.subheader("Upload & Preview")
st.markdown(
    "- **What this does:** reads your CSV and shows the first few rows so you can confirm columns and values.\n"
    "- **What to check:** your target has only two values (binary), and your sensitive column has meaningful groups."
)

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.empty or len(df) < 20:
    st.error("CSV is empty or too small. Provide at least ~20 rows.")
    st.stop()

st.dataframe(df.head(), use_container_width=True)

# Choose target + positive label + sensitive column
all_cols = df.columns.tolist()
with st.expander("Configuration", expanded=True):
    st.markdown(
        "**Explain:** Choose the label column, which value counts as the **positive** outcome, "
        "and the sensitive attribute for fairness breakdowns. The sensitive attribute is **not** used by the model."
    )
    target_col = st.selectbox("Target (label) column", options=all_cols, index=0)
    pos_choices = sorted(df[target_col].astype(str).unique().tolist())
    pos_label = st.selectbox("Positive label (the 'favorable' outcome)", options=pos_choices, index=0)

    sens_col = st.selectbox(
        "Sensitive attribute (for fairness)", options=[c for c in all_cols if c != target_col]
    )

# Basic cleaning/coercion
df_work = df.copy()
df_work[target_col] = df_work[target_col].astype(str).str.strip()
df_work[sens_col] = df_work[sens_col].astype(str).str.strip()

# Separate features/label
X_full = df_work.drop(columns=[target_col])
y_full = df_work[target_col]

# Split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=val_split, random_state=int(seed), stratify=y_full if y_full.nunique() == 2 else None
)

# Capture sensitive column for fairness (from the FULL matrices)
sens_valid = X_valid_full[sens_col].copy()

# Remove sensitive column from features for modeling
X_train = X_train_full.drop(columns=[sens_col], errors="ignore")
X_valid = X_valid_full.drop(columns=[sens_col], errors="ignore")

# Infer column types on TRAIN
cat_cols, num_cols = infer_column_types(X_train)
if len(cat_cols) + len(num_cols) == 0:
    st.error("No usable feature columns after removing target and sensitive column.")
    st.stop()

# Build & train
pipe = build_pipeline(cat_cols, num_cols)
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_valid)

# ---------- Performance ----------
st.subheader("Performance")
st.markdown(
    "- **What this shows:** overall accuracy and F1 for the positive label you chose.\n"
    "- **How to read it:** accuracy is the share of correct predictions; F1 balances precision & recall for the positive class."
)

acc = accuracy_score(y_valid.astype(str), y_pred.astype(str))
f1 = f1_score(y_valid.astype(str), y_pred.astype(str), pos_label=str(pos_label))

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric(f"F1 ({pos_label})", f"{f1:.3f}")

with st.expander("Classification report", expanded=False):
    report = classification_report(
        y_valid.astype(str), y_pred.astype(str), target_names=[str(x) for x in sorted(y_valid.astype(str).unique())]
    )
    st.code(report)

# ---------- Fairness ----------
st.subheader("Fairness")
st.markdown(
    "- **What this measures:** differences in outcomes across sensitive groups (e.g., Male/Female).\n"
    "- **Selection rate:** how often the model predicts the **positive** outcome per group.\n"
    "- **TPR / FPR:** true-positive and false-positive rates by group. Ideally, similar across groups."
)

# Flatten and coerce everything to strings
y_true = as_str_array(y_valid)
y_pred_arr = as_str_array(y_pred)
sens_arr = as_str_array(sens_valid)

# Sanity: aligned lengths
if not (len(y_true) == len(y_pred_arr) == len(sens_arr)):
    st.error(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred_arr)}, sens={len(sens_arr)}")
    st.stop()

# Safe callables that close over pos_label
sel_rate_fn = lambda yt, yp: safe_selection_rate(yt, yp, pos_label)
tpr_fn = lambda yt, yp: safe_tpr(yt, yp, pos_label)
fpr_fn = lambda yt, yp: safe_fpr(yt, yp, pos_label)

# MetricFrame on strings (robust to dtype issues)
mf = MetricFrame(
    metrics={"selection_rate": sel_rate_fn, "TPR": tpr_fn, "FPR": fpr_fn},
    y_true=y_true,
    y_pred=y_pred_arr,
    sensitive_features=sens_arr
)

# Summaries
group_tbl = mf.by_group.round(3)
group_rates = mf.by_group["selection_rate"]
dp_diff = float(abs(group_rates.max() - group_rates.min()))
dp_ratio = float(group_rates.min() / group_rates.max()) if group_rates.max() > 0 else float("nan")

st.write("**Group metrics**")
st.dataframe(group_tbl, use_container_width=True)
st.markdown(
    f"- **Demographic Parity Difference:** `{dp_diff:.4f}` (↓ better) — absolute gap between highest & lowest selection rate.\n"
    f"- **Demographic Parity Ratio:** `{dp_ratio:.4f}` (→ 1 is better) — lowest / highest selection rate."
)

st.altair_chart(
    _alt_selection_bars(group_rates, f"Selection Rate by {sens_col}"),
    use_container_width=True
)

st.caption(
    "The model never uses the sensitive attribute as an input. "
    "Fairness is assessed by comparing outcome rates across groups of the sensitive attribute."
)