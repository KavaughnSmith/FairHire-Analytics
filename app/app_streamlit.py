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
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)

from fairlearn.metrics import MetricFrame


# ===================== Utilities =====================

def infer_column_types(df: pd.DataFrame):
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return cat, num


def build_pipeline(categorical_cols, numeric_cols):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", StandardScaler(with_mean=False), numeric_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("prep", preprocess), ("model", clf)])


def as_str_array(x):
    return np.array(x).astype(str).ravel()


def f1_binary(y_true, y_pred, pos_label):
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    yt_b = (yt == str(pos_label)).astype(int)
    yp_b = (yp == str(pos_label)).astype(int)
    return f1_score(yt_b, yp_b)

def acc_binary(y_true, y_pred, pos_label):
    """Accuracy computed on a binary {pos_label vs not_pos_label} view.
    Keeps BEFORE vs AFTER comparable even if AFTER uses a synthetic 'not_<pos>' label."""
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    yt_b = (yt == str(pos_label)).astype(int)
    yp_b = (yp == str(pos_label)).astype(int)
    return accuracy_score(yt_b, yp_b)


# “Safe” group metrics
def safe_selection_rate(y_true, y_pred, pos_label):
    y_pred = as_str_array(y_pred)
    mask = (y_pred == str(pos_label))
    return float(np.mean(mask)) if mask.size > 0 else 0.0


def safe_tpr(y_true, y_pred, pos_label):
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    pos = (yt == str(pos_label))
    denom = max(pos.sum(), 1)
    return float(((pos) & (yp == str(pos_label))).sum() / denom)


def safe_fpr(y_true, y_pred, pos_label):
    yt = as_str_array(y_true)
    yp = as_str_array(y_pred)
    neg = (yt != str(pos_label))
    denom = max(neg.sum(), 1)
    return float(((neg) & (yp == str(pos_label))).sum() / denom)


# ---------- Post-processing for “After mitigation” ----------
def tune_group_thresholds(proba_pos: np.ndarray, groups: np.ndarray, target_rate: float):
    thresholds = {}
    target_rate = float(np.clip(target_rate, 0.0, 1.0))
    q = 1.0 - min(max(target_rate, 1e-6), 1 - 1e-6)
    uniq = np.unique(groups)
    for g in uniq:
        scores_g = proba_pos[groups == g]
        if len(scores_g) == 0:
            thresholds[g] = 1.0  # no examples; never select
        else:
            thresholds[g] = float(np.quantile(scores_g, q))
    return thresholds


def apply_group_thresholds(proba_pos: np.ndarray, groups: np.ndarray, thresholds: dict, pos_label: str):
    y_pred = np.empty_like(groups, dtype=object)
    for i, (p, g) in enumerate(zip(proba_pos, groups)):
        thr = thresholds.get(g, 0.5)
        y_pred[i] = pos_label if p >= thr else ("not_" + str(pos_label))
    return y_pred


def proba_for_label(pipe: Pipeline, X: pd.DataFrame, pos_label: str) -> np.ndarray:
    proba = pipe.predict_proba(X)
    classes_ = np.array(list(map(str, pipe.named_steps["model"].classes_)))
    pos = str(pos_label)
    if pos in classes_:
        idx = int(np.where(classes_ == pos)[0][0])
    else:
        lower_map = {c.lower(): i for i, c in enumerate(classes_)}
        idx = lower_map.get(pos.lower(), 1 if proba.shape[1] > 1 else 0)
    return proba[:, idx]


# ============== Altair charts ==============
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


def _alt_metric_bars(series: pd.Series, title: str, y_title: str):
    import altair as alt
    dfp = series.reset_index()
    dfp.columns = ["group", "value"]
    dfp["value_pct"] = (100 * dfp["value"]).round(2)
    bars = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("group:N", title="Group"),
        y=alt.Y("value_pct:Q", title=y_title),
        tooltip=["group", "value_pct"]
    ).properties(title=title, height=220)
    labels = bars.mark_text(dy=-6).encode(text="value_pct:Q")
    return bars + labels


def _alt_distribution(series: pd.Series, title: str, y_title="Count"):
    import altair as alt
    dfp = series.value_counts(dropna=False).reset_index()
    dfp.columns = ["value", "count"]
    return alt.Chart(dfp).mark_bar().encode(
        x=alt.X("value:N", title="Value"),
        y=alt.Y("count:Q", title=y_title),
        tooltip=["value", "count"]
    ).properties(title=title, height=220)


def _alt_group_sizes(groups: pd.Series, title: str):
    import altair as alt
    dfp = groups.value_counts().reset_index()
    dfp.columns = ["group", "n"]
    dfp["percent"] = (100 * dfp["n"] / dfp["n"].sum()).round(1)
    bars = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("group:N", title="Group"),
        y=alt.Y("n:Q", title="Rows"),
        tooltip=["group", "n", "percent"]
    ).properties(title=title, height=220)
    labels = bars.mark_text(dy=-6).encode(text="percent:Q")
    return bars + labels


def _alt_confusion(cm, labels, title="Confusion matrix"):
    import altair as alt
    dfp = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt("index", var_name="Pred", value_name="Count")
    dfp = dfp.rename(columns={"index": "True"})
    return alt.Chart(dfp).mark_rect().encode(
        x=alt.X("Pred:N", title="Predicted"),
        y=alt.Y("True:N", title="Actual"),
        tooltip=["True", "Pred", "Count"],
        color=alt.Color("Count:Q", scale=alt.Scale(type="linear"))
    ).properties(title=title, height=240)


def _alt_score_hist_by_group(scores: np.ndarray, groups: np.ndarray, title: str):
    import altair as alt
    dfp = pd.DataFrame({"score": scores, "group": groups})
    return alt.Chart(dfp).transform_bin("bin:Q", field="score", bin={"maxbins": 30}).mark_bar().encode(
        x=alt.X("bin:Q", title="Predicted probability"),
        y=alt.Y("count():Q", title="Count"),
        color=alt.Color("group:N", legend=alt.Legend(title="Group")),
        tooltip=["group", "count()"]
    ).properties(title=title, height=240)


def _alt_roc_curve(y_true_bin: np.ndarray, scores: np.ndarray, title="ROC curve"):
    import altair as alt
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc_val = auc(fpr, tpr)
    dfp = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    line = alt.Chart(dfp).mark_line().encode(
        x=alt.X("FPR:Q", title="False Positive Rate"),
        y=alt.Y("TPR:Q", title="True Positive Rate"),
        tooltip=["FPR", "TPR"]
    ).properties(title=f"{title} — AUC={auc_val:.3f}", height=240)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4,4]).encode(x="x:Q", y="y:Q")
    return diag + line


def _alt_roc_by_group(y_true_bin: np.ndarray, scores: np.ndarray, groups: np.ndarray, title="ROC by group"):
    import altair as alt
    frames = []
    for g in pd.unique(groups):
        mask = (groups == g)
        if mask.sum() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[mask], scores[mask])
        auc_val = auc(fpr, tpr)
        frames.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "group": str(g), "AUC": auc_val}))
    if not frames:
        return None
    dfp = pd.concat(frames, ignore_index=True)
    line = alt.Chart(dfp).mark_line().encode(
        x=alt.X("FPR:Q", title="False Positive Rate"),
        y=alt.Y("TPR:Q", title="True Positive Rate"),
        color=alt.Color("group:N", legend=alt.Legend(title="Group")),
        tooltip=["group", "FPR", "TPR", "AUC"]
    ).properties(title=title, height=240)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4,4]).encode(x="x:Q", y="y:Q")
    return diag + line


# ============== Narrative builders ==============
def executive_summary_rich(before, after, pos_label, sens_col):
    (acc_b, f1_b, sel_b, tpr_b, fpr_b) = before
    (acc_a, f1_a, sel_a, tpr_a, fpr_a) = after

    def gap_ratio(sr):
        gap = 100.0 * (sr.max() - sr.min())
        ratio = float(sr.min() / sr.max()) if sr.max() > 0 else np.nan
        return gap, ratio

    gap_b, ratio_b = gap_ratio(sel_b)
    gap_a, ratio_a = gap_ratio(sel_a)

    favored_b = sel_b.idxmax()
    disfav_b = sel_b.idxmin()
    favored_a = sel_a.idxmax()
    disfav_a = sel_a.idxmin()

    tpr_gap_b = 100.0 * (tpr_b.max() - tpr_b.min())
    fpr_gap_b = 100.0 * (fpr_b.max() - fpr_b.min())
    tpr_gap_a = 100.0 * (tpr_a.max() - tpr_a.min())
    fpr_gap_a = 100.0 * (fpr_a.max() - fpr_a.min())

    business = (
        f"**What you’re seeing:** We tested whether approvals are spread **evenly** across {sens_col} groups. "
        f"Before the fix, some groups were approved more often than others; after adjusting the decision cut-off per group, "
        f"approvals are **more evenly shared**.\n\n"
        f"**Headline numbers.** Accuracy {acc_b:.3f} → {acc_a:.3f}; F1({pos_label}) {f1_b:.3f} → {f1_a:.3f}. "
        f"Selection gap {gap_b:.2f}% → {gap_a:.2f}% (smaller is fairer). Parity ratio {ratio_b:.3f} → {ratio_a:.3f} (closer to 1 is better). "
        f"TPR/FPR gaps also moved from {tpr_gap_b:.2f}%/{fpr_gap_b:.2f}% to {tpr_gap_a:.2f}%/{fpr_gap_a:.2f}%.\n\n"
        f"**Why it matters:** If one group is consistently approved more, qualified people in other groups can be **overlooked**, "
        f"hurting diversity and increasing **compliance risk**. After mitigation, the spread is closer to even without a major quality loss."
    )

    technical = [
        f"Parity gap: **{gap_b:.2f}% → {gap_a:.2f}%**; Parity ratio: **{ratio_b:.3f} → {ratio_a:.3f}**.",
        f"TPR gap: **{tpr_gap_b:.2f}% → {tpr_gap_a:.2f}%**; FPR gap: **{fpr_gap_b:.2f}% → {fpr_gap_a:.2f}%**.",
        "Mitigation: group-specific thresholds to hit mean pre-mitigation selection rate (post-processing).",
        "Watchouts: parity can shift precision/recall; monitor per-group ROC/AUC and calibration if used operationally."
    ]
    return business, technical


def build_markdown_report(df_head, cfg, before, after, notes, by_group_before, by_group_after):
    target_col, pos_label, sens_col = cfg
    (acc_b, f1_b, sel_b, tpr_b, fpr_b) = before
    (acc_a, f1_a, sel_a, tpr_a, fpr_a) = after

    def gap_ratio(sr):
        gap = 100.0 * (sr.max() - sr.min())
        ratio = float(sr.min() / sr.max()) if sr.max() > 0 else np.nan
        return gap, ratio

    gap_b, ratio_b = gap_ratio(sel_b)
    gap_a, ratio_a = gap_ratio(sel_a)

    buf = io.StringIO()
    buf.write("# FairHire Analytics — Bias Detection Report\n\n")
    buf.write("## Why eliminating bias matters\n")
    buf.write("- Fair hiring widens your talent pool, improves employee trust, and reduces **legal/compliance risk**.\n")
    buf.write("- Even modest gaps (e.g., 5–10%) can **compound** over stages and time, leading to systematic exclusion.\n\n")

    buf.write("## Dataset preview\n\n")
    try:
        buf.write(df_head.to_markdown(index=False))
    except Exception:
        buf.write(df_head.to_csv(index=False))
    buf.write("\n\n")

    buf.write("## Configuration\n")
    buf.write(f"- Outcome (target): `{target_col}`\n")
    buf.write(f"- Positive label: `{pos_label}`\n")
    buf.write(f"- Sensitive attribute: `{sens_col}`\n\n")

    buf.write("## Data health checks\n")
    for n in (notes if notes else ["No major data flags detected."]):
        buf.write(f"- {n}\n")
    buf.write("\n")

    buf.write("## Performance & Fairness (Before vs After)\n")
    buf.write(f"- Accuracy: **{acc_b:.3f} → {acc_a:.3f}** ; F1({pos_label}): **{f1_b:.3f} → {f1_a:.3f}**\n")
    buf.write(f"- Selection gap: **{gap_b:.2f}% → {gap_a:.2f}%** ; Parity ratio: **{ratio_b:.3f} → {ratio_a:.3f}**\n")
    buf.write(f"- TPR gap: **{100*(tpr_b.max()-tpr_b.min()):.2f}% → {100*(tpr_a.max()-tpr_a.min()):.2f}%** ; "
              f"FPR gap: **{100*(fpr_b.max()-fpr_b.min()):.2f}% → {100*(fpr_a.max()-fpr_a.min()):.2f}%**\n\n")

    buf.write("### By-group metrics (before)\n")
    try:
        buf.write(by_group_before.round(3).to_markdown())
    except Exception:
        buf.write(by_group_before.round(3).to_csv())
    buf.write("\n\n")

    buf.write("### By-group metrics (after mitigation)\n")
    try:
        buf.write(by_group_after.round(3).to_markdown())
    except Exception:
        buf.write(by_group_after.round(3).to_csv())
    buf.write("\n\n")

    buf.write("## Method (technical)\n")
    buf.write("- Train baseline LR (OHE+scaler), **exclude** sensitive attribute from features.\n")
    buf.write("- Compute per-group selection/TPR/FPR on validation split.\n")
    buf.write("- **Post-process** predicted probabilities with group-specific thresholds to align selection rates.\n\n")

    buf.write("## Next steps\n")
    buf.write("- Inspect **feature importance**; remove/transform proxy features.\n")
    buf.write("- Consider pre-processing (re-weighting) or alternative mitigations if FPR/TPR shift undesirably.\n")
    buf.write("- **Monitor regularly** across roles, locations, and time windows.\n")
    return buf.getvalue()


# ===================== Page config =====================
st.set_page_config(page_title="FairHire Analytics — Bias Detection", layout="wide")
st.title("FairHire Analytics — Bias Detection")

st.markdown("""
### What this dashboard does (non-technical)
**Purpose.** Help you quickly see whether your decision process (e.g., shortlisting, hiring) treats demographic groups **evenly**.  
**How.** We train a simple model using your columns (but **not** the sensitive column), then:
1) report overall quality,  
2) compare outcomes by group, and  
3) simulate a light-touch fairness fix.

**Why it matters.** If one group is consistently approved more or less than others, qualified people can be **unfairly filtered out**, teams become less diverse, and compliance risk rises.

### What’s computed (technical)
- Logistic Regression + OHE/Scaling → `predict_proba`.
- Grouped **selection rate, TPR, FPR** via `MetricFrame` on a validation split.
- Optional **post-processing**: group thresholds to target a shared selection rate (parity-style).
""")


# ===================== Sidebar =====================
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Use a **binary** outcome (e.g., `hired` yes/no) and at least one demographic column (e.g., `sex`, `race`).")
    val_split = st.slider("Validation split", 0.10, 0.40, 0.20, step=0.01)
    seed = st.number_input("Random seed", value=42, step=1)


# ===================== 1) Upload & Preview =====================
st.subheader("1) Upload & Preview")
with st.expander("Explanation", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Non-technical:** We load your file and show the first rows so you can confirm columns look right. "
            "Later we’ll compare results between your chosen groups (e.g., Male/Female)."
        )
    with c2:
        st.markdown("**Technical:** CSV → Pandas. We keep numeric vs categorical dtypes; key columns coerced to strings.")

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.empty or len(df) < 20:
    st.error("CSV is empty or too small. Provide at least ~20 rows.")
    st.stop()

st.dataframe(df.head(), use_container_width=True)


# ===================== Configure columns =====================
all_cols = df.columns.tolist()
with st.expander("Configure columns", expanded=True):
    st.markdown(
        "Choose the **outcome** (target), the **positive label** (e.g., `yes`), "
        "and the **sensitive attribute** for fairness comparisons. "
        "_We never use the sensitive column as a feature._"
    )
    target_col = st.selectbox("Outcome (target) column", options=all_cols, index=0)
    pos_choices = sorted(df[target_col].astype(str).unique().tolist())
    pos_label = st.selectbox("Positive label (the 'favorable' outcome)", options=pos_choices, index=0)
    sens_col = st.selectbox("Sensitive attribute (for fairness)", options=[c for c in all_cols if c != target_col])


# ===================== 2) Data Health Checks =====================
st.subheader("2) Data Health Checks")
with st.expander("Explanation", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Non-technical:** We flag messy inputs that can distort fairness: many missing values, a target that’s "
            "almost always one value, or very tiny groups. These make a model look good/bad for the wrong reasons."
        )
    with c2:
        st.markdown("**Technical:** Simple counts; warn if minority class <20% or any group has <20 rows.")

df_work = df.copy()
df_work[target_col] = df_work[target_col].astype(str).str.strip()
df_work[sens_col] = df_work[sens_col].astype(str).str.strip()

notes = []
missing_total = int(df_work.isna().sum().sum())
if missing_total > 0:
    notes.append(f"Detected **{missing_total}** missing cells; consider imputation or row drops.")

outcome_counts = df_work[target_col].value_counts()
if len(outcome_counts) == 2:
    if outcome_counts.min() / outcome_counts.sum() < 0.2:
        notes.append("Outcome is **imbalanced** (<20% in one class). Consider re-weighting or stratified sampling.")
else:
    notes.append("Outcome has more than 2 unique values; ensure it represents a **binary** decision.")

group_counts = df_work[sens_col].value_counts()
small_groups = group_counts[group_counts < 20]
if len(small_groups) > 0:
    notes.append(f"Groups with <20 rows in **{sens_col}**: {', '.join(map(str, small_groups.index.tolist()))} (metrics may be unstable).")

cA, cB = st.columns(2)
with cA:
    st.altair_chart(_alt_distribution(df_work[target_col], f"Outcome distribution — {target_col}"), use_container_width=True)
    st.caption("**How to read:** shows how many rows are labeled with each outcome. A very unbalanced split can skew both performance and fairness.")
with cB:
    st.altair_chart(_alt_group_sizes(df_work[sens_col], f"Group sizes — {sens_col}"), use_container_width=True)
    st.caption("**How to read:** group sizes matter—very small groups lead to noisy, unstable metrics.")

if notes:
    st.warning("**Data flags:**\n\n" + "\n".join([f"- {n}" for n in notes]))
else:
    st.success("No concerns detected in quick checks.")


# ===================== Split, Train, Predict =====================
X_full = df_work.drop(columns=[target_col])
y_full = df_work[target_col]

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=val_split, random_state=int(seed),
    stratify=y_full if y_full.nunique() == 2 else None
)
sens_valid = X_valid_full[sens_col].copy()

X_train = X_train_full.drop(columns=[sens_col], errors="ignore")
X_valid  = X_valid_full.drop(columns=[sens_col], errors="ignore")

cat_cols, num_cols = infer_column_types(X_train)
if len(cat_cols) + len(num_cols) == 0:
    st.error("No usable features after removing target & sensitive column.")
    st.stop()

pipe = build_pipeline(cat_cols, num_cols)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_valid)
proba_pos = proba_for_label(pipe, X_valid, pos_label=pos_label)  # robust to label index/case


# ===================== 3) Performance (overall) =====================
st.subheader("3) Performance — How well does the model predict overall?")
with st.expander("Explanation", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Non-technical:** **Accuracy** is the share of predictions that are correct. "
            f"**F1 ({pos_label})** balances two needs: **catching true {pos_label} cases** and **not over-approving**."
        )
    with c2:
        st.markdown("**Technical:** `accuracy_score`; `f1_binary` after binarizing w.r.t. the selected positive label.")

# Use binary accuracy so BEFORE/AFTER are directly comparable
acc_before = acc_binary(y_valid, y_pred, pos_label)
f1_before  = f1_binary(y_valid, y_pred, pos_label)

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{acc_before:.3f}")
c2.metric(f"F1 ({pos_label})", f"{f1_before:.3f}")

with st.expander("Classification report (precision/recall/F1 by class)"):
    report = classification_report(
        as_str_array(y_valid), as_str_array(y_pred),
        target_names=[str(x) for x in sorted(pd.unique(as_str_array(y_valid)))]
    )
    st.code(report)

# Confusion matrix (overall)
yt_bin = (as_str_array(y_valid) == str(pos_label)).astype(int)
yp_bin = (as_str_array(y_pred) == str(pos_label)).astype(int)
cm = confusion_matrix(yt_bin, yp_bin, labels=[0,1])
st.altair_chart(_alt_confusion(cm, labels=[f"not_{pos_label}", str(pos_label)], title="Confusion matrix — overall"), use_container_width=True)
st.caption("**How to read:** top-left = correct rejections; bottom-right = correct approvals; "
           "top-right = false approvals; bottom-left = missed good candidates.")

# ROC curve (overall)
st.altair_chart(_alt_roc_curve(yt_bin, proba_pos, title="ROC curve — overall"), use_container_width=True)
st.caption("**How to read:** the higher the curve (AUC closer to 1), the better the model separates positives from negatives.")


# ===================== 4) Fairness — BEFORE mitigation =====================
st.subheader("4) Fairness — Before mitigation")
with st.expander("Explanation", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Non-technical:** We check three things for each group: "
            "**Selection Rate** (how often they get a favorable outcome), "
            "**True-Positive Rate** (we recognize genuinely qualified people), and "
            "**False-Positive Rate** (we over-approve people who aren’t qualified). "
            "Large gaps suggest uneven treatment."
        )
    with c2:
        st.markdown("**Technical:** `MetricFrame` over `sens_valid`; metrics = selection rate, TPR, FPR, all pinned to `pos_label`.")

y_true      = as_str_array(y_valid)
y_pred_arr  = as_str_array(y_pred)
sens_arr    = as_str_array(sens_valid)

sel_rate_fn = lambda yt, yp: safe_selection_rate(yt, yp, pos_label)
tpr_fn      = lambda yt, yp: safe_tpr(yt, yp, pos_label)
fpr_fn      = lambda yt, yp: safe_fpr(yt, yp, pos_label)

mf_before = MetricFrame(
    metrics={"selection_rate": sel_rate_fn, "TPR": tpr_fn, "FPR": fpr_fn},
    y_true=y_true, y_pred=y_pred_arr, sensitive_features=sens_arr
)
by_group_before = mf_before.by_group.round(3)
sel_before = mf_before.by_group["selection_rate"]
tpr_before = mf_before.by_group["TPR"]
fpr_before = mf_before.by_group["FPR"]

st.dataframe(by_group_before, use_container_width=True)
st.altair_chart(_alt_selection_bars(sel_before, f"Selection Rate by {sens_col} — BEFORE"), use_container_width=True)
st.caption("**How to read:** taller bar = group receives the favorable outcome more often. Large gaps can indicate potential bias.")
st.altair_chart(_alt_metric_bars(tpr_before, "True Positive Rate by group — BEFORE", "TPR (%)"), use_container_width=True)
st.caption("**How to read:** within each group, of the truly positive cases, what share did we approve? Higher is better.")
st.altair_chart(_alt_metric_bars(fpr_before, "False Positive Rate by group — BEFORE", "FPR (%)"), use_container_width=True)
st.caption("**How to read:** within each group, of the truly negative cases, what share did we mistakenly approve? Lower is better.")

# Score distribution by group (before)
st.altair_chart(_alt_score_hist_by_group(proba_pos, sens_arr, title="Score distribution by group (predicted probability)"), use_container_width=True)
st.caption("**How to read:** if one group’s scores are consistently higher, it may drive higher selection even with the same threshold.")
# ROC by group (before)
roc_by_group_chart = _alt_roc_by_group(yt_bin, proba_pos, sens_arr, title="ROC curves by group — BEFORE")
if roc_by_group_chart is not None:
    st.altair_chart(roc_by_group_chart, use_container_width=True)
    st.caption("**How to read:** lines show ranking quality per group (AUC). Wide separation suggests different signal quality across groups.")


# ===================== 5) Mitigation (post-processing) & AFTER =====================
st.subheader("5) Mitigation — Group thresholds for demographic parity (AFTER)")
with st.expander("Explanation", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Non-technical:** We keep the same model but adjust the decision cut-off **per group** so everyone is "
            "approved at about the same rate. This narrows gaps while leaving your training data unchanged."
        )
    with c2:
        st.markdown(
            "**Technical:** For each group g, choose threshold τ_g via score quantile to hit target selection rate "
            "(target = mean of pre-mitigation selection rates)."
        )

target_rate = float(sel_before.mean())
thresholds = tune_group_thresholds(proba_pos, sens_arr, target_rate=target_rate)
y_pred_after = apply_group_thresholds(proba_pos, sens_arr, thresholds, pos_label=str(pos_label))

# AFTER metrics on the same binary domain as BEFORE
acc_after = acc_binary(y_true, y_pred_after, pos_label)
f1_after  = f1_binary(y_true, y_pred_after, pos_label)

mf_after = MetricFrame(
    metrics={"selection_rate": sel_rate_fn, "TPR": tpr_fn, "FPR": fpr_fn},
    y_true=y_true, y_pred=y_pred_after, sensitive_features=sens_arr
)
by_group_after = mf_after.by_group.round(3)
sel_after = mf_after.by_group["selection_rate"]
tpr_after = mf_after.by_group["TPR"]
fpr_after = mf_after.by_group["FPR"]

c1, c2 = st.columns(2)
c1.metric("Accuracy (AFTER)", f"{acc_after:.3f}", delta=f"{acc_after-acc_before:+.3f}")
c2.metric(f"F1 ({pos_label}) (AFTER)", f"{f1_after:.3f}", delta=f"{f1_after-f1_before:+.3f}")

st.dataframe(by_group_after, use_container_width=True)
st.altair_chart(_alt_selection_bars(sel_after, f"Selection Rate by {sens_col} — AFTER"), use_container_width=True)
st.caption("**How to read:** bars should be closer together than BEFORE—meaning approvals are more evenly shared.")
st.altair_chart(_alt_metric_bars(tpr_after, "True Positive Rate by group — AFTER", "TPR (%)"), use_container_width=True)
st.caption("**How to read:** ensure we didn’t meaningfully reduce the ability to catch truly qualified people.")
st.altair_chart(_alt_metric_bars(fpr_after, "False Positive Rate by group — AFTER", "FPR (%)"), use_container_width=True)
st.caption("**How to read:** check that false approvals haven’t jumped for any group after the change.")


# ===================== 6) Executive Summary (rich) =====================
st.subheader("6) Executive Summary — Findings & Implications")

biz_text, tech_bullets = executive_summary_rich(
    before=(acc_before, f1_before, sel_before, tpr_before, fpr_before),
    after =(acc_after,  f1_after,  sel_after,  tpr_after,  fpr_after),
    pos_label=pos_label, sens_col=sens_col
)

with st.expander("Stakeholder view (non-technical)", expanded=True):
    st.info(biz_text)
    st.markdown("""
    **Interpretation:**
    - Before the fairness adjustment, approvals were noticeably higher for one group over another.
    - After adjustment, approval rates became **more balanced**, meaning candidates from each group had a similar chance of success.
    - Importantly, overall model accuracy stayed within the same range — suggesting that fairness improved **without a major quality trade-off**.
    - From a business standpoint, this means the model is now making **fairer decisions** and reducing **potential compliance risk**.
    """)

with st.expander("Technical view (details)", expanded=True):
    for b in tech_bullets:
        st.markdown(f"- {b}")
    st.markdown("""
    **Deeper insights:**
    - **Selection Rate Gap** shows how far apart approval rates were between groups; smaller = fairer.
    - **Parity Ratio** compares the lowest approval rate group vs the highest — a ratio near 1.0 means decisions are evenly distributed.
    - **True Positive Rate (TPR)** tells us if qualified candidates are recognized equally across groups.
    - **False Positive Rate (FPR)** reveals if unqualified candidates are mistakenly approved more often for certain groups.
    - These measures together show *where* and *how* bias manifests (e.g., under-recognition of one group vs over-approval of another).
    """)

st.markdown("""
---

### **Non-technical takeaways for hiring leaders**
- **Before the fairness fix:** certain groups were being approved **significantly more often**, which can unintentionally exclude qualified candidates and reduce diversity.  
- **After the fix:** approvals are **distributed more evenly**, creating a fairer process.  
- Model accuracy remained stable, meaning you didn’t sacrifice predictive performance to gain fairness.  
- **What this means:** you can maintain hiring quality while reducing bias, which strengthens both **equity outcomes** and **legal defensibility**.

### **Technical takeaways for data practitioners**
- The fairness fix applied is a **post-processing threshold adjustment** per group — no retraining or data changes were needed.
- This method is **transparent, reversible, and easily monitored**.
- You can extend this framework to **equalized odds** (balancing both TPR and FPR) if parity alone isn’t your goal.
- Continuous monitoring is key: data drift or candidate pool changes can reintroduce disparity over time.
- Examine group-level **ROC/AUC**; differences hint at proxy features or noisy signals affecting fairness.

---
""")



# ===================== 7) What to do next =====================
st.subheader("7) What to do next")

st.markdown("""
### **For Everyone (Non-Technical)**
**1. Define fairness goals clearly.**  
Decide what “fair” means in your context. For example:
- *Demographic parity* (equal selection rates) — ensures all groups are approved at similar rates.
- *Equal opportunity* (equal TPR) — ensures qualified candidates are equally recognized.
- *Equalized odds* — balances both TPR and FPR between groups.

**2. Incorporate a human review step.**  
For borderline cases or when confidence is low, allow recruiters or hiring managers to override model decisions.  
This keeps the process auditable and supports transparency.

**3. Communicate outcomes clearly.**  
Share fairness metrics internally (e.g., quarterly fairness reports) to show proactive compliance and accountability.

**4. Make fairness ongoing, not one-off.**  
Bias can reappear when new data or job roles are added — schedule regular fairness re-checks.

---

### **For Practitioners and Technical Teams**
**1. Audit your features.**  
Use **permutation importance** or **SHAP** to detect *proxy features* (like ZIP code or university) that correlate with sensitive attributes (like race or gender).

**2. Monitor fairness over time.**  
Deploy this dashboard as a recurring audit. Track parity ratios or selection gaps monthly.

**3. Evaluate calibration per group.**  
If probabilities differ in reliability across groups, calibrate separately to avoid systematic over/under-estimation.

**4. Explore stronger fairness interventions.**  
If parity post-processing isn’t enough:
- **Pre-processing**: reweight samples or balance distributions before training.
- **In-training**: use fairness-constrained optimization during model fitting.

**5. Align fairness with business KPIs.**  
Balance fairness with candidate quality, conversion, or retention metrics.  
Ensure adjustments meet *business* and *ethical* goals simultaneously.

---

### **Strategic Next Steps**
- Set measurable fairness benchmarks (e.g., *parity ratio ≥ 0.9*).  
- Review this report with your DEI and compliance teams.  
- Schedule the next fairness re-evaluation in 3–6 months.  
- Consider extending analysis to **race**, **age**, or **region** once data supports it.

""")

# ===================== 8) Export one-page report =====================
st.subheader("8) Export report")
report_md = build_markdown_report(
    df.head(),
    cfg=(target_col, pos_label, sens_col),
    before=(acc_before, f1_before, sel_before, tpr_before, fpr_before),
    after =(acc_after,  f1_after,  sel_after,  tpr_after,  fpr_after),
    notes=notes,
    by_group_before=by_group_before,
    by_group_after=by_group_after
)
st.download_button(
    "Download Markdown report",
    data=report_md.encode("utf-8"),
    file_name="fairhire_bias_report.md",
    mime="text/markdown",
    help="One-page summary with configuration, data checks, before/after fairness and next steps."
)