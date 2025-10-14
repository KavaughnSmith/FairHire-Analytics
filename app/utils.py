# app/utils.py
import json
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ---------- Artifact loaders (never crash) ----------

def load_pipeline(path: str = "models/pipeline.pkl"):
    """
    Try to load a persisted sklearn pipeline. If anything goes wrong,
    return None so the caller can train on the uploaded data instead.
    """
    try:
        if os.path.exists(path):
            return load(path)
    except Exception as e:
        print(f"[WARN] Could not load pipeline from {path}: {e}")
    return None


def load_thresholds(path: str = "models/group_thresholds.json") -> Dict[str, float]:
    """
    Load per-group thresholds if present. Supports several shapes:
    - { "Female": 0.47, "Male": 0.53 }
    - { "Female": {"threshold": 0.47}, "Male": {"threshold": 0.53} }
    - { "Female": {"value": 0.47}, ... } etc.

    Returns {} if not present or invalid.
    """
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)

        norm: Dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, (int, float, str)):
                try:
                    norm[str(k)] = float(v)
                except Exception:
                    continue
            elif isinstance(v, dict):
                # common keys we’ll accept
                for key in ("threshold", "value", "p", "prob", "cutoff"):
                    if key in v:
                        try:
                            norm[str(k)] = float(v[key])
                            break
                        except Exception:
                            pass
        return norm
    except Exception as e:
        print(f"[WARN] Could not load thresholds: {e}")
        return {}


# ---------- Generic preprocessing / training ----------

def infer_column_types(
    X: pd.DataFrame,
    target: str,
    sensitive: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """Infer categorical & numeric columns and exclude target/sensitive."""
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Exclude target & sensitive if they slipped in
    for col in (target, sensitive):
        if col and col in cat:
            cat.remove(col)
        if col and col in num:
            num.remove(col)
    return cat, num


def build_preprocessor(
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> ColumnTransformer:
    """
    For sklearn 1.2.x use OneHotEncoder(sparse=False) (no 'sparse_output' yet).
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
            ("num", StandardScaler(with_mean=False), numeric_cols),  # safe even if all dense
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def make_quick_pipe(
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> Pipeline:
    pre = build_preprocessor(categorical_cols, numeric_cols)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    return Pipeline(steps=[("prep", pre), ("model", clf)])