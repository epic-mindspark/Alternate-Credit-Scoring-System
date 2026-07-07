"""
Artifact loader — loads all inference artifacts once and caches them.
Call load_artifacts() from anywhere in the app.
"""

import pickle
import pathlib
import streamlit as st

# Resolve paths relative to this file's location
_APP_DIR = pathlib.Path(__file__).resolve().parent.parent   # .../app/
_REPO_ROOT = _APP_DIR.parent                                 # repo root
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_MODELS_DIR = _REPO_ROOT / "models"


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts() -> dict:
    """Load and return all inference artifacts as a single dict."""

    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    s1 = _load(_PROCESSED_DIR / "inference_artifacts_stage1.pkl")
    s2 = _load(_PROCESSED_DIR / "inference_artifacts_stage2.pkl")
    s4 = _load(_PROCESSED_DIR / "inference_artifacts_stage4.pkl")
    s6 = _load(_PROCESSED_DIR / "inference_artifacts_stage6.pkl")
    best_model = _load(_MODELS_DIR / "lr_l1.pkl")

    # Top-5 simulator features by mean |SHAP|
    top5_features = list(s6["shap_feature_importance"].head(5).index)

    return {
        # Stage 1 — preprocessing
        "simple_imputer": s1["simple_imputer"],
        "group_medians": s1["group_medians"],
        "global_medians": s1["global_medians"],
        "ohe": s1["ohe"],
        "scaler": s1["scaler"],
        "loan_bins": s1["loan_bins"],
        "income_bins": s1["income_bins"],
        "continuous_cols": s1["continuous_cols"],
        "feature_columns": s1["feature_columns"],           # 48 cols, pre cluster_id
        # Stage 2 — clustering
        "cluster_model": s2["cluster_model"],
        "cluster_profile_table": s2["cluster_profile_table"],
        "cluster_names": s2["cluster_names"],
        "clustering_features": s2["clustering_features"],
        "feature_columns_with_cluster": s2["feature_columns_with_cluster"],  # 49 cols
        # Stage 4 — thresholds
        "thresholds": s4["thresholds"],
        # Stage 6 — explainability
        "explainer": s6["explainer"],
        "feature_name_map": s6["feature_name_map"],
        "shap_feature_importance": s6["shap_feature_importance"],
        "top5_simulator_features": top5_features,
        # Model
        "best_model": best_model,
    }
