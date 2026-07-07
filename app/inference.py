"""
Full inference pipeline — replicates Stage 1 + Stage 2 preprocessing
exactly as done during training, then predicts with the best model.

Input : raw_input dict (from the form, may contain None for optional fields)
Output: result dict with probability, tier, SHAP values, cluster, etc.
"""

import numpy as np
import pandas as pd

# ── Constants (match notebook exactly) ──────────────────────────────────────
INCOME_COLS = [f"income_month_{i}" for i in range(1, 7)]
DIGITAL_FIELDS = [
    "upi_transactions_per_month",
    "upi_avg_transaction_amount",
    "upi_months_active",
    "mobile_wallet_used",
    "ecomm_orders_per_month",
    "prepaid_orders_ratio",
]
GROUP_COLS = [
    "upi_transactions_per_month",
    "upi_avg_transaction_amount",
    "upi_months_active",
    "mobile_wallet_used",
    "rent_paid_on_time_months",
    "total_rental_months",
]
LOAN_LABELS = ["Small", "Medium", "Large", "Very_Large"]
LOAN_CODES = {label: idx for idx, label in enumerate(LOAN_LABELS)}
INCOME_LABELS = ["Low", "Medium", "High"]
INCOME_CODES = {label: idx for idx, label in enumerate(INCOME_LABELS)}
STATE_TO_REGION = {
    "Delhi": "North", "Uttar Pradesh": "North", "Rajasthan": "North",
    "Karnataka": "South", "Tamil Nadu": "South", "Telangana": "South",
    "Maharashtra": "West", "Gujarat": "West",
    "Bihar": "East", "Odisha": "East",
    "Madhya Pradesh": "Central",
}
CATEGORICAL_COLS = ["region", "borrower_type", "employment_type", "loan_purpose"]
TELECOM_REF_YEAR = 2024
LOAN_STRESS_SENTINEL = 999


def _to_series(raw: dict) -> pd.Series:
    """Convert raw input dict to a Series with NaN for missing values."""
    return pd.Series({k: (np.nan if v is None else v) for k, v in raw.items()})


def run_inference(raw_input: dict, artifacts: dict, threshold_key: str = "default") -> dict:
    """
    Run full inference pipeline on a single raw input dict.

    Parameters
    ----------
    raw_input     : dict of raw form values (None = missing/optional)
    artifacts     : loaded artifact dict from load_artifacts()
    threshold_key : 'default' | 'max_f1' | 'conservative'

    Returns
    -------
    dict with keys: probability, risk_tier, decision, cluster_id, cluster_name,
                    shap_values, shap_feature_names, engineered_row,
                    data_confidence, model_confidence, top_positive, top_negative
    """
    row = _to_series(raw_input).to_frame().T.reset_index(drop=True)

    # ── Step 1: digital_footprint_density (pre-imputation) ───────────────────
    row["digital_footprint_density"] = (
        row[DIGITAL_FIELDS].notna().sum(axis=1) / len(DIGITAL_FIELDS)
    )

    # ── Step 2: missingness flags (pre-imputation) ────────────────────────────
    row["upi_data_missing"] = row["upi_transactions_per_month"].isna().astype(int)
    row["rent_data_missing"] = row["total_rental_months"].isna().astype(int)
    row["ecomm_data_missing"] = (
        row["ecomm_orders_per_month"].isna() | (row["ecomm_orders_per_month"] == 0)
    ).astype(int)

    # ── Step 3a: simple median imputation ────────────────────────────────────
    simple_cols = ["same_number_since_year", "survey_q4", "survey_q6"]
    row[simple_cols] = artifacts["simple_imputer"].transform(row[simple_cols])

    # ── Step 3b: group-wise imputation by borrower_type ───────────────────────
    gm = artifacts["group_medians"]   # DataFrame indexed by borrower_type
    gl = artifacts["global_medians"]  # Series fallback
    btype = str(raw_input.get("borrower_type", "gig"))
    for col in GROUP_COLS:
        if pd.isna(row[col].iloc[0]):
            if btype in gm.index and col in gm.columns:
                fill_val = gm.loc[btype, col]
            else:
                fill_val = gl[col]
            row[col] = fill_val

    # ── Step 3c: ecomm special imputation ─────────────────────────────────────
    if row["ecomm_data_missing"].iloc[0] == 1:
        row["ecomm_orders_per_month"] = 0.0
        row["ecomm_return_rate"] = 0.0
        row["prepaid_orders_ratio"] = 0.0

    # ── Step 4: derived features ──────────────────────────────────────────────
    util_total = row["utility_bills_total"].iloc[0]
    row["utility_payment_ratio"] = (
        0.5 if util_total == 0
        else row["utility_bills_paid"].iloc[0] / util_total
    )

    if row["rent_data_missing"].iloc[0] == 1:
        row["rent_consistency_score"] = 0.5
    else:
        rnt_total = row["total_rental_months"].iloc[0]
        row["rent_consistency_score"] = (
            0.5 if rnt_total == 0
            else row["rent_paid_on_time_months"].iloc[0] / rnt_total
        )

    row["income_mean"] = row[INCOME_COLS].mean(axis=1)
    row["income_std"] = row[INCOME_COLS].std(axis=1)
    inc_mean = row["income_mean"].iloc[0]
    row["income_regularity_index"] = (
        0.0 if inc_mean == 0
        else np.clip(1 - row["income_std"].iloc[0] / inc_mean, 0, 1)
    )
    row["income_trend"] = (
        row["income_month_6"].iloc[0] - row["income_month_1"].iloc[0]
    )
    row["telecom_stability_years"] = (
        TELECOM_REF_YEAR - row["same_number_since_year"].iloc[0]
    )
    denom = inc_mean * row["loan_tenure_months"].iloc[0]
    row["loan_to_income_ratio"] = (
        LOAN_STRESS_SENTINEL if denom == 0
        else row["loan_amount_requested"].iloc[0] / denom
    )
    row["upi_consistency_score"] = (
        (row["upi_transactions_per_month"].iloc[0] / 120.0)
        * (row["upi_months_active"].iloc[0] / 36.0)
    )

    drop_cols = INCOME_COLS + ["same_number_since_year",
                                "upi_transactions_per_month", "upi_months_active"]
    row = row.drop(columns=[c for c in drop_cols if c in row.columns])

    # ── Step 5: psychometric composites ──────────────────────────────────────
    q3_rev = 6 - row["survey_q3"].iloc[0]
    q7_rev = 6 - row["survey_q7"].iloc[0]
    row["financial_discipline_score"] = np.mean([
        row["survey_q1"].iloc[0], row["survey_q2"].iloc[0],
        q3_rev, row["survey_q4"].iloc[0], row["survey_q8"].iloc[0]
    ])
    row["future_planning_score"] = np.mean([
        row["survey_q5"].iloc[0], row["survey_q6"].iloc[0]
    ])
    row["risk_appetite_score"] = np.mean([q3_rev, q7_rev])

    survey_cols = [f"survey_q{i}" for i in range(1, 9)]
    row = row.drop(columns=[c for c in survey_cols if c in row.columns])

    # ── Step 6: state → region mapping + OHE ─────────────────────────────────
    state = str(raw_input.get("state", "Maharashtra"))
    row["region"] = STATE_TO_REGION.get(state, "West")

    # Drop identifier / non-model columns
    for c in ["borrower_id", "state"]:
        if c in row.columns:
            row = row.drop(columns=[c])

    ohe = artifacts["ohe"]
    cat_data = ohe.transform(row[CATEGORICAL_COLS])
    enc_cols = ohe.get_feature_names_out(CATEGORICAL_COLS)
    enc_df = pd.DataFrame(cat_data, columns=enc_cols)
    row = pd.concat(
        [row.drop(columns=CATEGORICAL_COLS).reset_index(drop=True), enc_df],
        axis=1
    )

    # ── Step 7: binning ───────────────────────────────────────────────────────
    loan_bins = artifacts["loan_bins"]
    income_bins = artifacts["income_bins"]

    row["loan_amount_bin"] = pd.cut(
        row["loan_amount_requested"], bins=loan_bins,
        labels=LOAN_LABELS, include_lowest=True
    ).map(LOAN_CODES).astype(int)

    row["income_level_bin"] = pd.cut(
        row["income_mean"], bins=income_bins,
        labels=INCOME_LABELS, include_lowest=True
    ).map(INCOME_CODES).astype(int)

    # ── Step 8: scale continuous columns ─────────────────────────────────────
    scaler = artifacts["scaler"]
    cont_cols = [c for c in artifacts["continuous_cols"] if c in row.columns]
    row[cont_cols] = scaler.transform(row[cont_cols])

    # ── Step 9: align to training feature columns (48 cols, no cluster_id) ───
    feature_cols = artifacts["feature_columns"]
    for c in feature_cols:
        if c not in row.columns:
            row[c] = 0
    row = row[feature_cols]

    # ── Step 10: cluster assignment ───────────────────────────────────────────
    clust_feats = artifacts["clustering_features"]
    X_clust = row[clust_feats]
    cluster_id = int(artifacts["cluster_model"].predict(X_clust)[0])
    cluster_name = artifacts["cluster_names"].get(cluster_id, f"Cluster {cluster_id}")

    row_with_cluster = row.copy()
    row_with_cluster["cluster_id"] = cluster_id

    # Align to full 49-col feature set
    full_feat_cols = artifacts["feature_columns_with_cluster"]
    for c in full_feat_cols:
        if c not in row_with_cluster.columns:
            row_with_cluster[c] = 0
    row_with_cluster = row_with_cluster[full_feat_cols]

    # ── Step 11: predict ──────────────────────────────────────────────────────
    model = artifacts["best_model"]
    prob = float(model.predict_proba(row_with_cluster)[0, 1])

    # ── Step 12: risk tier + decision ────────────────────────────────────────
    threshold = artifacts["thresholds"].get(threshold_key, 0.5)
    risk_tier = _risk_tier(prob)
    decision = _decision(prob, threshold)

    # ── Step 13: SHAP values ──────────────────────────────────────────────────
    explainer = artifacts["explainer"]
    shap_vals = explainer.shap_values(row_with_cluster)[0]     # shape (49,)
    feature_names = list(row_with_cluster.columns)

    # ── Step 14: factor extraction ────────────────────────────────────────────
    name_map = artifacts["feature_name_map"]
    top_pos, top_neg = _top_factors(shap_vals, feature_names, name_map)

    # ── Data & model confidence ───────────────────────────────────────────────
    dfd = float(raw_input.get("_digital_footprint_density_raw",
                              row["digital_footprint_density"].iloc[0]
                              if "digital_footprint_density" in row.columns
                              else 0.5))
    upi_miss = int(row_with_cluster["upi_data_missing"].iloc[0]) if "upi_data_missing" in row_with_cluster.columns else 0
    ecomm_miss = int(row_with_cluster["ecomm_data_missing"].iloc[0]) if "ecomm_data_missing" in row_with_cluster.columns else 0
    rent_miss = int(row_with_cluster["rent_data_missing"].iloc[0]) if "rent_data_missing" in row_with_cluster.columns else 0

    # Recompute raw dfd from original input for confidence scoring
    raw_dfd = sum(
        raw_input.get(f, None) is not None
        for f in DIGITAL_FIELDS
    ) / len(DIGITAL_FIELDS)

    data_confidence = _data_confidence(raw_dfd, upi_miss, ecomm_miss, rent_miss)
    model_confidence = _model_confidence(prob, threshold)

    return {
        "probability": prob,
        "risk_tier": risk_tier,
        "decision": decision,
        "threshold_used": threshold,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "shap_values": shap_vals,
        "feature_names": feature_names,
        "engineered_row": row_with_cluster,
        "data_confidence": data_confidence,
        "model_confidence": model_confidence,
        "top_positive_factors": top_pos,
        "top_negative_factors": top_neg,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    return "High"


def _decision(prob: float, threshold: float) -> str:
    if prob < threshold * 0.6:
        return "Approve"
    elif prob < threshold:
        return "Review"
    return "Reject"


def _top_factors(shap_vals, feature_names, name_map, n=3):
    pairs = list(zip(feature_names, shap_vals))
    # Positive factors = most negative SHAP (reduce default probability)
    top_pos = sorted(pairs, key=lambda x: x[1])[:n]
    # Negative factors = most positive SHAP (increase default probability)
    top_neg = sorted(pairs, key=lambda x: -x[1])[:n]

    def label(feat, val):
        return {
            "feature": feat,
            "label": name_map.get(feat, feat.replace("_", " ").title()),
            "shap": float(val),
        }

    return [label(f, v) for f, v in top_pos], [label(f, v) for f, v in top_neg]


def _data_confidence(dfd: float, upi_miss: int, ecomm_miss: int, rent_miss: int) -> str:
    miss_count = upi_miss + ecomm_miss + rent_miss
    if dfd >= 0.8 and upi_miss == 0 and ecomm_miss == 0:
        return "High"
    elif dfd < 0.4 or miss_count >= 2:
        return "Low"
    return "Medium"


def _model_confidence(prob: float, threshold: float) -> str:
    distance = abs(prob - threshold)
    if distance >= 0.25:
        return "High"
    elif distance >= 0.10:
        return "Medium"
    return "Low"


def generate_nl_summary(result: dict, borrower_type: str) -> str:
    """Generate a plain-English paragraph summarising the risk assessment."""
    prob = result["probability"]
    tier = result["risk_tier"]
    decision = result["decision"]
    dc = result["data_confidence"]
    mc = result["model_confidence"]
    pos = result["top_positive_factors"]
    neg = result["top_negative_factors"]
    cluster = result["cluster_name"]

    pos_labels = [f["label"] for f in pos[:2]]
    neg_labels = [f["label"] for f in neg[:2]]

    dc_meaning = {
        "High": "with high data completeness across digital channels",
        "Medium": "with moderate data coverage",
        "Low": "with limited data availability — interpret with caution",
    }[dc]

    rec_map = {
        "Approve": "Recommended for approval.",
        "Review": "Recommended for manual review before a credit decision.",
        "Reject": "Recommended for rejection based on the current risk profile.",
    }

    return (
        f"This {borrower_type} borrower has been assessed as a <b>{tier} Risk</b> "
        f"applicant with a default probability of <b>{prob:.1%}</b>, "
        f"placing them in the <b>{cluster}</b> peer group. "
        f"Key factors reducing risk include {pos_labels[0]} and {pos_labels[1]}. "
        f"Factors increasing risk include {neg_labels[0]} and {neg_labels[1]}. "
        f"The assessment was made {dc_meaning}, "
        f"with <b>{mc} model confidence</b>. "
        f"{rec_map[decision]}"
    )
