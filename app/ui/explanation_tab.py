"""
Explanation tab — Simple view (business-friendly factors) + Technical view
(SHAP waterfall, bar chart, peer radar comparison).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


_TIER_COLORS = {"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"}


# ── Simple View ────────────────────────────────────────────────────────────────

def _factor_card(label: str, shap_val: float, positive: bool) -> str:
    cls = "factor-positive" if positive else "factor-negative"
    direction = "Reduces" if positive else "Increases"
    icon = "↓" if positive else "↑"
    color = "#059669" if positive else "#DC2626"
    abs_v = abs(shap_val)
    return (
        f'<div class="factor-card {cls}">'
        f'<div class="factor-label">{icon} {label}</div>'
        f'<div class="factor-shap" style="color:{color}">'
        f'{direction} default risk &nbsp;|&nbsp; '
        f'Impact strength: {"High" if abs_v > 0.3 else "Medium" if abs_v > 0.1 else "Low"}'
        f'</div>'
        f'</div>'
    )


def _render_simple_view(result: dict, artifacts: dict):
    st.markdown('<p class="section-title">Key Decision Factors</p>',
                unsafe_allow_html=True)
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown(
            '<p style="font-size:0.78rem;font-weight:700;color:#059669;'
            'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem">'
            '✅ Positive Factors (Reducing Risk)</p>',
            unsafe_allow_html=True,
        )
        for f in result["top_positive_factors"]:
            st.markdown(
                _factor_card(f["label"], f["shap"], positive=True),
                unsafe_allow_html=True,
            )

    with col_neg:
        st.markdown(
            '<p style="font-size:0.78rem;font-weight:700;color:#DC2626;'
            'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem">'
            '⚠️ Risk Factors (Increasing Risk)</p>',
            unsafe_allow_html=True,
        )
        for f in result["top_negative_factors"]:
            st.markdown(
                _factor_card(f["label"], f["shap"], positive=False),
                unsafe_allow_html=True,
            )

    # ── Full feature impact table ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">All Feature Impacts</p>',
                unsafe_allow_html=True)
    name_map = artifacts["feature_name_map"]
    shap_vals  = result["shap_values"]
    feat_names = result["feature_names"]

    impact_df = pd.DataFrame({
        "Feature": [name_map.get(f, f) for f in feat_names],
        "SHAP Value": shap_vals,
        "Direction": ["Reduces Risk" if v < 0 else "Increases Risk" for v in shap_vals],
    }).sort_values("SHAP Value").reset_index(drop=True)

    def _style_direction(v):
        return "color: #059669; font-weight: 600" if v == "Reduces Risk" else "color: #DC2626; font-weight: 600"

    st.dataframe(
        impact_df.style
        .map(_style_direction, subset=["Direction"])
        .format({"SHAP Value": "{:+.4f}"}),
        use_container_width=True,
        height=400,
    )


# ── Technical View ─────────────────────────────────────────────────────────────

def _shap_waterfall(result: dict, artifacts: dict) -> go.Figure:
    """Build a waterfall-style SHAP chart (top 15 features by |SHAP|)."""
    shap_vals  = result["shap_values"]
    feat_names = result["feature_names"]
    name_map   = artifacts["feature_name_map"]

    idx = np.argsort(np.abs(shap_vals))[-15:][::-1]
    top_feats = [name_map.get(feat_names[i], feat_names[i]) for i in idx]
    top_shap  = [float(shap_vals[i]) for i in idx][::-1]
    top_feats = top_feats[::-1]

    colors = ["#059669" if v < 0 else "#DC2626" for v in top_shap]

    fig = go.Figure(go.Bar(
        x=top_shap,
        y=top_feats,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_shap],
        textposition="outside",
        textfont={"size": 10},
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#94A3B8")
    fig.update_layout(
        title={"text": "SHAP Feature Contributions (Top 15)",
               "font": {"size": 14, "family": "Inter", "color": "#1E293B"}},
        xaxis_title="SHAP Value (positive = increases default risk)",
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font={"family": "Inter", "color": "#475569"},
        margin=dict(l=10, r=60, t=50, b=40),
        height=520,
    )
    return fig


def _peer_radar(result: dict, artifacts: dict) -> go.Figure:
    """Radar chart: applicant vs cluster centroid on 8 clustering features."""
    clust_feats = artifacts["clustering_features"]
    name_map    = artifacts["feature_name_map"]
    cpt         = artifacts["cluster_profile_table"]
    cid         = result["cluster_id"]

    applicant_vals = result["engineered_row"][clust_feats].iloc[0].tolist()
    cluster_vals   = (
        cpt.loc[cid, clust_feats].tolist()
        if cid in cpt.index else [0.5] * len(clust_feats)
    )
    labels = [name_map.get(f, f) for f in clust_feats] + [name_map.get(clust_feats[0], clust_feats[0])]
    applicant_vals += [applicant_vals[0]]
    cluster_vals   += [cluster_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_vals, theta=labels,
        fill="toself",
        name=f"Cluster {cid} Avg",
        line_color="#CBD5E1",
        fillcolor="rgba(148,163,184,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=applicant_vals, theta=labels,
        fill="toself",
        name="This Applicant",
        line_color="#2563EB",
        fillcolor="rgba(37,99,235,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont={"size": 9, "family": "Inter"}, gridcolor="#E2E8F0"),
            angularaxis=dict(tickfont={"size": 10, "family": "Inter"}),
            bgcolor="#F8FAFC",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font={"family": "Inter"}),
        paper_bgcolor="white",
        font={"family": "Inter", "color": "#475569"},
        title={"text": "Applicant vs Peer Cluster — Behavioural Features",
               "font": {"size": 14, "family": "Inter", "color": "#1E293B"}},
        height=480,
        margin=dict(t=60, b=60),
    )
    return fig


def _render_technical_view(result: dict, artifacts: dict):
    col_w, col_r = st.columns([1.1, 1])

    with col_w:
        st.markdown('<p class="section-title">SHAP Feature Contributions</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(_shap_waterfall(result, artifacts),
                        use_container_width=True, theme="streamlit",
                        config={"displayModeBar": False})

    with col_r:
        st.markdown('<p class="section-title">Peer Comparison Radar</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(_peer_radar(result, artifacts),
                        use_container_width=True, theme="streamlit",
                        config={"displayModeBar": False})


# ── Entry point ────────────────────────────────────────────────────────────────

def render_explanation_tab(result: dict, artifacts: dict):
    view = st.radio(
        "View mode",
        ["Simple (Business View)", "Technical (SHAP View)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")
    if view.startswith("Simple"):
        _render_simple_view(result, artifacts)
    else:
        _render_technical_view(result, artifacts)
