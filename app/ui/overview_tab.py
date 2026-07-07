"""
Overview tab — Risk gauge, decision banner, score card, threshold toggle,
cluster intelligence, confidence metrics, and NL summary.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from inference import generate_nl_summary


# ── Colour helpers ─────────────────────────────────────────────────────────────

_TIER_COLORS = {
    "Low":    "#059669",
    "Medium": "#D97706",
    "High":   "#DC2626",
}

_DECISION_ICONS = {
    "Approve": "✅",
    "Review":  "⚠️",
    "Reject":  "🚫",
}

_DECISION_CLASS = {
    "Approve": "decision-approve",
    "Review":  "decision-review",
    "Reject":  "decision-reject",
}


def _gauge(prob: float, tier: str) -> go.Figure:
    color = _TIER_COLORS[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color,
                                        "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#94A3B8",
                "tickvals": [0, 30, 60, 100],
                "ticktext": ["0", "30", "60", "100"],
                "tickfont": {"size": 11, "color": "#64748B", "family": "Inter"},
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#D1FAE5"},
                {"range": [30, 60], "color": "#FEF3C7"},
                {"range": [60, 100],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#0F172A", "width": 3},
                "thickness": 0.85,
                "value": round(prob * 100, 1),
            },
        },
        title={"text": "Default Probability", "font": {"size": 14,
               "color": "#475569", "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        margin=dict(t=30, b=10, l=20, r=20),
        height=260,
    )
    return fig


def _badge(level: str) -> str:
    cls = f"badge-{level.lower()}"
    return f'<span class="badge {cls}">{level}</span>'


def render_overview_tab(result: dict, raw_input: dict, artifacts: dict):
    """Render the complete Overview tab."""

    prob     = result["probability"]
    tier     = result["risk_tier"]
    decision = result["decision"]

    # ── Threshold selector (top of main area) ─────────────────────────────────
    thr_map = {
        "Default (0.50)":               "default",
        f"Optimised F1 ({artifacts['thresholds']['max_f1']:.2f})":    "max_f1",
        f"Conservative ≥80% Recall ({artifacts['thresholds']['conservative']:.2f})": "conservative",
    }
    thr_col, _ = st.columns([2, 3])
    with thr_col:
        st.markdown('<p class="section-title">Threshold</p>', unsafe_allow_html=True)
        thr_label = st.radio(
            "Decision threshold",
            list(thr_map.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
    chosen_thr_key = thr_map[thr_label]
    chosen_thr     = artifacts["thresholds"][chosen_thr_key]

    # Recompute decision with chosen threshold
    from inference import _decision as _dec
    decision = _dec(prob, chosen_thr)

    st.markdown("---")

    # ── Decision banner ────────────────────────────────────────────────────────
    icon  = _DECISION_ICONS[decision]
    dcls  = _DECISION_CLASS[decision]
    thr_display = f"{chosen_thr:.0%}"
    st.markdown(
        f'<div class="decision-banner {dcls}">'
        f'{icon} &nbsp; <span>{decision.upper()}</span>'
        f'<span style="font-size:0.85rem;font-weight:400;margin-left:auto;opacity:0.75;">'
        f'Threshold: {thr_display}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Row 1: Gauge + Score metrics ──────────────────────────────────────────
    col_gauge, col_metrics = st.columns([1.1, 1])

    with col_gauge:
        st.plotly_chart(_gauge(prob, tier), use_container_width=True,
                        theme="streamlit", config={"displayModeBar": False})

    with col_metrics:
        st.markdown('<p class="section-title">Risk Overview</p>',
                    unsafe_allow_html=True)

        tier_color = _TIER_COLORS[tier]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Risk Tier</div>'
            f'<div class="metric-card-value" style="color:{tier_color}">{tier}</div>'
            f'<div class="metric-card-sub">Based on probability bands: Low &lt;30% | Medium 30–60% | High &gt;60%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        prob_cls = f"prob-{tier.lower()}"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Default Probability</div>'
            f'<div class="prob-display {prob_cls}">{prob:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        cid   = result["cluster_id"]
        cname = result["cluster_name"]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Peer Cluster</div>'
            f'<span class="cluster-badge">Cluster {cid} — {cname}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Row 2: Confidence + Cluster profile ───────────────────────────────────
    col_conf, col_clust = st.columns(2)

    with col_conf:
        st.markdown('<p class="section-title">Assessment Confidence</p>',
                    unsafe_allow_html=True)
        dc = result["data_confidence"]
        mc = result["model_confidence"]

        dc_desc = {
            "High":   "Full digital data coverage — prediction is well-informed.",
            "Medium": "Partial digital data — some fields estimated.",
            "Low":    "Limited data — missing multiple digital signals.",
        }[dc]
        mc_desc = {
            "High":   f"Probability is far from threshold ({chosen_thr:.0%}) — decision is clear.",
            "Medium": f"Moderate distance from threshold ({chosen_thr:.0%}).",
            "Low":    f"Probability is near threshold ({chosen_thr:.0%}) — borderline case.",
        }[mc]

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Data Confidence &nbsp;{_badge(dc)}</div>'
            f'<div style="font-size:0.85rem;color:#475569;margin-top:0.4rem">{dc_desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Model Confidence &nbsp;{_badge(mc)}</div>'
            f'<div style="font-size:0.85rem;color:#475569;margin-top:0.4rem">{mc_desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_clust:
        st.markdown('<p class="section-title">Cluster Intelligence</p>',
                    unsafe_allow_html=True)

        cpt = artifacts["cluster_profile_table"]
        clust_feats = artifacts["clustering_features"]
        name_map = artifacts["feature_name_map"]

        if cid in cpt.index:
            cluster_row = cpt.loc[cid, clust_feats]
            applicant_vals = result["engineered_row"][clust_feats].iloc[0]

            compare_df = pd.DataFrame({
                "Feature": [name_map.get(f, f) for f in clust_feats],
                "Applicant": applicant_vals.values,
                "Cluster Avg": cluster_row.values,
            }).round(3)
            compare_df["Δ vs Cluster"] = (
                compare_df["Applicant"] - compare_df["Cluster Avg"]
            ).round(3)

            def _color_delta(v):
                if v > 0.05:
                    return "color: #059669; font-weight:600"
                elif v < -0.05:
                    return "color: #DC2626; font-weight:600"
                return "color: #64748B"

            st.dataframe(
                compare_df.style
                .map(_color_delta, subset=["Δ vs Cluster"])
                .format({"Applicant": "{:.3f}", "Cluster Avg": "{:.3f}",
                         "Δ vs Cluster": "{:+.3f}"}),
                use_container_width=True,
                height=320,
            )

    st.markdown("---")

    # ── Row 3: Borrower Profile + Financial Health ───────────────────────────
    col_profile, col_health = st.columns(2)

    with col_profile:
        st.markdown('<p class="section-title">Borrower Profile</p>',
                    unsafe_allow_html=True)
        
        prof_cols = st.columns(2)
        with prof_cols[0]:
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Age & Household</div>'
                f'<div style="font-size:1rem;font-weight:600;">{int(raw_input["age"])} yrs | {int(raw_input["household_size"])} members</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Job Stability</div>'
                f'<div style="font-size:1rem;font-weight:600;">{int(raw_input["months_at_current_job"])} months in current role</div>'
                f'</div>', unsafe_allow_html=True
            )
        with prof_cols[1]:
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Location & Type</div>'
                f'<div style="font-size:1rem;font-weight:600;">{raw_input["state"]} | {raw_input["borrower_type"].title()}</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Telecom History</div>'
                f'<div style="font-size:1rem;font-weight:600;">Active since {int(raw_input["same_number_since_year"])}</div>'
                f'</div>', unsafe_allow_html=True
            )

    with col_health:
        st.markdown('<p class="section-title">Financial Health Indicators</p>',
                    unsafe_allow_html=True)
        
        row = result["engineered_row"]
        health_cols = st.columns(2)
        with health_cols[0]:
            lti = row["loan_to_income_ratio"].iloc[0]
            lti_val = f"{lti:.1%}" if lti < 10 else "High Stress"
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Loan-to-Income Stress</div>'
                f'<div style="font-size:1rem;font-weight:600;">{lti_val}</div>'
                f'</div>', unsafe_allow_html=True
            )
            reg = row["income_regularity_index"].iloc[0]
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Income Regularity</div>'
                f'<div style="font-size:1rem;font-weight:600;">{reg:.1%} consistency</div>'
                f'</div>', unsafe_allow_html=True
            )
        with health_cols[1]:
            upi = row["upi_consistency_score"].iloc[0]
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Digital Transaction Score</div>'
                f'<div style="font-size:1rem;font-weight:600;">{upi:.2f} (0-1 scale)</div>'
                f'</div>', unsafe_allow_html=True
            )
            util = row["utility_payment_ratio"].iloc[0]
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;">Utility Payment Ratio</div>'
                f'<div style="font-size:1rem;font-weight:600;">{util:.0%} on-time</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("---")

    # ── NL Summary ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Assessment Summary</p>',
                unsafe_allow_html=True)
    btype = str(raw_input.get("borrower_type", "gig"))
    summary = generate_nl_summary(result, btype)
    st.markdown(f'<div class="nl-summary-box">{summary}</div>',
                unsafe_allow_html=True)
