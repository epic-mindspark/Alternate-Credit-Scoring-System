"""
Counterfactual Simulator Tab.
Allows adjusting key input features to see the impact on default probability.
"""

import streamlit as st
import copy

from inference import run_inference


def render_simulator_tab(original_result: dict, raw_input: dict, artifacts: dict):
    st.markdown(
        """
        <p class="section-title">Counterfactual Simulator</p>
        <p style="font-size:0.85rem; color:#475569; margin-bottom: 1.5rem">
        Adjust the actionable parameters below to simulate "what-if" scenarios 
        and observe how the default probability and risk tier change in real-time.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_sliders, col_results = st.columns([1.2, 1])

    with col_sliders:
        st.markdown("**Adjustable Inputs**")
        
        # We target the raw inputs that drive the top SHAP features:
        # loan_amount_requested, income, utility_bills_paid, rent_paid_on_time_months, upi_transactions_per_month

        loan_val = raw_input.get("loan_amount_requested")
        sim_loan = st.slider(
            "Loan Amount Requested (₹)",
            min_value=5000, max_value=500000, step=5000,
            value=int(loan_val) if loan_val is not None else 150000,
        )
        
        avg_inc = sum(raw_input.get(f"income_month_{i}") or 0 for i in range(1, 7)) / 6
        sim_income = st.slider(
            "Average Monthly Income (₹)",
            min_value=5000, max_value=200000, step=1000,
            value=int(avg_inc),
            help="Overrides all 6 months of income history with this stable value."
        )

        util_total_val = raw_input.get("utility_bills_total")
        util_total = int(util_total_val) if util_total_val is not None else 12
        util_paid_val = raw_input.get("utility_bills_paid")
        sim_util_paid = st.slider(
            "Utility Bills Paid On Time",
            min_value=0, max_value=util_total, step=1,
            value=int(util_paid_val) if util_paid_val is not None else min(10, util_total),
        )

        rent_total_val = raw_input.get("total_rental_months")
        rent_total = int(rent_total_val) if rent_total_val is not None and rent_total_val > 0 else 12
        rent_paid_val = raw_input.get("rent_paid_on_time_months")
        sim_rent_paid = st.slider(
            "Rent Paid On Time (Months)",
            min_value=0, max_value=rent_total, step=1,
            value=int(rent_paid_val) if rent_paid_val is not None else min(10, rent_total),
        )

        upi_val = raw_input.get("upi_transactions_per_month")
        sim_upi = st.slider(
            "UPI Transactions per Month",
            min_value=0, max_value=200, step=5,
            value=int(upi_val) if upi_val is not None else 30,
        )

    # ── Run Inference on Simulated Data ───────────────────────────────────────
    sim_input = copy.deepcopy(raw_input)
    sim_input["loan_amount_requested"] = sim_loan
    for i in range(1, 7):
        sim_input[f"income_month_{i}"] = sim_income
    sim_input["utility_bills_paid"] = sim_util_paid
    sim_input["rent_paid_on_time_months"] = sim_rent_paid
    sim_input["upi_transactions_per_month"] = sim_upi

    # Pass the chosen threshold from Overview tab if it's in session state?
    # For now we'll just use default, or we can fetch it if we stored it.
    sim_result = run_inference(sim_input, artifacts, threshold_key="default")

    # ── Display Delta ─────────────────────────────────────────────────────────
    with col_results:
        st.markdown("**Simulation Outcome**")
        
        orig_prob = original_result["probability"]
        sim_prob  = sim_result["probability"]
        prob_delta = sim_prob - orig_prob

        orig_tier = original_result["risk_tier"]
        sim_tier  = sim_result["risk_tier"]

        orig_dec = original_result["decision"]
        sim_dec  = sim_result["decision"]

        # Color coding delta
        if prob_delta < -0.01:
            delta_cls = "delta-positive"
            delta_str = f"↓ {abs(prob_delta):.1%} Risk"
        elif prob_delta > 0.01:
            delta_cls = "delta-negative"
            delta_str = f"↑ {abs(prob_delta):.1%} Risk"
        else:
            delta_cls = "delta-neutral"
            delta_str = "No significant change"

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">New Default Probability</div>'
            f'<div class="metric-card-value">{sim_prob:.1%}</div>'
            f'<div class="{delta_cls}" style="margin-top:0.5rem">{delta_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">Risk Tier Transition</div>'
            f'<div style="font-size:1.1rem; font-weight:600; margin-top:0.25rem;">'
            f'<span style="color:#64748B">{orig_tier}</span> &nbsp;→&nbsp; '
            f'<span style="color:#0F172A">{sim_tier}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        dec_color = {"Approve": "#059669", "Review": "#D97706", "Reject": "#DC2626"}
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-title">New Decision</div>'
            f'<div style="font-size:1.4rem; font-weight:700; color:{dec_color[sim_dec]}; margin-top:0.25rem;">'
            f'{sim_dec.upper()}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.info("💡 **Tip:** Increasing 'Utility Bills Paid On Time' or 'Rent Paid On Time' generally improves the applicant's consistency scores, lowering default risk. Decreasing 'Loan Amount Requested' lowers the loan-to-income stress, which is often a strong positive factor.")
