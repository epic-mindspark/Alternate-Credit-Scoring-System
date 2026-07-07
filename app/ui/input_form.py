"""
Structured input form rendered in the sidebar.
Returns raw_input dict on submit (None = field left blank / optional).
"""

import streamlit as st
import numpy as np

# ── Option lists ──────────────────────────────────────────────────────────────
STATES = [
    "Delhi", "Uttar Pradesh", "Rajasthan",
    "Karnataka", "Tamil Nadu", "Telangana",
    "Maharashtra", "Gujarat",
    "Bihar", "Odisha",
    "Madhya Pradesh",
]
BORROWER_TYPES = ["gig", "migrant", "rural"]
EMPLOYMENT_TYPES = ["daily-wage", "salaried-gig", "seasonal", "self-employed"]
LOAN_PURPOSES = ["agriculture", "business", "consumption", "education", "medical"]

SURVEY_LABELS = {
    1: "I always pay bills on time",
    2: "I keep a personal budget",
    3: "I take financial risks for higher returns",   # reverse-scored
    4: "I set aside savings regularly",
    5: "I plan finances 6+ months ahead",
    6: "I think about retirement / long-term goals",
    7: "Borrowing money doesn't worry me",            # reverse-scored
    8: "I avoid unnecessary debt",
}


def render_input_form() -> dict | None:
    """
    Render the full input form in the sidebar.
    Returns raw_input dict when submitted, else None.
    """
    with st.sidebar:
        st.markdown(
            '<div class="app-header">'
            '<h1>AltScore</h1>'
            '<p>Alternate Credit Scoring System</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.form("applicant_form"):
            # ── Section 1: Basic Information ─────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Basic Information</p>',
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80,
                                      value=35, step=1)
                household_size = st.number_input("Household Size", min_value=1,
                                                 max_value=15, value=3, step=1)
            with col2:
                months_at_current_job = st.number_input(
                    "Months at Job", min_value=0, max_value=480, value=24, step=1)
                num_income_sources = st.number_input(
                    "Income Sources", min_value=1, max_value=10, value=1, step=1)

            state = st.selectbox("State", STATES, index=6)  # Maharashtra
            borrower_type = st.selectbox("Borrower Type", BORROWER_TYPES)
            employment_type = st.selectbox("Employment Type", EMPLOYMENT_TYPES, index=1)
            phone_year = st.number_input(
                "Phone Number Since Year",
                min_value=1990, max_value=2024, value=2018, step=1,
                help="Year the borrower got their current phone number (optional – leave 2018 if unknown)",
            )
            avg_recharge = st.number_input(
                "Avg Monthly Recharge (₹)", min_value=0.0, value=250.0, step=10.0)
            recharge_freq = st.number_input(
                "Recharges per Month", min_value=0.0, value=2.0, step=0.5)

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 2: Income Details ─────────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Monthly Income (₹)</p>',
                        unsafe_allow_html=True)
            inc_cols = st.columns(3)
            incomes = []
            for i in range(6):
                with inc_cols[i % 3]:
                    incomes.append(
                        st.number_input(f"Month {i+1}", min_value=0.0,
                                        value=20000.0, step=500.0, key=f"inc_{i}")
                    )

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 3: Rental Behaviour ───────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Rental Behaviour</p>',
                        unsafe_allow_html=True)
            no_rent_data = st.checkbox("No rental data available",
                                       help="Check if borrower has no rental history")
            if no_rent_data:
                total_rental_months = None
                rent_paid_on_time = None
            else:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    total_rental_months = st.number_input(
                        "Total Rental Months", min_value=0.0, value=12.0, step=1.0)
                with col_r2:
                    rent_paid_on_time = st.number_input(
                        "Paid On Time (Months)", min_value=0.0, value=10.0, step=1.0)

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 4: Digital Footprint ──────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Digital Footprint</p>',
                        unsafe_allow_html=True)
            no_upi = st.checkbox("No UPI data",
                                 help="Check if borrower has no UPI transaction history")
            if no_upi:
                upi_txn = None
                upi_avg = None
                upi_months = None
            else:
                col_u1, col_u2, col_u3 = st.columns(3)
                with col_u1:
                    upi_txn = st.number_input("UPI Txn/Month", min_value=0.0,
                                              value=30.0, step=1.0)
                with col_u2:
                    upi_avg = st.number_input("Avg Txn Amt (₹)", min_value=0.0,
                                              value=800.0, step=50.0)
                with col_u3:
                    upi_months = st.number_input("UPI Active Months", min_value=0.0,
                                                 max_value=36.0, value=18.0, step=1.0)

            mobile_wallet = st.selectbox("Mobile Wallet Used", ["Yes", "No"], index=0)
            mobile_wallet_val = 1.0 if mobile_wallet == "Yes" else 0.0

            no_ecomm = st.checkbox("No e-commerce data",
                                   help="Check if borrower has no e-commerce activity")
            if no_ecomm:
                ecomm_orders = None
                prepaid_ratio = None
                ecomm_return = None
            else:
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    ecomm_orders = st.number_input("Ecomm Orders/Month", min_value=0.0,
                                                   value=5.0, step=0.5)
                with col_e2:
                    prepaid_ratio = st.number_input("Prepaid Orders Ratio",
                                                    min_value=0.0, max_value=1.0,
                                                    value=0.7, step=0.05)
                with col_e3:
                    ecomm_return = st.number_input("Return Rate",
                                                   min_value=0.0, max_value=1.0,
                                                   value=0.1, step=0.05)

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 5: Utility Bills ──────────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Utility Bills</p>',
                        unsafe_allow_html=True)
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                util_paid = st.number_input("Bills Paid", min_value=0, value=10, step=1)
            with col_u2:
                util_total = st.number_input("Bills Total", min_value=0, value=12, step=1)

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 6: Loan Details ───────────────────────────────────────
            st.markdown('<p class="sidebar-section-header">Loan Details</p>',
                        unsafe_allow_html=True)
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                loan_amount = st.number_input("Loan Amount (₹)", min_value=1000.0,
                                              value=150000.0, step=5000.0)
            with col_l2:
                loan_tenure = st.number_input("Tenure (Months)", min_value=1,
                                              max_value=120, value=24, step=1)
            loan_purpose = st.selectbox("Loan Purpose", LOAN_PURPOSES, index=1)

            st.markdown('<hr>', unsafe_allow_html=True)

            # ── Section 7: Psychometric Survey ───────────────────────────────
            st.markdown('<p class="sidebar-section-header">Psychometric Survey</p>',
                        unsafe_allow_html=True)
            st.caption("Rate agreement with each statement  |  1 = Strongly Disagree · 5 = Strongly Agree")
            survey = {}
            for q in range(1, 9):
                is_reverse = q in (3, 7)
                suffix = " ↩ (reverse)" if is_reverse else ""
                survey[f"survey_q{q}"] = st.slider(
                    f"Q{q}: {SURVEY_LABELS[q]}{suffix}",
                    min_value=1, max_value=5, value=3,
                    key=f"survey_q{q}"
                )

            st.markdown('<hr>', unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "Assess Borrower",
                use_container_width=True,
                type="primary",
            )

        raw_input = {
            # Basic
            "age": float(age),
            "household_size": float(household_size),
            "months_at_current_job": float(months_at_current_job),
            "num_income_sources": float(num_income_sources),
            "state": state,
            "borrower_type": borrower_type,
            "employment_type": employment_type,
            "same_number_since_year": float(phone_year),
            "avg_monthly_recharge_amount": float(avg_recharge),
            "recharge_frequency_per_month": float(recharge_freq),
            # Income
            **{f"income_month_{i+1}": float(incomes[i]) for i in range(6)},
            # Rental
            "total_rental_months": total_rental_months,
            "rent_paid_on_time_months": rent_paid_on_time,
            # Digital
            "upi_transactions_per_month": upi_txn,
            "upi_avg_transaction_amount": upi_avg,
            "upi_months_active": upi_months,
            "mobile_wallet_used": mobile_wallet_val,
            "ecomm_orders_per_month": ecomm_orders,
            "prepaid_orders_ratio": prepaid_ratio,
            "ecomm_return_rate": ecomm_return,
            # Utility
            "utility_bills_paid": float(util_paid),
            "utility_bills_total": float(util_total),
            # Loan
            "loan_amount_requested": float(loan_amount),
            "loan_tenure_months": float(loan_tenure),
            "loan_purpose": loan_purpose,
            # Survey
            **survey,
        }

        if submitted:
            st.session_state["applicant_raw_input"] = raw_input

    return st.session_state.get("applicant_raw_input", None)
