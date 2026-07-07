"""
Enterprise CSS theme for the Alternate Credit Scoring System.
Clean, professional financial-services aesthetic.
"""

ENTERPRISE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #F0F4F8;
}

/* ── Main container ──────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1E3A5F;
    border-right: 1px solid #16304F;
}
[data-testid="stSidebar"] * {
    color: #E2EAF4 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1D4ED8;
}
[data-testid="stSidebar"] hr {
    border-color: #2D4A6E !important;
    margin: 1rem 0;
}

/* Sidebar section headings */
.sidebar-section-header {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7EA8D4 !important;
    padding: 0.5rem 0 0.25rem 0;
    border-bottom: 1px solid #2D4A6E;
    margin-bottom: 0.5rem;
}

/* ── Cards ───────────────────────────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.metric-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.metric-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}
.metric-card-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}

/* ── Decision Banner ─────────────────────────────────────────── */
.decision-banner {
    border-radius: 10px;
    padding: 1.25rem 1.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.decision-approve {
    background: #ECFDF5;
    border: 2px solid #059669;
    color: #065F46;
}
.decision-review {
    background: #FFFBEB;
    border: 2px solid #D97706;
    color: #92400E;
}
.decision-reject {
    background: #FEF2F2;
    border: 2px solid #DC2626;
    color: #991B1B;
}

/* ── Confidence Badges ───────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-high    { background: #D1FAE5; color: #065F46; }
.badge-medium  { background: #FEF3C7; color: #92400E; }
.badge-low     { background: #FEE2E2; color: #991B1B; }

/* ── Factor Cards ────────────────────────────────────────────── */
.factor-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.factor-positive { border-left-color: #059669; }
.factor-negative { border-left-color: #DC2626; }
.factor-label    { font-weight: 600; color: #1E293B; }
.factor-shap     { font-size: 0.76rem; color: #64748B; margin-top: 0.15rem; }

/* ── Section headings in main area ──────────────────────────── */
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #E2E8F0;
}

/* ── Tab styling ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 8px 8px 0 0;
    border: 1px solid #E2E8F0;
    border-bottom: none;
    padding: 0.25rem 0.5rem 0;
    gap: 0.25rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.88rem;
    color: #64748B;
    padding: 0.5rem 1.25rem;
    border-radius: 6px 6px 0 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1E3A5F;
    font-weight: 700;
    border-bottom: 2px solid #2563EB;
}

/* ── NL Summary box ──────────────────────────────────────────── */
.nl-summary-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #1E293B;
}

/* ── Probability display ─────────────────────────────────────── */
.prob-display {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}
.prob-low    { color: #059669; }
.prob-medium { color: #D97706; }
.prob-high   { color: #DC2626; }

/* ── Cluster info ────────────────────────────────────────────── */
.cluster-badge {
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1E40AF;
}

/* ── Simulator delta ─────────────────────────────────────────── */
.delta-positive { color: #059669; font-weight: 700; }
.delta-negative { color: #DC2626; font-weight: 700; }
.delta-neutral  { color: #64748B; font-weight: 600; }

/* ── Page header ─────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    border-radius: 10px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.25rem;
    color: white;
}
.app-header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
    color: white;
}
.app-header p {
    font-size: 0.85rem;
    color: #BFDBFE;
    margin: 0.25rem 0 0 0;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
"""


def inject_css():
    """Call this once at the top of app.py."""
    import streamlit as st
    st.markdown(f"<style>{ENTERPRISE_CSS}</style>", unsafe_allow_html=True)
