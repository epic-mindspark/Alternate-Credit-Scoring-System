"""
Main entry point for the Alternate Credit Scoring Streamlit application.
"""

import sys
import pathlib

# Add app/ to sys.path so we can import from local modules
_APP_DIR = pathlib.Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from utils.artifacts import load_artifacts
from inference import run_inference
from ui.input_form import render_input_form
from ui.overview_tab import render_overview_tab
from ui.explanation_tab import render_explanation_tab
from ui.simulator_tab import render_simulator_tab


def main():
    st.set_page_config(
        page_title="AltScore — Credit Decision System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Load custom component styles from static folder
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load artifacts. Did you run the pipeline to save them? Error: {e}")
        return

    # Render sidebar form
    raw_input = render_input_form()

    if raw_input is None:
        # Empty state
        st.markdown(
            '<div style="text-align:center; padding: 4rem 2rem; color: #64748B;">'
            '<h2 style="color: #475569;">Welcome to the Alternate Credit Scoring System</h2>'
            '<p style="font-size:1.1rem; max-width:600px; margin: 0 auto; line-height:1.6;">'
            'Fill out the applicant details in the left sidebar and click <b>Assess Borrower</b> '
            'to run the real-time inference pipeline. The system will evaluate the applicant\'s '
            'risk profile using behavioral and digital footprints.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    # Run inference
    with st.spinner("Running inference pipeline..."):
        try:
            result = run_inference(raw_input, artifacts, threshold_key="default")
        except Exception as e:
            st.error(f"Inference failed. Error: {e}")
            st.exception(e)
            return

    # Render tabs
    tab1, tab2, tab3 = st.tabs([
        "Risk Overview",
        "Factor Explanations",
        "Counterfactual Simulator"
    ])

    with tab1:
        render_overview_tab(result, raw_input, artifacts)
    
    with tab2:
        render_explanation_tab(result, artifacts)
        
    with tab3:
        render_simulator_tab(result, raw_input, artifacts)


if __name__ == "__main__":
    main()
