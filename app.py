"""
Streamlit application for the multi-agent stock prediction system.

This is the main entry point for the web interface, now using modular components.
"""

import streamlit as st
from components import (
    configure_app,
    render_app_header,
    render_sidebar,
    render_overview_tab,
    render_training_tab,
    render_results_tab,
    render_analysis_tab
)


def app():
    """Main Streamlit application function."""

    # Configure the app (page config and CSS)
    configure_app()

    # Render header
    render_app_header()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🤖 Training", "📈 Results", "🔍 Analysis"])

    with tab1:
        render_overview_tab(config['symbol'])

    with tab2:
        render_training_tab(config)

    with tab3:
        render_results_tab()

    with tab4:
        render_analysis_tab(config)


def main():
    """Entry point for the application."""
    app()


if __name__ == "__main__":
    main()
