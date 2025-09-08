"""
Application configuration and styling for the Streamlit application.
"""

import streamlit as st


def configure_app():
    """Configure the Streamlit application with page settings and custom CSS."""
    
    # Set page config
    st.set_page_config(
        page_title="Multi-Agent Stock Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .positive {
            color: #00FF00;
        }
        .negative {
            color: #FF0000;
        }
    </style>
    """, unsafe_allow_html=True)


def render_app_header():
    """Render the main application header."""
    st.markdown('<h1 class="main-header">📈 Multi-Agent Stock Predictor</h1>',
                unsafe_allow_html=True)
