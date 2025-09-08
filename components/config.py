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

    # Custom CSS for dark theme
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            font-size: 3rem;
            color: #00D4AA;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #3730A3 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            border: 1px solid #374151;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        /* Color indicators */
        .positive {
            color: #10B981;
            font-weight: bold;
        }
        .negative {
            color: #EF4444;
            font-weight: bold;
        }
        
        /* Dark theme enhancements */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #262730;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #262730;
            border-radius: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            color: #FAFAFA;
            margin: 0.5rem;
            width: 100%;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #00D4AA;
            color: #0E1117;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #00D4AA;
            color: #0E1117;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        
        .stButton > button:hover {
            background-color: #00B894;
            color: #0E1117;
        }
        
        /* Selectbox and input styling */
        .stSelectbox > div > div {
            background-color: #262730;
            color: #FAFAFA;
        }
        
        .stTextInput > div > div > input {
            background-color: #262730;
            color: #FAFAFA;
            border: 1px solid #374151;
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background-color: #00D4AA;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #1F2937;
            border: 1px solid #374151;
        }
        
        /* Success/Error styling */
        .stSuccess {
            background-color: #064E3B;
            border: 1px solid #10B981;
        }
        
        .stError {
            background-color: #7F1D1D;
            border: 1px solid #EF4444;
        }
        
        .stWarning {
            background-color: #78350F;
            border: 1px solid #F59E0B;
        }
        
        .stInfo {
            background-color: #1E3A8A;
            border: 1px solid #3B82F6;
        }
    </style>
    """, unsafe_allow_html=True)


def render_app_header():
    """Render the main application header."""
    st.markdown('<h1 class="main-header">📈 Multi-Agent Stock Predictor</h1>',
                unsafe_allow_html=True)
