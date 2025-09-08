"""
Web interface utilities for Stock Prediction Application.

This module contains web interface-related functions and logic.
"""

import subprocess
from .parameters import WebParameters


def run_web_interface(params: WebParameters) -> None:
    """
    Launch the Streamlit web interface.

    Args:
        params: WebParameters instance (currently unused but kept for consistency)
    """
    print("🚀 Starting Streamlit web interface...")
    print("📱 Open your browser to http://localhost:8501")
    subprocess.run(["uv", "run", "python", "-m", "streamlit", "run", "app.py"])
