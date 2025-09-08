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
    import os

    print("🚀 Starting Streamlit web interface...")

    # Check if running in production (Render, Heroku, etc.)
    port = os.environ.get('PORT', '8501')
    address = os.environ.get('STREAMLIT_SERVER_ADDRESS', 'localhost')

    if address == '0.0.0.0':
        print(f"🌐 Production mode: Server will be accessible on port {port}")
    else:
        print("📱 Development mode: Open your browser to http://localhost:8501")

    # Launch Streamlit with proper configuration
    subprocess.run([
        "uv", "run", "python", "-m", "streamlit", "run", "app.py",
        "--server.address", address,
        "--server.port", port,
        "--server.headless", "true"
    ])
