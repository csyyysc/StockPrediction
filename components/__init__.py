"""
Components package for the Streamlit application.

This package contains modular components for the stock prediction web interface.
"""

from .config import configure_app, render_app_header
from .sidebar import render_sidebar
from .overview import render_overview_tab
from .top_stocks import render_top_stocks_tab
from .training import render_training_tab
from .results import render_results_tab
from .analysis import render_analysis_tab
from .tools import (
    get_popular_stocks,
    get_top_stocks_analysis,
    get_category_stocks_analysis,
    get_stock_info,
    format_currency,
    save_web_plots
)

__all__ = [
    'configure_app',
    'render_app_header',
    'render_sidebar',
    'render_overview_tab',
    'render_top_stocks_tab',
    'render_training_tab',
    'render_results_tab',
    'render_analysis_tab',
    'get_popular_stocks',
    'get_top_stocks_analysis',
    'get_category_stocks_analysis',
    'get_stock_info',
    'format_currency',
    'save_web_plots'
]
