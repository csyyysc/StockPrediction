"""
Sidebar component for the Streamlit application.
"""

import os
import sys
import streamlit as st
import multiprocessing as mp

from agent import list_agents
from .tools import get_popular_stocks, get_stock_info, format_currency

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_sidebar():
    """Render the sidebar with configuration options."""

    st.sidebar.header("🔧 Configuration")
    st.sidebar.subheader("Agent Selection")
    available_agents = list_agents()

    agent_type = st.sidebar.selectbox(
        "Choose Agent",
        available_agents,
        help="Select the reinforcement learning algorithm to use"
    )

    num_workers = None
    if agent_type == 'a3c':
        st.sidebar.subheader("A3C Configuration")

        cpu_count = mp.cpu_count()

        worker_option = st.sidebar.selectbox(
            "Worker Threads",
            ["Auto (recommended)", "All CPU cores", "Custom"],
            help="Choose how many CPU cores to use for A3C training"
        )

        if worker_option == "Auto (recommended)":
            num_workers = None
            actual_workers = max(1, cpu_count - 1)
            st.sidebar.success(f"Using {actual_workers}/{cpu_count} CPU cores")
            agent_name = f'A3C ({actual_workers} workers)'
        elif worker_option == "All CPU cores":
            num_workers = -1
            st.sidebar.warning(
                f"Using all {cpu_count}/{cpu_count} CPU cores (may impact system responsiveness)")
            agent_name = f'A3C ({cpu_count} workers)'
        else:
            num_workers = st.sidebar.slider(
                "Number of workers",
                min_value=1,
                max_value=cpu_count,
                value=max(1, cpu_count // 2),
                help=f"Choose between 1 and {cpu_count} workers"
            )
            st.sidebar.info(f"Using {num_workers}/{cpu_count} CPU cores")
            agent_name = f'A3C ({num_workers} workers)'
    else:
        agent_name = 'Vanilla Actor-Critic'

    if agent_type == 'vac':
        agent_description = 'Classic actor-critic algorithm with shared feature extraction and synchronous updates'
    else:
        agent_description = 'Asynchronous Advantage Actor-Critic with parallel worker threads, GAE, and entropy regularization'

    st.sidebar.info(f"""
    **{agent_name}**
    
    {agent_description}
    """)

    st.sidebar.markdown("---")

    st.sidebar.subheader("Stock Selection")

    stock_categories = get_popular_stocks()
    category = st.sidebar.selectbox(
        "Choose Category", list(stock_categories.keys()))

    if category:
        popular_symbol = st.sidebar.selectbox(
            "Popular Stocks",
            stock_categories[category],
            help="Select from popular stocks in this category"
        )
    else:
        popular_symbol = "AAPL"

    custom_symbol = st.sidebar.text_input(
        "Or Enter Custom Symbol",
        value="",
        help="Enter any stock symbol (e.g., TSLA, GOOGL)"
    ).upper()

    symbol = custom_symbol if custom_symbol else popular_symbol

    if symbol:
        stock_info = get_stock_info(symbol)
        st.sidebar.info(f"""
        **{stock_info['name']}**
        - Sector: {stock_info['sector']}
        - Market Cap: {format_currency(stock_info['market_cap'])}
        - Current Price: ${stock_info['current_price']:.2f}
        """)

    st.sidebar.subheader("Training Parameters")

    window_size = st.sidebar.slider(
        "Training Window (days)",
        min_value=10,
        max_value=120,
        value=30,
        help="Number of days of historical data to use as input features"
    )

    train_period = st.sidebar.selectbox(
        "Data Period",
        ["1y", "2y", "3y", "5y"],
        index=1,
        help="Amount of historical data to fetch for training"
    )

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f"
    )

    num_episodes = st.sidebar.slider(
        "Training Episodes",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )

    return {
        'agent_type': agent_type,
        'agent_name': agent_name,
        'num_workers': num_workers,
        'symbol': symbol,
        'window_size': window_size,
        'train_period': train_period,
        'learning_rate': learning_rate,
        'num_episodes': num_episodes
    }
