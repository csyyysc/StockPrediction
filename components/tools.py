"""
Utility functions for the Streamlit application components.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from typing import Dict


@st.cache_data
def get_popular_stocks():
    """Get list of popular stocks."""

    return {
        'Technology': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'OSCR'],
        'Consumer': ['KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
        'ETFs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']
    }


@st.cache_data
def get_stock_info(symbol: str):
    """Get basic stock information."""

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'current_price': info.get('currentPrice', 0)
        }
    except:
        return {'name': symbol, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 0, 'current_price': 0}


def format_currency(value):
    """Format currency values."""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def save_web_plots(trainer, results_dir: str, final_eval: Dict, training_metrics: Dict):
    """Save individual plots for web interface results."""
    try:
        # 1. Portfolio value over time plot
        plt.figure(figsize=(12, 6))
        portfolio_values = final_eval['portfolio_values']
        steps = range(len(portfolio_values))

        plt.plot(steps, portfolio_values, linewidth=2,
                 color='green', label='Portfolio Value')
        plt.axhline(y=10000, color='red', linestyle='--',
                    label='Initial Investment', alpha=0.7)
        plt.title('Portfolio Value Over Time (Test Period)')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()

        portfolio_plot_path = os.path.join(
            results_dir, 'portfolio_value_plot.png')
        plt.savefig(portfolio_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Individual training metric plots
        if (training_metrics['episode_rewards'] and
            training_metrics['actor_losses'] and
                training_metrics['critic_losses']):

            # Episode Rewards Plot
            plt.figure(figsize=(10, 5))
            plt.plot(training_metrics['episode_rewards'],
                     linewidth=2, color='blue')
            plt.title('Episode Rewards During Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            rewards_plot_path = os.path.join(
                results_dir, 'episode_rewards_plot.png')
            plt.savefig(rewards_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Total Returns Plot
            plt.figure(figsize=(10, 5))
            returns_pct = [r * 100 for r in training_metrics['total_returns']]
            plt.plot(returns_pct, linewidth=2, color='green')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.title('Total Returns During Training (%)')
            plt.xlabel('Episode')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            returns_plot_path = os.path.join(
                results_dir, 'total_returns_plot.png')
            plt.savefig(returns_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Actor Loss Plot
            plt.figure(figsize=(10, 5))
            plt.plot(training_metrics['actor_losses'],
                     linewidth=2, color='orange')
            plt.title('Actor Loss During Training')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            actor_loss_path = os.path.join(results_dir, 'actor_loss_plot.png')
            plt.savefig(actor_loss_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Critic Loss Plot
            plt.figure(figsize=(10, 5))
            plt.plot(training_metrics['critic_losses'],
                     linewidth=2, color='red')
            plt.title('Critic Loss During Training')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            critic_loss_path = os.path.join(
                results_dir, 'critic_loss_plot.png')
            plt.savefig(critic_loss_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Trading Actions Distribution
            if 'actions_taken' in final_eval and final_eval['actions_taken']:
                plt.figure(figsize=(8, 6))
                actions_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                actions_taken = final_eval['actions_taken']
                action_counts = pd.Series(actions_taken).value_counts()
                action_labels = [actions_map.get(
                    i, f'Action {i}') for i in action_counts.index]

                colors = ['skyblue', 'lightgreen', 'salmon']
                plt.pie(action_counts.values, labels=action_labels, autopct='%1.1f%%',
                        startangle=90, colors=colors[:len(action_counts)])
                plt.title('Distribution of Trading Actions')
                plt.tight_layout()
                actions_plot_path = os.path.join(
                    results_dir, 'trading_actions_plot.png')
                plt.savefig(actions_plot_path, dpi=300, bbox_inches='tight')
                plt.close()

        print(f"✅ Individual plots saved to {results_dir}")

    except Exception as e:
        print(f"⚠️ Error saving individual plots: {e}")
