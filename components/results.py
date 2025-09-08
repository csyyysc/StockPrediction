"""
Results tab component for the Streamlit application.
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def render_results_tab():
    """Render the training results tab."""
    st.header("📈 Training Results")

    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        final_eval = results['final_evaluation']
        metrics = results['training_metrics']

        st.subheader("Performance Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Returns")
            st.metric("Algorithm Return",
                      f"{final_eval['total_return']:.2%}")
            st.metric("Buy & Hold Return",
                      f"{final_eval['buy_hold_return']:.2%}")
            st.metric("Excess Return",
                      f"{final_eval['excess_return']:.2%}")

        with col2:
            st.markdown("### 📈 Risk Metrics")
            st.metric("Sharpe Ratio", f"{final_eval['sharpe_ratio']:.3f}")
            st.metric("Maximum Drawdown",
                      f"{final_eval['max_drawdown']:.2%}")
            st.metric("Number of Trades", f"{final_eval['num_trades']}")

        st.subheader("Portfolio Value Over Time")
        portfolio_df = pd.DataFrame({
            'Step': range(len(final_eval['portfolio_values'])),
            'Portfolio Value': final_eval['portfolio_values']
        })

        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio_df['Step'], portfolio_df['Portfolio Value'],
                linewidth=2, color='green', label='Portfolio Value')
        ax.axhline(y=10000, color='red', linestyle='--',
                   label='Initial Investment', alpha=0.7)
        ax.set_title('Portfolio Value Over Time (Test Period)')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Training Progress")

        if (len(metrics['episode_rewards']) > 0 and
            len(metrics['actor_losses']) > 0 and
                len(metrics['critic_losses']) > 0):

            plt.clf()
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(15, 10))

            if metrics['episode_rewards']:
                ax1.plot(metrics['episode_rewards'],
                         linewidth=2, color='blue')
                ax1.set_title('Episode Rewards')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.grid(True, alpha=0.3)

            if metrics['total_returns']:
                returns_pct = [r * 100 for r in metrics['total_returns']]
                ax2.plot(returns_pct, linewidth=2, color='green')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                ax2.set_title('Total Returns (%)')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Return (%)')
                ax2.grid(True, alpha=0.3)

            if metrics['actor_losses']:
                ax3.plot(metrics['actor_losses'],
                         linewidth=2, color='orange')
                ax3.set_title('Actor Loss')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Loss')
                ax3.grid(True, alpha=0.3)

            if metrics['critic_losses']:
                ax4.plot(metrics['critic_losses'],
                         linewidth=2, color='red')
                ax4.set_title('Critic Loss')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Loss')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(
                "⚠️ Training metrics are empty. This might happen with very short training runs.")
            st.info(
                "💡 Try training for more episodes (50+) to see meaningful plots.")

    else:
        st.info("👆 Please run training first to see results.")
