"""
Analysis tab component for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render_analysis_tab(config: dict):
    """Render the detailed analysis tab."""
    st.header("🔍 Detailed Analysis")

    if 'training_results' in st.session_state and 'trainer' in st.session_state:
        results = st.session_state.training_results
        final_eval = results['final_evaluation']

        st.subheader("Trading Actions Analysis")

        actions_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        actions_taken = final_eval['actions_taken']
        action_counts = pd.Series(actions_taken).value_counts()
        action_labels = [actions_map[i] for i in action_counts.index]

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(action_counts.values, labels=action_labels,
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Trading Actions')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(actions_taken, 'o', markersize=3, alpha=0.6)
            ax.set_title('Trading Actions Over Time')
            ax.set_xlabel('Trading Day')
            ax.set_ylabel('Action (0=Hold, 1=Buy, 2=Sell)')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Hold', 'Buy', 'Sell'])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        if final_eval['trades']:
            st.subheader("Individual Trades")

            trades_df = pd.DataFrame(final_eval['trades'], columns=[
                                     'Action', 'Day', 'Price', 'Shares'])
            st.dataframe(trades_df, width='stretch')

            if len(trades_df) > 0:
                buy_trades = trades_df[trades_df['Action'] == 'BUY']
                sell_trades = trades_df[trades_df['Action'] == 'SELL']

                col1, col2 = st.columns(2)
                with col1:
                    if len(buy_trades) > 0:
                        st.metric("Average Buy Price",
                                  f"${buy_trades['Price'].mean():.2f}")
                        st.metric("Total Shares Bought",
                                  f"{buy_trades['Shares'].sum():,}")

                with col2:
                    if len(sell_trades) > 0:
                        st.metric("Average Sell Price",
                                  f"${sell_trades['Price'].mean():.2f}")
                        st.metric("Total Shares Sold",
                                  f"{sell_trades['Shares'].sum():,}")

        st.subheader("Model Configuration & Saved Files")

        col1, col2 = st.columns(2)

        with col1:
            config_info = f"""
            **Agent:** {st.session_state.trainer.agent.name}
            **Stock Symbol:** {st.session_state.symbol}
            **Training Window:** {config['window_size']} days
            **Data Period:** {config['train_period']}
            **Learning Rate:** {config['learning_rate']}
            **Training Episodes:** {config['num_episodes']}
            **State Dimension:** {st.session_state.trainer.env.get_state_size()}
            **Action Dimension:** {st.session_state.trainer.env.action_space_size}
            """
            st.info(config_info)

        with col2:
            if 'results_dir' in st.session_state.training_results:
                results_dir = st.session_state.training_results['results_dir']
                st.success(f"""
                **📁 Results Directory:**
                `{results_dir}`
                
                **📄 Saved Files:**
                **Models:** `best_model.pth`, `final_model.pth`
                **Data:** `training_metrics.json`, `best_evaluation.json`
                **Plots:** `training_progress.png` (combined)
                **Individual Plots:**
                - `portfolio_value_plot.png`
                - `episode_rewards_plot.png` 
                - `total_returns_plot.png`
                - `actor_loss_plot.png`
                - `critic_loss_plot.png`
                - `trading_actions_plot.png`
                """)
            else:
                st.warning("Results directory information not available.")

    else:
        st.info("👆 Please run training first to see detailed analysis.")
