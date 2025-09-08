"""
Training tab component for the Streamlit application.
"""

import os
import sys
import streamlit as st
import multiprocessing as mp

from .tools import save_web_plots
from trainers.trainer import StockTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_training_tab(config: dict):
    """Render the model training tab."""
    st.header("🤖 Model Training")

    agent_type = config['agent_type']
    agent_name = config['agent_name']
    symbol = config['symbol']
    window_size = config['window_size']
    train_period = config['train_period']
    learning_rate = config['learning_rate']
    num_episodes = config['num_episodes']
    num_workers = config['num_workers']

    config_text = f"""
    **Training Configuration:**
    - Agent: {agent_type.upper()} ({agent_name})
    - Stock Symbol: {symbol}
    - Window Size: {window_size} days
    - Data Period: {train_period}
    - Learning Rate: {learning_rate}
    - Episodes: {num_episodes}"""

    if agent_type == 'a3c':
        cpu_count = mp.cpu_count()
        if num_workers is None:
            actual_workers = max(1, cpu_count - 1)
            config_text += f"\n        - Workers: {actual_workers}/{cpu_count} CPU cores (auto)"
        elif num_workers == -1:
            config_text += f"\n        - Workers: {cpu_count}/{cpu_count} CPU cores (all)"
        else:
            config_text += f"\n        - Workers: {num_workers}/{cpu_count} CPU cores"

    st.info(config_text)

    if st.button("🚀 Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            trainer = StockTrainer(
                symbol=symbol,
                window_size=window_size,
                train_period=train_period,
                learning_rate=learning_rate,
                agent_type=agent_type,
                num_workers=num_workers
            )

            st.session_state.trainer = trainer
            st.session_state.symbol = symbol

            status_text.text("Initializing training...")

            status_text.text("Training in progress...")

            results = trainer.train(
                num_episodes=num_episodes,
                eval_frequency=max(50, num_episodes // 4),
                save_frequency=max(100, num_episodes // 2),
                verbose=False
            )

            progress_bar.progress(1.0)

            final_eval = results['final_evaluation']

            st.session_state.training_results = {
                'final_evaluation': final_eval,
                'training_metrics': results['training_metrics'],
                'results_dir': results['results_dir']
            }

            save_web_plots(
                trainer, results['results_dir'], final_eval, results['training_metrics'])

            status_text.text("✅ Training completed!")

            st.success(f"🎉 Training completed successfully!")
            st.info(f"📁 **Results saved to:** `{results['results_dir']}`\n\n" +
                    f"📄 **Files created:**\n" +
                    f"**Models:**\n" +
                    f"- `best_model.pth` - Best performing model\n" +
                    f"- `final_model.pth` - Final trained model\n\n" +
                    f"**Data:**\n" +
                    f"- `training_metrics.json` - Training history\n" +
                    f"- `best_evaluation.json` - Best evaluation results\n\n" +
                    f"**Plots:**\n" +
                    f"- `training_progress.png` - Combined training plots\n" +
                    f"- `portfolio_value_plot.png` - Portfolio performance\n" +
                    f"- `episode_rewards_plot.png` - Training rewards\n" +
                    f"- `total_returns_plot.png` - Return progression\n" +
                    f"- `actor_loss_plot.png` - Actor network loss\n" +
                    f"- `critic_loss_plot.png` - Critic network loss\n" +
                    f"- `trading_actions_plot.png` - Action distribution")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Final Return",
                    f"{final_eval['total_return']:.2%}",
                    delta=f"{final_eval['excess_return']:.2%}"
                )
            with col2:
                st.metric("Buy & Hold",
                          f"{final_eval['buy_hold_return']:.2%}")
            with col3:
                st.metric("Sharpe Ratio",
                          f"{final_eval['sharpe_ratio']:.3f}")
            with col4:
                st.metric("Max Drawdown",
                          f"{final_eval['max_drawdown']:.2%}")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)
