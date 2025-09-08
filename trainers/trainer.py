"""
Training module for reinforcement learning stock prediction models.
"""

import os
import json
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional
from datetime import datetime
from agent import get_agent, list_agents
from agent.a3c import A3CAgent
from envs.stock_env import StockTradingEnv


class StockTrainer:
    """Trainer for reinforcement learning stock prediction models."""

    def __init__(self,
                 symbol: str = "AAPL",
                 window_size: int = 30,
                 train_period: str = "2y",
                 learning_rate: float = 0.001,
                 agent_type: str = "vac",
                 num_workers: Optional[int] = None):
        """
        Initialize the trainer.

        Args:
            symbol: Stock symbol to trade
            window_size: Number of days for input features
            train_period: Period of historical data to use
            learning_rate: Learning rate for the optimizer
            agent_type: Type of agent to use ('vac' or 'a3c')
            num_workers: Number of worker threads for A3C (None=auto, -1=all cores)
        """
        self.symbol = symbol
        self.window_size = window_size
        self.train_period = train_period
        self.learning_rate = learning_rate
        self.agent_type = agent_type
        self.num_workers = num_workers

        # Validate agent type
        available_agents = list_agents()
        if agent_type not in available_agents:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {available_agents}")

        # Initialize environment
        self.env = StockTradingEnv(symbol, window_size, train_period)

        # Initialize agent with num_workers for A3C
        state_size = self.env.get_state_size()
        action_size = self.env.action_space_size

        if agent_type == 'a3c':
            self.agent = get_agent(agent_type, state_size, action_size, learning_rate,
                                   num_workers=num_workers)
        else:
            self.agent = get_agent(
                agent_type, state_size, action_size, learning_rate)

        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'portfolio_values': [],
            'actor_losses': [],
            'critic_losses': [],
            'total_returns': [],
            'excess_returns': []
        }

        # Create results directory with agent type
        self.results_dir = f"results_{agent_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

    def train_episode(self) -> Dict:
        """Train for one episode."""
        state = self.env.reset(mode='train')
        episode_reward = 0.0
        steps = 0

        while True:
            # Select action
            action = self.agent.select_action(state)

            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)

            # Store transition
            self.agent.store_transition(action, reward)

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Update agent
        actor_loss, critic_loss = self.agent.update()

        # Get performance metrics
        performance = self.env.get_portfolio_performance()

        return {
            'episode_reward': episode_reward,
            'steps': steps,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'portfolio_value': self.env.portfolio_value,
            'total_return': performance['total_return'],
            'excess_return': performance['excess_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'num_trades': performance['num_trades']
        }

    def evaluate_episode(self) -> Dict:
        """Evaluate the agent on test data."""
        state = self.env.reset(mode='test')
        episode_reward = 0.0
        steps = 0

        # Store actions for analysis
        actions_taken = []
        states_history = []

        while True:
            # Select action (no exploration)
            with torch.no_grad():
                action = self.agent.select_action(state)

            actions_taken.append(action)
            states_history.append(state.copy())

            # Take step in environment
            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Get performance metrics
        performance = self.env.get_portfolio_performance()

        return {
            'episode_reward': episode_reward,
            'steps': steps,
            'portfolio_value': self.env.portfolio_value,
            'total_return': performance['total_return'],
            'excess_return': performance['excess_return'],
            'buy_hold_return': performance['buy_hold_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'num_trades': performance['num_trades'],
            'portfolio_values': self.env.portfolio_values,
            'actions_taken': actions_taken,
            'trades': self.env.trades
        }

    def train(self,
              num_episodes: int = 1000,
              eval_frequency: int = 100,
              save_frequency: int = 500,
              verbose: bool = True) -> Dict:
        """
        Train the actor-critic agent.

        Args:
            num_episodes: Number of training episodes
            eval_frequency: How often to evaluate on test set
            save_frequency: How often to save the model
            verbose: Whether to print progress

        Returns:
            Dictionary containing training history and final evaluation
        """
        best_return = float('-inf')

        # Check if using threaded A3C - handle differently
        if isinstance(self.agent, A3CAgent):
            return self._train_threaded_a3c(num_episodes, eval_frequency, save_frequency, verbose)

        for episode in range(num_episodes):
            # Train episode
            train_results = self.train_episode()

            # Store metrics
            self.training_metrics['episode_rewards'].append(
                train_results['episode_reward'])
            self.training_metrics['portfolio_values'].append(
                train_results['portfolio_value'])
            self.training_metrics['actor_losses'].append(
                train_results['actor_loss'])
            self.training_metrics['critic_losses'].append(
                train_results['critic_loss'])
            self.training_metrics['total_returns'].append(
                train_results['total_return'])
            self.training_metrics['excess_returns'].append(
                train_results['excess_return'])

            # Print progress
            if verbose and (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Reward: {train_results['episode_reward']:.4f}")
                print(f"  Total Return: {train_results['total_return']:.2%}")
                print(f"  Excess Return: {train_results['excess_return']:.2%}")
                print(f"  Actor Loss: {train_results['actor_loss']:.4f}")
                print(f"  Critic Loss: {train_results['critic_loss']:.4f}")
                print()

            # Evaluate periodically
            if (episode + 1) % eval_frequency == 0:
                eval_results = self.evaluate_episode()

                if verbose:
                    print(f"=== Evaluation at Episode {episode + 1} ===")
                    print(
                        f"Test Total Return: {eval_results['total_return']:.2%}")
                    print(
                        f"Test Excess Return: {eval_results['excess_return']:.2%}")
                    print(
                        f"Buy & Hold Return: {eval_results['buy_hold_return']:.2%}")
                    print(f"Sharpe Ratio: {eval_results['sharpe_ratio']:.3f}")
                    print(f"Max Drawdown: {eval_results['max_drawdown']:.2%}")
                    print(f"Number of Trades: {eval_results['num_trades']}")
                    print("=" * 40)
                    print()

                # Save best model
                if eval_results['total_return'] > best_return:
                    best_return = eval_results['total_return']
                    model_path = os.path.join(
                        self.results_dir, 'best_model.pth')
                    self.agent.save_model(model_path)

                    # Save evaluation results
                    eval_path = os.path.join(
                        self.results_dir, 'best_evaluation.json')
                    with open(eval_path, 'w') as f:
                        # Convert numpy arrays to lists for JSON serialization
                        eval_results_json = eval_results.copy()
                        eval_results_json['portfolio_values'] = [
                            float(x) for x in eval_results_json['portfolio_values']]
                        eval_results_json['actions_taken'] = [
                            int(x) for x in eval_results_json['actions_taken']]
                        json.dump(eval_results_json, f, indent=2)

            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                model_path = os.path.join(
                    self.results_dir, f'model_episode_{episode + 1}.pth')
                self.agent.save_model(model_path)

        # Final evaluation
        final_eval = self.evaluate_episode()

        # Save training metrics
        metrics_path = os.path.join(self.results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)

        # Save final model
        final_model_path = os.path.join(self.results_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)

        return {
            'training_metrics': self.training_metrics,
            'final_evaluation': final_eval,
            'best_return': best_return,
            'results_dir': self.results_dir
        }

    def _train_threaded_a3c(self, num_episodes: int, eval_frequency: int, save_frequency: int, verbose: bool) -> Dict:
        """Special training method for threaded A3C agent."""
        if verbose:
            print(
                f"Starting threaded A3C training with {self.agent.num_workers} workers...")

        best_return = float('-inf')

        # Create environment factory for workers (reuse loaded data)
        def env_creator():
            # Create new environment instance but copy the already loaded data
            new_env = StockTradingEnv.__new__(StockTradingEnv)
            # Copy all attributes from the master environment
            for key, value in self.env.__dict__.items():
                if key in ['features', 'prices', 'dates', 'train_features', 'train_prices',
                           'test_features', 'test_prices', 'scaler']:
                    # Share the same data arrays (read-only)
                    setattr(new_env, key, value)
                else:
                    # Copy other attributes
                    setattr(new_env, key, value)
            # Reset the trading state for this worker
            new_env.reset()
            return new_env

        # Start the async training and get episode metrics
        try:
            episode_metrics = self.agent.train_synchronous_episodes(
                num_episodes, env_creator)

            # Process the metrics to match expected format
            for i, metrics in enumerate(episode_metrics):
                self.training_metrics['episode_rewards'].append(
                    metrics['episode_reward'])
                self.training_metrics['portfolio_values'].append(
                    metrics['portfolio_value'])
                self.training_metrics['actor_losses'].append(
                    metrics['actor_loss'])
                self.training_metrics['critic_losses'].append(
                    metrics['critic_loss'])
                self.training_metrics['total_returns'].append(
                    metrics['total_return'])
                self.training_metrics['excess_returns'].append(
                    metrics['excess_return'])

                # Print progress
                if verbose and (i + 1) % 50 == 0:
                    print(f"Episode {i + 1}/{num_episodes}")
                    print(f"  Reward: {metrics['episode_reward']:.4f}")
                    print(f"  Total Return: {metrics['total_return']:.2%}")
                    print(f"  Excess Return: {metrics['excess_return']:.2%}")
                    print()

                # Evaluate periodically
                if (i + 1) % eval_frequency == 0:
                    eval_results = self.evaluate_episode()

                    if verbose:
                        print(f"=== Evaluation at Episode {i + 1} ===")
                        print(
                            f"Test Total Return: {eval_results['total_return']:.2%}")
                        print(
                            f"Test Excess Return: {eval_results['excess_return']:.2%}")
                        print(
                            f"Buy & Hold Return: {eval_results['buy_hold_return']:.2%}")
                        print(
                            f"Sharpe Ratio: {eval_results['sharpe_ratio']:.3f}")
                        print(
                            f"Max Drawdown: {eval_results['max_drawdown']:.2%}")
                        print(
                            f"Number of Trades: {eval_results['num_trades']}")
                        print("=" * 40)
                        print()

                    # Save best model
                    if eval_results['total_return'] > best_return:
                        best_return = eval_results['total_return']
                        model_path = os.path.join(
                            self.results_dir, 'best_model.pth')
                        self.agent.save_model(model_path)

                        # Save evaluation results
                        eval_path = os.path.join(
                            self.results_dir, 'best_evaluation.json')
                        with open(eval_path, 'w') as f:
                            eval_results_json = eval_results.copy()
                            eval_results_json['portfolio_values'] = [
                                float(x) for x in eval_results_json['portfolio_values']]
                            eval_results_json['actions_taken'] = [
                                int(x) for x in eval_results_json['actions_taken']]
                            json.dump(eval_results_json, f, indent=2)

                # Save model periodically
                if (i + 1) % save_frequency == 0:
                    model_path = os.path.join(
                        self.results_dir, f'model_episode_{i + 1}.pth')
                    self.agent.save_model(model_path)

        except Exception as e:
            print(f"Error during threaded A3C training: {e}")
            # Stop the agent properly
            if hasattr(self.agent, 'stop_training'):
                self.agent.stop_training()
            raise

        # Final evaluation
        final_eval = self.evaluate_episode()

        # Save training metrics
        metrics_path = os.path.join(self.results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)

        # Save final model
        final_model_path = os.path.join(self.results_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)

        if verbose:
            print("Threaded A3C training completed!")

        return {
            'training_metrics': self.training_metrics,
            'final_evaluation': final_eval,
            'best_return': best_return,
            'results_dir': self.results_dir
        }

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f'Training Progress - {self.agent.name} - {self.symbol}', fontsize=16)

        # Episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # Portfolio values
        axes[0, 1].plot(self.training_metrics['portfolio_values'])
        axes[0, 1].axhline(y=10000, color='r',
                           linestyle='--', label='Initial Value')
        axes[0, 1].set_title('Portfolio Values')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portfolio Value ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Total returns
        returns_pct = [r * 100 for r in self.training_metrics['total_returns']]
        axes[0, 2].plot(returns_pct)
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_title('Total Returns')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Return (%)')
        axes[0, 2].grid(True)

        # Actor losses
        axes[1, 0].plot(self.training_metrics['actor_losses'])
        axes[1, 0].set_title('Actor Losses')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)

        # Critic losses
        axes[1, 1].plot(self.training_metrics['critic_losses'])
        axes[1, 1].set_title('Critic Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)

        # Excess returns
        excess_returns_pct = [
            r * 100 for r in self.training_metrics['excess_returns']]
        axes[1, 2].plot(excess_returns_pct)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_title('Excess Returns vs Buy & Hold')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Excess Return (%)')
        axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def load_trained_model(self, model_path: str):
        """Load a trained model."""
        self.agent.load_model(model_path)

    def get_trading_summary(self, eval_results: Dict) -> str:
        """Generate a trading summary report."""
        summary = f"""
=== Trading Summary for {self.symbol} ===
Agent: {self.agent.name}
Training Window: {self.window_size} days
Training Period: {self.train_period}

Performance Metrics:
- Total Return: {eval_results['total_return']:.2%}
- Buy & Hold Return: {eval_results['buy_hold_return']:.2%}
- Excess Return: {eval_results['excess_return']:.2%}
- Sharpe Ratio: {eval_results['sharpe_ratio']:.3f}
- Maximum Drawdown: {eval_results['max_drawdown']:.2%}
- Number of Trades: {eval_results['num_trades']}

Final Portfolio Value: ${eval_results['portfolio_value']:,.2f}
Initial Investment: $10,000.00
Profit/Loss: ${eval_results['portfolio_value'] - 10000:,.2f}
"""
        return summary
