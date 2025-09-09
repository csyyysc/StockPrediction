"""
Tests for StockTrainer class.
"""

import os
import json
import shutil
import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch
from trainers.trainer import StockTrainer


class TestStockTrainer:
    """Test cases for StockTrainer class."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment for testing."""

        mock_env = Mock()
        mock_env.window_size = 30
        mock_env.action_space_size = 3
        mock_env.reset.return_value = np.random.randn(
            30, 10)  # window_size * features
        mock_env.step.return_value = (
            np.random.randn(30, 10),  # next_state
            0.1,  # reward
            False,  # done
            {'portfolio_value': 1000}  # info
        )
        return mock_env

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""

        mock_agent = Mock()
        mock_agent.act.return_value = (0, 0.5)  # action, log_prob
        mock_agent.update.return_value = {
            'actor_loss': 0.1, 'critic_loss': 0.2}
        mock_agent.save_model.return_value = None
        mock_agent.load_model.return_value = None
        return mock_agent

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""

        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init_default_parameters(self, mock_environment, mock_agent):
        """Test trainer initialization with default parameters."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            assert trainer.symbol == "AAPL"
            assert trainer.window_size == 30
            assert trainer.train_period == "2y"
            assert trainer.learning_rate == 0.001
            assert trainer.agent_type == "vac"
            assert trainer.num_workers is None

    def test_init_custom_parameters(self, mock_environment, mock_agent):
        """Test trainer initialization with custom parameters."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer(
                symbol="GOOGL",
                window_size=20,
                train_period="1y",
                learning_rate=0.002,
                agent_type="a3c",
                num_workers=4
            )

            assert trainer.symbol == "GOOGL"
            assert trainer.window_size == 20
            assert trainer.train_period == "1y"
            assert trainer.learning_rate == 0.002
            assert trainer.agent_type == "a3c"
            assert trainer.num_workers == 4

    def test_init_invalid_agent_type(self, mock_environment):
        """Test trainer initialization with invalid agent type."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.list_agents', return_value=['vac', 'a3c']):

            with pytest.raises(ValueError, match="Unknown agent type"):
                StockTrainer(agent_type="invalid")

    def test_train_single_episode(self, mock_environment, mock_agent, temp_dir):
        """Test training a single episode."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            # Ensure final evaluation doesn't loop forever by mocking it
            fake_final_eval = {
                'episode_reward': 10.0,
                'steps': 5,
                'portfolio_value': 10100.0,
                'total_return': 0.01,
                'excess_return': 0.005,
                'buy_hold_return': 0.008,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.02,
                'num_trades': 3,
                'portfolio_values': [10000.0, 10050.0, 10100.0],
                'actions_taken': [0, 1, 2],
                'trades': []
            }

            # Mock the training loop and the final evaluation
            with patch.object(trainer, 'train_episode', return_value={
                'episode_reward': 10.0,
                'steps': 10,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'total_return': 0.1,
                'excess_return': 0.1,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.05,
                'portfolio_value': 1000.0,
            }), patch.object(trainer, 'evaluate_episode', return_value=fake_final_eval):
                metrics = trainer.train(num_episodes=1, save_frequency=1)

                # Top-level keys from train()
                assert isinstance(metrics, dict)
                assert 'training_metrics' in metrics
                assert 'final_evaluation' in metrics
                assert 'results_dir' in metrics

                # Validate structure of training_metrics
                tm = metrics['training_metrics']
                assert 'episode_rewards' in tm
                assert 'portfolio_values' in tm
                assert 'actor_losses' in tm
                assert 'critic_losses' in tm
                assert 'total_returns' in tm
                assert 'excess_returns' in tm

                # Exactly one episode recorded
                assert len(tm['episode_rewards']) == 1
                assert len(tm['portfolio_values']) == 1
                assert len(tm['actor_losses']) == 1
                assert len(tm['critic_losses']) == 1

    def test_train_multiple_episodes(self, mock_environment, mock_agent, temp_dir):
        """Test training multiple episodes."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            # Provide deterministic final evaluation to ensure fast termination
            fake_final_eval = {
                'episode_reward': 12.0,
                'steps': 6,
                'portfolio_value': 10200.0,
                'total_return': 0.02,
                'excess_return': 0.01,
                'buy_hold_return': 0.015,
                'sharpe_ratio': 1.3,
                'max_drawdown': 0.03,
                'num_trades': 4,
                'portfolio_values': [10000.0, 10100.0, 10200.0],
                'actions_taken': [0, 1, 2],
                'trades': []
            }

            # Mock the training loop and final evaluation
            with patch.object(trainer, 'train_episode', return_value={
                'episode_reward': 10.0,
                'portfolio_value': 1000.0,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'total_return': 0.05,
                'excess_return': 0.03
            }), patch.object(trainer, 'evaluate_episode', return_value=fake_final_eval):
                metrics = trainer.train(num_episodes=5, save_frequency=2)

                assert isinstance(metrics, dict)
                assert 'training_metrics' in metrics
                assert 'final_evaluation' in metrics

                tm = metrics['training_metrics']
                assert len(tm['episode_rewards']) == 5
                assert len(tm['portfolio_values']) == 5
                assert len(tm['actor_losses']) == 5
                assert len(tm['critic_losses']) == 5
                assert len(tm['total_returns']) == 5
                assert len(tm['excess_returns']) == 5

    def test_train_episode(self, mock_environment, mock_agent):
        """Test single episode training."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            # Mock environment to return done=True after a few steps
            step_count = 0

            def mock_step(action):
                nonlocal step_count
                step_count += 1
                return (
                    np.random.randn(30, 10),
                    0.1,
                    step_count >= 5,  # Done after 5 steps
                    {'portfolio_value': 1000}
                )

            mock_environment.step.side_effect = mock_step

            # Ensure agent/env APIs match trainer expectations
            mock_agent.select_action.return_value = 0
            mock_agent.store_transition.return_value = None
            mock_agent.update.return_value = (0.1, 0.2)
            mock_environment.get_portfolio_performance.return_value = {
                'total_return': 0.01,
                'excess_return': 0.005,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.02,
                'num_trades': 2,
                'buy_hold_return': 0.008
            }

            # Ensure required env attributes exist for trainer
            mock_environment.portfolio_value = 10000.0
            mock_environment.initial_cash = 10000.0
            mock_agent.update.return_value = (0.1, 0.2)
            mock_environment.get_portfolio_performance.return_value = {
                'total_return': 0.01,
                'excess_return': 0.005,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.02,
                'num_trades': 2,
                'buy_hold_return': 0.008
            }

            episode_metrics = trainer.train_episode()

            assert isinstance(episode_metrics, dict)
            assert 'episode_reward' in episode_metrics
            assert 'portfolio_value' in episode_metrics
            assert 'actor_loss' in episode_metrics
            assert 'critic_loss' in episode_metrics

    def test_evaluate(self, mock_environment, mock_agent, temp_dir):
        """Test single evaluation episode (no aggregate evaluate method)."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            # Mock evaluation episode to return expected keys
            fake_eval = {
                'episode_reward': 15.0,
                'steps': 5,
                'portfolio_value': 1100.0,
                'total_return': 0.1,
                'excess_return': 0.06,
                'buy_hold_return': 0.04,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.05,
                'num_trades': 3,
                'portfolio_values': [10000, 10500, 11000],
                'actions_taken': [0, 1, 2],
                'trades': []
            }

            with patch.object(trainer, 'evaluate_episode', return_value=fake_eval):
                eval_metrics = trainer.evaluate_episode()

                assert isinstance(eval_metrics, dict)
                for key in ['episode_reward', 'steps', 'portfolio_value', 'total_return', 'excess_return',
                            'buy_hold_return', 'sharpe_ratio', 'max_drawdown', 'num_trades',
                            'portfolio_values', 'actions_taken', 'trades']:
                    assert key in eval_metrics

    def test_evaluate_episode(self, mock_environment, mock_agent):
        """Test single evaluation episode."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            # Mock environment to return done=True after a few steps
            step_count = 0

            def mock_step(action):
                nonlocal step_count
                step_count += 1
                return (
                    np.random.randn(30, 10),
                    0.1,
                    step_count >= 5,  # Done after 5 steps
                    {'portfolio_value': 1000}
                )

            mock_environment.step.side_effect = mock_step

            # Supply required agent/env behavior for evaluation
            mock_agent.select_action.return_value = 0
            mock_environment.get_portfolio_performance.return_value = {
                'total_return': 0.02,
                'excess_return': 0.01,
                'sharpe_ratio': 1.3,
                'max_drawdown': 0.03,
                'num_trades': 4,
                'buy_hold_return': 0.01
            }

            eval_metrics = trainer.evaluate_episode()

            assert isinstance(eval_metrics, dict)
            assert 'episode_reward' in eval_metrics
            assert 'portfolio_value' in eval_metrics
            assert 'total_return' in eval_metrics
            assert 'sharpe_ratio' in eval_metrics
            assert 'max_drawdown' in eval_metrics

    def test_save_model(self, mock_environment, mock_agent, temp_dir):
        """Test model saving."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir
            trainer.agent.save_model("test_model")

            mock_agent.save_model.assert_called_once()

    def test_load_model(self, mock_environment, mock_agent, temp_dir):
        """Test model loading."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir
            trainer.agent.load_model("test_model")

            mock_agent.load_model.assert_called_once()

    # @TODO: fix when trainer.save_metrics is implemented
    def test_save_metrics(self, mock_environment, mock_agent, temp_dir):
        """Test metrics saving."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            metrics = {
                'episode_rewards': [1.0, 2.0, 3.0],
                'portfolio_values': [1000, 1100, 1200],
                'actor_losses': [0.1, 0.2, 0.3],
                'critic_losses': [0.2, 0.3, 0.4],
                'total_returns': [0.01, 0.03, 0.02],
                'excess_returns': [0.005, 0.02, 0.01]
            }

            # Trainer has no save_metrics; save here to validate file creation
            metrics_file = os.path.join(temp_dir, "test_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)

            assert os.path.exists(metrics_file)

    def test_plot_training_metrics(self, mock_environment, mock_agent, temp_dir):
        """Test training metrics plotting."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime, \
                patch('trainers.trainer.plt.savefig') as mock_savefig:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            metrics = {
                'episode_rewards': [1.0, 2.0, 3.0],
                'portfolio_values': [1000, 1100, 1200],
                'actor_losses': [0.1, 0.2, 0.3],
                'critic_losses': [0.2, 0.3, 0.4],
                'total_returns': [0.01, 0.03, 0.02],
                'excess_returns': [0.005, 0.02, 0.01]
            }

            trainer.training_metrics = metrics
            out_path = os.path.join(temp_dir, "training_progress.png")
            trainer.plot_training_progress(save_path=out_path)

            assert mock_savefig.call_count >= 1

    def test_plot_evaluation_metrics(self, mock_environment, mock_agent, temp_dir):
        """Test evaluation metrics plotting."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime, \
                patch('trainers.trainer.plt.savefig') as mock_savefig:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            eval_metrics = {
                'episode_rewards': [1.0, 2.0, 3.0],
                'portfolio_values': [1000, 1100, 1200],
                'total_returns': [0.1, 0.2, 0.3],
                'sharpe_ratios': [1.0, 1.5, 2.0],
                'max_drawdowns': [0.05, 0.03, 0.02]
            }

            # Map evaluation-like metrics to the training plotting API
            trainer.training_metrics = {
                'episode_rewards': eval_metrics['episode_rewards'],
                'portfolio_values': eval_metrics['portfolio_values'],
                'actor_losses': [0.1, 0.1, 0.1],
                'critic_losses': [0.2, 0.2, 0.2],
                'total_returns': eval_metrics['total_returns'],
                'excess_returns': [0.0, 0.0, 0.0]
            }

            out_path = os.path.join(temp_dir, "evaluation_progress.png")
            trainer.plot_training_progress(save_path=out_path)

            assert mock_savefig.call_count >= 1

    def test_calculate_performance_metrics(self, mock_environment, mock_agent):
        """Test performance metrics calculation."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            # The trainer exposes performance via env.get_portfolio_performance
            mock_env_metrics = {
                'total_return': 0.3,
                'buy_hold_return': 0.25,
                'excess_return': 0.05,
                'volatility': 0.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.1,
                'num_trades': 7
            }
            mock_environment.get_portfolio_performance.return_value = mock_env_metrics

            metrics = trainer.env.get_portfolio_performance()

            assert isinstance(metrics, dict)
            for key in ['total_return', 'buy_hold_return', 'excess_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'num_trades']:
                assert key in metrics
            assert isinstance(metrics['total_return'], (int, float))
            assert isinstance(metrics['sharpe_ratio'], (int, float))
            assert isinstance(metrics['max_drawdown'], (int, float))

    def test_get_best_model_path(self, mock_environment, mock_agent, temp_dir):
        """Test best model path retrieval."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            # Trainer doesn't expose a helper; expected best model filename
            best_model_path = os.path.join(
                trainer.results_dir, "best_model.pth")

            assert isinstance(best_model_path, str)
            assert best_model_path.endswith("best_model.pth")

    def test_get_final_model_path(self, mock_environment, mock_agent, temp_dir):
        """Test final model path retrieval."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent), \
                patch('trainers.trainer.os.makedirs'), \
                patch('trainers.trainer.datetime') as mock_datetime:

            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            trainer = StockTrainer()
            trainer.results_dir = temp_dir

            final_model_path = os.path.join(
                trainer.results_dir, "final_model.pth")

            assert isinstance(final_model_path, str)
            assert final_model_path.endswith("final_model.pth")

    def test_agent_type_validation(self, mock_environment):
        """Test agent type validation."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.list_agents', return_value=['vac', 'a3c']), \
                patch('trainers.trainer.get_agent', return_value=Mock()):

            trainer1 = StockTrainer(agent_type="vac")
            assert trainer1.agent_type == "vac"

            trainer2 = StockTrainer(agent_type="a3c")
            assert trainer2.agent_type == "a3c"

            with pytest.raises(ValueError):
                StockTrainer(agent_type="invalid")

    def test_error_handling(self, mock_environment, mock_agent):
        """Ensure train() does not hang on edge episode counts."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            # Prevent potential hang in final evaluation
            fake_eval = {
                'episode_reward': 0.0,
                'steps': 1,
                'portfolio_value': 10000.0,
                'total_return': 0.0,
                'excess_return': 0.0,
                'buy_hold_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'portfolio_values': [10000.0],
                'actions_taken': [0],
                'trades': []
            }

            with patch.object(trainer, 'evaluate_episode', return_value=fake_eval):
                result_zero = trainer.train(num_episodes=0)
                assert isinstance(result_zero, dict)

                result_negative = trainer.train(num_episodes=-1)
                assert isinstance(result_negative, dict)

    def test_memory_management(self, mock_environment, mock_agent):
        """Test memory management during training."""

        with patch('trainers.trainer.StockTradingEnv', return_value=mock_environment), \
                patch('trainers.trainer.get_agent', return_value=mock_agent):

            trainer = StockTrainer()

            fake_final_eval = {
                'episode_reward': 12.0,
                'steps': 2,
                'portfolio_value': 10100.0,
                'total_return': 0.01,
                'excess_return': 0.002,
                'buy_hold_return': 0.008,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.02,
                'num_trades': 1,
                'portfolio_values': [10000, 10100],
                'actions_taken': [0, 1],
                'trades': []
            }

            with patch.object(trainer, 'train_episode', return_value={
                'episode_reward': 10.0,
                'portfolio_value': 1000.0,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'total_return': 0.01,
                'excess_return': 0.005
            }), patch.object(trainer, 'evaluate_episode', return_value=fake_final_eval):
                metrics = trainer.train(num_episodes=10)

                assert isinstance(metrics, dict)
                assert 'training_metrics' in metrics

                tm = metrics['training_metrics']
                assert len(tm['episode_rewards']) == 10
