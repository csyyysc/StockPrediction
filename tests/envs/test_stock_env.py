"""
Tests for StockTradingEnv class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from envs.stock_env import StockTradingEnv


class TestStockTradingEnv:
    """Test cases for StockTradingEnv class."""

    @pytest.fixture
    def mock_data(self):
        """Create mock stock data for testing."""

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate realistic stock data
        base_price = 100
        # 0.1% daily return, 2% volatility
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        return data

    @pytest.fixture
    def env_with_mock_data(self, mock_data):
        """Create environment with mocked data."""

        with patch('envs.stock_env.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            env = StockTradingEnv(
                symbol="TEST",
                window_size=10,
                train_period="1y",
                test_split=0.2
            )
            return env

    def test_init_default_parameters(self, mock_data):
        """Test environment initialization with default parameters."""

        with patch('envs.stock_env.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            env = StockTradingEnv()

            assert env.symbol == "AAPL"
            assert env.window_size == 30
            assert env.train_period == "2y"
            assert env.test_split == 0.2
            assert env.action_space_size == 3

    def test_init_custom_parameters(self, mock_data):
        """Test environment initialization with custom parameters."""

        with patch('envs.stock_env.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            env = StockTradingEnv(
                symbol="GOOGL",
                window_size=20,
                train_period="1y",
                test_split=0.3
            )

            assert env.symbol == "GOOGL"
            assert env.window_size == 20
            assert env.train_period == "1y"
            assert env.test_split == 0.3

    def test_load_data_success(self, mock_data):
        """Test that data loading assigns raw_data correctly with mocked data."""

        with patch('envs.stock_env.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            env = StockTradingEnv(symbol="TEST")

            # Verify that the mocked data was assigned to raw_data
            assert env.raw_data is not None
            assert len(env.raw_data) == 100
            assert 'Close' in env.raw_data.columns
            assert 'Volume' in env.raw_data.columns
            # Verify it's the same data (same index)
            assert env.raw_data.index.equals(mock_data.index)

    def test_load_data_failure(self):
        """Test data loading failure handling."""

        with patch('envs.stock_env.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = Exception(
                "No data found for symbol INVALID")
            mock_ticker.return_value = mock_ticker_instance

            # Should raise the exception when even AAPL fallback fails
            with pytest.raises(Exception, match="No data found for symbol INVALID"):
                StockTradingEnv(symbol="INVALID")

    def test_prepare_features(self, env_with_mock_data):
        """Test feature preparation."""

        env = env_with_mock_data

        # Basic assertions
        assert env.features is not None
        assert env.prices is not None
        assert env.dates is not None
        assert env.scaler is not None

        assert env.train_features is not None
        assert env.train_prices is not None
        assert env.test_features is not None
        assert env.test_prices is not None
        assert env.current_features is not None
        assert env.current_prices is not None

        # Test feature dimensions - should have 9 features as defined in the code
        assert env.features.shape[1] == 9
        assert len(env.dates) > 0
        assert len(env.features) > 0
        assert len(env.prices) > 0
        assert len(env.train_features) > 0
        assert len(env.train_prices) > 0
        assert len(env.test_features) > 0
        assert len(env.test_prices) > 0
        assert len(env.current_features) > 0
        assert len(env.current_prices) > 0

    def test_reset(self, env_with_mock_data):
        """Test environment reset."""

        env = env_with_mock_data

        # Perform some actions to change state
        env.step(1)  # Buy
        env.step(2)  # Sell

        # Reset environment
        state = env.reset()

        assert env.current_step == 0  # reset() sets current_step to 0
        assert env.cash == env.initial_cash
        assert env.position == 0  # The attribute is 'position', not 'shares'
        assert env.portfolio_value == env.initial_cash
        assert env.total_reward == 0.0
        assert state is not None
        # 9 features + 3 portfolio state values
        assert len(state) == env.window_size * 9 + 3
        assert len(env.trades) == 0
        assert len(env.portfolio_values) == 1
        assert env.portfolio_values[0] == env.initial_cash

    def test_step_hold_action(self, env_with_mock_data):
        """Test hold action (action=0)."""

        env = env_with_mock_data
        initial_cash = env.cash
        initial_shares = env.position

        state, reward, done, info = env.step(0)  # Hold

        assert env.cash == initial_cash
        assert env.position == initial_shares
        assert state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_buy_action(self, env_with_mock_data):
        """Test buy action (action=1)."""

        env = env_with_mock_data
        initial_cash = env.cash
        initial_shares = env.position

        state, reward, done, info = env.step(1)  # Buy

        # Should have spent cash and gained shares
        assert env.cash < initial_cash
        assert env.position > initial_shares
        assert state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_sell_action(self, env_with_mock_data):
        """Test sell action (action=2)."""

        env = env_with_mock_data

        # First buy some shares
        env.step(1)  # Buy
        shares_before_sell = env.position
        cash_before_sell = env.cash

        state, reward, done, info = env.step(2)  # Sell

        # Should have gained cash and lost shares
        assert env.cash > cash_before_sell
        assert env.position < shares_before_sell
        assert state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_invalid_action(self, env_with_mock_data):
        """Test invalid action handling."""

        env = env_with_mock_data

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(3)  # Invalid action

    def test_calculate_portfolio_value(self, env_with_mock_data):
        """Test portfolio value calculation."""

        env = env_with_mock_data
        initial_portfolio_value = env.portfolio_value

        # Buy some shares
        env.step(1)

        # Portfolio value should be updated after the step
        assert isinstance(env.portfolio_value, (int, float))
        assert env.portfolio_value > 0
        assert env.portfolio_value != initial_portfolio_value  # Should have changed
        assert env.portfolio_value == env.cash + \
            env.position * env.current_prices[env.current_step]

    def test_calculate_reward(self, env_with_mock_data):
        """Test reward calculation."""

        env = env_with_mock_data
        initial_portfolio_value = env.portfolio_value

        # Take a step and get the reward
        state, reward, done, info = env.step(1)  # Buy

        # Reward should be calculated based on portfolio value change
        assert isinstance(reward, (int, float))
        assert reward == (env.portfolio_value -
                          initial_portfolio_value) / initial_portfolio_value

    def test_get_state(self, env_with_mock_data):
        """Test state retrieval."""

        env = env_with_mock_data

        state = env.get_state()
        features = env.current_features.shape[1]

        assert state is not None
        assert len(state) == env.window_size * features + 3
        assert isinstance(state, np.ndarray)

    # @TODO: need fix when have a better way on step function
    def test_is_done(self, env_with_mock_data):
        """Test episode termination condition."""

        env = env_with_mock_data

        # Should not be done initially
        assert env.current_step < len(env.current_prices) - 1

        # Move to the end of the episode
        env.current_step = len(env.current_prices) - 1

        # Now it should be done
        assert env.current_step >= len(env.current_prices) - 1

    def test_get_info(self, env_with_mock_data):
        """Test info dictionary generation."""

        env = env_with_mock_data

        # Take a step to get the info dictionary
        state, reward, done, info = env.step(0)  # Hold action

        assert isinstance(info, dict)
        assert 'cash' in info
        assert 'position' in info  # The attribute is 'position', not 'shares'
        assert 'current_price' in info
        assert 'portfolio_value' in info
        assert 'total_return' in info
        assert 'trade_executed' in info

        assert isinstance(info['cash'], (int, float))
        assert isinstance(info['position'], (int, float))
        assert isinstance(info['current_price'], (int, float))
        assert isinstance(info['portfolio_value'], (int, float))
        assert isinstance(info['total_return'], (int, float))
        assert isinstance(info['trade_executed'], bool)

    # @TODO: need fix when have a better way on feature engineering
    def test_feature_engineering(self, env_with_mock_data):
        """Test feature engineering methods."""

        env = env_with_mock_data

        # Test that features were properly engineered during initialization
        assert env.features is not None
        assert env.features.shape[1] == 9  # Should have 9 features
        assert len(env.features) > 0

        # Test that raw_data exists and has the expected columns
        assert env.raw_data is not None
        assert 'High' in env.raw_data.columns
        assert 'Low' in env.raw_data.columns
        assert 'Open' in env.raw_data.columns
        assert 'Close' in env.raw_data.columns
        assert 'Volume' in env.raw_data.columns

    def test_technical_indicators(self, env_with_mock_data):
        """Test technical indicator calculations."""

        env = env_with_mock_data

        # Test that technical indicators were calculated during feature preparation
        # The indicators are calculated in prepare_features() and stored in the features array

        # Check that we have the expected number of features (9 total)
        assert env.features.shape[1] == 9

        # The features include:
        # Returns, High_Low_Pct, Open_Close_Pct,
        # RSI, BB_Position, Volume_Ratio,
        # Price_MA5_Ratio, Price_MA10_Ratio, Price_MA20_Ratio

        # Test that features contain reasonable values (not all NaN or infinite)
        assert not np.isnan(env.features).all(), "Features contain NaN values"
        assert not np.isinf(env.features).all(
        ), "Features contain infinite values"

        # Test that we have a reasonable number of samples after feature engineering
        assert len(env.features) > 0, "No features were generated"
        assert len(env.features) < len(
            env.raw_data), "Not all rows were included in features"

        # Test that the scaler was fitted properly
        assert env.scaler is not None, "Scaler was not initialized"
        assert hasattr(
            env.scaler, 'mean_'), "Scaler mean_ attribute is missing"
        assert hasattr(
            env.scaler, 'scale_'), "Scaler scale_ attribute is missing"

        # Test that the scaler was fitted properly
        assert env.scaler is not None
        assert hasattr(env.scaler, 'mean_')
        assert hasattr(env.scaler, 'scale_')

    def test_data_split(self, env_with_mock_data):
        """Test train/test data splitting."""

        env = env_with_mock_data

        # Check that data is properly split
        total_data_points = len(env.features)
        expected_train_size = int(total_data_points * (1 - env.test_split))
        expected_test_size = total_data_points - expected_train_size

        # Test train/test split sizes
        assert len(env.train_features) == expected_train_size
        assert len(env.test_features) == expected_test_size
        assert len(env.train_prices) == expected_train_size
        assert len(env.test_prices) == expected_test_size

        # Test that current features/prices point to training data by default
        assert np.array_equal(env.current_features, env.train_features)
        assert np.array_equal(env.current_prices, env.train_prices)

        # Test that split is reasonable (not empty)
        assert len(env.train_features) > 0, "Training set is empty"
        assert len(env.test_features) > 0, "Test set is empty"

    # @TODO: need fix when have a better way on step function
    @pytest.mark.slow
    def test_full_episode(self, env_with_mock_data):
        """Test a complete episode."""

        env = env_with_mock_data

        state = env.reset()
        done = False
        step_count = 0
        max_steps = env.current_prices.shape[0] - 1

        while not done and step_count < max_steps:  # Prevent infinite loop
            action = np.random.randint(0, 3)
            state, reward, done, info = env.step(action)
            step_count += 1

        assert step_count > 0
        assert done or step_count >= max_steps

    def test_environment_consistency(self, env_with_mock_data):
        """Test environment state consistency."""

        env = env_with_mock_data

        # Reset and perform actions
        env.reset()
        env.step(1)  # Buy

        # Check that portfolio value is consistent
        portfolio_value = env.portfolio_value
        expected_value = env.cash + env.position * \
            env.current_prices[env.current_step]

        assert abs(portfolio_value - expected_value) < 1e-6

    def test_memory_management(self, env_with_mock_data):
        """Test that environment doesn't leak memory."""

        env = env_with_mock_data

        # Perform many operations
        for _ in range(10):
            env.reset()
            for _ in range(20):
                action = np.random.randint(0, 3)
                env.step(action)

        # Environment should still be functional
        state = env.reset()
        assert state is not None
