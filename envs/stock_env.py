"""
Stock trading environment for reinforcement learning.
"""

import warnings
import numpy as np
import yfinance as yf
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class StockTradingEnv:
    """Stock trading environment using historical data."""

    def __init__(self,
                 symbol: str = "AAPL",
                 window_size: int = 30,
                 train_period: str = "2y",
                 test_split: float = 0.2):
        """
        Initialize the stock trading environment.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            window_size: Number of days to use as features
            train_period: Period of data to fetch (e.g., '1y', '2y', '5y')
            test_split: Fraction of data to use for testing
        """
        self.symbol = symbol
        self.window_size = window_size
        self.train_period = train_period
        self.test_split = test_split

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space_size = 3

        # Load and prepare data
        self.load_data()
        self.prepare_features()

        # Trading state
        self.reset()

    def load_data(self) -> None:
        """Load stock data from Yahoo Finance."""

        try:
            ticker = yf.Ticker(self.symbol)
            self.raw_data = ticker.history(period=self.train_period)

            if self.raw_data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            print(
                f"Loaded {len(self.raw_data)} days of data for {self.symbol}")

        except Exception as e:
            print(f"Error loading data for {self.symbol}: {e}")

            # Fallback to AAPL if symbol fails
            if self.symbol != "AAPL":
                print("Falling back to AAPL...")
                self.symbol = "AAPL"
                ticker = yf.Ticker(self.symbol)
                self.raw_data = ticker.history(period=self.train_period)
            else:
                raise e

    def prepare_features(self) -> None:
        """Prepare features from raw stock data."""

        df = self.raw_data.copy()

        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']

        # Technical indicators
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        # Relative strength index (simplified)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / \
            (df['BB_Upper'] - df['BB_Lower'])

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Price position relative to moving averages
        df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
        df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']

        # Select features for the model
        feature_columns = [
            'Returns', 'High_Low_Pct', 'Open_Close_Pct',
            'RSI', 'BB_Position', 'Volume_Ratio',
            'Price_MA5_Ratio', 'Price_MA10_Ratio', 'Price_MA20_Ratio'
        ]

        # Drop NaN values
        df = df.dropna()

        # Prepare features and targets
        self.features = df[feature_columns].values
        self.prices = df['Close'].values
        self.dates = df.index

        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Split data
        split_idx = int(len(self.features) * (1 - self.test_split))
        self.train_features = self.features[:split_idx]
        self.train_prices = self.prices[:split_idx]
        self.test_features = self.features[split_idx:]
        self.test_prices = self.prices[split_idx:]

        # Current dataset (training by default)
        self.current_features = self.train_features
        self.current_prices = self.train_prices

        print(
            f"Prepared {len(self.features)} samples with {self.features.shape[1]} features")
        print(
            f"Training samples: {len(self.train_features)}, Test samples: {len(self.test_features)}")

    def reset(self, mode: str = 'train') -> np.ndarray:
        """Reset the environment."""

        if mode == 'train':
            self.current_features = self.train_features
            self.current_prices = self.train_prices
        else:
            self.current_features = self.test_features
            self.current_prices = self.test_prices

        self.current_step = 0
        self.initial_cash = 10000.0
        self.cash = self.initial_cash
        self.position = 0  # Number of shares held
        self.portfolio_value = self.initial_cash
        self.total_reward = 0.0

        # Trading history
        self.trades = []
        self.portfolio_values = [self.initial_cash]

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current state (features for the window)."""

        if self.current_step < self.window_size:
            # Pad with zeros if we don't have enough history
            state = np.zeros(
                (self.window_size, self.current_features.shape[1]))
            available_steps = self.current_step + 1
            state[-available_steps:] = self.current_features[:available_steps]
        else:
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            state = self.current_features[start_idx:end_idx]

        # Add portfolio state (position, cash, portfolio value)
        portfolio_state = np.array([
            self.position,
            self.cash / self.initial_cash,  # Normalized cash
            self.portfolio_value / self.initial_cash  # Normalized portfolio value
        ])

        # Flatten state and concatenate with portfolio state
        flattened_state = state.flatten()
        full_state = np.concatenate([flattened_state, portfolio_state])

        return full_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""

        if self.current_step >= len(self.current_prices) - 1:
            return self.get_state(), 0.0, True, {}

        # Validate action
        if action not in [0, 1, 2]:
            raise ValueError(
                f"Invalid action: {action}. Valid actions are 0 (Hold), 1 (Buy), 2 (Sell)")

        # Get current and next prices
        current_price = self.current_prices[self.current_step]
        next_price = self.current_prices[self.current_step + 1]

        # Execute action
        reward = 0.0
        trade_executed = False

        if action == 0:  # Hold
            # No action taken
            pass
        elif action == 1:  # Buy
            shares_to_buy = self.cash // current_price
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.position += shares_to_buy
                self.cash -= cost
                trade_executed = True
                self.trades.append(
                    ('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # Sell
            if self.position > 0:
                revenue = self.position * current_price
                self.cash += revenue
                self.trades.append(
                    ('SELL', self.current_step, current_price, self.position))
                self.position = 0
                trade_executed = True

        # Move to next step
        self.current_step += 1

        # Calculate new portfolio value
        new_portfolio_value = self.cash + self.position * next_price

        # Calculate reward (portfolio return)
        reward = (new_portfolio_value - self.portfolio_value) / \
            self.portfolio_value

        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        self.portfolio_values.append(self.portfolio_value)
        self.total_reward += reward

        # Check if episode is done
        done = self.current_step >= len(self.current_prices) - 1

        # Additional info
        info = {
            'cash': self.cash,
            'position': self.position,
            'current_price': next_price,
            'portfolio_value': self.portfolio_value,
            'trade_executed': trade_executed,
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash
        }

        return self.get_state(), reward, done, info

    def get_state_size(self) -> int:
        """Get the size of the state space."""

        # Window size * number of features + portfolio state (3 values)
        return self.window_size * self.current_features.shape[1] + 3

    def set_mode(self, mode: str):
        """Set environment mode (train/test)."""

        if mode == 'train':
            self.current_features = self.train_features
            self.current_prices = self.train_prices
        else:
            self.current_features = self.test_features
            self.current_prices = self.test_prices

    def get_portfolio_performance(self) -> Dict:
        """Get portfolio performance metrics."""

        returns = np.array(self.portfolio_values)
        daily_returns = np.diff(returns) / returns[:-1]

        total_return = (self.portfolio_value -
                        self.initial_cash) / self.initial_cash

        # Buy and hold return for comparison
        start_price = self.current_prices[0]
        end_price = self.current_prices[self.current_step - 1]
        buy_hold_return = (end_price - start_price) / start_price

        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'volatility': np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0,
            'sharpe_ratio': (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'num_trades': len(self.trades)
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown) if len(drawdown) > 0 else 0.0
