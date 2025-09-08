"""
Base agent interface for reinforcement learning algorithms.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseAgent(ABC):
    """Abstract base class for all reinforcement learning agents."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def store_transition(self, action: int, reward: float):
        """
        Store a transition for learning.

        Args:
            action: Action taken
            reward: Reward received
        """
        pass

    @abstractmethod
    def update(self, gamma: float = 0.99) -> Tuple[float, float]:
        """
        Update the agent's policy and value function.

        Args:
            gamma: Discount factor

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save the agent's model.

        Args:
            filepath: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load the agent's model.

        Args:
            filepath: Path to load the model from
        """
        pass

    def reset_episode_data(self):
        """Reset episode data storage. Override if needed."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the agent."""
        pass
