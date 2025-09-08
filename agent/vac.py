"""
Vanilla Actor-Critic implementation for stock prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List

from .base import BaseAgent


class ActorCritic(nn.Module):
    """Vanilla Actor-Critic neural network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic networks."""
        shared_features = self.shared_layers(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Sample action from policy distribution."""
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class VanillaActorCritic(BaseAgent):
    """Vanilla Actor-Critic algorithm implementation."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        super().__init__(state_dim, action_dim, learning_rate)

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Storage for episode data
        self.reset_episode_data()

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return "Vanilla Actor-Critic"

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Classic actor-critic algorithm with shared feature extraction and synchronous updates"

    def reset_episode_data(self):
        """Reset episode data storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action based on current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.model.get_action(state_tensor)

        # Store for training
        _, value = self.model.forward(state_tensor)
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())

        return action

    def store_transition(self, action: int, reward: float):
        """Store action and reward for the transition."""
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns."""
        returns = []
        discounted_sum = 0

        for reward in reversed(self.rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        return returns

    def update(self, gamma: float = 0.99) -> Tuple[float, float]:
        """Update the actor-critic model."""
        if len(self.rewards) == 0:
            return 0.0, 0.0

        # Ensure we have matching numbers of values/log_probs and rewards
        # We need to exclude the last value/log_prob since it doesn't have a corresponding reward
        num_steps = len(self.rewards)
        if len(self.values) > num_steps:
            # Remove the last value and log_prob (from the final state with no reward)
            self.values = self.values[:num_steps]
            self.log_probs = self.log_probs[:num_steps]

        if len(self.values) == 0 or len(self.log_probs) == 0:
            return 0.0, 0.0

        # Compute returns
        returns = self.compute_returns(gamma)
        returns = torch.FloatTensor(returns).to(self.device)

        # Convert stored data to tensors
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        # Ensure all tensors have the same length
        min_length = min(len(returns), len(log_probs), len(values))
        returns = returns[:min_length]
        log_probs = log_probs[:min_length]
        values = values[:min_length]

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - values.detach()

        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = actor_loss + critic_loss

        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Reset episode data
        self.reset_episode_data()

        return actor_loss.item(), critic_loss.item()

    def save_model(self, filepath: str):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load_model(self, filepath: str):
        """Load the model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
