"""
A3C (Asynchronous Advantage Actor-Critic) implementation with proper threading for stock prediction.
"""

import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Optional, Dict

from .base import BaseAgent


class A3CNetwork(nn.Module):
    """A3C neural network with shared feature extraction."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(A3CNetwork, self).__init__()

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights with smaller variance to prevent instability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic networks."""
        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get value, entropy, and log probability."""
        action_logits, value = self.forward(state)

        # Clamp logits to prevent extreme values
        action_logits = torch.clamp(action_logits, -20, 20)

        # Create categorical distribution with more stable softmax
        action_probs = F.softmax(action_logits, dim=-1)

        # Check for NaN values and handle them
        if torch.isnan(action_probs).any():
            print(
                f"Warning: NaN detected in action probabilities, using uniform distribution")
            action_probs = torch.ones_like(
                action_probs) / action_probs.shape[-1]

        # Ensure probabilities sum to 1 and are positive
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        action_probs = torch.clamp(action_probs, 1e-8, 1.0)

        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, value.squeeze(), entropy


class A3CWorker(threading.Thread):
    """Worker thread for A3C algorithm."""

    def __init__(self, worker_id: int, global_network: A3CNetwork, optimizer: torch.optim.Optimizer,
                 env_creator, state_dim: int, action_dim: int,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5, update_frequency: int = 20,
                 gamma: float = 0.99, lambda_gae: float = 0.95, gradient_lock=None, a3c_agent=None):
        super(A3CWorker, self).__init__()

        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env_creator = env_creator
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.gradient_lock = gradient_lock
        self.a3c_agent = a3c_agent  # Reference to main A3C agent for loss tracking

        # Create local network
        self.local_network = A3CNetwork(state_dim, action_dim)
        self.sync_with_global()

        # Create environment once and reuse it
        self.env = self.env_creator()

        # Episode storage
        self.reset_episode_data()

        # Control flags
        self.stop_flag = threading.Event()
        self.episode_count = 0
        self.total_steps = 0

    def sync_with_global(self):
        """Synchronize local network with global network safely."""

        state_dict = {k: v.clone().detach()
                      for k, v in self.global_network.state_dict().items()}
        self.local_network.load_state_dict(state_dict)

    def reset_episode_data(self):
        """Reset episode data storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def compute_gae_returns(self) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation (GAE) returns and advantages."""
        returns = []
        advantages = []

        if len(self.rewards) == 0:
            return returns, advantages

        # Add bootstrap value for the last state (assumed to be 0 if terminal)
        values = [v.item() for v in self.values] + [0.0]

        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return returns, advantages

    def update_global_network(self):
        """Update global network with local gradients."""
        if len(self.rewards) == 0:
            return 0.0, 0.0

        # Compute GAE returns and advantages
        returns, advantages = self.compute_gae_returns()

        if len(returns) == 0:
            return 0.0, 0.0

        # Convert to tensors
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)

        # Normalize advantages (disabled for debugging)
        # if len(advantages) > 1:
        #     advantages = (advantages - advantages.mean()) / \
        #         (advantages.std() + 1e-8)

        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        # Reduce value coefficient to match VAC (which uses 1.0 implicitly)
        total_loss = actor_loss + 1.0 * critic_loss + self.entropy_coeff * entropy_loss

        # Compute local gradients
        total_loss.backward()

        # Copy local gradients to global network with thread safety
        with self.gradient_lock:
            # Clear global gradients inside the lock
            self.optimizer.zero_grad()

            for local_param, global_param in zip(self.local_network.parameters(),
                                                 self.global_network.parameters()):
                if local_param.grad is not None:
                    # Check for NaN gradients
                    if torch.isnan(local_param.grad).any():
                        print(
                            f"Warning: NaN gradients detected in worker {self.worker_id}, skipping update")
                        return 0.0, 0.0

                    # Use detach() and clone() to avoid memory corruption
                    grad_copy = local_param.grad.detach().clone()
                    if global_param.grad is not None:
                        global_param.grad += grad_copy
                    else:
                        global_param.grad = grad_copy

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.global_network.parameters(), self.max_grad_norm)

            # Update global network
            self.optimizer.step()

        # Sync local network with updated global network
        self.sync_with_global()

        # Clear local gradients to free memory
        for param in self.local_network.parameters():
            if param.grad is not None:
                param.grad = None

        # Record losses for metrics tracking
        actor_loss_val = actor_loss.item()
        critic_loss_val = critic_loss.item()

        # Add to shared loss tracking through A3C agent reference
        if self.a3c_agent is not None:
            with self.a3c_agent.loss_lock:
                self.a3c_agent.shared_losses['actor_losses'].append(
                    actor_loss_val)
                self.a3c_agent.shared_losses['critic_losses'].append(
                    critic_loss_val)

        return actor_loss_val, critic_loss_val

    def run_episode(self):
        """Run a single episode."""
        # Reuse the existing environment instead of creating new one
        state = self.env.reset(mode='train')
        episode_reward = 0.0
        steps = 0

        self.reset_episode_data()

        while True:
            # Select action using local network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value, entropy = self.local_network.get_action_and_value(
                state_tensor)

            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)

            # Store transition
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.entropies.append(entropy)

            episode_reward += reward
            state = next_state
            steps += 1
            self.total_steps += 1

            # Update global network periodically or at episode end
            if len(self.rewards) >= self.update_frequency or done:
                actor_loss, critic_loss = self.update_global_network()
                self.reset_episode_data()

            if done:
                break

        self.episode_count += 1
        return episode_reward, steps, actor_loss if 'actor_loss' in locals() else 0.0, critic_loss if 'critic_loss' in locals() else 0.0

    def run(self):
        """Main worker thread loop."""
        print(f"Worker {self.worker_id} started")

        while not self.stop_flag.is_set():
            try:
                episode_reward, steps, actor_loss, critic_loss = self.run_episode()

                if self.episode_count % 10 == 0:
                    print(f"Worker {self.worker_id} - Episode {self.episode_count}, "
                          f"Reward: {episode_reward:.2f}, Steps: {steps}, "
                          f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                break

        print(f"Worker {self.worker_id} stopped")

    def stop(self):
        """Stop the worker thread."""
        self.stop_flag.set()


class A3CAgent(BaseAgent):
    """A3C (Asynchronous Advantage Actor-Critic) algorithm with proper threading."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.0003,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5, num_workers: Optional[int] = None,
                 update_frequency: int = 10):
        super().__init__(state_dim, action_dim, learning_rate)

        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency

        # Threading lock for gradient updates
        self.gradient_lock = threading.Lock()

        # Shared loss tracking for metrics
        self.shared_losses = {'actor_losses': [], 'critic_losses': []}
        self.loss_lock = threading.Lock()

        # Determine number of workers with user customization
        cpu_count = mp.cpu_count()
        if num_workers is None:
            # Default: use all CPU cores but leave one free, minimum 1
            self.num_workers = max(1, cpu_count - 1)
        elif num_workers == -1:
            # Special case: use all available cores
            self.num_workers = cpu_count
        else:
            # User specified: ensure it's between 1 and available cores
            self.num_workers = max(1, min(num_workers, cpu_count))

        print(f"A3C using {self.num_workers}/{cpu_count} CPU cores")

        # Global network
        self.global_network = A3CNetwork(state_dim, action_dim).to(self.device)
        # Note: Using threading instead of multiprocessing, so we don't need share_memory()

        # Global optimizer
        self.optimizer = torch.optim.Adam(
            self.global_network.parameters(), lr=learning_rate)

        # Worker threads
        self.workers = []
        self.is_training = False

        # For single-threaded evaluation
        self.local_network = A3CNetwork(state_dim, action_dim).to(self.device)
        self.sync_networks()

        # Episode storage for evaluation
        self.reset_episode_data()

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"A3C (Threaded - {self.num_workers} workers)"

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return f"Asynchronous Advantage Actor-Critic with {self.num_workers} parallel worker threads, GAE, and entropy regularization"

    def sync_networks(self):
        """Synchronize local network with global network."""
        self.local_network.load_state_dict(self.global_network.state_dict())

    def reset_episode_data(self):
        """Reset episode data storage (used for evaluation)."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action based on current policy (used for evaluation)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action, log_prob, value, entropy = self.local_network.get_action_and_value(
            state_tensor)

        # Store for evaluation
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        return action

    def store_transition(self, action: int, reward: float):
        """Store action and reward for the transition (used for evaluation)."""
        self.actions.append(action)
        self.rewards.append(reward)

    def start_training(self, env_creator, max_episodes_total: int = 100):
        """Start asynchronous training with worker threads."""
        if self.is_training:
            print("Training already in progress!")
            return

        self.is_training = True
        print(f"Starting A3C training with {self.num_workers} workers...")

        # Create and start worker threads
        for i in range(self.num_workers):
            worker = A3CWorker(
                worker_id=i,
                global_network=self.global_network,
                optimizer=self.optimizer,
                env_creator=env_creator,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                entropy_coeff=self.entropy_coeff,
                value_coeff=self.value_coeff,
                max_grad_norm=self.max_grad_norm,
                update_frequency=self.update_frequency,
                gradient_lock=self.gradient_lock,
                a3c_agent=self  # Pass reference to this A3C agent
            )
            self.workers.append(worker)
            worker.start()

        # Wait for workers to complete their episodes
        start_time = time.time()
        try:
            while self.is_training:
                time.sleep(1)

                # Check if total episodes across all workers reached the limit
                total_episodes = sum(
                    worker.episode_count for worker in self.workers)
                if total_episodes >= max_episodes_total:
                    break

                # Print progress
                if int(time.time() - start_time) % 30 == 0:  # Every 30 seconds
                    total_steps = sum(
                        worker.total_steps for worker in self.workers)
                    print(
                        f"Training progress: {total_episodes} episodes, {total_steps} total steps")

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            self.stop_training()

    def stop_training(self):
        """Stop all worker threads."""
        if not self.is_training:
            return

        print("Stopping A3C training...")

        # Stop all workers
        for worker in self.workers:
            worker.stop()

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        self.is_training = False

        # Sync local network with final global state
        self.sync_networks()

        print("A3C training stopped")

    def train_synchronous_episodes(self, num_episodes: int, env_creator) -> List[Dict]:
        """Train synchronously for compatibility with existing trainer interface."""
        print(f"Training A3C synchronously for {num_episodes} episodes...")

        # Start async training in background
        training_thread = threading.Thread(
            target=self.start_training,
            # num_episodes is now total episodes, not per worker
            args=(env_creator, num_episodes)
        )
        training_thread.start()

        # Monitor progress and collect metrics
        episode_metrics = []
        last_episode_count = 0

        while training_thread.is_alive():
            time.sleep(1)

            # Get current progress from workers
            current_episodes = sum(
                worker.episode_count for worker in self.workers) if self.workers else 0

            # Simulate episode completion for trainer compatibility
            if current_episodes > last_episode_count:
                for _ in range(current_episodes - last_episode_count):
                    # Run a quick evaluation episode for metrics
                    eval_metrics = self._quick_evaluation(env_creator())
                    episode_metrics.append(eval_metrics)

                last_episode_count = current_episodes

            # Stop when we have enough episodes OR training is complete
            if len(episode_metrics) >= num_episodes:
                break

        # Wait for training to complete
        training_thread.join()

        # Ensure we have at least the requested number of metrics
        # Fill any remaining slots with final evaluation
        while len(episode_metrics) < num_episodes:
            eval_metrics = self._quick_evaluation(env_creator())
            episode_metrics.append(eval_metrics)

        # Return exactly the requested number of episodes
        return episode_metrics[:num_episodes]

    def _quick_evaluation(self, env) -> Dict:
        """Quick evaluation episode for metrics collection."""
        state = env.reset(mode='train')
        episode_reward = 0.0
        steps = 0

        while True:
            # Use local network for action selection
            action = self.get_deterministic_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        performance = env.get_portfolio_performance()

        # Get average recent losses from workers
        actor_loss = 0.0
        critic_loss = 0.0

        with self.loss_lock:
            if self.shared_losses['actor_losses']:
                # Get average of recent losses (last 50 updates)
                recent_actor = self.shared_losses['actor_losses'][-50:]
                recent_critic = self.shared_losses['critic_losses'][-50:]
                actor_loss = sum(recent_actor) / len(recent_actor)
                critic_loss = sum(recent_critic) / len(recent_critic)

        return {
            'episode_reward': episode_reward,
            'steps': steps,
            'actor_loss': actor_loss,  # Actual average losses from workers
            'critic_loss': critic_loss,
            'portfolio_value': env.portfolio_value,
            'total_return': performance['total_return'],
            'excess_return': performance['excess_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'num_trades': performance['num_trades']
        }

    def update(self, gamma: float = 0.99) -> Tuple[float, float]:
        """Update method for compatibility (actual updates happen in workers)."""
        # This is called during evaluation, but actual training happens in worker threads
        # Return recent average losses from workers
        actor_loss = 0.0
        critic_loss = 0.0

        with self.loss_lock:
            if self.shared_losses['actor_losses']:
                # Get average of recent losses (last 10 updates)
                recent_actor = self.shared_losses['actor_losses'][-10:]
                recent_critic = self.shared_losses['critic_losses'][-10:]
                actor_loss = sum(recent_actor) / len(recent_actor)
                critic_loss = sum(recent_critic) / len(recent_critic)

        return actor_loss, critic_loss

    def save_model(self, filepath: str):
        """Save the global model state."""
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'local_network_state_dict': self.local_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coeff': self.entropy_coeff,
            'value_coeff': self.value_coeff,
            'max_grad_norm': self.max_grad_norm,
            'num_workers': self.num_workers
        }, filepath)

    def load_model(self, filepath: str):
        """Load the model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_network.load_state_dict(
            checkpoint['global_network_state_dict'])
        self.local_network.load_state_dict(
            checkpoint['local_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load hyperparameters if available
        if 'entropy_coeff' in checkpoint:
            self.entropy_coeff = checkpoint['entropy_coeff']
        if 'value_coeff' in checkpoint:
            self.value_coeff = checkpoint['value_coeff']
        if 'max_grad_norm' in checkpoint:
            self.max_grad_norm = checkpoint['max_grad_norm']

    def get_deterministic_action(self, state: np.ndarray) -> int:
        """Get deterministic action for evaluation (no exploration)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.local_network.get_action_and_value(
                state_tensor, deterministic=True)

        return action
