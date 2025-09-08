"""
Agent package for stock trading reinforcement learning algorithms.
"""

from .base import BaseAgent
from .vac import VanillaActorCritic
from .a3c import A3CAgent

__all__ = ['BaseAgent', 'VanillaActorCritic', 'A3CAgent']

# Agent registry for easy selection
AVAILABLE_AGENTS = {
    'vac': VanillaActorCritic,
    'a3c': A3CAgent  # Multi-threaded A3C (proper implementation)
}


def get_agent(agent_type: str, *args, **kwargs):
    """Get agent instance by type."""
    if agent_type not in AVAILABLE_AGENTS:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AVAILABLE_AGENTS.keys())}")

    return AVAILABLE_AGENTS[agent_type](*args, **kwargs)


def list_agents():
    """List all available agent types."""

    return list(AVAILABLE_AGENTS.keys())
