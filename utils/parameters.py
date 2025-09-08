"""
Parameter management for Stock Prediction Application.

This module contains parameter classes and argument parser setup
for the stock prediction application.
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List, Literal

# Type aliases for better type safety
AppMode = Literal["web", "train", "eval"]
AgentType = Literal["vac", "a3c"]


@dataclass
class TrainingParameters:
    """Parameters for training mode."""
    symbol: str = "AAPL"
    window_size: int = 30
    train_period: str = "2y"
    learning_rate: float = 0.001
    num_episodes: int = 500
    agent_type: str = "vac"
    num_workers: Optional[int] = None
    verbose: bool = True


@dataclass
class EvaluationParameters:
    """Parameters for evaluation mode."""
    model_path: str
    symbol: str = "AAPL"
    window_size: int = 30
    train_period: str = "2y"
    agent_type: AgentType = "vac"
    num_workers: Optional[int] = None


@dataclass
class WebParameters:
    """Parameters for web interface mode."""
    # Currently no specific parameters for web mode
    pass


def create_argument_parser(available_agents: List[AgentType]) -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the application.

    Args:
        available_agents: List of available agent types

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Stock Prediction Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  uv run main.py                                          # Launch web interface
  uv run main.py --mode train --symbol TSLA --agent a3c  # Train A3C on Tesla stock
  uv run main.py --mode eval --model path.pth --agent vac # Evaluate VAC model

Available agents: {', '.join(available_agents)}
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["web", "train", "eval"],
        default="web",
        help="Application mode (default: web)"
    )

    # Stock symbol
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Stock symbol to trade (default: AAPL)"
    )

    # Agent type
    parser.add_argument(
        "--agent",
        choices=available_agents,
        default="vac",
        help=f"Agent type to use (default: vac). Available: {', '.join(available_agents)}"
    )

    # Worker threads (for A3C)
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads for A3C (default: auto, -1: all cores, 1-N: specific count)"
    )

    # Training parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Training window size in days (default: 30)"
    )

    parser.add_argument(
        "--train-period",
        default="2y",
        help="Training data period (default: 2y)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)"
    )

    # Model path (for evaluation)
    parser.add_argument(
        "--model",
        help="Path to saved model (for evaluation mode)"
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    return parser


def parse_arguments(available_agents: List[AgentType]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        available_agents: List of available agent types

    Returns:
        Parsed arguments namespace
    """
    parser = create_argument_parser(available_agents)
    return parser.parse_args()


def create_training_parameters(args: argparse.Namespace) -> TrainingParameters:
    """
    Create TrainingParameters from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        TrainingParameters instance
    """
    return TrainingParameters(
        symbol=args.symbol.upper(),
        window_size=args.window_size,
        train_period=args.train_period,
        learning_rate=args.learning_rate,
        num_episodes=args.episodes,
        agent_type=args.agent,
        num_workers=args.workers,
        verbose=not args.quiet
    )


def create_evaluation_parameters(args: argparse.Namespace) -> EvaluationParameters:
    """
    Create EvaluationParameters from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        EvaluationParameters instance

    Raises:
        ValueError: If model path is not provided
    """
    if not args.model:
        raise ValueError("--model path required for evaluation mode")

    return EvaluationParameters(
        model_path=args.model,
        symbol=args.symbol.upper(),
        window_size=args.window_size,
        train_period=args.train_period,
        agent_type=args.agent,
        num_workers=args.workers
    )


def create_web_parameters(args: argparse.Namespace) -> WebParameters:
    """
    Create WebParameters from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        WebParameters instance
    """
    return WebParameters()
