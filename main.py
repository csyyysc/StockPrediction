"""
Multi-Agent Stock Prediction Application

This application uses reinforcement learning algorithms to learn stock trading strategies.
Supports multiple agent types including Vanilla Actor-Critic (VAC) and A3C.
It can be run in different modes:

1. Web Interface (default): Interactive Streamlit app
2. Command Line Training: Direct training without GUI
3. Evaluation Mode: Load and evaluate pre-trained models
"""

import sys

from agent import list_agents
from utils import (
    parse_arguments,
    create_training_parameters,
    create_evaluation_parameters,
    create_web_parameters,
    run_training,
    run_evaluation,
    run_web_interface
)
from utils.parameters import AppMode


def main() -> None:
    """Main entry point."""

    available_agents = list_agents()
    args = parse_arguments(available_agents)
    mode: AppMode = args.mode  # type: ignore[assignment]

    try:
        if mode == "web":
            web_params = create_web_parameters(args)
            run_web_interface(web_params)

        elif mode == "train":
            train_params = create_training_parameters(args)
            run_training(train_params)

        elif mode == "eval":
            eval_params = create_evaluation_parameters(args)
            run_evaluation(eval_params)

        else:
            # This should never happen due to argparse choices, but provides type safety
            raise ValueError(f"Invalid mode: {mode}")

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
