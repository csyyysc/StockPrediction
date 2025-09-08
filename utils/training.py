"""
Training utilities for Stock Prediction Application.

This module contains training-related functions and logic.
"""

import os
import multiprocessing as mp
from typing import Optional

from trainers.trainer import StockTrainer
from .parameters import TrainingParameters


def run_training(params: TrainingParameters) -> None:
    """
    Run training from command line.

    Args:
        params: TrainingParameters instance containing all training configuration
    """
    print(f"🤖 Training {params.agent_type.upper()} model for {params.symbol}")
    print(f"📊 Configuration:")
    print(f"  - Agent Type: {params.agent_type}")
    print(f"  - Symbol: {params.symbol}")
    print(f"  - Window Size: {params.window_size} days")
    print(f"  - Training Period: {params.train_period}")
    print(f"  - Learning Rate: {params.learning_rate}")
    print(f"  - Episodes: {params.num_episodes}")

    _print_worker_info(params.agent_type, params.num_workers)
    print()

    trainer = StockTrainer(
        symbol=params.symbol,
        window_size=params.window_size,
        train_period=params.train_period,
        learning_rate=params.learning_rate,
        agent_type=params.agent_type,
        num_workers=params.num_workers
    )

    results = trainer.train(
        num_episodes=params.num_episodes,
        eval_frequency=100,
        save_frequency=250,
        verbose=params.verbose
    )

    _print_training_results(trainer, results)

    _generate_training_plots(trainer, results)


def _print_worker_info(agent_type: str, num_workers: Optional[int]) -> None:
    """Print worker thread information for A3C agent."""
    if agent_type == 'a3c':
        cpu_count = mp.cpu_count()
        if num_workers is None:
            actual_workers = max(1, cpu_count - 1)
            print(
                f"  - Workers: {actual_workers}/{cpu_count} CPU cores (auto)")
        elif num_workers == -1:
            print(f"  - Workers: {cpu_count}/{cpu_count} CPU cores (all)")
        else:
            actual_workers = max(1, min(num_workers, cpu_count))
            print(
                f"  - Workers: {actual_workers}/{cpu_count} CPU cores (custom)")


def _print_training_results(trainer: StockTrainer, results: dict) -> None:
    """Print training completion results."""
    final_eval = results['final_evaluation']
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETED!")
    print("="*50)
    print(trainer.get_trading_summary(final_eval))
    print(f"📁 Results saved to: {results['results_dir']}")


def _generate_training_plots(trainer: StockTrainer, results: dict) -> None:
    """Generate and save training plots."""
    print("📊 Generating training plots...")
    plot_path = os.path.join(results['results_dir'], 'training_progress.png')
    trainer.plot_training_progress(save_path=plot_path)
    print(f"📈 Training plots saved to: {plot_path}")
