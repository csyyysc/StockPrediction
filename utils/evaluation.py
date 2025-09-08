"""
Evaluation utilities for Stock Prediction Application.

This module contains evaluation-related functions and logic.
"""

import os
from trainers.trainer import StockTrainer
from .parameters import EvaluationParameters


def run_evaluation(params: EvaluationParameters) -> None:
    """
    Evaluate a pre-trained model.

    Args:
        params: EvaluationParameters instance containing all evaluation configuration
    """
    print(
        f"🔍 Evaluating {params.agent_type.upper()} model: {params.model_path}")
    print(f"📊 Symbol: {params.symbol}")

    if not os.path.exists(params.model_path):
        print(f"❌ Error: Model file not found: {params.model_path}")
        return

    # Initialize trainer and load model
    trainer = _create_trainer_for_evaluation(params)
    trainer.load_trained_model(params.model_path)

    # Run evaluation
    eval_results = trainer.evaluate_episode()

    # Print results
    _print_evaluation_results(trainer, eval_results)


def _create_trainer_for_evaluation(params: EvaluationParameters) -> StockTrainer:
    """Create and configure trainer for evaluation."""
    return StockTrainer(
        symbol=params.symbol,
        window_size=params.window_size,
        train_period=params.train_period,
        agent_type=params.agent_type,
        num_workers=params.num_workers
    )


def _print_evaluation_results(trainer: StockTrainer, eval_results: dict) -> None:
    """Print evaluation results."""
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(trainer.get_trading_summary(eval_results))
