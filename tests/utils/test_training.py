import os
from typing import Any, Dict
from utils.training import run_training
from utils.parameters import TrainingParameters


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def train(self, num_episodes: int, eval_frequency: int, save_frequency: int, verbose: bool) -> Dict[str, Any]:
        return {
            "final_evaluation": {"return": 1.23},
            "results_dir": "/tmp/results"
        }

    def get_trading_summary(self, eval_results: dict) -> str:
        return f"summary:{eval_results.get('return')}"

    def plot_training_progress(self, save_path: str) -> None:
        # no file system side effects in tests
        pass


def test_run_training_a3c_worker_messages(monkeypatch, capsys, tmp_path):
    # Patch StockTrainer with DummyTrainer

    from utils import training as training_mod
    monkeypatch.setattr(training_mod, "StockTrainer", DummyTrainer)

    params = TrainingParameters(
        symbol="AAPL",
        window_size=30,
        train_period="2y",
        learning_rate=0.001,
        num_episodes=10,
        agent_type="a3c",
        num_workers=-1,
        verbose=True,
    )

    # Ensure results path write succeeds
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    run_training(params)
    out = capsys.readouterr().out
    assert "🤖 Training A3C model for AAPL" in out
    assert "Workers:" in out
    assert "🎉 TRAINING COMPLETED!" in out
