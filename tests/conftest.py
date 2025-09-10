import sys
import pytest
from pathlib import Path


# Ensure project root is importable (so `components` and others can be imported)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Use a non-interactive matplotlib backend during tests
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass


@pytest.fixture(autouse=True)
def disable_side_effects(monkeypatch):
    """Disable filesystem writes and heavy training during tests."""

    # No-op plotting in utils.training
    try:
        import utils.training as training_mod
        if hasattr(training_mod, "_generate_training_plots"):
            monkeypatch.setattr(
                training_mod, "_generate_training_plots", lambda *a, **k: None)
    except Exception:
        pass

    # Provide a lightweight dummy StockTrainer globally
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, num_episodes=1, eval_frequency=1, save_frequency=1, verbose=False):
            return {
                "final_evaluation": {
                    "total_return": 0.0,
                    "buy_hold_return": 0.0,
                    "excess_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                },
                "training_metrics": {
                    "episode_rewards": [],
                    "total_returns": [],
                    "actor_losses": [],
                    "critic_losses": [],
                },
                "results_dir": "/tmp",
            }

        def load_trained_model(self, path: str):
            return None

        def evaluate_episode(self):
            return {"return": 0.0}

        def get_trading_summary(self, eval_results: dict) -> str:
            return "summary:0.0"

        def plot_training_progress(self, *args, **kwargs):
            return None

    try:
        import trainers.trainer as trainer_mod
        if hasattr(trainer_mod, "StockTrainer"):
            monkeypatch.setattr(trainer_mod, "StockTrainer", DummyTrainer)
    except Exception:
        pass

    # No-op save_web_plots in components.training if loaded
    try:
        import components.training as comp_training
        if hasattr(comp_training, "save_web_plots"):
            monkeypatch.setattr(
                comp_training, "save_web_plots", lambda *a, **k: None)
    except Exception:
        pass
