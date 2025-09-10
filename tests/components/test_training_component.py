from components.training import render_training_tab


def test_training_button_flow(monkeypatch):
    # Stub streamlit interactions
    class Progress:
        def progress(self, *_a, **_k):
            pass

    class Empty:
        def text(self, *_a, **_k):
            pass

    class ST:
        def __init__(self):
            self.session_state = {}

        def header(self, *_a, **_k):
            pass

        def info(self, *_a, **_k):
            pass

        def button(self, *_a, **_k):
            # Simulate clicking the Start Training button
            return True

        def progress(self, *_a, **_k):
            return Progress()

        def empty(self, *_a, **_k):
            return Empty()

        def columns(self, n):
            # Return objects with metric method
            class Col:
                def metric(self, *_a, **_k):
                    pass

            return [Col() for _ in range(n)]

        def success(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

        def exception(self, *_a, **_k):
            pass

    st = ST()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", st)

    # Stub save_web_plots to no-op
    from components import training as training_mod
    monkeypatch.setattr(training_mod, "save_web_plots", lambda *a, **k: None)

    # Dummy trainer used by component
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, num_episodes, eval_frequency, save_frequency, verbose):
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

    monkeypatch.setattr(training_mod, "StockTrainer", DummyTrainer)

    config = {
        "agent_type": "vac",
        "agent_name": "Vanilla Actor-Critic",
        "symbol": "AAPL",
        "window_size": 30,
        "train_period": "2y",
        "learning_rate": 0.001,
        "num_episodes": 10,
        "num_workers": None,
    }

    render_training_tab(config)
