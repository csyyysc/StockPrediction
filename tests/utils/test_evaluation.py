from utils.evaluation import run_evaluation
from utils.parameters import EvaluationParameters


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def load_trained_model(self, path: str) -> None:  # noqa: D401
        self.loaded = path

    def evaluate_episode(self) -> dict:
        return {"return": 1.0}

    def get_trading_summary(self, eval_results: dict) -> str:
        return f"summary:{eval_results.get('return')}"

    def plot_training_progress(self, *args, **kwargs):
        pass


def test_run_evaluation_missing_model(tmp_path, capsys):
    # Model path does not exist
    params = EvaluationParameters(
        model_path=str(tmp_path / "missing.pth"),
        symbol="AAPL",
        window_size=30,
        train_period="2y",
        agent_type="vac",
        num_workers=None,
    )

    run_evaluation(params)
    out = capsys.readouterr().out
    assert "Model file not found" in out


def test_run_evaluation_happy_path(tmp_path, monkeypatch, capsys):
    # Create a fake model file
    model_file = tmp_path / "model.pth"
    model_file.write_text("dummy")

    # Patch trainer factory to use DummyTrainer
    from utils import evaluation as evaluation_mod

    monkeypatch.setattr(
        evaluation_mod, "_create_trainer_for_evaluation", lambda params: DummyTrainer())

    params = EvaluationParameters(
        model_path=str(model_file),
        symbol="AAPL",
        window_size=30,
        train_period="2y",
        agent_type="vac",
        num_workers=None,
    )

    run_evaluation(params)

    out = capsys.readouterr().out
    assert "Evaluating VAC model" in out
    assert "📈 EVALUATION RESULTS" in out
