from utils.parameters import (
    create_argument_parser,
    create_training_parameters,
    create_evaluation_parameters,
    create_web_parameters,
)


def build_parser():
    return create_argument_parser(["vac", "a3c"])  # available agents


def test_default_args_and_training_parameters():
    parser = build_parser()
    args = parser.parse_args([])

    # Defaults
    assert args.mode == "web"
    assert args.symbol == "AAPL"
    assert args.agent == "vac"
    assert args.window_size == 30
    assert args.train_period == "2y"
    assert args.learning_rate == 0.001
    assert args.episodes == 500
    assert args.model is None

    # Create training params with defaults
    train_params = create_training_parameters(args)
    assert train_params.symbol == "AAPL"  # uppercased in creator
    assert train_params.window_size == 30
    assert train_params.train_period == "2y"
    assert train_params.learning_rate == 0.001
    assert train_params.num_episodes == 500
    assert train_params.agent_type == "vac"
    assert train_params.num_workers is None
    assert train_params.verbose is True


def test_custom_training_args_mapping():
    parser = build_parser()
    args = parser.parse_args([
        "--mode", "train",
        "--symbol", "tsla",
        "--agent", "a3c",
        "--window-size", "60",
        "--train-period", "3y",
        "--learning-rate", "0.0005",
        "--episodes", "1000",
        "--workers", "-1",
        "--quiet",
    ])

    assert args.mode == "train"
    train_params = create_training_parameters(args)
    assert train_params.symbol == "TSLA"  # creator uppercases
    assert train_params.agent_type == "a3c"
    assert train_params.window_size == 60
    assert train_params.train_period == "3y"
    assert train_params.learning_rate == 0.0005
    assert train_params.num_episodes == 1000
    assert train_params.num_workers == -1
    assert train_params.verbose is False


def test_evaluation_parameters_require_model():
    parser = build_parser()
    args = parser.parse_args(["--mode", "eval"])  # no --model

    try:
        create_evaluation_parameters(args)
        assert False, "Expected ValueError when model is missing"
    except ValueError as e:
        assert "--model path required" in str(e)


def test_evaluation_parameters_mapping():
    parser = build_parser()
    args = parser.parse_args([
        "--mode", "eval",
        "--model", "data/model.pth",
        "--symbol", "nvda",
        "--agent", "vac",
        "--window-size", "45",
        "--train-period", "1y",
        "--workers", "2",
    ])

    eval_params = create_evaluation_parameters(args)
    assert eval_params.model_path == "data/model.pth"
    assert eval_params.symbol == "NVDA"  # creator uppercases
    assert eval_params.agent_type == "vac"
    assert eval_params.window_size == 45
    assert eval_params.train_period == "1y"
    assert eval_params.num_workers == 2


def test_web_parameters_creation():
    parser = build_parser()
    args = parser.parse_args(["--mode", "web"])
    web_params = create_web_parameters(args)

    # WebParameters currently has no fields, just ensure instance created
    assert web_params is not None
