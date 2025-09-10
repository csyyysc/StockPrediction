from typing import List
from utils.web import run_web_interface
from utils.parameters import WebParameters


class SubprocessRunSpy:
    def __init__(self) -> None:
        self.called = False
        self.args: List[str] | None = None

    def __call__(self, args: List[str]):  # type: ignore[override]
        self.called = True
        self.args = args
        return 0


def test_run_web_interface_defaults(monkeypatch, capsys):

    # Ensure env vars are unset to use defaults
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_ADDRESS", raising=False)

    spy = SubprocessRunSpy()
    monkeypatch.setattr("subprocess.run", spy)

    run_web_interface(WebParameters())

    captured = capsys.readouterr().out
    assert "🚀 Starting Streamlit web interface..." in captured
    # Default address is 0.0.0.0 -> production message
    assert "🌐 Production mode: Server will be accessible on port 8502" in captured

    assert spy.called is True
    assert spy.args is not None
    # Expected CLI invocation
    assert spy.args[:6] == [
        "uv",
        "run",
        "python",
        "-m",
        "streamlit",
        "run",
    ]
    # Ensure app and server flags are passed with defaults
    assert "app.py" in spy.args
    assert "--server.address" in spy.args
    assert "0.0.0.0" in spy.args
    assert "--server.port" in spy.args
    assert "8502" in spy.args
    assert "--server.headless" in spy.args
    assert "true" in spy.args


def test_run_web_interface_dev_mode(monkeypatch, capsys):
    # Dev address should change message; PORT left default so message matches code's text
    monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    monkeypatch.delenv("PORT", raising=False)

    spy = SubprocessRunSpy()
    monkeypatch.setattr("subprocess.run", spy)

    run_web_interface(WebParameters())

    captured = capsys.readouterr().out
    # Current implementation prints a dev message pointing to localhost:8502
    assert "📱 Development mode: Open your browser to http://localhost:8502" in captured

    assert spy.called is True
    # Should pass through env-derived address and default port
    assert "--server.address" in spy.args
    assert "localhost" in spy.args
    assert "--server.port" in spy.args
    assert "8502" in spy.args
