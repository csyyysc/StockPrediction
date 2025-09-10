import sys
import importlib
from types import SimpleNamespace


def test_overview_handles_no_symbol(monkeypatch):

    class ST:
        def header(self, *_a, **_k):
            pass

        def markdown(self, *_a, **_k):
            pass

        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

        def columns(self, *args, **kwargs):
            # Return stubs for metric columns
            return [SimpleNamespace(metric=lambda *a, **k: None) for _ in range(4)]

        def pyplot(self, *_a, **_k):
            pass

    # Ensure clean import state and stub streamlit
    if "components.overview" in sys.modules:
        del sys.modules["components.overview"]
    monkeypatch.setitem(sys.modules, "streamlit", ST())

    # Mock yfinance to return empty history so error path is exercised
    class MockTicker:
        def history(self, period="1mo"):
            class DF:
                empty = True
            return DF()

    monkeypatch.setitem(sys.modules, "yfinance",
                        SimpleNamespace(Ticker=lambda *_a, **_k: MockTicker()))

    overview_module = importlib.import_module("components.overview")
    overview_module.render_overview_tab("")
    overview_module.render_overview_tab("AAPL")
