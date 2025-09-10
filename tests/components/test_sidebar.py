import sys
import types
import importlib
from types import SimpleNamespace


def test_render_sidebar_basic(monkeypatch):
    # Mock agent list
    monkeypatch.setitem(__import__("sys").modules, "agent",
                        SimpleNamespace(list_agents=lambda: ["vac", "a3c"]))

    # Minimal streamlit stubs capturing calls/returns
    class Sidebar:
        def __init__(self):
            self.values = {}

        def header(self, *_a, **_k):
            pass

        def subheader(self, *_a, **_k):
            pass

        def selectbox(self, label, options=None, index=0, help=None):
            # Return first option to keep deterministic
            return options[index] if options else None

        def text_input(self, *_a, **_k):
            return ""

        def slider(self, label, min_value=None, max_value=None, value=None, **_k):
            return value

        def success(self, *_a, **_k):
            pass

        def warning(self, *_a, **_k):
            pass

        def info(self, *_a, **_k):
            pass

        def markdown(self, *_a, **_k):
            pass

    # Build a proper streamlit module stub
    st_mod = types.ModuleType("streamlit")
    st_mod.sidebar = Sidebar()

    # Ensure clean import state and inject stub streamlit
    for name in ["components.sidebar", "streamlit"]:
        if name in sys.modules:
            del sys.modules[name]
    monkeypatch.setitem(sys.modules, "streamlit", st_mod)

    sidebar_module = importlib.import_module("components.sidebar")

    cfg = sidebar_module.render_sidebar()

    assert set(["agent_type", "agent_name", "num_workers", "symbol", "window_size",
               "train_period", "learning_rate", "num_episodes"]) <= set(cfg.keys())
    # Defaults given our stub should be stable
    assert cfg["agent_type"] in ("vac", "a3c")
    assert isinstance(cfg["window_size"], int)
