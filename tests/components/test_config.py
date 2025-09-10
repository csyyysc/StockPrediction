import sys
import types


def test_configure_app(monkeypatch):
    calls = {"set_page_config": 0, "markdown": 0}

    # Build a fake streamlit module
    st_mod = types.ModuleType("streamlit")

    def set_page_config(**kwargs):
        calls["set_page_config"] += 1

    def markdown(*_args, **_kwargs):
        calls["markdown"] += 1
    st_mod.set_page_config = set_page_config
    st_mod.markdown = markdown

    # Ensure clean imports
    for name in ["streamlit", "components.config"]:
        if name in sys.modules:
            del sys.modules[name]

    # Inject stub before importing the module under test
    monkeypatch.setitem(sys.modules, "streamlit", st_mod)

    import importlib
    config_module = importlib.import_module("components.config")

    config_module.configure_app()
    assert calls["set_page_config"] == 1
    assert calls["markdown"] >= 1


def test_render_app_header(monkeypatch):
    called = {"markdown": 0}

    st_mod = types.ModuleType("streamlit")

    def markdown(*_args, **_kwargs):
        called["markdown"] += 1
    st_mod.markdown = markdown

    for name in ["streamlit", "components.config"]:
        if name in sys.modules:
            del sys.modules[name]

    # Inject stub before importing the module under test
    monkeypatch.setitem(sys.modules, "streamlit", st_mod)

    import importlib
    config_module = importlib.import_module("components.config")

    config_module.render_app_header()
    assert called["markdown"] == 1
