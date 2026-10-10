import importlib.util
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_gallery_runs_without_training_dependencies():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value("Browse Final Comparisons").run()
    assert not app.exception
    assert not app.error
    assert len(app.selectbox) == 2
    app.selectbox[0].set_value("make_older").run()
    assert not app.exception
    assert not app.error
    app.sidebar.radio[0].set_value("Interactive Editor").run()
    assert not app.exception
    if importlib.util.find_spec("torch") is None:
        assert app.info
        assert not app.button


def test_missing_lora_fails_before_model_loading(tmp_path, monkeypatch):
    from src.inference import pipeline

    monkeypatch.setitem(pipeline.PATHS, "exports_dir", tmp_path / "exports")
    with pytest.raises(FileNotFoundError, match="LoRA weights"):
        pipeline.load_pipeline("sd15", run_dir=tmp_path)
