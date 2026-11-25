import importlib.util
from pathlib import Path

import pytest

from cairn.serve.engine import set_torch_threads

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"


def load_script():
    spec = importlib.util.spec_from_file_location("serve_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_service_refuses_to_start_without_a_key_unless_told_to(monkeypatch, capsys):
    monkeypatch.delenv("CAIRN_API_KEY", raising=False)
    assert load_script().main(["--llm", "fake"]) == 1
    assert "CAIRN_API_KEY" in capsys.readouterr().err


def test_a_short_key_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("CAIRN_API_KEY", "short")
    assert load_script().main(["--llm", "fake"]) == 1
    assert "16 characters" in capsys.readouterr().err


def test_there_is_no_command_line_option_for_the_key():
    with pytest.raises(SystemExit):
        load_script().main(["--api-key", "x" * 20])


def test_torch_threads_are_set_for_the_process():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        set_torch_threads(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
