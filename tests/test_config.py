"""Phase 0: settings come from environment variables, are validated, and never leak the API key."""
import dataclasses

import pytest

from cairn import config
from cairn.config import ConfigError, Settings, load_settings


def test_config_error_is_an_exception():
    assert issubclass(ConfigError, Exception)


def test_settings_is_a_frozen_dataclass():
    assert dataclasses.is_dataclass(Settings)
    s = load_settings({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.top_k = 99


def test_defaults_when_nothing_is_set():
    s = load_settings({})
    assert s.data_dir == config.PROJECT_ROOT / "data"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.top_k == 5
    assert s.log_level == "INFO"
    assert s.anthropic_api_key is None
    assert s.corpus_repo_url == "https://github.com/fastapi/fastapi.git"
    assert s.corpus_tag == "0.143.0"


def test_environment_overrides_defaults():
    s = load_settings({
        "CAIRN_DATA_DIR": "somewhere",
        "CAIRN_EMBEDDING_MODEL": "other/model",
        "CAIRN_TOP_K": "7",
        "CAIRN_LOG_LEVEL": "debug",
        "CAIRN_CORPUS_REPO_URL": "https://example.com/repo.git",
        "CAIRN_CORPUS_TAG": "1.2.3",
    })
    assert str(s.data_dir) == "somewhere"
    assert s.embedding_model == "other/model"
    assert s.top_k == 7
    assert s.log_level == "DEBUG"          # normalised to upper case
    assert s.corpus_repo_url == "https://example.com/repo.git"
    assert s.corpus_tag == "1.2.3"


def test_blank_values_count_as_unset():
    s = load_settings({"CAIRN_TOP_K": "", "CAIRN_LOG_LEVEL": "   ", "ANTHROPIC_API_KEY": "  "})
    assert s.top_k == 5
    assert s.log_level == "INFO"
    assert s.anthropic_api_key is None


@pytest.mark.parametrize("level", ["debug", "INFO", "Warning", "error"])
def test_valid_log_levels(level):
    assert load_settings({"CAIRN_LOG_LEVEL": level}).log_level == level.upper()


@pytest.mark.parametrize("env,variable", [
    ({"CAIRN_TOP_K": "abc"}, "CAIRN_TOP_K"),
    ({"CAIRN_TOP_K": "0"}, "CAIRN_TOP_K"),
    ({"CAIRN_TOP_K": "-3"}, "CAIRN_TOP_K"),
    ({"CAIRN_TOP_K": "2.5"}, "CAIRN_TOP_K"),
    ({"CAIRN_LOG_LEVEL": "LOUD"}, "CAIRN_LOG_LEVEL"),
])
def test_invalid_values_raise_config_error_naming_the_variable(env, variable):
    with pytest.raises(ConfigError, match=variable):
        load_settings(env)


def test_api_key_is_loaded_but_never_printed():
    s = load_settings({"ANTHROPIC_API_KEY": "sk-secret-123"})
    assert s.anthropic_api_key == "sk-secret-123"
    assert "sk-secret-123" not in repr(s)
    assert "sk-secret-123" not in str(s)


def test_reads_the_real_environment_when_called_without_arguments(monkeypatch):
    monkeypatch.setenv("CAIRN_TOP_K", "9")
    assert load_settings().top_k == 9
