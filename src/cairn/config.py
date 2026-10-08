"""Settings for Cairn, read from environment variables."""
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str
    top_k: int
    log_level: str
    corpus_repo_url: str
    corpus_tag: str
    anthropic_api_key: str | None = field(default=None, repr=False)


def _read(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is not None:
        value = value.strip()
        if not value:
            return None
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    raw_level = _read(env, "CAIRN_LOG_LEVEL")
    log_level = (raw_level or "INFO").upper()
    if log_level not in LOG_LEVELS:
        raise ConfigError(
            f"CAIRN_LOG_LEVEL must be one of {sorted(LOG_LEVELS)}, got {raw_level!r}"
        )

    raw_top_k = _read(env, "CAIRN_TOP_K")
    if raw_top_k is None:
        top_k = 5
    else:
        try:
            top_k = int(raw_top_k)
        except ValueError as e:
            raise ConfigError(f"CAIRN_TOP_K must be a whole number, got {raw_top_k!r}") from e
        if top_k < 1:
            raise ConfigError(f"CAIRN_TOP_K must be at least 1, got {top_k}")

    return Settings(
        data_dir=Path(_read(env, "CAIRN_DATA_DIR") or PROJECT_ROOT / "data"),
        embedding_model=_read(env, "CAIRN_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
        top_k=top_k,
        log_level=log_level,
        corpus_repo_url=_read(env, "CAIRN_CORPUS_REPO_URL") or "https://github.com/fastapi/fastapi.git",
        corpus_tag=_read(env, "CAIRN_CORPUS_TAG") or "0.143.0",
        anthropic_api_key=_read(env, "ANTHROPIC_API_KEY"),
    )
