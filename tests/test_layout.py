"""Phase 0: does the project skeleton exist and is it set up sensibly?"""
import tomllib

import pytest

import cairn
from cairn import config

ROOT = config.PROJECT_ROOT


def test_package_has_a_version_string():
    assert isinstance(cairn.__version__, str) and cairn.__version__


@pytest.mark.parametrize("folder", ["src/cairn", "tests", "eval", "bench", "docs", "scripts"])
def test_expected_folders_exist(folder):
    assert (ROOT / folder).is_dir()


def test_gitignore_covers_the_essentials():
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for pattern in [".venv/", "__pycache__/", ".env", "data/raw/"]:
        assert pattern in lines, f"{pattern!r} missing from .gitignore"


def test_env_example_documents_the_key_without_containing_a_secret():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in text
    assert "sk-" not in text


def test_pyproject_basics():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "cairn"
    assert "requires-python" in project
    assert "numpy" in " ".join(project["dependencies"]).lower()
    dev = " ".join(project["optional-dependencies"]["dev"]).lower()
    assert "pytest" in dev and "ruff" in dev
    assert "ruff" in data["tool"]
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
