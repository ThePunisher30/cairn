"""Phase 2, step 7: the command-line ingest script, scripts/ingest.py."""

import subprocess
import sys

import pytest

from cairn import config
from cairn.ingest.pipeline import read_chunks

SCRIPT = config.PROJECT_ROOT / "scripts" / "ingest.py"


@pytest.fixture
def repo(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": "# Title\nsome words here\n\n## Part\nmore words\n",
        "fastapi/m.py": "def f(x):\n    return x + 1\n",
    }.items():
        path = tmp_path / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_ingest_writes_the_chunks_and_prints_the_statistics(repo):
    out = repo / "data" / "chunks.jsonl"
    result = run("--root", str(repo / "corpus"), "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert out.exists() and len(read_chunks(out)) > 0
    assert "chunks" in result.stdout and "documents" in result.stdout


def test_a_smaller_prose_budget_gives_more_chunks(repo):
    big = run(
        "--root", str(repo / "corpus"), "--out", str(repo / "a.jsonl"), "--prose-tokens", "500"
    )
    small = run(
        "--root", str(repo / "corpus"), "--out", str(repo / "b.jsonl"), "--prose-tokens", "2"
    )
    assert big.returncode == 0 and small.returncode == 0
    assert len(read_chunks(repo / "b.jsonl")) > len(read_chunks(repo / "a.jsonl"))


def test_a_missing_root_is_a_clean_one_line_error(repo):
    result = run("--root", str(repo / "does-not-exist"), "--out", str(repo / "x.jsonl"))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
    assert not (repo / "x.jsonl").exists()
