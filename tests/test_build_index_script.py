"""Phase 3, step 6a: the command-line script scripts/build_index.py."""

import subprocess
import sys

import pytest

from cairn import config
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import ENGLISH_STOPWORDS, Tokenizer

SCRIPT = config.PROJECT_ROOT / "scripts" / "build_index.py"


@pytest.fixture
def chunks_file(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": "# Title\nsome words here\n\n## Part\nmore words about the part\n",
        "fastapi/m.py": "def get_item(x):\n    return x + 1\n",
    }.items():
        path = tmp_path / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out = tmp_path / "chunks.jsonl"
    ingest(tmp_path / "corpus", out)
    return out


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_it_builds_an_index_and_reports_what_it_made(chunks_file, tmp_path):
    out = tmp_path / "out" / "bm25.pkl"  # the folder does not exist yet
    result = run("--chunks", str(chunks_file), "--out", str(out))
    assert result.returncode == 0, result.stderr
    index = InvertedIndex.load(out)
    assert index.n_docs == len(read_chunks(chunks_file)) > 0
    assert "chunks" in result.stdout and "terms" in result.stdout


def test_by_default_identifiers_are_split_and_nothing_is_a_stopword(chunks_file, tmp_path):
    out = tmp_path / "bm25.pkl"
    assert run("--chunks", str(chunks_file), "--out", str(out)).returncode == 0
    index = InvertedIndex.load(out)
    assert index.tokenizer == Tokenizer()
    assert index.doc_freq("get_item") == 1 and index.doc_freq("item") == 1


def test_no_split_identifiers_switches_splitting_off(chunks_file, tmp_path):
    out = tmp_path / "bm25.pkl"
    result = run("--chunks", str(chunks_file), "--out", str(out), "--no-split-identifiers")
    assert result.returncode == 0, result.stderr
    index = InvertedIndex.load(out)
    assert index.tokenizer.split_identifiers is False
    assert index.doc_freq("get_item") == 1 and index.doc_freq("item") == 0


def test_stopwords_flag_uses_the_english_list(chunks_file, tmp_path):
    out = tmp_path / "bm25.pkl"
    result = run("--chunks", str(chunks_file), "--out", str(out), "--stopwords")
    assert result.returncode == 0, result.stderr
    index = InvertedIndex.load(out)
    assert index.tokenizer.stopwords == ENGLISH_STOPWORDS
    assert index.doc_freq("the") == 0 and index.doc_freq("words") >= 1


def test_a_missing_chunks_file_is_a_clean_one_line_error(tmp_path):
    out = tmp_path / "bm25.pkl"
    result = run("--chunks", str(tmp_path / "missing.jsonl"), "--out", str(out))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "ingest" in result.stderr  # tells you which script makes the chunks file
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_an_empty_chunks_file_is_a_clean_error(tmp_path):
    empty = tmp_path / "chunks.jsonl"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "bm25.pkl"
    result = run("--chunks", str(empty), "--out", str(out))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_the_generated_index_is_not_tracked_by_git():
    lines = {
        line.strip()
        for line in (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "data/bm25.pkl" in lines
