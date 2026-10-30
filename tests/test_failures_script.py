"""Phase 6: the reading aid scripts/list_failures.py.

Provided, not part of the exercise. Offline: the `hashing` embedding model and the `overlap`
reranker. The script must never read the test split, so the workspace holds one test question
and the tests check that it never appears.
"""

import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.index import InvertedIndex

SCRIPT = config.PROJECT_ROOT / "scripts" / "list_failures.py"


@pytest.fixture
def workspace(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": "# Title\nsome words here\n\n## Part\nmore words about cherries\n",
        "fastapi/m.py": "def get_item(x):\n    return x + 1\n",
    }.items():
        path = tmp_path / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    ingest(tmp_path / "corpus", tmp_path / "chunks.jsonl")
    chunks = read_chunks(tmp_path / "chunks.jsonl")
    InvertedIndex.build(chunks).save(tmp_path / "bm25.pkl")
    build_dense_index(chunks, load_embedder("hashing"))[0].save(tmp_path / "dense.npz")
    rows = [
        ("q1", "lexical", "dev", "Which part talks about cherries?", "docs/en/docs/a.md", 4, 5),
        ("q2", "code", "dev", "What does get_item return?", "fastapi/m.py", 1, 2),
        ("q3", "semantic", "test", "Held out question?", "docs/en/docs/a.md", 1, 2),
    ]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": qid,
                    "category": category,
                    "split": split,
                    "question": question,
                    "gold": [{"path": path, "start_line": start, "end_line": end}],
                }
            )
            for qid, category, split, question, path, start, end in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def run(ws, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions", str(ws / "questions.jsonl"),
            "--chunks", str(ws / "chunks.jsonl"),
            "--bm25-index", str(ws / "bm25.pkl"),
            "--dense-index", str(ws / "dense.npz"),
            "--model", "hashing",
            "--reranker", "overlap",
            *extra,
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip


def test_it_lists_every_row_for_every_dev_question(workspace):
    result = run(workspace, "--all")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    header = next(line for line in lines if line.startswith("id"))
    assert header.split() == ["id", "category", "bm25", "dense", "rrf", "minmax"]
    assert any(line.startswith("q1") for line in lines)
    assert any(line.startswith("q2") for line in lines)


def test_the_test_split_is_never_read(workspace):
    for extra in ((), ("--all",), ("--markdown",), ("--rerank", "--all")):
        result = run(workspace, *extra)
        assert result.returncode == 0, result.stderr
        assert "q3" not in result.stdout and "Held out" not in result.stdout


def test_the_summary_counts_add_up(workspace):
    result = run(workspace)
    assert "2 dev questions" in result.stdout
    assert "has the gold at rank 1 for" in result.stdout
    assert "lost:" in result.stdout and "gained:" in result.stdout


def test_rerank_adds_a_column_and_markdown_adds_a_table(workspace):
    result = run(workspace, "--rerank", "--all", "--markdown")
    assert result.returncode == 0, result.stderr
    assert "rrf+rerank" in result.stdout
    assert (
        "| Q | Category |" in result.stdout and "| Cause | What the text shows |" in result.stdout
    )


def test_a_missing_file_is_a_clean_error(workspace):
    (workspace / "bm25.pkl").unlink()
    result = run(workspace)
    assert result.returncode == 1
    assert "not found" in result.stderr and "Traceback" not in result.stderr
