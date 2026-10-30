"""Phase 6: the ablation script scripts/ablate_hybrid.py.

Provided, not part of the exercise. It needs your `fusion`, `hybrid`, `rerank` and `context` modules
to work. The tests use the offline `hashing` embedding model and the `overlap` reranker, so they run
in seconds with no download.
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

ABLATE = config.PROJECT_ROOT / "scripts" / "ablate_hybrid.py"

DEV_ROWS = [
    ("q1", "lexical", "Which part talks about cherries?", "docs/en/docs/a.md", 4, 5),
    ("q2", "code", "What does get_item return?", "fastapi/m.py", 1, 2),
]


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
        {
            "id": qid,
            "category": category,
            "split": "dev",
            "question": question,
            "gold": [{"path": path, "start_line": start, "end_line": end}],
        }
        for qid, category, question, path, start, end in DEV_ROWS
    ]
    rows.append(
        {
            "id": "q3",
            "category": "semantic",
            "split": "test",
            "question": "Held out?",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 1, "end_line": 2}],
        }
    )
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def ablate(ws, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ABLATE),
            "--questions", str(ws / "questions.jsonl"),
            "--chunks", str(ws / "chunks.jsonl"),
            "--bm25-index", str(ws / "bm25.pkl"),
            "--dense-index", str(ws / "dense.npz"),
            "--test-log", str(ws / "test.log"),
            "--model", "hashing",
            "--reranker", "overlap",
            *extra,
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip


ROWS = [
    "bm25",
    "dense",
    "hybrid rrf (k=60)",
    "hybrid weighted minmax",
    "hybrid weighted zscore",
    "hybrid raw sum",
    "bm25 + rerank",
    "dense + rerank",
    "hybrid rrf + rerank",
]


def test_the_table_has_a_row_for_every_retriever(workspace):
    result = ablate(workspace)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "2 dev questions, k = 10" in out  # the test question is not used
    for name in ROWS:
        assert name in out, name
    assert "hit@1" in out and "mrr" in out and "p50 ms" in out


def test_without_the_reranker_there_are_no_rerank_rows(workspace):
    result = ablate(workspace, "--no-rerank")
    assert result.returncode == 0, result.stderr
    assert "rerank" not in result.stdout
    assert "hybrid rrf (k=60)" in result.stdout


def test_by_category_adds_an_mrr_table(workspace):
    result = ablate(workspace, "--by-category")
    assert result.returncode == 0, result.stderr
    assert "MRR by category" in result.stdout
    assert "lexical" in result.stdout and "code" in result.stdout


def test_the_sweep_tries_several_rrf_ks_and_weights(workspace):
    result = ablate(workspace, "--sweep")
    assert result.returncode == 0, result.stderr
    assert "k = 1000" in result.stdout and "w = 0.75" in result.stdout


def test_the_expansion_table_compares_chunks_with_bigger_contexts(workspace):
    result = ablate(workspace, "--expansion", "--expand-tokens", "50", "400")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "chunks as retrieved" in out and "gold covered" in out
    assert "section, up to 50" in out and "parent, up to 400" in out


def test_the_test_split_is_refused_without_the_flag(workspace):
    result = ablate(workspace, "--split", "test")
    assert result.returncode == 2
    assert "--use-test-set" in result.stderr
    assert not (workspace / "test.log").exists()


def test_a_test_run_is_logged_and_the_third_one_is_refused(workspace):
    first = ablate(workspace, "--split", "test", "--use-test-set", "--no-rerank")
    assert first.returncode == 0, first.stderr
    assert "1 test questions" in first.stdout
    assert ablate(workspace, "--split", "test", "--use-test-set", "--no-rerank").returncode == 0
    lines = (workspace / "test.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and "ablate_hybrid" in lines[0]
    third = ablate(workspace, "--split", "test", "--use-test-set", "--no-rerank")
    assert third.returncode == 1 and third.stderr.startswith("error:")
    assert len((workspace / "test.log").read_text(encoding="utf-8").splitlines()) == 2


def test_tuning_on_the_test_split_is_refused(workspace):
    result = ablate(workspace, "--split", "test", "--use-test-set", "--sweep")
    assert result.returncode == 2
    assert not (workspace / "test.log").exists()


def test_missing_files_are_a_clean_one_line_error(workspace):
    (workspace / "bm25.pkl").unlink()
    result = ablate(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:") and "build_index" in result.stderr
    assert "Traceback" not in result.stderr


def test_bad_numbers_are_refused(workspace):
    for bad in (["--k", "0"], ["--depth", "0"], ["--weight", "1.5"]):
        result = ablate(workspace, *bad)
        assert result.returncode == 1 and result.stderr.startswith("error:"), bad
