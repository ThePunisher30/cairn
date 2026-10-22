"""Phase 4: the report scripts ablate_dense.py, bench_embed.py and compare_retrievers.py.

All three are provided, not part of the exercise. They need your dense package to work. The tests
use the offline `hashing` models, so they run in seconds with no download.
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

ABLATE = config.PROJECT_ROOT / "scripts" / "ablate_dense.py"
BENCH = config.PROJECT_ROOT / "scripts" / "bench_embed.py"
COMPARE = config.PROJECT_ROOT / "scripts" / "compare_retrievers.py"


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
            "id": "q1",
            "category": "lexical",
            "split": "dev",
            "question": "Which part talks about cherries?",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 4, "end_line": 5}],
        },
        {
            "id": "q2",
            "category": "code",
            "split": "dev",
            "question": "What does get_item return?",
            "gold": [{"path": "fastapi/m.py", "start_line": 1, "end_line": 2}],
        },
        {
            "id": "q3",
            "category": "semantic",
            "split": "test",
            "question": "Held out?",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 1, "end_line": 2}],
        },
    ]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def ablate(ws, *extra):
    return run(
        ABLATE,
        "--corpus", str(ws / "corpus"),
        "--questions", str(ws / "questions.jsonl"),
        "--cache", str(ws / "cache.npz"),
        *extra,
    )  # fmt: skip


def bench(ws, *extra):
    return run(BENCH, "--chunks", str(ws / "chunks.jsonl"), *extra)


def compare(ws, *extra):
    return run(
        COMPARE,
        "--questions", str(ws / "questions.jsonl"),
        "--chunks", str(ws / "chunks.jsonl"),
        "--bm25-index", str(ws / "bm25.pkl"),
        "--dense-index", str(ws / "dense.npz"),
        "--model", "hashing",
        *extra,
    )  # fmt: skip


# ----------------------------------------------------------------- ablate_dense
def test_ablation_reports_one_row_per_model_and_chunker(workspace):
    result = ablate(workspace, "--models", "hashing")
    assert result.returncode == 0, result.stderr
    row = next(r for r in result.stdout.splitlines() if r.startswith("hashing"))
    assert "structure-300" in row and "none" in row
    assert "hit@10" in result.stdout and "mrr" in result.stdout


def test_a_model_with_a_query_prefix_gets_an_on_and_an_off_row(workspace):
    result = ablate(workspace, "--models", "hashing-prefixed")
    assert result.returncode == 0, result.stderr
    rows = [r for r in result.stdout.splitlines() if r.startswith("hashing-prefixed")]
    assert len(rows) == 2
    assert " on " in rows[0] and " off " in rows[1]


def test_all_chunkers_can_be_asked_for(workspace):
    result = ablate(workspace, "--models", "hashing", "--chunkers", "all")
    assert result.returncode == 0, result.stderr
    for name in ("fixed-200", "fixed-400", "recursive-400", "structure-300"):
        assert any(r.startswith("hashing") and name in r for r in result.stdout.splitlines()), name


def test_the_second_run_takes_everything_from_the_cache(workspace):
    first = ablate(workspace, "--models", "hashing")
    second = ablate(workspace, "--models", "hashing")
    assert first.returncode == 0 and second.returncode == 0
    assert "(0 computed" in second.stderr
    assert (workspace / "cache.npz").exists()


def test_by_category_adds_a_column_per_category(workspace):
    result = ablate(workspace, "--models", "hashing", "--by-category")
    assert "mrr lexical" in result.stdout and "mrr code" in result.stdout


def test_ablation_refuses_the_test_split_and_unknown_chunkers(workspace):
    assert "frozen" in ablate(workspace, "--split", "test").stderr
    assert ablate(workspace, "--split", "test").returncode == 2
    assert ablate(workspace, "--chunkers", "nope").returncode == 2


def test_ablation_with_an_empty_corpus_is_a_clean_error(workspace, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(
        ABLATE, "--corpus", str(empty), "--questions", str(workspace / "questions.jsonl"),
        "--models", "hashing", "--cache", str(tmp_path / "c.npz"),
    )  # fmt: skip
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


# ------------------------------------------------------------------ bench_embed
def test_bench_reports_speed_and_memory(workspace):
    result = bench(workspace, "--model", "hashing", "--n", "4", "--batch-sizes", "2", "4")
    assert result.returncode == 0, result.stderr
    for expected in ("chunks/s", "original order", "sorted by length", "memory per 100,000 chunks"):
        assert expected in result.stdout, result.stdout
    assert "102.4 MB" in result.stdout  # 100,000 x 256 numbers x 4 bytes


def test_bench_with_missing_chunks_or_bad_numbers_is_a_clean_error(workspace, tmp_path):
    missing = run(BENCH, "--chunks", str(tmp_path / "nope.jsonl"), "--model", "hashing")
    assert missing.returncode == 1 and missing.stderr.startswith("error:")
    bad = bench(workspace, "--model", "hashing", "--n", "0")
    assert bad.returncode == 1 and bad.stderr.startswith("error:")
    assert "Traceback" not in missing.stderr + bad.stderr


# ----------------------------------------------------------- compare_retrievers
def test_compare_lists_every_dev_question_with_both_ranks(workspace):
    result = compare(workspace)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert any(line.startswith("q1 ") for line in lines)
    assert any(line.startswith("q2 ") for line in lines)
    assert not any(line.startswith("q3 ") for line in lines)  # a test question: never shown
    assert "rank 1: both" in result.stdout


def test_compare_shows_the_failing_questions_with_gold_and_top_results(workspace):
    result = compare(
        workspace, "--fail-rank", "0"
    )  # nothing ranks above 0: every question is listed
    assert result.returncode == 0, result.stderr
    assert "gold:  docs/en/docs/a.md:4-5" in result.stdout
    assert "bm25  (rank" in result.stdout and "dense (rank" in result.stdout


def test_compare_names_the_script_that_builds_a_missing_file(workspace):
    (workspace / "dense.npz").unlink()
    result = compare(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:") and "embed_chunks" in result.stderr
    (workspace / "bm25.pkl").unlink()
    assert "build_index" in compare(workspace).stderr


def test_compare_with_chunks_that_do_not_match_the_indexes_is_a_clean_error(workspace):
    (workspace / "chunks.jsonl").write_text(
        json.dumps(
            {
                "id": "x", "path": "z.md", "start_line": 1, "end_line": 1, "kind": "prose",
                "heading_path": [], "content_hash": "x", "token_count": 1, "text": "z",
            }
        )
        + "\n",
        encoding="utf-8",
    )  # fmt: skip
    result = compare(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
