"""Phase 3: the report scripts scripts/bench_search.py and scripts/ablate_bm25.py.

Both are provided, not part of the exercise. They need your retrieval package to work.
"""

import dataclasses
import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.index import InvertedIndex

BENCH = config.PROJECT_ROOT / "scripts" / "bench_search.py"
ABLATE = config.PROJECT_ROOT / "scripts" / "ablate_bm25.py"


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
    InvertedIndex.build(read_chunks(tmp_path / "chunks.jsonl")).save(tmp_path / "bm25.pkl")
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


def bench(ws, *extra):
    return run(
        BENCH,
        "--index",
        str(ws / "bm25.pkl"),
        "--chunks",
        str(ws / "chunks.jsonl"),
        "--questions",
        str(ws / "questions.jsonl"),
        *extra,
    )


def ablate(ws, *extra):
    return run(
        ABLATE,
        "--corpus",
        str(ws / "corpus"),
        "--questions",
        str(ws / "questions.jsonl"),
        *extra,
    )


# ------------------------------------------------------------------ bench_search
def test_bench_reports_the_percentiles_and_the_verdict(workspace):
    result = bench(workspace, "--repeat", "3")
    assert result.returncode == 0, result.stderr
    for expected in ("searches: 6", "p50", "p95", "max", "target"):
        assert expected in result.stdout, result.stdout
    assert "OK" in result.stdout  # a 2-chunk index is far below 20 ms


def test_bench_uses_the_dev_questions_only(workspace):
    result = bench(workspace, "--repeat", "1")
    assert "searches: 2 (2 questions x 1)" in result.stdout  # q3 is a test question


def test_bench_with_a_missing_index_is_a_clean_error(workspace):
    (workspace / "bm25.pkl").unlink()
    result = bench(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "build_index" in result.stderr
    assert "Traceback" not in result.stderr


def test_bench_with_chunks_that_do_not_match_the_index_is_a_clean_error(workspace):
    other = workspace / "other.jsonl"
    rows = [dataclasses.asdict(c) for c in read_chunks(workspace / "chunks.jsonl")][:1]
    other.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = bench(workspace, "--chunks", str(other))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")


# ------------------------------------------------------------------- ablate_bm25
def test_ablation_lists_every_chunker_and_tokenizer_variant(workspace):
    result = ablate(workspace)
    assert result.returncode == 0, result.stderr
    rows = result.stdout.splitlines()
    for chunker in ("fixed-200", "fixed-400", "recursive-400", "structure-300"):
        for variant in ("plain", "split", "plain+stop", "split+stop"):
            assert any(r.startswith(chunker) and f" {variant} " in r for r in rows), (
                chunker,
                variant,
            )
    assert "mrr" in result.stdout and "hit@10" in result.stdout


def test_by_category_adds_a_column_per_category(workspace):
    result = ablate(workspace, "--by-category")
    assert result.returncode == 0, result.stderr
    assert "mrr lexical" in result.stdout and "mrr code" in result.stdout


def test_tune_prints_a_k1_by_b_grid(workspace):
    result = ablate(workspace, "--tune", "structure-300", "--tokenizer", "split")
    assert result.returncode == 0, result.stderr
    assert "b=0" in result.stdout and "b=0.75" in result.stdout and "b=1" in result.stdout
    assert any(line.startswith("k1=1.2") for line in result.stdout.splitlines())
    assert any(line.startswith("k1=0 ") for line in result.stdout.splitlines())


def test_the_test_split_is_refused(workspace):
    result = ablate(workspace, "--split", "test")
    assert result.returncode == 2
    assert "frozen" in result.stderr


def test_an_empty_corpus_is_a_clean_error(workspace, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(ABLATE, "--corpus", str(empty), "--questions", str(workspace / "questions.jsonl"))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
