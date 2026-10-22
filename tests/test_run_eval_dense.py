"""Phase 4, step 5b: scripts/run_eval.py can run the dense retriever (--retriever dense).

New options: --model (a key of cairn.dense.models.MODELS, default bge-small), --dense-index (the
saved index; default <data folder>/dense-<model>.npz), --no-query-prefix (leave the query prefix
out, to measure what it is worth). The chunks file is --chunks, as for BM25.
"""

import dataclasses
import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.chunkers import make_chunk
from cairn.jsonl import write_jsonl

SCRIPT = config.PROJECT_ROOT / "scripts" / "run_eval.py"


def save_chunks(path, chunks):
    write_jsonl(path, (dataclasses.asdict(c) for c in chunks))


@pytest.fixture
def workspace(tmp_path):
    chunks = [
        make_chunk("docs/a.md", 1, 3, "apple banana apple", "prose"),
        make_chunk("docs/a.md", 4, 6, "cherry dates", "prose"),
        make_chunk("fastapi/m.py", 1, 2, "def dog(): pass", "code"),
    ]
    save_chunks(tmp_path / "chunks.jsonl", chunks)
    index, _ = build_dense_index(chunks, load_embedder("hashing"))
    index.save(tmp_path / "dense.npz")
    rows = [
        {
            "id": "q1",
            "category": "lexical",
            "split": "dev",
            "question": "Which fruit is cherry?",
            "gold": [{"path": "docs/a.md", "start_line": 4, "end_line": 6}],
        },
        {
            "id": "q2",
            "category": "code",
            "split": "dev",
            "question": "dog",
            "gold": [{"path": "fastapi/m.py", "start_line": 1, "end_line": 2}],
        },
        {
            "id": "q3",
            "category": "semantic",
            "split": "test",
            "question": "Held out?",
            "gold": [{"path": "docs/a.md", "start_line": 1, "end_line": 2}],
        },
    ]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def run(ws, *extra):
    args = [
        sys.executable, str(SCRIPT),
        "--retriever", "dense", "--model", "hashing",
        "--questions", str(ws / "questions.jsonl"),
        "--dense-index", str(ws / "dense.npz"),
        "--chunks", str(ws / "chunks.jsonl"),
        "--test-log", str(ws / "test_runs.log"),
        *extra,
    ]  # fmt: skip
    return subprocess.run(args, capture_output=True, text=True)


def test_dense_finds_the_gold_chunks(workspace):
    result = run(workspace)
    assert result.returncode == 0, result.stderr
    overall = next(row for row in result.stdout.splitlines() if row.startswith("overall"))
    assert "1.000" in overall
    assert "misses (0): none" in result.stdout


def test_the_query_prefix_can_be_left_out(workspace):
    result = run(workspace, "--no-query-prefix")
    assert result.returncode == 0, result.stderr


def test_a_missing_index_tells_you_which_script_builds_it(workspace):
    (workspace / "dense.npz").unlink()
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "embed_chunks" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_chunks_file_is_a_clean_error(workspace):
    (workspace / "chunks.jsonl").unlink()
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_chunks_that_do_not_match_the_index_are_refused(workspace):
    save_chunks(
        workspace / "chunks.jsonl", [make_chunk("docs/z.md", 1, 1, "something else", "prose")]
    )
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_a_file_that_is_not_a_dense_index_is_a_clean_error(workspace):
    (workspace / "dense.npz").write_bytes(b"not an index")
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_an_unknown_model_is_refused(workspace):
    result = run(workspace, "--model", "no-such-model")
    assert result.returncode == 2


def test_the_test_split_is_still_frozen_for_dense(workspace):
    result = run(workspace, "--split", "test")
    assert result.returncode == 2
    assert "frozen" in result.stderr
    assert not (workspace / "test_runs.log").exists()
