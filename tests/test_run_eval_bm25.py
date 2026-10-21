"""Phase 3, step 6b: scripts/run_eval.py can run the BM25 retriever (--retriever bm25).

New options: --index (the saved index), --chunks (the chunks file it was built from),
--k1 and --b (the BM25 knobs, used when searching).
"""

import dataclasses
import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.ingest.chunkers import make_chunk
from cairn.jsonl import write_jsonl
from cairn.retrieval.index import InvertedIndex

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
    InvertedIndex.build(chunks).save(tmp_path / "bm25.pkl")
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
        sys.executable,
        str(SCRIPT),
        "--retriever",
        "bm25",
        "--questions",
        str(ws / "questions.jsonl"),
        "--index",
        str(ws / "bm25.pkl"),
        "--chunks",
        str(ws / "chunks.jsonl"),
        "--test-log",
        str(ws / "test_runs.log"),
        *extra,
    ]
    return subprocess.run(args, capture_output=True, text=True)


def test_bm25_finds_the_gold_chunks(workspace):
    result = run(workspace)
    assert result.returncode == 0, result.stderr
    overall = next(row for row in result.stdout.splitlines() if row.startswith("overall"))
    assert "1.000" in overall  # both dev questions are answered at rank 1
    assert "misses (0): none" in result.stdout


def test_the_knobs_are_accepted(workspace):
    result = run(workspace, "--k1", "2.0", "--b", "0.5")
    assert result.returncode == 0, result.stderr


def test_a_knob_out_of_range_is_a_clean_error(workspace):
    result = run(workspace, "--b", "2")
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_a_missing_index_tells_you_which_script_builds_it(workspace):
    (workspace / "bm25.pkl").unlink()
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "build_index" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_chunks_file_is_a_clean_error(workspace):
    (workspace / "chunks.jsonl").unlink()
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_chunks_that_do_not_match_the_index_are_refused(workspace):
    other = [make_chunk("docs/z.md", 1, 1, "something else entirely", "prose")]
    save_chunks(workspace / "chunks.jsonl", other)  # the index was built from different chunks
    result = run(workspace)
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_the_test_split_is_still_frozen_for_bm25(workspace):
    result = run(workspace, "--split", "test")
    assert result.returncode == 2
    assert "frozen" in result.stderr
    assert not (workspace / "test_runs.log").exists()
