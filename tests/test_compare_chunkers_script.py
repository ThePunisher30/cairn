"""Phase 2: the report script scripts/compare_chunkers.py (provided, not part of the exercise)."""

import json
import subprocess
import sys

import pytest

from cairn import config

SCRIPT = config.PROJECT_ROOT / "scripts" / "compare_chunkers.py"


@pytest.fixture
def workspace(tmp_path):
    doc = tmp_path / "corpus" / "docs" / "en" / "docs" / "a.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# Title\nsome words here\n\n## Part\nmore words here\n", encoding="utf-8")
    code = tmp_path / "corpus" / "fastapi" / "m.py"
    code.parent.mkdir(parents=True)
    code.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    rows = [
        {
            "id": "q1",
            "category": "lexical",
            "split": "dev",
            "question": "What is in the part?",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 4, "end_line": 5}],
        },
        {
            "id": "q2",
            "category": "code",
            "split": "dev",
            "question": "What does f do?",
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
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"corpus": tmp_path / "corpus", "questions": questions}


def run(ws, *extra):
    args = [
        sys.executable,
        str(SCRIPT),
        "--corpus",
        str(ws["corpus"]),
        "--questions",
        str(ws["questions"]),
        *extra,
    ]
    return subprocess.run(args, capture_output=True, text=True)


def test_the_report_lists_every_strategy(workspace):
    result = run(workspace)
    assert result.returncode == 0, result.stderr
    for name in ("fixed-200", "recursive-400", "structure-300", "intact"):
        assert name in result.stdout


def test_a_big_chunk_budget_keeps_the_tiny_answers_intact(workspace):
    line = next(
        row for row in run(workspace).stdout.splitlines() if row.startswith("structure-300")
    )
    assert "1.000" in line  # both dev answers lie inside one chunk


def test_the_test_split_is_refused(workspace):
    result = run(workspace, "--split", "test")
    assert result.returncode == 2
    assert "frozen" in result.stderr


def test_an_empty_corpus_is_a_clean_error(workspace, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus",
            str(empty),
            "--questions",
            str(workspace["questions"]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_show_broken_lists_the_spans_that_are_cut(workspace):
    rows = [
        json.loads(line) for line in workspace["questions"].read_text(encoding="utf-8").splitlines()
    ]
    rows.append(
        {
            "id": "q9",
            "category": "lexical",
            "split": "dev",
            "question": "A span that crosses two sections?",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 1, "end_line": 5}],
        }
    )
    workspace["questions"].write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    result = run(workspace, "--show-broken", "structure-300")
    assert result.returncode == 0, result.stderr
    assert "q9" in result.stdout and "docs/en/docs/a.md:1-5" in result.stdout
    assert (
        "q1 " not in result.stdout.split("cuts across chunks")[1]
    )  # the intact ones are not listed


def test_show_broken_says_so_when_nothing_is_cut(workspace):
    result = run(workspace, "--show-broken", "structure-300")
    assert "(none)" in result.stdout
