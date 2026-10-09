"""Phase 1, step 8: the command-line runner scripts/run_eval.py."""
import json
import subprocess
import sys

import pytest

from cairn import config

SCRIPT = config.PROJECT_ROOT / "scripts" / "run_eval.py"


@pytest.fixture
def workspace(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    body = "".join(f"line {i}\n" for i in range(1, 11))
    (corpus / "docs" / "a.md").write_text(body, encoding="utf-8")
    rows = [
        {"id": "q1", "category": "lexical", "split": "dev", "question": "What is line 4?",
         "gold": [{"path": "docs/a.md", "start_line": 3, "end_line": 5}]},
        {"id": "q2", "category": "code", "split": "dev", "question": "Where is line 8?",
         "gold": [{"path": "docs/a.md", "start_line": 8, "end_line": 8}]},
        {"id": "q3", "category": "semantic", "split": "test", "question": "Held out?",
         "gold": [{"path": "docs/a.md", "start_line": 1, "end_line": 2}]},
    ]
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"corpus": corpus, "questions": questions, "log": tmp_path / "test_runs.log"}


def run(ws, *extra):
    args = [
        sys.executable, str(SCRIPT),
        "--questions", str(ws["questions"]),
        "--corpus", str(ws["corpus"]),
        "--test-log", str(ws["log"]),
        *extra,
    ]
    return subprocess.run(args, capture_output=True, text=True)


def test_oracle_scores_perfectly(workspace):
    result = run(workspace, "--retriever", "oracle")
    assert result.returncode == 0, result.stderr
    assert "overall" in result.stdout and "1.000" in result.stdout


@pytest.mark.parametrize("name", ["random", "first"])
def test_baselines_run_and_print_a_report(workspace, name):
    result = run(workspace, "--retriever", name, "--window", "5")
    assert result.returncode == 0, result.stderr
    assert "overall" in result.stdout and "hit@5" in result.stdout


def test_unknown_retriever_name_is_a_usage_error(workspace):
    assert run(workspace, "--retriever", "magic").returncode == 2


def test_the_test_split_is_refused_without_the_flag(workspace):
    result = run(workspace, "--retriever", "oracle", "--split", "test")
    assert result.returncode == 2
    assert "--use-test-set" in result.stderr
    assert not workspace["log"].exists()


def test_using_the_test_split_is_allowed_with_the_flag_and_is_logged(workspace):
    first = run(workspace, "--retriever", "oracle", "--split", "test", "--use-test-set")
    assert first.returncode == 0, first.stderr
    run(workspace, "--retriever", "first", "--split", "test", "--use-test-set", "--window", "5")
    lines = workspace["log"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "oracle" in lines[0] and "first" in lines[1]


def test_a_broken_question_file_is_a_clean_one_line_error(workspace):
    workspace["questions"].write_text("not json\n", encoding="utf-8")
    result = run(workspace, "--retriever", "oracle")
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
