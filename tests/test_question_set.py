"""Phase 1 acceptance: checks on YOUR real question file, eval/questions.jsonl.

They validate whatever questions exist so far, so you can run them while you write, batch by
batch. Only `test_question_set_is_complete` demands the full 60, so it stays red until you finish.
Tests are SKIPPED (not passed) if the question file or the downloaded corpus is missing.
"""
from collections import Counter

import pytest

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.questions import load_questions
from cairn.types import CATEGORIES

QUESTIONS_PATH = config.PROJECT_ROOT / "eval" / "questions.jsonl"
CORPUS_ROOT = load_settings().data_dir / "raw" / "fastapi"
MAX_SPAN_LINES = 60


@pytest.fixture(scope="module")
def questions():
    if not QUESTIONS_PATH.exists():
        pytest.skip("eval/questions.jsonl does not exist yet")
    return load_questions(QUESTIONS_PATH)


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_ROOT.is_dir():
        pytest.skip("corpus not downloaded: run scripts/fetch_corpus.py")
    return CORPUS_ROOT


def count_lines(path):
    # Count lines the way an editor does: only newline characters split lines.
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def span_lines(path, start, end):
    with open(path, encoding="utf-8") as f:
        return [line for i, line in enumerate(f, start=1) if start <= i <= end]


def test_file_loads_and_is_not_empty(questions):
    assert len(questions) > 0


def test_no_question_text_is_repeated(questions):
    seen = Counter(" ".join(q.text.lower().split()) for q in questions)
    repeated = [t for t, n in seen.items() if n > 1]
    assert not repeated, f"repeated questions: {repeated}"


def test_gold_files_exist_and_line_numbers_are_in_range(questions, corpus):
    problems = []
    for q in questions:
        for g in q.gold:
            f = corpus / g.path
            if not f.is_file():
                problems.append(f"{q.id}: no such file {g.path}")
                continue
            n = count_lines(f)
            if g.end_line > n:
                problems.append(
                    f"{q.id}: {g.path} has {n} lines but the span ends at {g.end_line}"
                )
    assert not problems, "\n".join(problems)


def test_gold_paths_are_inside_the_corpus_scope(questions):
    allowed = ("docs/en/docs/", "fastapi/", "docs_src/")
    outside = [
        f"{q.id}: {g.path}"
        for q in questions
        for g in q.gold
        if not g.path.startswith(allowed) or g.path.endswith("release-notes.md")
    ]
    assert not outside, (
        "gold must be under docs/en/docs/, fastapi/ or docs_src/ (and not the changelog):\n"
        + "\n".join(outside)
    )


def test_gold_spans_are_tight(questions):
    too_long = [
        f"{q.id}: {g.path}:{g.start_line}-{g.end_line} ({g.end_line - g.start_line + 1} lines)"
        for q in questions
        for g in q.gold
        if g.end_line - g.start_line + 1 > MAX_SPAN_LINES
    ]
    assert not too_long, f"gold spans longer than {MAX_SPAN_LINES} lines:\n" + "\n".join(too_long)


def test_gold_spans_contain_text(questions, corpus):
    blank = []
    for q in questions:
        for g in q.gold:
            f = corpus / g.path
            if f.is_file() and not any(
                line.strip() for line in span_lines(f, g.start_line, g.end_line)
            ):
                blank.append(f"{q.id}: {g.path}:{g.start_line}-{g.end_line} is all blank lines")
    assert not blank, "\n".join(blank)


def test_multi_chunk_questions_have_at_least_two_gold_spans(questions):
    single = [q.id for q in questions if q.category == "multi_chunk" and len(q.gold) < 2]
    assert not single, f"multi_chunk questions with only one gold span: {single}"


def test_question_set_is_complete(questions):
    """The finish line: 15 per category, split 10 dev + 5 test inside every category."""
    counts = Counter((q.category, q.split) for q in questions)
    expected = {(c, "dev"): 10 for c in CATEGORIES} | {(c, "test"): 5 for c in CATEGORIES}
    assert len(questions) == 60, f"{len(questions)} questions so far, need 60"
    assert dict(counts) == expected, f"counts per (category, split) are {dict(counts)}"
