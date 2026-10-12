"""Phase 1, step 4: loading and validating the question file."""

import json

import pytest

from cairn.evaluation.questions import QuestionError, load_questions
from cairn.types import Question, Span


def row(**overrides):
    base = {
        "id": "q1",
        "category": "lexical",
        "split": "dev",
        "question": "What does response_model_exclude_unset do?",
        "gold": [{"path": "docs/a.md", "start_line": 3, "end_line": 9}],
    }
    base.update(overrides)
    return base


def write(tmp_path, rows, name="questions.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_loads_rows_into_question_objects(tmp_path):
    path = write(
        tmp_path,
        [
            row(),
            row(
                id="q2",
                category="multi_chunk",
                split="test",
                question="Two places?",
                gold=[
                    {"path": "a.md", "start_line": 1, "end_line": 2},
                    {"path": "b/c.py", "start_line": 10, "end_line": 12},
                ],
            ),
        ],
    )
    qs = load_questions(path)
    assert [q.id for q in qs] == ["q1", "q2"]
    assert isinstance(qs[0], Question)
    assert qs[0].text == "What does response_model_exclude_unset do?"
    assert qs[0].gold == (Span("docs/a.md", 3, 9),)
    assert qs[1].category == "multi_chunk" and qs[1].split == "test"
    assert qs[1].gold == (Span("a.md", 1, 2), Span("b/c.py", 10, 12))


def test_split_filter(tmp_path):
    path = write(tmp_path, [row(id="q1"), row(id="q2", split="test"), row(id="q3")])
    assert [q.id for q in load_questions(path, split="dev")] == ["q1", "q3"]
    assert [q.id for q in load_questions(path, split="test")] == ["q2"]
    assert len(load_questions(path)) == 3


def test_unknown_split_argument_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_questions(write(tmp_path, [row()]), split="validation")


def test_bad_json_names_file_and_line(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(row()) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(QuestionError, match="line 2"):
        load_questions(path)


@pytest.mark.parametrize("field", ["id", "category", "split", "question", "gold"])
def test_missing_field_is_named(tmp_path, field):
    r = row()
    del r[field]
    with pytest.raises(QuestionError, match=field):
        load_questions(write(tmp_path, [r]))


def test_unknown_category_and_split_are_named(tmp_path):
    with pytest.raises(QuestionError, match="semantics"):
        load_questions(write(tmp_path, [row(category="semantics")]))
    with pytest.raises(QuestionError, match="validation"):
        load_questions(write(tmp_path, [row(split="validation")]))


def test_blank_question_text_is_rejected_and_names_the_id(tmp_path):
    with pytest.raises(QuestionError, match="q7"):
        load_questions(write(tmp_path, [row(id="q7", question="   ")]))


def test_empty_gold_is_rejected(tmp_path):
    with pytest.raises(QuestionError, match="gold"):
        load_questions(write(tmp_path, [row(gold=[])]))


@pytest.mark.parametrize(
    "bad",
    [
        {"path": "a.md", "start_line": 0, "end_line": 5},
        {"path": "a.md", "start_line": 9, "end_line": 5},
    ],
)
def test_impossible_line_numbers_are_rejected_with_the_question_id(tmp_path, bad):
    with pytest.raises(QuestionError, match="q9"):
        load_questions(write(tmp_path, [row(id="q9", gold=[bad])]))


def test_gold_entry_missing_a_key_is_named(tmp_path):
    with pytest.raises(QuestionError, match="end_line"):
        load_questions(write(tmp_path, [row(gold=[{"path": "a.md", "start_line": 1}])]))


@pytest.mark.parametrize("path", ["docs\\a.md", "/docs/a.md", "C:/docs/a.md"])
def test_gold_paths_must_be_relative_with_forward_slashes(tmp_path, path):
    bad = [{"path": path, "start_line": 1, "end_line": 2}]
    with pytest.raises(QuestionError, match="path"):
        load_questions(write(tmp_path, [row(gold=bad)]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(QuestionError, match="q1"):
        load_questions(write(tmp_path, [row(id="q1"), row(id="q1", question="Another?")]))
