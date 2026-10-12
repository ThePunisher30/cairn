"""Load and validate the hand-written question file (eval/questions.jsonl)."""

from pathlib import Path

from cairn.jsonl import JsonlError, read_jsonl
from cairn.types import CATEGORIES, SPLITS, Question, Span


class QuestionError(Exception):
    pass


def _need(row: dict, field: str, where: str):
    """Return row[field], or raise a QuestionError that names the missing field."""
    if field not in row:
        raise QuestionError(f"{where}: missing field {field!r}")
    return row[field]


def _parse_span(entry: dict, where: str) -> Span:
    """Turn one {"path", "start_line", "end_line"} entry into a Span, or explain what is wrong."""
    path = _need(entry, "path", where)
    start_line = _need(entry, "start_line", where)
    end_line = _need(entry, "end_line", where)

    if not isinstance(path, str):
        raise QuestionError(f"{where}: gold path must be text, got {path!r}")
    if "\\" in path or path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise QuestionError(f"{where}: gold path {path!r} must be relative with forward slashes")
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise QuestionError(f"{where}: start_line and end_line must be whole numbers")

    try:
        return Span(path, start_line, end_line)
    except ValueError as e:
        raise QuestionError(f"{where}: {e}") from e


def load_questions(path, split: str | None = None) -> list[Question]:
    """Read, check and return every question in the file (optionally only one split).

    Every row is validated, so a typo in a hand-written file is reported with the file name
    and the question id instead of surfacing later as a confusing failure.
    """
    path = Path(path)
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    questions: list[Question] = []
    seen_ids: set[str] = set()
    try:
        for number, row in enumerate(read_jsonl(path), start=1):
            where = f"{path.name}: row {number}"
            question_id = _need(row, "id", where)
            where = f"{path.name}: question {question_id!r}"  # from here on, name the question

            category = _need(row, "category", where)
            row_split = _need(row, "split", where)
            text = _need(row, "question", where)
            gold = _need(row, "gold", where)

            if category not in CATEGORIES:
                raise QuestionError(f"{where}: unknown category {category!r}, use {CATEGORIES}")
            if row_split not in SPLITS:
                raise QuestionError(f"{where}: unknown split {row_split!r}, use {SPLITS}")
            if not isinstance(text, str) or not text.strip():
                raise QuestionError(f"{where}: the question text is blank")
            if not isinstance(gold, list) or not gold:
                raise QuestionError(f"{where}: gold must be a non-empty list of spans")
            if question_id in seen_ids:
                raise QuestionError(f"{where}: duplicate id")
            seen_ids.add(question_id)

            spans = tuple(_parse_span(entry, where) for entry in gold)
            questions.append(Question(question_id, category, row_split, text, spans))
    except JsonlError as e:
        raise QuestionError(str(e)) from e

    return [q for q in questions if split is None or q.split == split]
