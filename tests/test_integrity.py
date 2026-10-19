"""Phase 2, step 6: gold integrity, how often a chunker keeps an answer in ONE piece.

A gold span is INTACT when some chunk of the same file contains it completely
(chunk.start_line <= span.start_line and chunk.end_line >= span.end_line).
Cut an answer across two chunks and no single chunk can ever contain the whole answer.
"""

import pytest

from cairn.evaluation.integrity import broken_spans, gold_integrity
from cairn.ingest.chunkers import make_chunk
from cairn.types import Question, Span


def chunk(path, start, end):
    return make_chunk(path, start, end, "text", "prose")


def question(qid, category, *spans):
    return Question(qid, category, "dev", f"question {qid}?", tuple(Span(*s) for s in spans))


def test_a_span_inside_a_chunk_is_intact():
    result = gold_integrity([chunk("a.md", 1, 50)], [question("q1", "lexical", ("a.md", 10, 20))])
    overall = result["overall"]
    assert (overall["questions"], overall["spans"], overall["spans_intact"]) == (1, 1, 1)
    assert overall["span_rate"] == 1.0 and overall["question_rate"] == 1.0


def test_a_span_exactly_equal_to_a_chunk_is_intact():
    result = gold_integrity([chunk("a.md", 10, 20)], [question("q1", "code", ("a.md", 10, 20))])
    assert result["overall"]["span_rate"] == 1.0


def test_a_span_cut_across_two_chunks_is_not_intact():
    chunks = [chunk("a.md", 1, 15), chunk("a.md", 16, 30)]
    result = gold_integrity(chunks, [question("q1", "lexical", ("a.md", 10, 20))])
    assert result["overall"]["spans_intact"] == 0
    assert result["overall"]["span_rate"] == 0.0


def test_a_chunk_in_another_file_does_not_count():
    result = gold_integrity([chunk("b.md", 1, 50)], [question("q1", "lexical", ("a.md", 10, 20))])
    assert result["overall"]["spans_intact"] == 0


def test_a_chunk_that_is_one_line_short_is_not_enough():
    result = gold_integrity([chunk("a.md", 1, 19)], [question("q1", "lexical", ("a.md", 10, 20))])
    assert result["overall"]["spans_intact"] == 0
    result = gold_integrity([chunk("a.md", 11, 30)], [question("q1", "lexical", ("a.md", 10, 20))])
    assert result["overall"]["spans_intact"] == 0


def test_a_question_counts_as_intact_only_if_all_its_spans_are():
    chunks = [chunk("a.md", 1, 50)]
    q = question("q1", "multi_chunk", ("a.md", 10, 20), ("b.md", 1, 5))  # the second one is missing
    overall = gold_integrity(chunks, [q])["overall"]
    assert (overall["spans"], overall["spans_intact"]) == (2, 1)
    assert overall["span_rate"] == 0.5
    assert overall["question_rate"] == 0.0


def test_numbers_are_split_by_category_in_the_usual_order():
    chunks = [chunk("a.md", 1, 50)]
    questions = [
        question("q1", "code", ("a.md", 1, 5)),  # intact
        question("q2", "lexical", ("a.md", 60, 70)),  # no chunk there
        question("q3", "lexical", ("a.md", 5, 9)),  # intact
    ]
    result = gold_integrity(chunks, questions)
    assert list(result["by_category"]) == ["lexical", "code"]  # CATEGORIES order, present ones only
    assert result["by_category"]["lexical"]["questions"] == 2
    assert result["by_category"]["lexical"]["span_rate"] == pytest.approx(0.5)
    assert result["by_category"]["code"]["span_rate"] == 1.0
    assert result["overall"]["span_rate"] == pytest.approx(2 / 3)


def test_no_questions_is_an_error():
    with pytest.raises(ValueError):
        gold_integrity([chunk("a.md", 1, 5)], [])


def test_no_chunks_means_nothing_is_intact():
    result = gold_integrity([], [question("q1", "lexical", ("a.md", 1, 2))])
    assert result["overall"]["span_rate"] == 0.0


# ------------------------------------------------------------------ broken_spans
def test_broken_spans_lists_the_spans_that_no_chunk_contains():
    chunks = [chunk("a.md", 1, 15), chunk("a.md", 16, 30), chunk("b.md", 1, 99)]
    cut = question("q1", "lexical", ("a.md", 10, 20))  # cut across two chunks
    fine = question("q2", "code", ("b.md", 5, 9))
    mixed = question(
        "q3", "multi_chunk", ("b.md", 1, 2), ("c.md", 1, 2)
    )  # second span has no chunk
    assert broken_spans(chunks, [cut, fine, mixed]) == [
        (cut, Span("a.md", 10, 20)),
        (mixed, Span("c.md", 1, 2)),
    ]


def test_broken_spans_is_empty_when_everything_is_intact():
    assert broken_spans([chunk("a.md", 1, 50)], [question("q1", "lexical", ("a.md", 3, 4))]) == []


def test_broken_spans_agrees_with_gold_integrity():
    chunks = [chunk("a.md", 1, 15), chunk("a.md", 16, 30)]
    questions = [
        question("q1", "lexical", ("a.md", 10, 20)),
        question("q2", "code", ("a.md", 1, 5)),
    ]
    overall = gold_integrity(chunks, questions)["overall"]
    assert overall["spans"] - overall["spans_intact"] == len(broken_spans(chunks, questions)) == 1
