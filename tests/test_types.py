"""Phase 1, step 2: the shared vocabulary (Span, Hit, Question, Retriever)."""

import dataclasses

import pytest

from cairn.types import CATEGORIES, SPLITS, Hit, Question, Retriever, Span


def test_constants():
    assert CATEGORIES == ("lexical", "semantic", "multi_chunk", "code")
    assert SPLITS == ("dev", "test")


def test_span_fields_equality_and_hashing():
    s = Span("docs/a.md", 3, 7)
    assert (s.path, s.start_line, s.end_line) == ("docs/a.md", 3, 7)
    assert s == Span("docs/a.md", 3, 7)
    assert hash(s) == hash(Span("docs/a.md", 3, 7))


def test_span_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Span("a.md", 1, 2).start_line = 5


def test_single_line_span_is_allowed():
    assert Span("a.md", 3, 3).end_line == 3


@pytest.mark.parametrize("start,end", [(0, 5), (-1, 3), (5, 4)])
def test_span_rejects_impossible_lines(start, end):
    with pytest.raises(ValueError):
        Span("a.md", start, end)


def test_hit_defaults_and_frozen():
    h = Hit("a.md", 1, 2)
    assert h.score == 0.0 and h.text == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.score = 1.0


def test_hit_keeps_what_it_is_given():
    h = Hit(path="a.md", start_line=4, end_line=9, score=0.75, text="hello")
    assert (h.path, h.start_line, h.end_line, h.score, h.text) == ("a.md", 4, 9, 0.75, "hello")


def test_question_holds_gold_spans():
    gold = (Span("a.md", 1, 5), Span("b.py", 10, 20))
    q = Question(id="q1", category="code", split="dev", text="Where?", gold=gold)
    assert q.gold == gold and q.text == "Where?"
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.id = "other"


def test_any_class_with_search_satisfies_retriever():
    class Fake:  # no inheritance: structural typing
        def search(self, query, k):
            return [Hit("a.md", 1, 2)]

    def use(r: Retriever):
        return r.search("x", 1)

    assert use(Fake()) == [Hit("a.md", 1, 2)]
    assert hasattr(Retriever, "search")
