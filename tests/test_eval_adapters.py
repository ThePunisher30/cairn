import pytest

from cairn.evaluation.adapters import (
    AdapterError,
    CharSpanRetriever,
    RetrievedNode,
    char_to_line,
    locate,
    span_lines,
)
from cairn.evaluation.harness import run_retriever, summarize
from cairn.types import Question, Span

TEXT = "alpha\nbeta\ngamma\ndelta\n"  # lines: 1 alpha, 2 beta, 3 gamma, 4 delta


def test_a_character_offset_gives_the_line_that_holds_it():
    assert [char_to_line(TEXT, i) for i in (0, 4, 5, 6, 10, 11, 16, 17)] == [1, 1, 1, 2, 2, 3, 3, 4]
    assert char_to_line(TEXT, -5) == 1  # clamped
    assert char_to_line(TEXT, 10_000) == 5  # past the end: after the last newline


def test_a_span_covers_the_lines_of_its_first_and_last_character():
    assert span_lines(TEXT, 0, 5) == (1, 1)  # "alpha"
    assert span_lines(TEXT, 6, 16) == (2, 3)  # "beta\ngamma"
    assert span_lines(TEXT, 3, 8) == (1, 2)  # starts inside alpha, ends inside beta


def test_a_trailing_newline_does_not_add_a_line_but_a_leading_one_does():
    assert span_lines(TEXT, 6, 11) == (2, 2)  # "beta\n": the newline belongs to line 2
    assert span_lines(TEXT, 5, 10) == (1, 2)  # "\nbeta": starts on the newline of line 1


def test_an_empty_span_is_the_line_it_sits_on():
    assert span_lines(TEXT, 7, 7) == (2, 2)
    assert span_lines(TEXT, 9, 3) == (2, 2)


def test_locate_finds_the_first_copy_and_refuses_a_missing_text():
    assert locate("one two one", "one") == (0, 3)
    with pytest.raises(AdapterError, match="not in the document"):
        locate("one two", "three")


def make(nodes, texts=None, **kwargs):
    return CharSpanRetriever(lambda query, k: nodes, texts or {"a.md": TEXT}, **kwargs)


def test_nodes_become_hits_with_line_numbers_scores_and_text():
    nodes = [
        RetrievedNode("a.md", 6, 16, 0.9, "beta\ngamma"),
        RetrievedNode("a.md", 0, 5, 0.4, "alpha"),
    ]
    hits = make(nodes).search("q", 10)
    assert [(h.path, h.start_line, h.end_line, h.score, h.text) for h in hits] == [
        ("a.md", 2, 3, 0.9, "beta\ngamma"),
        ("a.md", 1, 1, 0.4, "alpha"),
    ]


def test_it_never_returns_more_than_k_even_if_the_framework_does():
    nodes = [RetrievedNode("a.md", 0, 5, 1.0 - i / 10, "alpha") for i in range(5)]
    assert len(make(nodes).search("q", 2)) == 2
    assert make(nodes).search("q", 0) == []


def test_a_node_from_an_unknown_file_is_an_error_not_a_guess():
    with pytest.raises(AdapterError, match="not in the corpus"):
        make([RetrievedNode("zzz.md", 0, 5, 1.0, "x")]).search("q", 3)


def test_a_node_without_offsets_is_refused_unless_locating_is_allowed():
    node = RetrievedNode("a.md", None, None, 0.5, "gamma")
    with pytest.raises(AdapterError, match="no character offsets"):
        make([node]).search("q", 3)
    retriever = make([node], locate_missing=True)
    hit = retriever.search("q", 3)[0]
    assert (hit.start_line, hit.end_line) == (3, 3) and retriever.located == 1


def test_the_harness_scores_a_framework_row_like_any_other_retriever():
    question = Question("q1", "lexical", "dev", "where is gamma", (Span("a.md", 3, 3),))
    right = RetrievedNode("a.md", 11, 16, 0.9, "gamma")
    wrong = RetrievedNode("a.md", 0, 5, 0.9, "alpha")
    for node, expected in ((right, 1.0), (wrong, 0.0)):
        results = run_retriever(make([node]), [question], 5)
        assert summarize(results, ks=(1, 5))["overall"]["hit@1"] == expected
