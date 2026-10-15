"""Phase 2, step 4a: FixedChunker, the naive baseline (pack lines until the budget is full)."""

import pytest

from cairn.ingest.chunkers import FixedChunker
from cairn.tokens import count_tokens
from cairn.types import Document


def doc_of(lines, path="f.md", kind="prose"):
    return Document(path, "\n".join(lines) + "\n", kind)


def spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


def test_lines_are_packed_until_the_budget_is_full():
    doc = doc_of(["a b c"] * 5)  # every line is 3 tokens
    # two lines = 6 tokens fit in 7, a third would make 9
    assert spans(FixedChunker(max_tokens=7).split(doc)) == [(1, 2), (3, 4), (5, 5)]


def test_everything_fits_in_one_chunk_when_the_budget_is_big():
    assert spans(FixedChunker(max_tokens=1000).split(doc_of(["a b c"] * 5))) == [(1, 5)]


def test_a_line_that_is_over_budget_on_its_own_gets_its_own_chunk():
    doc = doc_of(["a b", "x y z w v u t s r q", "c"])  # 2, 10, 1 tokens
    assert spans(FixedChunker(max_tokens=4).split(doc)) == [(1, 1), (2, 2), (3, 3)]


def test_blank_lines_inside_a_chunk_cost_nothing():
    doc = doc_of(["a b", "", "", "c d", "e f"])  # 2, 0, 0, 2, 2
    assert spans(FixedChunker(max_tokens=4).split(doc)) == [(1, 4), (5, 5)]


def test_chunks_never_start_or_end_on_a_blank_line():
    doc = doc_of(["a b c", "", "", "", "d e f"])
    assert spans(FixedChunker(max_tokens=3).split(doc)) == [(1, 1), (5, 5)]


def test_chunk_text_and_metadata():
    doc = doc_of(["a b c", "d e f", "g h i"], path="docs/x.md")
    first, second = FixedChunker(max_tokens=6).split(doc)
    assert first.text == "a b c\nd e f"
    assert second.text == "g h i"
    assert first.path == "docs/x.md" and first.kind == "prose"
    assert first.heading_path == ()
    assert first.token_count == count_tokens(first.text)


def test_the_kind_is_taken_from_the_document():
    assert FixedChunker(10).split(doc_of(["x = 1"], kind="code"))[0].kind == "code"


@pytest.mark.parametrize("bad", [0, -5])
def test_max_tokens_must_be_positive(bad):
    with pytest.raises(ValueError):
        FixedChunker(max_tokens=bad)


def test_empty_and_blank_documents_give_no_chunks():
    assert FixedChunker(10).split(Document("f.md", "", "prose")) == []
    assert FixedChunker(10).split(Document("f.md", "\n\n   \n", "prose")) == []
