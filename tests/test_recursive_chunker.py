"""Phase 2, step 4b: RecursiveChunker (paragraphs first, lines only when a paragraph is too long).

Algorithm, in words:
1. A paragraph is a run of consecutive non-blank lines. Blank lines separate paragraphs.
2. A paragraph that fits the token budget stays whole. One that does not is cut into pieces
   by packing lines greedily, exactly like FixedChunker.
3. Consecutive pieces are merged while the merged chunk (blank lines between them included)
   still fits the budget.
4. Overlap: every chunk after the first is extended backwards so that it starts with the last
   `overlap_lines` lines of the previous chunk, but never reaching the previous chunk's first line
   (so at most len(previous) - 1 lines), and leading blank lines are trimmed.
"""

import pytest

from cairn.ingest.chunkers import RecursiveChunker
from cairn.types import Document

#  1 "aa bb"  2 "cc dd"  3 ""  4 "ee ff"  5 "gg hh"  6 ""  7 "ii jj"   (every text line is 2 tokens)
THREE_PARAGRAPHS = ["aa bb", "cc dd", "", "ee ff", "gg hh", "", "ii jj"]
ONE_LONG_PARAGRAPH = ["aa bb"] * 6


def doc_of(lines, path="f.md", kind="prose"):
    return Document(path, "\n".join(lines) + "\n", kind)


def spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


def test_everything_in_one_chunk_when_the_budget_is_big():
    assert spans(RecursiveChunker(max_tokens=100).split(doc_of(THREE_PARAGRAPHS))) == [(1, 7)]


def test_paragraphs_are_merged_while_they_fit():
    # paragraphs 1-2 and 4-5 are 4 tokens each: together (lines 1..5) 8 <= 9, adding 7 makes 10
    assert spans(RecursiveChunker(max_tokens=9).split(doc_of(THREE_PARAGRAPHS))) == [(1, 5), (7, 7)]


def test_paragraphs_that_do_not_fit_together_stay_apart():
    assert spans(RecursiveChunker(max_tokens=4).split(doc_of(THREE_PARAGRAPHS))) == [
        (1, 2),
        (4, 5),
        (7, 7),
    ]


def test_a_paragraph_that_is_too_long_is_cut_by_lines():
    doc = doc_of(ONE_LONG_PARAGRAPH)  # one paragraph of 12 tokens
    assert spans(RecursiveChunker(max_tokens=5).split(doc)) == [(1, 2), (3, 4), (5, 6)]


def test_overlap_repeats_the_end_of_the_previous_chunk():
    doc = doc_of(ONE_LONG_PARAGRAPH)
    assert spans(RecursiveChunker(5, overlap_lines=1).split(doc)) == [(1, 2), (2, 4), (4, 6)]


def test_overlap_never_swallows_the_whole_previous_chunk():
    doc = doc_of(ONE_LONG_PARAGRAPH)  # previous chunks have only 2 lines, so at most 1 is repeated
    assert spans(RecursiveChunker(5, overlap_lines=5).split(doc)) == [(1, 2), (2, 4), (4, 6)]


@pytest.mark.parametrize(
    "overlap,expected",
    [
        (0, [(1, 5), (7, 7)]),
        (1, [(1, 5), (5, 7)]),
        (2, [(1, 5), (4, 7)]),
        (3, [(1, 5), (4, 7)]),  # the third line back is blank, so the leading blank is trimmed
    ],
)
def test_overlap_across_a_paragraph_break(overlap, expected):
    doc = doc_of(THREE_PARAGRAPHS)
    assert spans(RecursiveChunker(9, overlap_lines=overlap).split(doc)) == expected


def test_chunk_text_and_metadata():
    doc = doc_of(THREE_PARAGRAPHS, path="docs/y.md")
    first, second = RecursiveChunker(max_tokens=9).split(doc)
    assert first.text == "aa bb\ncc dd\n\nee ff\ngg hh"
    assert second.text == "ii jj"
    assert first.path == "docs/y.md" and first.kind == "prose" and first.heading_path == ()


@pytest.mark.parametrize("bad", [0, -1])
def test_max_tokens_must_be_positive(bad):
    with pytest.raises(ValueError):
        RecursiveChunker(max_tokens=bad)


def test_overlap_cannot_be_negative():
    with pytest.raises(ValueError):
        RecursiveChunker(max_tokens=10, overlap_lines=-1)


def test_empty_and_blank_documents_give_no_chunks():
    assert RecursiveChunker(10).split(Document("f.md", "", "prose")) == []
    assert RecursiveChunker(10).split(Document("f.md", "\n \n", "prose")) == []
