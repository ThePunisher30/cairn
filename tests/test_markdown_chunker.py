"""Phase 2, step 4c: MarkdownChunker (one chunk per section, labelled with its heading path).

Rules:
- A heading is a line starting with 1-6 '#' followed by a space, at the start of the line, and NOT
  inside a fenced code block (a line starting with ``` or ~~~ opens or closes a fence).
- A section runs from its heading line to the line before the next heading of any level. Text
  before the
  first heading is a section with an empty heading path.
- heading_path is the chain of titles above and including the section: when a heading of level L
  arrives, every remembered heading of level >= L is forgotten first.
- A title is the heading text without the '#'s and without a trailing ' { #anchor }' attribute.
- Sections are trimmed of blank lines at both ends. A section over the token budget is split with
  the
  same paragraph logic as RecursiveChunker (no overlap); all of its pieces share the heading path.
"""

import pytest

from cairn.ingest.chunkers import MarkdownChunker
from cairn.types import Document

SAMPLE = [
    "Intro text",  # 1
    "",  # 2
    "# Title { #title }",  # 3
    "body",  # 4
    "",  # 5
    "## Sub A",  # 6
    "a1",  # 7
    "",  # 8
    "```python",  # 9
    "# not a heading",  # 10
    "```",  # 11
    "",  # 12
    "### Deep",  # 13
    "d1",  # 14
    "",  # 15
    "## Sub B",  # 16
    "b1",  # 17
]


def doc_of(lines, path="docs/a.md"):
    return Document(path, "\n".join(lines) + "\n", "prose")


def summary(chunks):
    return [(c.start_line, c.end_line, c.heading_path) for c in chunks]


def test_sections_and_their_heading_paths():
    chunks = MarkdownChunker(max_tokens=1000).split(doc_of(SAMPLE))
    assert summary(chunks) == [
        (1, 1, ()),
        (3, 4, ("Title",)),
        (6, 11, ("Title", "Sub A")),  # the fenced block stays inside its section
        (13, 14, ("Title", "Sub A", "Deep")),
        (16, 17, ("Title", "Sub B")),
    ]


def test_chunk_text_is_the_section_lines_unchanged():
    chunks = MarkdownChunker(max_tokens=1000).split(doc_of(SAMPLE))
    assert chunks[1].text == "# Title { #title }\nbody"
    assert chunks[2].text.splitlines()[-2:] == ["# not a heading", "```"]
    assert all(c.kind == "prose" and c.path == "docs/a.md" for c in chunks)


def test_the_index_text_starts_with_the_breadcrumb():
    chunk = MarkdownChunker(max_tokens=1000).split(doc_of(SAMPLE))[2]
    assert chunk.index_text.startswith("Title > Sub A\n")


def test_skipped_heading_levels():
    doc = doc_of(["# A", "x", "### C", "y", "## B", "z"])
    paths = [c.heading_path for c in MarkdownChunker(1000).split(doc)]
    assert paths == [("A",), ("A", "C"), ("A", "B")]


def test_anchor_attributes_are_removed_and_backticks_are_kept():
    doc = doc_of(["## `Query` as the default value { #query-as-the-default-value }", "text"])
    assert MarkdownChunker(1000).split(doc)[0].heading_path == ("`Query` as the default value",)


def test_a_hash_without_a_space_is_not_a_heading():
    doc = doc_of(["#hashtag", "text"])
    assert summary(MarkdownChunker(1000).split(doc)) == [(1, 2, ())]


def test_an_indented_hash_line_is_not_a_heading():
    doc = doc_of(["intro", "    # indented", "more"])
    assert summary(MarkdownChunker(1000).split(doc)) == [(1, 3, ())]


def test_a_tilde_fence_hides_headings_too():
    doc = doc_of(["# A", "~~~", "# inside", "~~~", "end"])
    assert summary(MarkdownChunker(1000).split(doc)) == [(1, 5, ("A",))]


def test_a_heading_with_no_body_is_still_a_chunk():
    doc = doc_of(["# A", "## B", "text"])
    assert summary(MarkdownChunker(1000).split(doc)) == [(1, 1, ("A",)), (2, 3, ("A", "B"))]


def test_a_document_without_headings_is_one_chunk_with_an_empty_path():
    assert summary(MarkdownChunker(1000).split(doc_of(["just", "text"]))) == [(1, 2, ())]


def test_a_section_over_budget_is_split_and_every_piece_keeps_the_heading_path():
    doc = doc_of(["# T", "", "aa bb", "", "cc dd", "", "ee ff"])  # the section is 8 tokens
    chunks = MarkdownChunker(max_tokens=5).split(doc)
    assert summary(chunks) == [(1, 3, ("T",)), (5, 7, ("T",))]


@pytest.mark.parametrize("bad", [0, -3])
def test_max_tokens_must_be_positive(bad):
    with pytest.raises(ValueError):
        MarkdownChunker(max_tokens=bad)


def test_empty_and_blank_documents_give_no_chunks():
    assert MarkdownChunker(100).split(Document("f.md", "", "prose")) == []
    assert MarkdownChunker(100).split(Document("f.md", "\n\n", "prose")) == []
