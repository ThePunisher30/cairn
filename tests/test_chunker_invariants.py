"""Phase 2, step 4e: promises EVERY chunker must keep, whatever the document looks like."""

import pytest

from cairn.ingest.chunkers import (
    FixedChunker,
    MarkdownChunker,
    PythonChunker,
    RecursiveChunker,
    make_chunk,
)
from cairn.types import Document

MARKDOWN = "\n".join(
    [
        "Intro text",
        "",
        "# Title { #title }",
        "body line one",
        "body line two",
        "",
        "## Sub A",
        "a1",
        "",
        "```python",
        "# not a heading",
        "x = 1",
        "```",
        "",
        "### Deep",
        "d1",
        "",
        "## Sub B",
        "b1",
        "b2",
    ]
)

PYTHON = '''\
"""Docstring."""
import os

# comment above
def helper(x):
    return x + 1


class Greeter:
    """Hello."""
    greeting = "hi"

    def __init__(self):
        self.n = 0

    def name(self):
        return "x"

x = Greeter()
# trailing
'''

DOCS = {
    "markdown": Document("docs/a.md", MARKDOWN + "\n", "prose"),
    "python": Document("fastapi/m.py", PYTHON, "code"),
    "no-final-newline": Document("a.md", "one\ntwo\n\nthree", "prose"),
    "single-line": Document("a.md", "just one line", "prose"),
    "blank-heavy": Document("a.md", "\n\na b c\n\n\n\nd e f\n\n", "prose"),
    "broken-python": Document("fastapi/b.py", "def broken(:\n    pass\nx = 1\n", "code"),
}

# (chunker factory, may its chunks overlap?)
CHUNKERS = {
    "fixed-6": (lambda: FixedChunker(6), False),
    "fixed-40": (lambda: FixedChunker(40), False),
    "recursive-8": (lambda: RecursiveChunker(8), False),
    "recursive-40": (lambda: RecursiveChunker(40), False),
    "recursive-8-overlap2": (lambda: RecursiveChunker(8, overlap_lines=2), True),
    "markdown-10": (lambda: MarkdownChunker(10), False),
    "markdown-300": (lambda: MarkdownChunker(300), False),
    "python": (lambda: PythonChunker(), False),
    "python-small-fallback": (lambda: PythonChunker(fallback_max_tokens=6), False),
}

CASES = [(c, d) for c in CHUNKERS for d in DOCS]


@pytest.mark.parametrize("chunker_name,doc_name", CASES)
def test_the_chunker_keeps_its_promises(chunker_name, doc_name):
    factory, overlapping = CHUNKERS[chunker_name]
    doc = DOCS[doc_name]
    lines = doc.lines
    chunks = factory().split(doc)

    covered = set()
    previous = None
    for chunk in chunks:
        # lines are real and in order
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        # the text is exactly those lines
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        # it belongs to this document
        assert chunk.path == doc.path and chunk.kind == doc.kind
        # it starts and ends on a real line, never on a blank one
        assert lines[chunk.start_line - 1].strip() and lines[chunk.end_line - 1].strip()
        # its identity, hash and size are the ones make_chunk would compute
        assert chunk == make_chunk(
            chunk.path, chunk.start_line, chunk.end_line, chunk.text, chunk.kind, chunk.heading_path
        )
        if previous is not None:
            assert (
                chunk.start_line > previous.start_line
            )  # in order, never starting at the same line
            if not overlapping:
                assert chunk.start_line > previous.end_line  # no line is in two chunks
        previous = chunk
        covered |= set(range(chunk.start_line, chunk.end_line + 1))

    # nothing is lost: every non-blank line of the document is in some chunk
    non_blank = {n for n, line in enumerate(lines, start=1) if line.strip()}
    assert non_blank <= covered
    # ids are unique
    assert len({c.id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("chunker_name", list(CHUNKERS))
def test_splitting_twice_gives_the_same_chunks(chunker_name):
    factory, _ = CHUNKERS[chunker_name]
    doc = DOCS["markdown"]
    assert factory().split(doc) == factory().split(doc)


@pytest.mark.parametrize("chunker_name", list(CHUNKERS))
def test_a_blank_document_has_no_chunks(chunker_name):
    factory, _ = CHUNKERS[chunker_name]
    assert factory().split(Document("a.md", "\n  \n\n", "prose")) == []
