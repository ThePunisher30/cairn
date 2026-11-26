"""Putting another library's retriever into the harness (Phase 11).

The harness scores `Hit`s that carry a file and a line range. A framework returns nodes that carry
a file and a *character* range inside the document text it read. `CharSpanRetriever` converts one
into the other, so a framework row is scored by exactly the same code, and the same overlap rule,
as a Cairn chunk. Nothing here imports a framework: the framework-specific code lives in
`bench/frameworks/` and only has to hand over `RetrievedNode`s.
"""

from dataclasses import dataclass

from cairn.types import Hit


class AdapterError(Exception):
    pass


@dataclass(frozen=True)
class RetrievedNode:
    """What a framework retrieved: where it came from and what it said."""

    path: str  # relative to the corpus root, forward slashes (as in the gold spans)
    start_char: int | None  # offset in the document text; None if the framework gave none
    end_char: int | None  # exclusive
    score: float
    text: str  # the node's own text, without any metadata the framework may add


def char_to_line(text, index):
    """The 1-based number of the line that holds the character at `index`. Lines are counted
    by "\\n" as everywhere else in Cairn."""
    index = max(0, min(index, len(text)))
    return text.count("\n", 0, index) + 1


def span_lines(text, start_char, end_char):
    """(first line, last line) of the characters `start_char` up to, not including, `end_char`.
    An empty span counts as the line it sits on. A trailing newline does not add a line."""
    if end_char <= start_char:
        line = char_to_line(text, start_char)
        return line, line
    first = char_to_line(text, start_char)
    last = char_to_line(text, end_char - 1)  # the last character that is inside the span
    return first, last


def locate(text, node_text):
    """Where `node_text` starts in `text`, for a framework that gives no offsets. The first
    occurrence is used, so a node whose text repeats in the file can be placed on the wrong
    copy: the caller should say so. Raises if the text is not found."""
    start = text.find(node_text)
    if start < 0:
        raise AdapterError("the node's text is not in the document: its lines cannot be found")
    return start, start + len(node_text)


class CharSpanRetriever:
    """`retrieve(query, k)` returns `RetrievedNode`s; `texts` maps each path to the document text
    the framework saw (the offsets refer to that text, not to a re-read of the file)."""

    def __init__(self, retrieve, texts, *, locate_missing=False):
        self._retrieve = retrieve
        self._texts = texts
        self._locate_missing = locate_missing
        self.located = 0  # nodes placed by searching for their text (see `locate`)

    def search(self, query, k):
        if k <= 0:
            return []
        hits = []
        for node in self._retrieve(query, k)[:k]:
            try:
                text = self._texts[node.path]
            except KeyError as exc:
                raise AdapterError(
                    f"a retrieved node names a file not in the corpus: {node.path}"
                ) from exc
            start, end = node.start_char, node.end_char
            if start is None or end is None:
                if not self._locate_missing:
                    raise AdapterError(f"{node.path}: the node has no character offsets")
                start, end = locate(text, node.text)
                self.located += 1
            first, last = span_lines(text, start, end)
            hits.append(Hit(node.path, first, last, node.score, node.text))
        return hits
