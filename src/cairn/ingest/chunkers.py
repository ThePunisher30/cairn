"""The chunkers, and make_chunk, the one way a Chunk gets built."""

import hashlib

from cairn.tokens import count_tokens
from cairn.types import Chunk, Document


def make_chunk(path, start_line, end_line, text, kind, heading_path=()) -> Chunk:
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid line range: {start_line}-{end_line}")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    place_and_content = f"{path}:{start_line}:{end_line}:{content_hash}"
    chunk_id = hashlib.sha256(place_and_content.encode("utf-8")).hexdigest()[:16]

    return Chunk(
        id=chunk_id,
        text=text,
        path=path,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        content_hash=content_hash,
        token_count=count_tokens(text),
        heading_path=tuple(heading_path),
    )


def _is_blank(line: str) -> bool:
    return not line.strip()


def _trim(lines, start, end):
    """Move start and end inwards past blank lines.

    Returns (start, end), or None if every line in the range is blank.
    Line numbers are 1-based and inclusive, so line n is lines[n - 1].
    """
    while start <= end and _is_blank(lines[start - 1]):
        start += 1
    while end >= start and _is_blank(lines[end - 1]):
        end -= 1
    if start > end:
        return None
    return start, end


def _pack_lines(lines, start, end, max_tokens):
    """Cut lines start..end into ranges that each stay under max_tokens (greedy)."""
    spans = []
    chunk_start = start
    total = 0
    for i in range(start, end + 1):
        tokens = count_tokens(lines[i - 1])
        if total > 0 and total + tokens > max_tokens:
            spans.append((chunk_start, i - 1))  # close the current chunk before line i
            chunk_start = i
            total = 0
        total += tokens
    spans.append((chunk_start, end))
    result = []
    for s, e in spans:
        trimmed = _trim(lines, s, e)
        if trimmed is not None:
            result.append(trimmed)
    return result


def _chunks_from_spans(doc, spans, heading_path=()):
    lines = doc.lines
    return [
        make_chunk(doc.path, s, e, "\n".join(lines[s - 1 : e]), doc.kind, heading_path)
        for s, e in spans
    ]


class FixedChunker:
    def __init__(self, max_tokens=200):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.max_tokens = max_tokens

    def split(self, doc: Document) -> list[Chunk]:
        lines = doc.lines
        if not lines:
            return []
        spans = _pack_lines(lines, 1, len(lines), self.max_tokens)
        return _chunks_from_spans(doc, spans)


def _tokens(lines, start, end):
    return sum(count_tokens(line) for line in lines[start - 1 : end])


def _paragraphs(lines, start, end):
    """Runs of consecutive non-blank lines, as (first line, last line) pairs."""
    paragraphs = []
    paragraph_start = None  # None means "not inside a paragraph right now"
    for i in range(start, end + 1):
        if _is_blank(lines[i - 1]):
            if paragraph_start is not None:  # a blank line ends the paragraph
                paragraphs.append((paragraph_start, i - 1))
                paragraph_start = None
        elif paragraph_start is None:  # first non-blank line: a paragraph begins
            paragraph_start = i
    if paragraph_start is not None:  # the file ended inside a paragraph
        paragraphs.append((paragraph_start, end))
    return paragraphs


def _pack_paragraphs(lines, start, end, max_tokens):
    """Keep paragraphs whole when they fit; cut long ones by lines; merge neighbours."""
    pieces = []
    for p_start, p_end in _paragraphs(lines, start, end):
        if _tokens(lines, p_start, p_end) > max_tokens:
            pieces.extend(_pack_lines(lines, p_start, p_end, max_tokens))  # too big: cut by lines
        else:
            pieces.append((p_start, p_end))  # fits: keep it whole

    merged = []
    for p_start, p_end in pieces:
        if merged and _tokens(lines, merged[-1][0], p_end) <= max_tokens:
            merged[-1] = (merged[-1][0], p_end)  # extend the previous chunk over this piece
        else:
            merged.append((p_start, p_end))  # start a new chunk
    return merged


def _apply_overlap(lines, spans, overlap_lines):
    """Each span after the first starts with the last `overlap_lines` lines of the one before."""
    if not spans:
        return spans
    result = [spans[0]]
    for (previous_start, previous_end), (start, end) in zip(spans, spans[1:], strict=False):
        new_start = max(previous_end - overlap_lines + 1, previous_start + 1)
        trimmed = _trim(lines, min(new_start, start), end)
        result.append(trimmed)
    return result


def _chunks_from_spans(doc, spans, heading_path=()):
    lines = doc.lines
    return [
        make_chunk(doc.path, s, e, "\n".join(lines[s - 1 : e]), doc.kind, heading_path)
        for s, e in spans
    ]


class RecursiveChunker:
    def __init__(self, max_tokens=200, overlap_lines=0):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if overlap_lines < 0:
            raise ValueError("overlap_lines cannot be negative")
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines

    def split(self, doc: Document) -> list[Chunk]:
        lines = doc.lines
        if not lines:
            return []
        spans = _pack_paragraphs(lines, 1, len(lines), self.max_tokens)
        if self.overlap_lines > 0:
            spans = _apply_overlap(lines, spans, self.overlap_lines)
        return _chunks_from_spans(doc, spans)
