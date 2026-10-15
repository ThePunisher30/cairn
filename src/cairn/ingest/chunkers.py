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
        return [make_chunk(doc.path, s, e, "\n".join(lines[s - 1 : e]), doc.kind) for s, e in spans]
