"""The chunkers, and make_chunk, the one way a Chunk gets built."""

import ast
import hashlib
import re

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


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")


def _headings(lines):
    """(line number, level, title) for every real heading, skipping fenced code blocks."""
    found = []
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(line)
        if match:
            title = _ANCHOR.sub("", match.group(2)).strip()
            found.append((number, len(match.group(1)), title))
    return found


def _sections(lines):
    """(first line, last line, heading path) for every section of a
    markdown document."""
    headings = _headings(lines)
    sections = []
    first_heading = headings[0][0] if headings else len(lines) + 1
    if first_heading > 1:
        sections.append((1, first_heading - 1, ()))
    stack = []  # (level, title) of the headings above the current one
    for index, (number, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        is_last = index + 1 == len(headings)
        end = len(lines) if is_last else headings[index + 1][0] - 1
        sections.append((number, end, tuple(t for _, t in stack)))
    return sections


class MarkdownChunker:
    def __init__(self, max_tokens=300):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.max_tokens = max_tokens

    def split(self, doc: Document) -> list[Chunk]:
        lines = doc.lines
        chunks = []
        for start, end, heading_path in _sections(lines):
            trimmed = _trim(lines, start, end)
            if trimmed is None:
                continue
            if _tokens(lines, *trimmed) > self.max_tokens:
                spans = _pack_paragraphs(lines, *trimmed, self.max_tokens)  # too big: split it
            else:
                spans = [trimmed]
            chunks.extend(_chunks_from_spans(doc, spans, heading_path))
        return chunks


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _first_line(node):
    """First line of a statement, counting its decorators."""
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _next_nonblank(lines, i):
    while i <= len(lines) and _is_blank(lines[i - 1]):
        i += 1
    return i


def _python_units(lines, statements, prefix, previous_end, units):
    """Append (start, end, heading path) for each unit among `statements`.

    Returns the end line of the last unit, so the caller knows where to continue.
    """
    # 1. group the statements: every def/class alone, runs of other statements together
    groups = []
    for statement in statements:
        if isinstance(statement, _DEFS):
            groups.append([statement])
        elif groups and not isinstance(groups[-1][0], _DEFS):
            groups[-1].append(statement)
        else:
            groups.append([statement])

    # 2. turn each group into a unit
    for group in groups:
        node = group[0]
        start = _next_nonblank(lines, previous_end + 1)  # comments/decorators above belong to it
        if isinstance(node, ast.ClassDef):
            body = node.body
            leading = 0  # how many statements before the first def: docstring, attributes...
            while leading < len(body) and not isinstance(body[leading], _DEFS):
                leading += 1
            if leading > 0:
                header_end = body[leading - 1].end_lineno
            else:  # the body starts with a method: the header is the lines before it
                header_end = _first_line(body[0]) - 1
                while header_end > start and _is_blank(lines[header_end - 1]):
                    header_end -= 1
            path = prefix + (node.name,)
            units.append((start, header_end, path))
            # the methods (and nested classes) are handled the same way, one level deeper
            previous_end = _python_units(lines, body[leading:], path, header_end, units)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append((start, node.end_lineno, prefix + (node.name,)))
            previous_end = node.end_lineno
        else:  # a run of imports, constants, docstring...
            end = group[-1].end_lineno
            units.append((start, end, prefix))
            previous_end = end
    return previous_end


class PythonChunker:
    def __init__(self, fallback_max_tokens=200):
        if fallback_max_tokens < 1:
            raise ValueError("fallback_max_tokens must be at least 1")
        self.fallback = FixedChunker(fallback_max_tokens)

    def split(self, doc: Document) -> list[Chunk]:
        lines = doc.lines
        if all(_is_blank(line) for line in lines):  # also true for an empty file
            return []
        try:
            tree = ast.parse(doc.text)
        except SyntaxError:
            return self.fallback.split(doc)

        units = []
        _python_units(lines, tree.body, (), 0, units)
        if not units:  # a file with only comments
            return self.fallback.split(doc)

        # a comment after the last statement belongs to the last chunk
        last_nonblank = max(n for n, line in enumerate(lines, start=1) if not _is_blank(line))
        if units[-1][1] < last_nonblank:
            start, _, path = units[-1]
            units[-1] = (start, last_nonblank, path)

        chunks = []
        for start, end, path in units:
            chunks.extend(_chunks_from_spans(doc, [(start, end)], path))
        return chunks
