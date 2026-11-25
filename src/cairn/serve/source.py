"""Reading a few lines of a corpus file for the source viewer (Phase 10).

The path comes from the browser, so it is untrusted input exactly like a path from the model in
Phase 8: the same checks apply (no absolute path, no `..`, nothing that resolves outside the
corpus), plus one more: only the files that were indexed can be read, not everything under the
folder (the checkout also holds `.git`, build files and so on).
"""

from pathlib import Path

from cairn.agent.tools import ToolError, _safe_path
from cairn.ingest.documents import EXCLUDED, SCOPE
from cairn.types import Document


class SourceError(Exception):
    """The request cannot be served; `status` is the HTTP status to use."""

    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def in_scope(path):
    norm = path.replace("\\", "/")
    if norm in EXCLUDED:
        return False
    return any(
        norm.startswith(folder + "/") and norm.endswith(suffix) for folder, suffix, _ in SCOPE
    )


def read_source(root, path, start, end=None, *, context=6, max_lines=400):
    """The lines around `start`..`end` (1-based, inclusive) of `path` under `root`, with
    `context` lines on each side, and which of them are the cited ones."""
    if root is None:
        raise SourceError("the source files are not available on this server", 404)
    if not in_scope(path):
        raise SourceError("that file is not part of the indexed corpus", 404)
    try:
        full = _safe_path(Path(root).resolve(), path)
    except ToolError as exc:
        raise SourceError(str(exc), 404) from exc
    lines = Document(path, full.read_text(encoding="utf-8", errors="replace"), "prose").lines
    end = start if end is None else end
    if end < start:
        raise SourceError("end must not be before start", 422)
    if start > len(lines):
        raise SourceError(f"the file has only {len(lines)} lines", 404)
    end = min(end, len(lines))
    first = max(1, start - context)
    last = min(len(lines), end + context, first + max_lines - 1)
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "first_line": first,
        "last_line": last,
        "total_lines": len(lines),
        "lines": [
            {"n": n, "text": lines[n - 1], "cited": start <= n <= end}
            for n in range(first, last + 1)
        ],
    }
