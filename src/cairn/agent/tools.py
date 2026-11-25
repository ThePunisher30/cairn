"""The tools the agent may call, and the registry that runs them safely.

The model never runs anything: it sends a request, our code runs it and sends back text. The
model's output is untrusted input, so every request is checked, and every failure comes back as a
message the model can read ("Error: ..."), never as a crash.
"""

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cairn.generation.types import ToolCall, ToolSpec
from cairn.types import Document


class ToolError(Exception):
    """Raised by a tool with a message meant for the model to read."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # a JSON schema of the arguments
    func: Callable[..., str]  # called as func(**arguments)


@dataclass(frozen=True)
class ToolResult:
    text: str  # what the model reads: the result, or "Error: " + error
    error: str | None = None
    truncated: bool = False


def _fail(error):
    return ToolResult("Error: " + error, error=error)


# ------------------------------------------------------------------ argument checking
def _is_type(value, name):
    # bool is a subclass of int in Python, but a model sending true for k is making a mistake
    if name == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int)
    if name == "number":
        return isinstance(value, int | float)
    return True  # a type we do not check


def _check_arguments(schema, arguments):
    """The first problem found with `arguments` against a small subset of JSON schema, or None."""
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            return f"missing required argument '{name}'"
    for name in arguments:
        if name not in properties:
            known = ", ".join(properties) or "none"
            return f"unknown argument '{name}' (the arguments are: {known})"
    for name, value in arguments.items():
        rule = properties[name]
        wanted = rule.get("type")
        if wanted and not _is_type(value, wanted):
            return f"argument '{name}' must be of type {wanted}, got {value!r}"
        if wanted in ("integer", "number"):
            if "minimum" in rule and value < rule["minimum"]:
                return f"argument '{name}' must be at least {rule['minimum']}, got {value}"
            if "maximum" in rule and value > rule["maximum"]:
                return f"argument '{name}' must be at most {rule['maximum']}, got {value}"
    return None


class ToolRegistry:
    def __init__(self, tools, *, max_result_chars=6000):
        self._tools = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"two tools are called '{tool.name}'")
            self._tools[tool.name] = tool
        self.max_result_chars = max_result_chars

    @property
    def names(self):
        return tuple(self._tools)

    def specs(self):
        return [ToolSpec(t.name, t.description, t.parameters) for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises an Exception: failures become an error result."""
        if call.error is not None:
            return _fail(call.error)
        tool = self._tools.get(call.name)
        if tool is None:
            return _fail(f"unknown tool '{call.name}'; the tools are: {', '.join(self.names)}")
        problem = _check_arguments(tool.parameters, call.arguments)
        if problem is not None:
            return _fail(problem)
        try:
            text = tool.func(**call.arguments)
        except ToolError as exc:
            return _fail(str(exc))
        except Exception as exc:  # a bug in a tool must not end the run
            return _fail(f"tool '{call.name}' failed: {type(exc).__name__}: {exc}")
        if len(text) > self.max_result_chars:
            cut = len(text) - self.max_result_chars
            marker = f"\n[truncated: {cut} more characters]"
            return ToolResult(text[: self.max_result_chars] + marker, truncated=True)
        return ToolResult(text)


# ------------------------------------------------------------------ the corpus tools
class CorpusTools:
    """The four tools and the session state they share: which chunks `search` has returned."""

    def __init__(self):
        self.seen = []  # the Hits returned so far; the chunk with id n is seen[n - 1]
        self.ids = {}  # (path, start_line, end_line) -> id
        self.registry = None

    def reset(self):
        """A new question starts clean."""
        self.seen.clear()
        self.ids.clear()


def _safe_path(root, path):
    """The file `path` names inside `root`, or a ToolError. The file is not opened here.

    Refused: absolute paths, any '..' component, NUL, and anything that, once symbolic links are
    resolved, is not inside the corpus (the model must not read ~/.ssh or the .env file).
    """
    if "\0" in path:
        raise ToolError("invalid path")
    norm = path.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise ToolError("path is outside the corpus (absolute paths are not allowed)")
    if ".." in norm.split("/"):
        raise ToolError("path is outside the corpus ('..' is not allowed)")
    full = (root / norm).resolve()
    if not full.is_relative_to(root):
        raise ToolError("path is outside the corpus")
    if not full.exists():
        raise ToolError(f"no such file: {path}")
    if full.is_dir():
        raise ToolError(f"{path} is a directory, not a file")
    return full


def _read_lines(full, path):
    text = full.read_text(encoding="utf-8", errors="replace")
    return Document(path, text, "prose").lines  # lines counted as everywhere else in Cairn


def _symbols(tree):
    """(indent, keyword, name, first line, last line) for classes and functions, in file order."""
    found = []

    def visit(body, depth):
        for node in body:
            if isinstance(node, ast.ClassDef):
                found.append((depth, "class", node.name, node.lineno, node.end_lineno))
                visit(node.body, depth + 1)  # methods; functions inside functions are skipped
            elif isinstance(node, ast.AsyncFunctionDef):
                found.append((depth, "async def", node.name, node.lineno, node.end_lineno))
            elif isinstance(node, ast.FunctionDef):
                found.append((depth, "def", node.name, node.lineno, node.end_lineno))

    visit(tree.body, 0)
    return found


def build_corpus_tools(
    retriever, root, *, max_k=10, snippet_chars=400, max_lines=120, max_result_chars=6000
):
    root = Path(root).resolve()
    state = CorpusTools()
    default_k = min(5, max_k)

    def chunk_id(hit):
        key = (hit.path, hit.start_line, hit.end_line)
        if key not in state.ids:
            state.seen.append(hit)
            state.ids[key] = len(state.seen)
        return state.ids[key]

    def search(query, k=default_k, path_filter=None):
        if not query.strip():
            raise ToolError("the query is empty")
        if path_filter:
            # the filter drops hits, so ask for many more than k to have something left
            hits = retriever.search(query, max(k * 10, 50))
            hits = [h for h in hits if h.path.startswith(path_filter)]
        else:
            hits = retriever.search(query, k)
        hits = hits[:k]
        if not hits:
            return "No results."
        blocks = []
        for hit in hits:
            snippet = hit.text[:snippet_chars]
            if len(hit.text) > snippet_chars:
                snippet += " ..."
            blocks.append(
                f"[{chunk_id(hit)}] {hit.path}:{hit.start_line}-{hit.end_line}\n{snippet}"
            )
        return "\n\n".join(blocks)

    def read_chunk(id):
        if id > len(state.seen):
            raise ToolError(f"there is no chunk with id {id}; search returned {len(state.seen)}")
        hit = state.seen[id - 1]
        return f"[{id}] {hit.path}:{hit.start_line}-{hit.end_line}\n{hit.text}"

    def read_file(path, start, end):
        lines = _read_lines(_safe_path(root, path), path)
        if end < start:
            raise ToolError(f"end ({end}) is before start ({start})")
        if start > len(lines):
            raise ToolError(f"{path} has {len(lines)} lines; start ({start}) is past the end")
        end = min(end, len(lines))
        stop = min(end, start + max_lines - 1)
        out = [f"{n}: {lines[n - 1]}" for n in range(start, stop + 1)]
        if stop < end:
            out.append(f"[stopped after {max_lines} lines; continue from line {stop + 1}]")
        return "\n".join(out)

    def list_symbols(path):
        full = _safe_path(root, path)
        if full.suffix != ".py":
            raise ToolError(f"{path} is not a Python file")
        try:
            tree = ast.parse(full.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError) as exc:
            raise ToolError(f"{path} could not be parsed: {exc}") from exc
        found = _symbols(tree)
        if not found:
            return "No symbols."
        return "\n".join(
            f"{'  ' * depth}{keyword} {name} (lines {first}-{last})"
            for depth, keyword, name, first, last in found
        )

    path_rule = {"type": "string"}
    state.registry = ToolRegistry(
        [
            Tool(
                "search",
                "Search the FastAPI docs and source code. Returns numbered chunks "
                "[n] path:lines with a snippet. Cite a chunk as [n] in your answer.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1, "maximum": max_k},
                        "path_filter": {"type": "string"},
                    },
                    "required": ["query"],
                },
                search,
            ),
            Tool(
                "read_chunk",
                "Read the full text of a chunk that search returned earlier, by its number.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "minimum": 1}},
                    "required": ["id"],
                },
                read_chunk,
            ),
            Tool(
                "read_file",
                "Read lines start..end (1-based, inclusive) of a file in the corpus.",
                {
                    "type": "object",
                    "properties": {
                        "path": path_rule,
                        "start": {"type": "integer", "minimum": 1},
                        "end": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start", "end"],
                },
                read_file,
            ),
            Tool(
                "list_symbols",
                "List the classes, functions and methods of a Python file with their line ranges.",
                {
                    "type": "object",
                    "properties": {"path": path_rule},
                    "required": ["path"],
                },
                list_symbols,
            ),
        ],
        max_result_chars=max_result_chars,
    )
    return state
