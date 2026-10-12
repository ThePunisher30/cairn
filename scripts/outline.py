"""Print the outline of a corpus file, with line ranges you can use for gold spans.

Usage:
    python scripts/outline.py docs/en/docs/tutorial/background-tasks.md
    python scripts/outline.py fastapi/routing.py
    python scripts/outline.py fastapi/routing.py --grep route     # only names containing "route"

For a Markdown file: each section (heading) with the lines it covers.
For a Python file: each class / function / method with the lines it covers.
The path is relative to the corpus folder (data/raw/fastapi).
"""

import argparse
import ast
import sys
from pathlib import Path

from cairn.config import load_settings


def count_lines(text: str) -> int:
    return len(text.split("\n")) - (1 if text.endswith("\n") else 0)


def markdown_outline(text: str) -> list[tuple[int, int, str]]:
    """(start, end, heading) for every section; ignores '#' lines inside code fences."""
    lines = text.split("\n")
    total = count_lines(text)
    heads: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heads.append((number, "  " * (level - 1) + line.strip()))
    return [
        (start, (heads[i + 1][0] - 1 if i + 1 < len(heads) else total), title)
        for i, (start, title) in enumerate(heads)
    ]


def python_outline(text: str) -> list[tuple[int, int, str]]:
    """(start, end, name) for every class, function and method (decorators included)."""
    found: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                kind = "class" if isinstance(child, ast.ClassDef) else "def"
                label = "  " * depth + f"{kind} {child.name}"
                found.append((start, child.end_lineno or start, label))
                visit(child, depth + 1)

    visit(ast.parse(text), 0)
    return found


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy encoding: never crash on characters like arrows or emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Outline of a corpus file with line ranges.")
    parser.add_argument("path", help="path inside the corpus folder, with forward slashes")
    parser.add_argument("--grep", help="only show entries whose text contains this word")
    parser.add_argument("--corpus", type=Path, default=load_settings().data_dir / "raw" / "fastapi")
    args = parser.parse_args(argv)

    file = args.corpus / args.path
    if not file.is_file():
        print(f"error: no such file: {file}", file=sys.stderr)
        return 1

    text = file.read_text(encoding="utf-8")
    if file.suffix == ".md":
        entries = markdown_outline(text)
    elif file.suffix == ".py":
        entries = python_outline(text)
    else:
        print("error: only .md and .py files are supported", file=sys.stderr)
        return 1

    for start, end, label in entries:
        if args.grep is None or args.grep.lower() in label.lower():
            print(f"{start:5d}-{end:<5d} {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
