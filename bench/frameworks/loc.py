"""Lines of code, counted one way for everything (Phase 11).

    python bench/frameworks/loc.py

The rule: a line counts if it is not blank and does not start with `#`. Docstrings and the
comments after code count (they are code you wrote and maintain); that favours neither side much,
and the rule is stated so that it can be argued with. The framework count covers only what you
write (`llamaindex_minimal.py`), never the library; Cairn's count covers every module its
BM25 + dense + hybrid pipeline needs, because there is no library under it. So the two numbers
answer "how much do you write", not "how good is the code".
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "cairn"

# everything the Cairn BM25 + dense + hybrid pipeline needs, from raw files to ranked hits
CAIRN_PIPELINE = [
    "types.py",
    "tokens.py",
    "jsonl.py",
    "ingest/documents.py",
    "ingest/chunkers.py",
    "ingest/pipeline.py",
    "retrieval/tokenizer.py",
    "retrieval/index.py",
    "retrieval/bm25.py",
    "retrieval/fusion.py",
    "retrieval/hybrid.py",
    "dense/embedder.py",
    "dense/models.py",
    "dense/cache.py",
    "dense/flat_index.py",
    "dense/retriever.py",
]
# written on top of that and not counted above: the approximate indexes (ann/), reranking,
# generation, the agent, the service, the evaluation harness


def count(text):
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def function_source(path, name):
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    raise KeyError(name)


def imports_of(path):
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            out.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cairn = {name: count((SRC / name).read_text(encoding="utf-8")) for name in CAIRN_PIPELINE}
    minimal = Path(__file__).with_name("llamaindex_minimal.py")
    imports = count(imports_of(minimal))
    default = count(function_source(minimal, "default_pipeline"))
    selected = count(function_source(minimal, "selected_pipeline"))
    report = {
        "rule": "non-blank lines that do not start with #",
        "cairn_modules": cairn,
        "cairn_total": sum(cairn.values()),
        "llamaindex_imports": imports,
        "llamaindex_default_pipeline": default,
        "llamaindex_selected_pipeline": selected,
        "llamaindex_default_total": imports + default,
        "llamaindex_selected_total": imports + selected,
    }
    for name, lines in cairn.items():
        print(f"{lines:>5}  src/cairn/{name}")
    print(f"{report['cairn_total']:>5}  Cairn, the whole BM25 + dense + hybrid pipeline")
    print(f"{imports:>5}  LlamaIndex imports (shared by both)")
    print(
        f"{default:>5}  LlamaIndex default pipeline (vector search)"
        f"  -> {imports + default} with imports"
    )
    print(f"{selected:>5}  LlamaIndex selected pipeline (Markdown + code, vector + BM25 fusion)"
          f"  -> {imports + selected} with imports")  # fmt: skip
    (Path(__file__).parent / "results" / "loc.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
