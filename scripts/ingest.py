"""Chunk the corpus and write data/chunks.jsonl.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --prose-tokens 400 --code-tokens 300
"""

import argparse
import sys
from pathlib import Path

from cairn.config import load_settings
from cairn.ingest.chunkers import MarkdownChunker, PythonChunker
from cairn.ingest.pipeline import format_stats, ingest


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Chunk the corpus into chunks.jsonl.")
    parser.add_argument("--root", type=Path, default=settings.data_dir / "raw" / "fastapi")
    parser.add_argument("--out", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--prose-tokens", type=int, default=300)
    parser.add_argument("--code-tokens", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: {args.root} is not a folder (run scripts/fetch_corpus.py?)", file=sys.stderr)
        return 1

    chunkers = {
        "prose": MarkdownChunker(args.prose_tokens),
        "code": PythonChunker(args.code_tokens),
    }
    stats = ingest(args.root, args.out, chunkers)
    print(format_stats(stats))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
