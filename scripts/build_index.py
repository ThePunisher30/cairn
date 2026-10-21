"""Build the BM25 index from the chunks file.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --no-split-identifiers --stopwords
"""

import argparse
import sys
from pathlib import Path

from cairn.config import load_settings
from cairn.ingest.pipeline import read_chunks
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import ENGLISH_STOPWORDS, Tokenizer


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Build the BM25 index from chunks.jsonl.")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--out", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--no-split-identifiers", action="store_true")
    parser.add_argument("--stopwords", action="store_true")
    args = parser.parse_args(argv)

    if not args.chunks.is_file():
        print(f"error: {args.chunks} not found (run scripts/ingest.py first)", file=sys.stderr)
        return 1
    chunks = read_chunks(args.chunks)
    if not chunks:
        print(f"error: {args.chunks} has no chunks", file=sys.stderr)
        return 1

    tokenizer = Tokenizer(
        split_identifiers=not args.no_split_identifiers,
        stopwords=ENGLISH_STOPWORDS if args.stopwords else frozenset(),
    )
    index = InvertedIndex.build(chunks, tokenizer)
    index.save(args.out)
    print(
        f"indexed {index.n_docs} chunks, {len(index.postings)} distinct terms, "
        f"mean length {index.avg_length:.1f} tokens"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
