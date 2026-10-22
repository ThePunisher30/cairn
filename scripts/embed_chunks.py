"""Embed the chunks and save the dense index.

Usage:
    python scripts/embed_chunks.py
    python scripts/embed_chunks.py --model  minilim
"""

import argparse
import sys
import time
from pathlib import Path

from cairn.config import load_settings
from cairn.dense.cache import EmbeddingCache
from cairn.dense.models import MODELS, load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.pipeline import read_chunks


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Embed chunks.jsonl into a dense index.")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=settings.data_dir / "embedding_cache.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        print("error: --batch-size must be at least 1", file=sys.stderr)
        return 1
    if not args.chunks.is_file():
        print(f"error: {args.chunks} not found (run scripts/ingest.py first)", file=sys.stderr)
        return 1
    chunks = read_chunks(args.chunks)
    if not chunks:
        print(f"error: {args.chunks} has no chunks", file=sys.stderr)
        return 1

    out = args.out or settings.data_dir / f"dense-{args.model}.npz"
    embedder = load_embedder(args.model, batch_size=args.batch_size)
    cache = EmbeddingCache(args.cache)
    truncated = embedder.n_truncated([chunk.index_text for chunk in chunks])

    start = time.perf_counter()
    index, computed = build_dense_index(chunks, embedder, cache)
    seconds = time.perf_counter() - start
    cache.save()
    index.save(out)

    print(
        f"embedded {index.n_docs} chunks with {args.model} (dim {index.dim}): "
        f"{computed} computed, {index.n_docs - computed} from cache, {seconds:.1f}s"
    )
    print(f"{truncated} chunks are longer than the {embedder.max_tokens}-token limit (truncated)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
