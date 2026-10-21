"""How fast does a model embed chunks on this machine, and how much memory do the vectors take?

Takes a reproducible sample of the chunks (every k-th one), embeds it without the cache for a few
batch sizes, and also in the original order without length sorting, to show what sorting buys.

Usage:
    python scripts/bench_embed.py --model bge-small
    python scripts/bench_embed.py --model minilm --n 100 --batch-sizes 8 16 32
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from cairn.config import load_settings
from cairn.dense.embedder import l2_normalize
from cairn.dense.models import MODELS, load_embedder
from cairn.ingest.pipeline import read_chunks


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Time the embedding of a sample of chunks.")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--n", type=int, default=160, help="how many chunks to embed")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args(argv)

    if args.n < 1 or any(size < 1 for size in args.batch_sizes):
        print("error: --n and --batch-sizes must be at least 1", file=sys.stderr)
        return 1
    if not args.chunks.is_file():
        print(f"error: {args.chunks} not found (run scripts/ingest.py first)", file=sys.stderr)
        return 1
    chunks = read_chunks(args.chunks)
    if not chunks:
        print(f"error: {args.chunks} has no chunks", file=sys.stderr)
        return 1

    step = max(len(chunks) // args.n, 1)
    texts = [chunk.index_text for chunk in chunks[::step][: args.n]]
    embedder = load_embedder(args.model)
    embedder.embed_documents(texts[:4])  # warm-up: the first calls are slower
    lengths = [len(text) for text in texts]

    print(f"model {args.model}: dim {embedder.dim}, reads at most {embedder.max_tokens} tokens")
    print(f"{len(texts)} chunks sampled, mean {np.mean(lengths):.0f} characters")
    print(f"\n{'variant':<22}{'batch':>6}{'chunks/s':>10}{'seconds':>9}")
    best = 0.0

    def report(label, batch_size, seconds):
        nonlocal best
        rate = len(texts) / seconds
        best = max(best, rate)
        print(f"{label:<22}{batch_size:>6}{rate:>10.1f}{seconds:>9.2f}", flush=True)

    unsorted_size = max(args.batch_sizes)
    start = time.perf_counter()
    parts = [
        embedder.model.encode(texts[i : i + unsorted_size])
        for i in range(0, len(texts), unsorted_size)
    ]
    l2_normalize(np.vstack(parts))
    report("original order", unsorted_size, time.perf_counter() - start)
    for batch_size in args.batch_sizes:
        embedder.batch_size = batch_size
        start = time.perf_counter()
        embedder.embed_documents(texts)
        report("sorted by length", batch_size, time.perf_counter() - start)

    total = len(chunks)
    print(f"\nthe whole corpus ({total} chunks) at the best speed: about {total / best:.0f} s")
    megabytes = total * embedder.dim * 4 / 1e6
    per_100k = 100_000 * embedder.dim * 4 / 1e6
    print(f"vectors in memory: {total} x {embedder.dim} x 4 bytes = {megabytes:.1f} MB")
    print(f"memory per 100,000 chunks: {per_100k:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
