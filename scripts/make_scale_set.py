"""Make a large set of vectors to test approximate indexes on.

The corpus has only 3,267 chunks, where brute force takes under a millisecond and no index can
help. Embedding 100,000 real chunks would take about 5 hours on a CPU, so the large set is made
from the real vectors: each new vector is a mix of two real chunk vectors plus a little random
noise, scaled to length 1. The queries are made the same way from other random draws, so they are
not in the index. This keeps the shape of real embeddings (384 numbers, unit length, the same
kind of neighbourhoods) but it is synthetic: real data will behave differently.

Usage:
    python scripts/make_scale_set.py
    python scripts/make_scale_set.py --n 200000 --queries 2000 --out data/scale-200000.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from cairn.config import load_settings
from cairn.dense.flat_index import FlatIndex


def make_vectors(source, n, rng, noise):
    """n unit vectors, each a mix of two rows of `source` plus noise."""
    dim = source.shape[1]
    first = source[rng.integers(len(source), size=n)]
    second = source[rng.integers(len(source), size=n)]
    weight = rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32)
    jitter = rng.normal(size=(n, dim)).astype(np.float32) * (noise / np.sqrt(dim))
    mixed = weight * first + (1.0 - weight) * second + jitter
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    return (mixed / norms).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Make a large synthetic vector set.")
    parser.add_argument("--source", type=Path, default=settings.data_dir / "dense-bge-small.npz")
    parser.add_argument("--n", type=int, default=100_000, help="how many vectors")
    parser.add_argument("--queries", type=int, default=1000, help="how many query vectors")
    parser.add_argument("--noise", type=float, default=0.3, help="length of the random noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.n < 1 or args.queries < 1 or args.noise < 0:
        print("error: --n and --queries must be at least 1, --noise at least 0", file=sys.stderr)
        return 1
    if not args.source.is_file():
        print(f"error: {args.source} not found (run scripts/embed_chunks.py)", file=sys.stderr)
        return 1
    try:
        source = FlatIndex.load(args.source).vectors
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = args.out or settings.data_dir / f"scale-{args.n}.npz"

    rng = np.random.default_rng(args.seed)
    vectors = make_vectors(source, args.n, rng, args.noise)
    queries = make_vectors(source, args.queries, rng, args.noise)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        np.savez(f, vectors=vectors, queries=queries)
    print(
        f"{args.n} vectors and {args.queries} queries of {vectors.shape[1]} numbers "
        f"from {len(source)} real vectors, noise {args.noise}, seed {args.seed}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
