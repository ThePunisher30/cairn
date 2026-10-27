"""Two ablations of the HNSW index: the neighbour-selection heuristic and the upper layers.

    --experiment heuristic   build the graph with the simple "keep the M closest" choice and
                             compare it with the heuristic graph, ef by ef
    --experiment layers      search the same graph twice: with the upper layers, and starting the
                             layer-0 beam at the entry node (no upper layers)

Recall and distance computations are the numbers to compare; queries per second are too noisy.
The heuristic graph is the same file bench/ann.py saves, so it is not built again if it is there.

Usage:
    python bench/ann_ablate.py --experiment heuristic --scale data/scale-100000.npz
    python bench/ann_ablate.py --experiment layers --scale data/scale-100000.npz
    python bench/ann_ablate.py --experiment layers --real
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from ann import build_or_load, load_real, load_scale, mb, measure, seconds_text

from cairn.ann.hnsw import HNSWIndex
from cairn.config import PROJECT_ROOT, load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS


class WithoutUpperLayers:
    """The same graph, searched from the entry node straight on layer 0."""

    def __init__(self, index):
        self.index = index
        self.last_n_distances = 0

    def search(self, query, k, ef=None):
        index = self.index
        ef = max(index.ef_search if ef is None else ef, k)
        index.last_n_distances = 0
        query = np.asarray(query, dtype=np.float32)
        found = index._search_layer(query, [index.entry_point], ef, 0)
        self.last_n_distances = index.last_n_distances
        return [(int(row), float(score)) for score, row in found[:k]]


def get_hnsw(args, vectors, ids, key, *, heuristic):
    """Load the saved graph, or build and save it. The name matches bench/ann.py's."""
    name = f"hnsw-{key}-M{args.M}-efc{args.ef_construction}-seed{args.seed}"
    path = args.index_dir / (name + ("" if heuristic else "-simple") + ".npz")
    args.index_dir.mkdir(parents=True, exist_ok=True)

    def build():
        return HNSWIndex(
            vectors,
            ids,
            M=args.M,
            ef_construction=args.ef_construction,
            seed=args.seed,
            heuristic=heuristic,
        )

    return build_or_load(HNSWIndex, path, build)


def describe(label, index, seconds):
    print(
        f"{label}: {seconds_text(seconds)}, {index.n_links} links, "
        f"extra memory {mb(index.extra_nbytes):.2f} MB, top layer {index.max_level}"
    )


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Ablations of the HNSW index.")
    parser.add_argument("--experiment", choices=["heuristic", "layers"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scale", type=Path, help="a file from scripts/make_scale_set.py")
    source.add_argument("--real", action="store_true", help="the real vectors and dev questions")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument(
        "--dense-index", type=Path, default=settings.data_dir / "dense-bge-small.npz"
    )
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=None, help="use only the first N queries")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-searches", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-dir", type=Path, default=settings.data_dir / "ann")
    args = parser.parse_args(argv)

    if (
        args.k < 1
        or (args.n_queries is not None and args.n_queries < 1)
        or args.M < 2
        or args.ef_construction < 1
        or any(e < 1 for e in args.ef_searches)
    ):
        print(
            "error: --k, --n-queries, --ef-construction and --ef-searches must be at least 1, "
            "--M at least 2",
            file=sys.stderr,
        )
        return 1
    try:
        vectors, queries = load_real(args, settings) if args.real else load_scale(args.scale)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.n_queries is not None:
        queries = queries[: args.n_queries]
    n, dim = vectors.shape
    ids = [str(i) for i in range(n)]
    key = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()[:10]

    print(f"{n} vectors of {dim} numbers, {len(queries)} queries, k = {args.k}")
    flat = FlatIndex(vectors, ids)
    truth = [{row for row, _ in flat.search(q, args.k)} for q in queries]

    if args.experiment == "heuristic":
        on, on_seconds = get_hnsw(args, vectors, ids, key, heuristic=True)
        off, off_seconds = get_hnsw(args, vectors, ids, key, heuristic=False)
        print()
        describe("heuristic on ", on, on_seconds)
        describe("heuristic off", off, off_seconds)
        print("\n        heuristic on           heuristic off")
        print("  ef    recall@k  distances    recall@k  distances")
        for ef in args.ef_searches:
            recall_on, _, dists_on = measure(on, queries, truth, args.k, ef=ef)
            recall_off, _, dists_off = measure(off, queries, truth, args.k, ef=ef)
            print(
                f"{ef:5d}  {recall_on:8.3f}  {dists_on:9.0f}    {recall_off:8.3f}  {dists_off:9.0f}"
            )
    else:
        index, seconds = get_hnsw(args, vectors, ids, key, heuristic=True)
        print()
        describe("HNSW", index, seconds)
        sizes = [int((index.levels >= layer).sum()) for layer in range(index.max_level + 1)]
        print(f"nodes on layers 0, 1, 2, ...: {sizes}")
        flat_walk = WithoutUpperLayers(index)
        print("\n        with the layers        without the upper layers")
        print("  ef    recall@k  distances    recall@k  distances")
        for ef in args.ef_searches:
            recall_with, _, dists_with = measure(index, queries, truth, args.k, ef=ef)
            recall_without, _, dists_without = measure(flat_walk, queries, truth, args.k, ef=ef)
            print(
                f"{ef:5d}  {recall_with:8.3f}  {dists_with:9.0f}    "
                f"{recall_without:8.3f}  {dists_without:9.0f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
