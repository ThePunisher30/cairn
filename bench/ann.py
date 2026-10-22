"""Recall against speed for the flat, IVF and HNSW indexes.

For every query the exact answer comes from the flat index. An approximate index is scored by
recall@k (how many of the exact top k it also found), by queries per second (one query at a time,
one thread) and by the mean number of distance computations per query (a count that does not
depend on the machine).

Two kinds of data:
    --scale PATH   a file made by scripts/make_scale_set.py (vectors and queries)
    --real         the real chunk vectors, and the dev questions embedded with the model

Built indexes are saved under --index-dir and loaded next time (HNSW takes minutes to build).

Usage:
    python bench/ann.py --scale data/scale-100000.npz --plot data/ann.png
    python bench/ann.py --real
    python bench/ann.py --scale data/scale-100000.npz --skip hnsw --nprobes 1 4 16 64
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from cairn.ann.hnsw import HNSWIndex
from cairn.ann.ivf import IVFIndex
from cairn.config import PROJECT_ROOT, load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS, load_embedder
from cairn.evaluation.questions import QuestionError, load_questions


def load_scale(path):
    with np.load(path, allow_pickle=False) as data:
        if "vectors" not in data or "queries" not in data:
            raise ValueError(f"{path} is not a scale set (needs 'vectors' and 'queries')")
        return data["vectors"], data["queries"]


def load_real(args, settings):
    if not args.dense_index.is_file():
        raise ValueError(f"{args.dense_index} not found (run scripts/embed_chunks.py)")
    vectors = FlatIndex.load(args.dense_index).vectors
    try:
        questions = load_questions(args.questions, split="dev")
    except (OSError, QuestionError) as e:
        raise ValueError(str(e)) from e
    embedder = load_embedder(args.model)
    return vectors, np.vstack([embedder.embed_query(q.text) for q in questions])


def measure(index, queries, truth, k, **kwargs):
    """recall@k, queries per second and mean distance computations for one setting."""
    found, distances = [], 0
    start = time.perf_counter()
    for query in queries:
        found.append({row for row, _ in index.search(query, k, **kwargs)})
        distances += index.last_n_distances
    seconds = time.perf_counter() - start
    recall = np.mean([len(want & got) / len(want) for want, got in zip(truth, found, strict=True)])
    return float(recall), len(queries) / seconds, distances / len(queries)


def build_or_load(cls, path, build):
    """Load a saved index, or build and save it. Returns (index, build seconds or None)."""
    side = path.with_suffix(".json")
    if path.is_file():
        try:
            seconds = json.loads(side.read_text())["build_seconds"]
        except (OSError, ValueError, KeyError):
            seconds = None
        return cls.load(path), seconds
    start = time.perf_counter()
    index = build()
    seconds = time.perf_counter() - start
    index.save(path)
    side.write_text(json.dumps({"build_seconds": seconds}))
    return index, seconds


def mb(n_bytes):
    return n_bytes / 1e6


def seconds_text(seconds):
    return "built earlier" if seconds is None else f"built in {seconds:.1f}s"


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Recall against speed for ANN indexes.")
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
    parser.add_argument("--nlist", type=int, default=None, help="default: round(sqrt(n))")
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-searches", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", choices=["ivf", "hnsw"], action="append", default=[])
    parser.add_argument("--index-dir", type=Path, default=settings.data_dir / "ann")
    parser.add_argument("--plot", type=Path, default=None, help="write a recall/speed picture")
    args = parser.parse_args(argv)

    if args.k < 1 or (args.n_queries is not None and args.n_queries < 1):
        print("error: --k and --n-queries must be at least 1", file=sys.stderr)
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
    nlist = args.nlist or max(1, round(n**0.5))
    if (
        not 1 <= nlist <= n
        or any(p < 1 for p in args.nprobes)
        or any(e < 1 for e in args.ef_searches)
    ):
        print(
            "error: --nlist must be between 1 and n; --nprobes and --ef-searches at least 1",
            file=sys.stderr,
        )
        return 1
    key = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()[:10]

    print(f"{n} vectors of {dim} numbers, {len(queries)} queries, k = {args.k}")
    flat = FlatIndex(vectors, ids)
    truth = [{row for row, _ in flat.search(q, args.k)} for q in queries]
    start = time.perf_counter()
    for q in queries:
        flat.search(q, args.k)
    flat_qps = len(queries) / (time.perf_counter() - start)
    print(
        f"\nflat (exact): recall 1.000, {flat_qps:.0f} queries/s, {n} distances per query, "
        f"vectors {mb(flat.nbytes):.1f} MB"
    )
    curves = {"flat": [(1.0, flat_qps)]}

    if "ivf" not in args.skip:
        path = args.index_dir / f"ivf-{key}-nlist{nlist}-seed{args.seed}.npz"
        args.index_dir.mkdir(parents=True, exist_ok=True)
        ivf, secs = build_or_load(
            IVFIndex, path, lambda: IVFIndex(vectors, ids, nlist, seed=args.seed)
        )
        print(
            f"\nIVF nlist={nlist}, {seconds_text(secs)}, extra memory {mb(ivf.extra_nbytes):.2f} MB"
        )
        print("  nprobe  recall@k  queries/s  distances")
        curves["ivf"] = []
        for nprobe in args.nprobes:
            recall, qps, dists = measure(ivf, queries, truth, args.k, nprobe=nprobe)
            print(f"  {nprobe:6d}  {recall:8.3f}  {qps:9.0f}  {dists:9.0f}")
            curves["ivf"].append((recall, qps))

    if "hnsw" not in args.skip:
        name = f"hnsw-{key}-M{args.M}-efc{args.ef_construction}-seed{args.seed}.npz"
        path = args.index_dir / name
        args.index_dir.mkdir(parents=True, exist_ok=True)
        hnsw, secs = build_or_load(
            HNSWIndex,
            path,
            lambda: HNSWIndex(
                vectors, ids, M=args.M, ef_construction=args.ef_construction, seed=args.seed
            ),
        )
        print(
            f"\nHNSW M={args.M} efConstruction={args.ef_construction}, {seconds_text(secs)}, "
            f"extra memory {mb(hnsw.extra_nbytes):.2f} MB, top layer {hnsw.max_level}"
        )
        print("  ef      recall@k  queries/s  distances")
        curves["hnsw"] = []
        for ef in args.ef_searches:
            recall, qps, dists = measure(hnsw, queries, truth, args.k, ef=ef)
            print(f"  {ef:6d}  {recall:8.3f}  {qps:9.0f}  {dists:9.0f}")
            curves["hnsw"].append((recall, qps))

    if args.plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("error: --plot needs matplotlib (pip install matplotlib)", file=sys.stderr)
            return 1
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, points in curves.items():
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel(f"recall@{args.k}")
        ax.set_ylabel("queries per second")
        ax.set_yscale("log")
        ax.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
