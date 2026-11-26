"""FAISS against Cairn's own indexes on the Phase 5 synthetic set (Phase 11).

Same vectors and queries as `bench/ann.py --scale data/scale-100000.npz`, k = 10, one query at a
time. The exact answer comes from Cairn's flat index. Cairn's HNSW is the one built and saved in
Phase 5 (M 16, efConstruction 100, seed 0), loaded, not rebuilt. FAISS HNSW is built with the same
M and efConstruction. FAISS is measured with one thread and with its default threads.

    python bench/frameworks/faiss_vs_cairn.py
    python bench/frameworks/faiss_vs_cairn.py --out bench/frameworks/results/faiss-vs-cairn.json

The data is synthetic (mixes of real chunk vectors plus noise), see docs/results.md, Phase 5.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common import ROOT, peak_memory_mb, say, write_json  # noqa: E402

from cairn.ann.hnsw import HNSWIndex  # noqa: E402
from cairn.dense.flat_index import FlatIndex  # noqa: E402

K = 10
EF_SEARCHES = (10, 30, 60, 80, 160, 320)


def qps_and_recall(search, queries, truth):
    """Queries per second (one at a time) and recall@K against the exact answer."""
    found = []
    began = time.perf_counter()
    for query in queries:
        found.append(search(query))
    seconds = time.perf_counter() - began
    recall = float(
        np.mean([len(want & got) / len(want) for want, got in zip(truth, found, strict=True)])
    )
    return len(queries) / seconds, recall


def main(argv=None):
    import faiss

    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=Path, default=ROOT / "data" / "scale-100000.npz")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "ann")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument(
        "--repeats", type=int, default=3, help="measurements per setting; best kept"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    with np.load(args.scale, allow_pickle=False) as data:
        vectors, queries = data["vectors"], data["queries"]
    n, dim = vectors.shape
    key = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()[:10]
    saved = args.index_dir / f"hnsw-{key}-M{args.M}-efc{args.ef_construction}-seed0.npz"
    say(
        f"{n} vectors x {dim}, {len(queries)} queries; "
        f"saved Cairn HNSW: {saved.name} ({saved.is_file()})"
    )

    flat = FlatIndex(vectors, [str(i) for i in range(n)])
    truth = [{row for row, _ in flat.search(q, K)} for q in queries]

    def best(search_factory, repeats=args.repeats):
        runs = [qps_and_recall(search_factory, queries, truth) for _ in range(repeats)]
        return max(r[0] for r in runs), runs[0][1]

    report = {"vectors": n, "dim": dim, "queries": len(queries), "k": K, "rows": []}

    def add(name, qps, recall, **extra):
        row = {
            "index": name,
            "queries_per_second": round(qps, 1),
            "recall": round(recall, 4),
            **extra,
        }
        report["rows"].append(row)
        say(f"  {name:38s} recall {recall:.3f}  {qps:9.1f} queries/s  {extra}")

    qps, recall = best(lambda q: {r for r, _ in flat.search(q, K)})
    add("Cairn flat (numpy)", qps, recall)

    for threads in (1, None):
        label = "1 thread" if threads == 1 else f"default threads ({faiss.omp_get_max_threads()})"
        faiss.omp_set_num_threads(threads or faiss.omp_get_max_threads())
        flat_faiss = faiss.IndexFlatIP(dim)
        flat_faiss.add(vectors.astype(np.float32))
        qps, recall = best(
            lambda q, index=flat_faiss: set(index.search(q[None, :], K)[1][0].tolist())
        )
        add(f"FAISS IndexFlatIP, {label}", qps, recall)

    if saved.is_file():
        began = time.perf_counter()
        hnsw = HNSWIndex.load(saved)
        say(f"loaded Cairn HNSW in {time.perf_counter() - began:.1f}s")
        for ef in EF_SEARCHES:
            qps, recall = best(lambda q, ef=ef: {r for r, _ in hnsw.search(q, K, ef=ef)})
            add(f"Cairn HNSW M{args.M} ef{ef}", qps, recall)
    else:
        say("no saved Cairn HNSW for this vector set: its rows are skipped")

    faiss.omp_set_num_threads(1)
    began = time.perf_counter()
    fh = faiss.IndexHNSWFlat(dim, args.M, faiss.METRIC_INNER_PRODUCT)
    fh.hnsw.efConstruction = args.ef_construction
    fh.add(vectors.astype(np.float32))
    build_seconds = time.perf_counter() - began
    report["faiss_hnsw_build_seconds"] = round(build_seconds, 1)
    say(f"FAISS HNSW built in {build_seconds:.1f}s (1 thread)")
    for threads in (1, None):
        label = "1 thread" if threads == 1 else "default threads"
        faiss.omp_set_num_threads(threads or faiss.omp_get_max_threads())
        for ef in EF_SEARCHES:
            fh.hnsw.efSearch = ef
            qps, recall = best(
                lambda q: set(fh.search(q[None, :].astype(np.float32), K)[1][0].tolist())
            )
            add(f"FAISS HNSW M{args.M} ef{ef}, {label}", qps, recall)

    report["peak_memory_mb"] = round(peak_memory_mb(), 1)
    if args.out:
        write_json(args.out, report)
        say(f"wrote {args.out}")
    print(json.dumps({"rows": len(report["rows"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
