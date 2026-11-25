"""What does a tombstone cost? (Phase 9)

Loads the real index, deletes files until a chosen fraction of the slots are tombstones (with
compaction switched off), and times BM25 and dense search at each level. Then compacts and times
again. The queries are the dev questions; the dense query vectors are random unit vectors (a dense
search costs the same whatever the vector, it is one matrix-vector product).

    python bench/tombstone_cost.py
"""

import json
import random
import sys
import time

import numpy as np

from cairn import config
from cairn.dense.flat_index import FlatIndex
from cairn.ingest.pipeline import read_chunks
from cairn.serve.live import LiveBM25, LiveIndex

LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75)


def time_searches(live, queries, vectors, repeats=5):
    bm25 = LiveBM25(live)
    best = {"bm25": [], "dense": []}
    for _ in range(repeats):
        began = time.perf_counter()
        for q in queries:
            bm25.search(q, 10)
        best["bm25"].append((time.perf_counter() - began) / len(queries))
        began = time.perf_counter()
        with live.lock.read():
            for v in vectors:
                live.dense_search(v, 10)
        best["dense"].append((time.perf_counter() - began) / len(vectors))
    return {k: 1000 * min(v) for k, v in best.items()}  # the fastest repeat: least noise


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = config.PROJECT_ROOT / "data"
    chunks = read_chunks(data / "chunks.jsonl")
    flat = FlatIndex.load(data / "dense-bge-small.npz")
    questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    rows = [json.loads(line) for line in questions.open(encoding="utf-8")]
    queries = [r["question"] for r in rows if r["split"] == "dev"]
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, flat.dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    began = time.perf_counter()
    live = LiveIndex.build(chunks, flat.vectors, compact_threshold=1.0, min_dead_to_compact=10**9)
    built = time.perf_counter() - began
    print(f"built {live.n_alive} chunks, {len(live._postings)} terms in {built:.1f}s")
    paths = sorted({c.path for c in chunks})
    random.Random(0).shuffle(paths)
    cursor = 0
    print(f"{'tombstones':>10} {'alive':>6} {'slots':>6} {'bm25 ms':>9} {'dense ms':>9}")
    base = None
    for level in LEVELS:
        while live.dead_fraction < level and cursor < len(paths):
            live.apply(remove_paths=[paths[cursor]])
            cursor += 1
        t = time_searches(live, queries, vectors)
        base = base or t
        print(f"{live.dead_fraction:>10.0%} {live.n_alive:>6} {live.n_slots:>6} "
              f"{t['bm25']:>9.3f} {t['dense']:>9.3f}")  # fmt: skip
    began = time.perf_counter()
    live.compact()
    took = time.perf_counter() - began
    t = time_searches(live, queries, vectors)
    print(f"after compact ({took * 1000:.0f} ms under the write lock): {live.n_alive} alive, "
          f"{live.n_slots} slots, bm25 {t['bm25']:.3f} ms, dense {t['dense']:.3f} ms")  # fmt: skip
    print(f"(no-tombstone BM25 time for reference: {base['bm25']:.3f} ms)")


if __name__ == "__main__":
    sys.exit(main())
