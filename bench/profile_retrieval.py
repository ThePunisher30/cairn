"""Where does one retrieval spend its time, and how does it scale with threads? (Phase 9)

Times the query embedding, BM25 and the whole hybrid search on the dev questions, then measures
how many hybrid searches per second 1, 4 and 20 worker threads get through, for several values of
torch's thread count. This is what found the Phase 9 bottleneck (see docs/results.md).

    python bench/profile_retrieval.py
"""

import json
import sys
import threading
import time

import torch

from cairn import config
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import load_embedder
from cairn.ingest.pipeline import read_chunks
from cairn.retrieval.hybrid import HybridRetriever
from cairn.serve.live import LiveBM25, LiveDense, LiveIndex

THREAD_COUNTS = (1, 2, 4, 12)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = config.PROJECT_ROOT / "data"
    flat = FlatIndex.load(data / "dense-bge-small.npz")
    live = LiveIndex.build(read_chunks(data / "chunks.jsonl"), flat.vectors)
    embedder = load_embedder("bge-small")
    path = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    queries = [r["question"] for r in rows if r["split"] == "dev"]
    bm25 = LiveBM25(live)
    hybrid = HybridRetriever(
        [bm25, LiveDense(embedder, live)], depth=50, method="weighted", normalize="minmax"
    )
    for q in queries[:3]:
        hybrid.search(q, 5)  # warm up

    def mean_ms(fn, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            began = time.perf_counter()
            for q in queries:
                fn(q)
            best = min(best, (time.perf_counter() - began) / len(queries))
        return 1000 * best

    def throughput(workers, seconds=6):
        stop = time.monotonic() + seconds
        done = [0] * workers

        def run(i):
            n = 0
            while time.monotonic() < stop:
                hybrid.search(queries[n % len(queries)], 5)
                n += 1
            done[i] = n

        threads = [threading.Thread(target=run, args=(i,)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return sum(done) / seconds

    print(f"{'torch threads':>13} | {'embed ms':>8} {'bm25 ms':>8} {'hybrid ms':>9} | "
          f"{'1 worker/s':>10} {'4 workers/s':>11} {'20 workers/s':>12}")  # fmt: skip
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        print(f"{threads:>13} | {mean_ms(embedder.embed_query):>8.1f} "
              f"{mean_ms(lambda q: bm25.search(q, 50)):>8.1f} "
              f"{mean_ms(lambda q: hybrid.search(q, 5)):>9.1f} | "
              f"{throughput(1):>10.1f} {throughput(4):>11.1f} {throughput(20):>12.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
