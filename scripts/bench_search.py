"""Measure how long one BM25 search takes (the Phase 3 target: p95 under 20 ms on the full corpus).

Every dev question is searched `--repeat` times after one warm-up pass; the time of each single
search is recorded.

Usage:
    python scripts/bench_search.py
    python scripts/bench_search.py --repeat 50 --k1 1.5 --b 0.5
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.metrics import percentile
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.pipeline import read_chunks
from cairn.jsonl import JsonlError
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex
from cairn.timing import Timer

TARGET_P95_MS = 20.0


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    parser = argparse.ArgumentParser(description="Time BM25 searches.")
    parser.add_argument("--index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)

    for path, script in ((args.index, "build_index"), (args.chunks, "ingest")):
        if not path.is_file():
            print(f"error: {path} not found (run scripts/{script}.py)", file=sys.stderr)
            return 1
    try:
        questions = load_questions(args.questions, split="dev")  # dev only, never the test split
        retriever = BM25Retriever(
            InvertedIndex.load(args.index), read_chunks(args.chunks), k1=args.k1, b=args.b
        )
    except (QuestionError, JsonlError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not questions:
        print("error: no dev questions", file=sys.stderr)
        return 1

    for question in questions:  # warm-up: first calls are slower (caches, lazy imports)
        retriever.search(question.text, args.k)
    times = []
    for _ in range(max(args.repeat, 1)):
        for question in questions:
            with Timer() as timer:
                retriever.search(question.text, args.k)
            times.append(timer.seconds * 1000)

    p95 = percentile(times, 95)
    print(f"chunks: {retriever.index.n_docs}, distinct terms: {len(retriever.index.postings)}")
    print(f"searches: {len(times)} ({len(questions)} questions x {max(args.repeat, 1)})")
    print(f"p50: {percentile(times, 50):.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"max: {max(times):.2f} ms")
    verdict = "OK" if p95 < TARGET_P95_MS else "SLOW"
    print(f"target: p95 under {TARGET_P95_MS:.0f} ms -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
