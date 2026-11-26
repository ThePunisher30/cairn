"""Shared helpers for the framework comparison (Phase 11)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
CORPUS = ROOT / "data" / "raw" / "fastapi"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
K = 10


def rss_mb():
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def peak_memory_mb():
    """The highest working set the process has had (Windows), or the peak resident size."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return peak / 2**20


class Stopwatch:
    def __enter__(self):
        self.began = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.began


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def say(*parts):
    print(*parts, file=sys.stderr, flush=True)
