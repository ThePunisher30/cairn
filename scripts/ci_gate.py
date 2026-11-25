"""The retrieval-regression gate for CI: score retrieval on the fixture corpus, fail if it drops.

    python scripts/ci_gate.py                 # exit 1 if any metric is below eval/ci_gate.json
    python scripts/ci_gate.py --update        # print the numbers as a new thresholds file

It runs the real ingest (chunkers), the real BM25 and the real fusion over the small corpus in
tests/fixtures/ci_corpus and asks the 40 dev questions. The dense half is the word-hashing model
(no weights, no download), so the run is deterministic, needs no network and costs nothing.
It is a check on the code (chunker, tokenizer, scoring, fusion, index), NOT a measure of how well
the real embedding model does: those numbers are in docs/results.md.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from cairn import config
from cairn.dense.embedder import Embedder
from cairn.dense.models import HashingModel
from cairn.dense.retriever import DenseRetriever, build_dense_index
from cairn.evaluation.harness import run_retriever, summarize
from cairn.evaluation.questions import load_questions
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex

ROOT = config.PROJECT_ROOT
# How far below the measured value a metric may fall. The run is deterministic, so there is no
# noise to allow for: hit@5 may not drop at all (one lost question is 0.025); the mrr gets a little
# slack because harmless ties can reorder a few ranks.
MARGINS = {"hit@5": 0.0, "mrr": 0.01}
K = 10


def measure(corpus, questions_path):
    """{retriever name: {metric: value}} for BM25, hashing-dense and their hybrid."""
    questions = load_questions(questions_path, split="dev")
    with tempfile.TemporaryDirectory() as tmp:
        chunks_path = Path(tmp) / "chunks.jsonl"
        ingest(corpus, chunks_path)
        chunks = read_chunks(chunks_path)
    bm25 = BM25Retriever(InvertedIndex.build(chunks), chunks)
    embedder = Embedder(HashingModel(), name="hashing")
    index, _ = build_dense_index(chunks, embedder)
    dense = DenseRetriever(embedder, index, chunks)
    hybrid = HybridRetriever([bm25, dense], depth=50, method="weighted", normalize="minmax")
    numbers = {}
    for name, retriever in (("bm25", bm25), ("dense_hashing", dense), ("hybrid", hybrid)):
        overall = summarize(run_retriever(retriever, questions, K), ks=(1, 5, K))["overall"]
        numbers[name] = {m: round(overall[m], 4) for m in ("hit@1", "hit@5", "hit@10", "mrr")}
    numbers["_chunks"] = len(chunks)
    numbers["_questions"] = len(questions)
    return numbers


def check(numbers, thresholds):
    """The list of failures: every gated metric that is below its threshold (or missing)."""
    failures = []
    for name, wanted in thresholds.items():
        if name.startswith("_"):
            continue
        for metric, minimum in wanted.items():
            got = numbers.get(name, {}).get(metric)
            if got is None or got < minimum:
                failures.append(f"{name} {metric}: {got} is below the minimum {minimum}")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieval-regression gate for CI.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "tests/fixtures/ci_corpus")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "eval/ci_gate.json")
    parser.add_argument("--update", action="store_true", help="print thresholds, do not check")
    args = parser.parse_args(argv)

    if not args.corpus.is_dir():
        print(f"error: {args.corpus} not found (run scripts/make_ci_fixture.py)", file=sys.stderr)
        return 1
    numbers = measure(args.corpus, args.questions)
    print(f"{numbers['_questions']} dev questions over {numbers['_chunks']} chunks")
    print(f"{'retriever':>14} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'mrr':>7}")
    for name in ("bm25", "dense_hashing", "hybrid"):
        r = numbers[name]
        print(
            f"{name:>14} {r['hit@1']:>7.3f} {r['hit@5']:>7.3f} {r['hit@10']:>7.3f} {r['mrr']:>7.3f}"
        )

    if args.update:
        gate = {
            name: {m: round(numbers[name][m] - margin, 4) for m, margin in MARGINS.items()}
            for name in ("bm25", "hybrid")
        }
        print(json.dumps(gate, indent=2))
        return 0

    if not args.thresholds.is_file():
        print(f"error: {args.thresholds} not found (run with --update)", file=sys.stderr)
        return 1
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    failures = check(numbers, thresholds)
    if failures:
        print("\nRETRIEVAL REGRESSION:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nretrieval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
