"""BM25 and dense retrieval side by side, question by question: the failure-analysis worksheet.

For every dev question it prints the rank at which each retriever first returns a chunk that
overlaps the gold answer ("-" = never among the chunks it can return). Then, for the questions
where either retriever is worse than `--fail-rank`, it prints the question, the gold span and the
top result of each retriever, which is what you classify in eval/failures/phase4.md.

Usage:
    python scripts/compare_retrievers.py
    python scripts/compare_retrievers.py --model minilm --fail-rank 3
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS, load_embedder
from cairn.dense.retriever import DenseRetriever
from cairn.evaluation.metrics import overlaps
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.pipeline import read_chunks
from cairn.jsonl import JsonlError
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex


def first_rank(retriever, question, depth):
    """Rank (1 = best) of the first returned chunk that overlaps a gold span, or None."""
    for rank, hit in enumerate(retriever.search(question.text, depth), start=1):
        if any(overlaps(hit, gold) for gold in question.gold):
            return rank
    return None


def where(hit):
    return f"{hit.path}:{hit.start_line}-{hit.end_line}" if hit else "(nothing)"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    parser = argparse.ArgumentParser(description="Compare BM25 and dense retrieval per question.")
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--fail-rank", type=int, default=1, help="list questions ranked below this")
    args = parser.parse_args(argv)

    dense_path = args.dense_index or settings.data_dir / f"dense-{args.model}.npz"
    for path, script in (
        (args.chunks, "ingest"),
        (args.bm25_index, "build_index"),
        (dense_path, "embed_chunks"),
    ):
        if not path.is_file():
            print(f"error: {path} not found (run scripts/{script}.py)", file=sys.stderr)
            return 1
    try:
        questions = load_questions(args.questions, split="dev")  # dev only, never the test split
        chunks = read_chunks(args.chunks)
        bm25 = BM25Retriever(InvertedIndex.load(args.bm25_index), chunks)
        dense = DenseRetriever(load_embedder(args.model), FlatIndex.load(dense_path), chunks)
    except (QuestionError, JsonlError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    depth = len(chunks)
    rows = []
    for question in questions:
        rows.append(
            (question, first_rank(bm25, question, depth), first_rank(dense, question, depth))
        )

    def cell(rank):
        return "-" if rank is None else str(rank)

    print(f"{len(questions)} dev questions, {len(chunks)} chunks, dense model {args.model}\n")
    print(f"{'id':<6}{'category':<13}{'bm25':>6}{'dense':>7}  question")
    for question, b, d in rows:
        print(
            f"{question.id:<6}{question.category:<13}{cell(b):>6}{cell(d):>7}  {question.text[:70]}"
        )

    both = sum(1 for _, b, d in rows if b == 1 and d == 1)
    only_bm25 = sum(1 for _, b, d in rows if b == 1 and d != 1)
    only_dense = sum(1 for _, b, d in rows if d == 1 and b != 1)
    neither = sum(1 for _, b, d in rows if b != 1 and d != 1)
    print(
        f"\nrank 1: both {both}, only BM25 {only_bm25}, only dense {only_dense}, neither {neither}"
    )

    def bad(rank):
        return rank is None or rank > args.fail_rank

    failing = [(q, b, d) for q, b, d in rows if bad(b) or bad(d)]
    print(f"\n{len(failing)} questions where a retriever ranks the answer below {args.fail_rank}:")
    for question, b, d in failing:
        print(f"\n{question.id} [{question.category}] {question.text}")
        for gold in question.gold:
            print(f"  gold:  {gold.path}:{gold.start_line}-{gold.end_line}")
        top_b, top_d = bm25.search(question.text, 1), dense.search(question.text, 1)
        print(f"  bm25  (rank {cell(b)}): top = {where(top_b[0] if top_b else None)}")
        print(f"  dense (rank {cell(d)}): top = {where(top_d[0] if top_d else None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
