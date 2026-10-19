"""Compare chunking strategies by how often they keep a gold answer inside ONE chunk.

For every strategy: chunk the whole corpus, then report how many chunks it made, how big they are,
how many exceed the embedding model's 512-token limit, and the gold-integrity rate (the share of
gold
spans that fit completely inside a single chunk) overall and per question category.

Usage:
    python scripts/compare_chunkers.py
    python scripts/compare_chunkers.py --split dev
"""

import argparse
import sys
from pathlib import Path
from statistics import fmean

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.integrity import broken_spans, gold_integrity
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.chunkers import FixedChunker, MarkdownChunker, PythonChunker, RecursiveChunker
from cairn.ingest.documents import load_documents
from cairn.types import CATEGORIES

TOKEN_LIMIT = 512


def strategies() -> dict[str, dict]:
    """Every strategy as {kind: chunker}, where kind is "prose" (Markdown) or "code" (Python)."""
    return {
        "fixed-100": {"prose": FixedChunker(100), "code": FixedChunker(100)},
        "fixed-200": {"prose": FixedChunker(200), "code": FixedChunker(200)},
        "fixed-400": {"prose": FixedChunker(400), "code": FixedChunker(400)},
        "recursive-200": {"prose": RecursiveChunker(200), "code": RecursiveChunker(200)},
        "recursive-400": {"prose": RecursiveChunker(400), "code": RecursiveChunker(400)},
        "recursive-200+overlap3": {
            "prose": RecursiveChunker(200, overlap_lines=3),
            "code": RecursiveChunker(200, overlap_lines=3),
        },
        "structure-150": {"prose": MarkdownChunker(150), "code": PythonChunker()},
        "structure-300": {"prose": MarkdownChunker(300), "code": PythonChunker()},
        "structure-600": {"prose": MarkdownChunker(600), "code": PythonChunker()},
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    default_corpus = load_settings().data_dir / "raw" / "fastapi"
    parser = argparse.ArgumentParser(description="Compare chunking strategies.")
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--show-broken",
        choices=list(strategies()),
        metavar="STRATEGY",
        help="also list the gold spans this strategy cuts across several chunks",
    )
    args = parser.parse_args(argv)

    if args.split == "test":
        parser.error("choose chunkers on the dev questions only: the test split stays frozen")
    try:
        questions = load_questions(args.questions, split=args.split)
    except QuestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    documents = load_documents(args.corpus)
    if not documents:
        print(f"error: no documents found under {args.corpus}", file=sys.stderr)
        return 1

    categories = [c for c in CATEGORIES if any(q.category == c for q in questions)]
    header = (
        f"{'strategy':<24}{'chunks':>8}{'mean tok':>10}{'>' + str(TOKEN_LIMIT):>6}{'intact':>8}"
    )
    header += "".join(f"{c:>13}" for c in categories)
    print(f"{len(documents)} documents, {len(questions)} {args.split} questions\n")
    print(header)
    for name, chunkers in strategies().items():
        chunks = [chunk for doc in documents for chunk in chunkers[doc.kind].split(doc)]
        over = sum(c.token_count > TOKEN_LIMIT for c in chunks)
        result = gold_integrity(chunks, questions)
        row = f"{name:<24}{len(chunks):>8}{fmean(c.token_count for c in chunks):>10.1f}{over:>6}"
        row += f"{result['overall']['span_rate']:>8.3f}"
        row += "".join(f"{result['by_category'][c]['span_rate']:>13.3f}" for c in categories)
        print(row)
    print("\nintact = share of gold spans that fit completely inside one chunk (higher is better)")

    if args.show_broken:
        chunkers = strategies()[args.show_broken]
        chunks = [chunk for doc in documents for chunk in chunkers[doc.kind].split(doc)]
        broken = broken_spans(chunks, questions)
        print(f"\nspans that {args.show_broken} cuts across chunks ({len(broken)}):")
        for question, span in broken:
            where = f"{span.path}:{span.start_line}-{span.end_line}"
            print(f"  {question.id}  {question.category:<12}{where}")
        if not broken:
            print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
