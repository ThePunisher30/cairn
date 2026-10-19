"""BM25 experiments on the dev questions: tokenizer variants x chunkers, and a k1 x b grid.

Usage:
    python scripts/ablate_bm25.py                   # every chunker x every tokenizer variant
    python scripts/ablate_bm25.py --by-category     # the same, with MRR per question category
    python scripts/ablate_bm25.py --tune structure-300 --tokenizer split
                                                    # a k1 x b grid for one configuration

Only the dev questions are used: the test split stays frozen. With 40 questions, one question is
worth 0.025, so treat differences of a few hundredths as noise.
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.harness import run_retriever, summarize
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.chunkers import FixedChunker, MarkdownChunker, PythonChunker, RecursiveChunker
from cairn.ingest.documents import load_documents
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import ENGLISH_STOPWORDS, Tokenizer
from cairn.types import CATEGORIES

K = 10
K1_VALUES = (0.0, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0)
B_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("mrr", "ndcg@10", "hit@10", "hit@1", "recall@10")


def chunkers() -> dict[str, dict]:
    return {
        "fixed-200": {"prose": FixedChunker(200), "code": FixedChunker(200)},
        "fixed-400": {"prose": FixedChunker(400), "code": FixedChunker(400)},
        "recursive-400": {"prose": RecursiveChunker(400), "code": RecursiveChunker(400)},
        "structure-300": {"prose": MarkdownChunker(300), "code": PythonChunker()},
    }


def tokenizers() -> dict[str, Tokenizer]:
    stop = ENGLISH_STOPWORDS
    return {
        "plain": Tokenizer(split_identifiers=False),
        "split": Tokenizer(split_identifiers=True),
        "plain+stop": Tokenizer(split_identifiers=False, stopwords=stop),
        "split+stop": Tokenizer(split_identifiers=True, stopwords=stop),
    }


def chunk_corpus(documents, strategy):
    chunks = [chunk for doc in documents for chunk in strategy[doc.kind].split(doc)]
    return sorted(chunks, key=lambda c: (c.path, c.start_line))


def evaluate(index, chunks, questions, k1=1.2, b=0.75):
    retriever = BM25Retriever(index, chunks, k1=k1, b=b)
    return summarize(run_retriever(retriever, questions, k=K))


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    default_corpus = load_settings().data_dir / "raw" / "fastapi"
    parser = argparse.ArgumentParser(description="BM25 ablations on the dev questions.")
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--by-category", action="store_true")
    parser.add_argument("--tune", choices=list(chunkers()), metavar="CHUNKER")
    parser.add_argument("--tokenizer", choices=list(tokenizers()), default="split")
    parser.add_argument("--metric", choices=METRICS, default="mrr")
    args = parser.parse_args(argv)

    if args.split == "test":
        parser.error("tune on the dev questions only: the test split stays frozen")
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
    print(f"{len(documents)} documents, {len(questions)} {args.split} questions, k = {K}\n")

    if args.tune:
        chunks = chunk_corpus(documents, chunkers()[args.tune])
        # k1 and b only matter when searching, so one index serves the whole grid
        index = InvertedIndex.build(chunks, tokenizers()[args.tokenizer])
        print(f"{args.metric} for {args.tune} + {args.tokenizer} ({len(chunks)} chunks)")
        print(f"{'':<8}" + "".join(f"{'b=' + format(b, 'g'):>9}" for b in B_VALUES))
        for k1 in K1_VALUES:
            row = f"{'k1=' + format(k1, 'g'):<8}"
            for b in B_VALUES:
                summary = evaluate(index, chunks, questions, k1=k1, b=b)
                row += f"{summary['overall'][args.metric]:>9.3f}"
            print(row)
        return 0

    header = f"{'chunker':<16}{'tokenizer':<12}{'terms':>8}{'hit@1':>8}{'hit@10':>8}"
    header += f"{'mrr':>8}{'ndcg@10':>9}"
    if args.by_category:
        header += "".join(f"{'mrr ' + c:>18}" for c in categories)
    print(header)
    for chunker_name, strategy in chunkers().items():
        chunks = chunk_corpus(documents, strategy)
        for tokenizer_name, tokenizer in tokenizers().items():
            index = InvertedIndex.build(chunks, tokenizer)
            summary = evaluate(index, chunks, questions, k1=args.k1, b=args.b)
            overall = summary["overall"]
            row = f"{chunker_name:<16}{tokenizer_name:<12}{len(index.postings):>8}"
            row += f"{overall['hit@1']:>8.3f}{overall['hit@10']:>8.3f}"
            row += f"{overall['mrr']:>8.3f}{overall['ndcg@10']:>9.3f}"
            if args.by_category:
                row += "".join(f"{summary['by_category'][c]['mrr']:>18.3f}" for c in categories)
            print(row)
    print(f"\nk1 = {args.k1}, b = {args.b}; higher is better for every column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
