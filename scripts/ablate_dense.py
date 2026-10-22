"""Dense-retrieval experiments on the dev questions: models, query prefix on/off, chunkers.

Usage:
    python scripts/ablate_dense.py                          # structure-300, bge-small and minilm
    python scripts/ablate_dense.py --models bge-small --by-category
    python scripts/ablate_dense.py --models bge-small --chunkers all    # every chunker (slow once)

Every vector is stored in the embedding cache (`--cache`, default data/embedding_cache.npz), so a
second run, or another script that embeds the same chunks, costs nothing. Only the dev questions
are used: the test split stays frozen. With 40 questions, one question is worth 0.025.
"""

import argparse
import sys
from pathlib import Path

from ablate_bm25 import chunk_corpus, chunkers  # the sibling script: same four chunkers

from cairn import config
from cairn.config import load_settings
from cairn.dense.cache import EmbeddingCache
from cairn.dense.models import MODELS, load_embedder, load_model
from cairn.dense.retriever import DenseRetriever, build_dense_index
from cairn.evaluation.harness import run_retriever, summarize
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.documents import load_documents
from cairn.types import CATEGORIES

K = 10


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Dense-retrieval ablations on the dev questions.")
    parser.add_argument("--corpus", type=Path, default=settings.data_dir / "raw" / "fastapi")
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=["bge-small", "minilm"]
    )
    parser.add_argument("--chunkers", nargs="+", default=["structure-300"])
    parser.add_argument("--cache", type=Path, default=settings.data_dir / "embedding_cache.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--by-category", action="store_true")
    args = parser.parse_args(argv)

    if args.split == "test":
        parser.error("tune on the dev questions only: the test split stays frozen")
    names = list(chunkers()) if args.chunkers == ["all"] else args.chunkers
    unknown = [name for name in names if name not in chunkers()]
    if unknown:
        parser.error(f"unknown chunker(s) {unknown}; choose from {list(chunkers())} or 'all'")
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
    header = f"{'model':<11}{'chunker':<16}{'q-prefix':<9}{'chunks':>7}{'hit@1':>7}{'hit@10':>8}"
    header += f"{'mrr':>7}{'ndcg@10':>9}"
    if args.by_category:
        header += "".join(f"{'mrr ' + c:>18}" for c in categories)
    print(header, flush=True)

    cache = EmbeddingCache(args.cache)
    loaded = {}
    for chunker_name in names:
        chunks = chunk_corpus(documents, chunkers()[chunker_name])
        for model_key in args.models:
            spec = MODELS[model_key]
            if model_key not in loaded:
                loaded[model_key] = load_model(spec)
            embedder = load_embedder(model_key, args.batch_size, True, model=loaded[model_key])
            print(
                f"  embedding {len(chunks)} chunks with {model_key} ...",
                file=sys.stderr,
                flush=True,
            )
            index, computed = build_dense_index(chunks, embedder, cache)
            cache.save()
            print(f"  ({computed} computed, {len(chunks) - computed} cached)", file=sys.stderr)
            settings_to_try = [("on", True)] if spec.query_prefix else [("none", True)]
            if spec.query_prefix:
                settings_to_try.append(("off", False))
            for label, use_prefix in settings_to_try:
                variant = load_embedder(
                    model_key, args.batch_size, use_prefix, model=loaded[model_key]
                )
                retriever = DenseRetriever(variant, index, chunks)
                summary = summarize(run_retriever(retriever, questions, k=K))
                overall = summary["overall"]
                row = f"{model_key:<11}{chunker_name:<16}{label:<9}{len(chunks):>7}"
                row += f"{overall['hit@1']:>7.3f}{overall['hit@10']:>8.3f}"
                row += f"{overall['mrr']:>7.3f}{overall['ndcg@10']:>9.3f}"
                if args.by_category:
                    row += "".join(f"{summary['by_category'][c]['mrr']:>18.3f}" for c in categories)
                print(row, flush=True)
    print("\nq-prefix: on = trained prefix, off = prefix left out, none = no prefix needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
