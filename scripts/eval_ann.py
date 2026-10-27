"""Do the approximate indexes change the answers that matter? (hit@10 and MRR against the gold)

bench/ann.py measures recall against the flat index. That is not the number the project cares
about: a question counts as answered when a gold passage is in the top 10. This script builds an
IVF and an HNSW index from the real chunk vectors, plugs each into DenseRetriever, and runs the
dev questions, next to the flat index. The last column counts the questions whose top-k list is
different from the flat index's (any chunk, any order).

Only the dev split is used, there is no option for the test split.

Usage:
    python scripts/eval_ann.py
    python scripts/eval_ann.py --nprobes 1 4 16 --ef-searches 10 40 160
"""

import argparse
import sys
from pathlib import Path

from cairn.ann.hnsw import HNSWIndex
from cairn.ann.ivf import IVFIndex
from cairn.config import PROJECT_ROOT, load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS, load_embedder
from cairn.dense.retriever import DenseRetriever
from cairn.evaluation.harness import run_retriever, summarize
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.pipeline import read_chunks
from cairn.jsonl import JsonlError


def lists_of(results):
    """For every question, the set of passages it got back."""
    return [{(h.path, h.start_line, h.end_line) for h in r.hits} for r in results]


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="hit@k and MRR for flat, IVF and HNSW.")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--no-query-prefix", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=None, help="default: round(sqrt(n))")
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-searches", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dense_index = args.dense_index or settings.data_dir / f"dense-{args.model}.npz"
    for path, hint in (
        (dense_index, "scripts/embed_chunks.py"),
        (args.chunks, "scripts/ingest.py"),
    ):
        if not path.is_file():
            print(f"error: {path} not found (run {hint})", file=sys.stderr)
            return 1
    try:
        questions = load_questions(args.questions, split="dev")
        flat = FlatIndex.load(dense_index)
        chunks = read_chunks(args.chunks)
        embedder = load_embedder(args.model, use_query_prefix=not args.no_query_prefix)
        n = flat.n_docs
        nlist = args.nlist or max(1, round(n**0.5))
        if args.k < 1 or any(p < 1 for p in args.nprobes) or any(e < 1 for e in args.ef_searches):
            raise ValueError("--k, --nprobes and --ef-searches must be at least 1")
        ivf = IVFIndex(flat.vectors, flat.doc_ids, nlist, seed=args.seed)
        hnsw = HNSWIndex(
            flat.vectors,
            flat.doc_ids,
            M=args.M,
            ef_construction=args.ef_construction,
            seed=args.seed,
        )
        flat_run = run_retriever(DenseRetriever(embedder, flat, chunks), questions, args.k)
    except (OSError, ValueError, QuestionError, JsonlError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    reference = lists_of(flat_run)
    k = args.k
    print(f"{n} chunks, {len(questions)} dev questions, k = {k}\n")
    print(f"{'index':26s} {'hit@1':>6s} {f'hit@{k}':>7s} {'mrr':>6s}  different from flat")

    def row(label, results):
        overall = summarize(results, ks=(1, k))["overall"]
        different = sum(a != b for a, b in zip(lists_of(results), reference, strict=True))
        print(
            f"{label:26s} {overall['hit@1']:6.3f} {overall[f'hit@{k}']:7.3f} "
            f"{overall['mrr']:6.3f}  {different} of {len(questions)}"
        )

    row("flat (exact)", flat_run)
    for nprobe in args.nprobes:
        ivf.nprobe = nprobe
        row(
            f"IVF nlist={nlist} nprobe={nprobe}",
            run_retriever(DenseRetriever(embedder, ivf, chunks), questions, k),
        )
    for ef in args.ef_searches:
        hnsw.ef_search = ef
        row(
            f"HNSW M={args.M} ef={ef}",
            run_retriever(DenseRetriever(embedder, hnsw, chunks), questions, k),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
