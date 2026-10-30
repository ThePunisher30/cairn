"""The Phase 6 ablation table: BM25, dense, hybrid, and hybrid + rerank, on one split.

    python scripts/ablate_hybrid.py                      the table, dev split
    python scripts/ablate_hybrid.py --by-category        add MRR per question category
    python scripts/ablate_hybrid.py --sweep              RRF k and fusion weight, dev only
    python scripts/ablate_hybrid.py --expansion          what small-to-big expansion adds, and costs
    python scripts/ablate_hybrid.py --no-rerank          skip the reranker (no model download)

Every row searches the same chunks. A row with "+rerank" re-scores the top `--rerank-depth` chunks
of the retriever above it with a cross-encoder. Each rerank row gets its own empty score cache, so
its latency includes the model time (pass --share-rerank-cache to reuse scores: faster, but the
latency of the later rows is then too small to mean anything).

The test split is frozen. `--split test` needs `--use-test-set`, writes a line to the test log, and
is refused once the log has two lines (the project allows two test runs: after Phase 6 and at the
end). `run_eval.py` writes to the same log. Tuning (`--sweep`) is dev only.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS, load_embedder
from cairn.dense.retriever import DenseRetriever
from cairn.evaluation.harness import run_retriever, summarize
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.pipeline import read_chunks
from cairn.jsonl import JsonlError
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.context import ContextExpander
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.rerank import RerankingRetriever
from cairn.retrieval.rerank_models import RERANKERS, load_model, load_reranker
from cairn.tokens import count_tokens
from cairn.types import CATEGORIES

MAX_TEST_RUNS = 2


def gold_coverage(hits, gold):
    """The share of the gold lines that some hit contains."""
    wanted = {(g.path, line) for g in gold for line in range(g.start_line, g.end_line + 1)}
    got = {(h.path, line) for h in hits for line in range(h.start_line, h.end_line + 1)}
    return len(wanted & got) / len(wanted)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def print_table(rows, k):
    last = f"hit@{k}"
    print(
        f"{'retriever':<30}{'hit@1':>7}{'hit@5':>7}{last:>7}{'mrr':>7}{f'ndcg@{k}':>8}{'p50 ms':>9}"
    )
    for name, summary in rows:
        o = summary["overall"]
        print(
            f"{name:<30}{o['hit@1']:>7.3f}{o['hit@5']:>7.3f}{o[f'hit@{k}']:>7.3f}"
            f"{o['mrr']:>7.3f}{o[f'ndcg@{k}']:>8.3f}{o['latency_p50_ms']:>9.1f}"
        )


def print_by_category(rows):
    present = [c for c in CATEGORIES if c in rows[0][1]["by_category"]]
    print(f"\nMRR by category\n{'retriever':<30}" + "".join(f"{c:>13}" for c in present))
    for name, summary in rows:
        cells = "".join(f"{summary['by_category'][c]['mrr']:>13.3f}" for c in present)
        print(f"{name:<30}{cells}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Ablation table for hybrid retrieval.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--use-test-set", action="store_true")
    parser.add_argument(
        "--test-log", type=Path, default=config.PROJECT_ROOT / "eval" / "test_set_runs.log"
    )
    parser.add_argument(
        "--questions", type=Path, default=config.PROJECT_ROOT / "eval" / "questions.jsonl"
    )
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--reranker", choices=list(RERANKERS), default="minilm-ce")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=50, help="chunks taken from each retriever")
    parser.add_argument("--rerank-depth", type=int, default=30, help="chunks the reranker scores")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--weight", type=float, default=0.5, help="dense weight, BM25 gets 1 - it")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--share-rerank-cache", action="store_true")
    parser.add_argument("--by-category", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--expansion", action="store_true")
    parser.add_argument("--expand-tokens", type=int, nargs="+", default=[400, 800, 1600])
    args = parser.parse_args(argv)

    if args.k < 1 or args.depth < 1 or args.rerank_depth < 1 or not 0 <= args.weight <= 1:
        print(
            "error: --k, --depth and --rerank-depth must be >= 1, --weight in [0, 1]",
            file=sys.stderr,
        )
        return 1
    if args.split == "test" and not args.use_test_set:
        parser.error("the test split is frozen: pass --use-test-set to spend a run on it")
    if args.split == "test" and args.sweep:
        parser.error("tuning is for the dev split only: --sweep cannot be used with --split test")
    if args.split == "test" and args.test_log.is_file():
        used = [line for line in args.test_log.read_text(encoding="utf-8").splitlines() if line]
        if len(used) >= MAX_TEST_RUNS:
            print(
                f"error: the test split has already been used {len(used)} times "
                f"(see {args.test_log})",
                file=sys.stderr,
            )
            return 1

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
        questions = load_questions(args.questions, split=args.split)
        chunks = read_chunks(args.chunks)
        bm25 = BM25Retriever(InvertedIndex.load(args.bm25_index), chunks)
        dense = DenseRetriever(load_embedder(args.model), FlatIndex.load(dense_path), chunks)
    except (QuestionError, JsonlError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.split == "test":
        args.test_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.test_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} retriever=ablate_hybrid\n")

    k = args.k
    print(f"{len(chunks)} chunks, {len(questions)} {args.split} questions, k = {k}\n")

    def hybrid(**options):
        return HybridRetriever([bm25, dense], depth=args.depth, **options)

    rrf = hybrid(method="rrf", rrf_k=args.rrf_k)
    weights = [1 - args.weight, args.weight]

    if args.sweep:
        print("RRF: mrr and hit@5 for each k (the constant in 1 / (k + rank))")
        for rrf_k in (1, 10, 30, 60, 100, 1000):
            s = summarize(run_retriever(hybrid(method="rrf", rrf_k=rrf_k), questions, k), (1, 5, k))
            print(f"  k = {rrf_k:<5}{s['overall']['mrr']:>7.3f}{s['overall']['hit@5']:>7.3f}")
        print("\nweighted min-max fusion: mrr and hit@5 for each dense weight (BM25 gets 1 - it)")
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            retriever = hybrid(method="weighted", weights=[1 - weight, weight])
            s = summarize(run_retriever(retriever, questions, k), (1, 5, k))
            print(f"  w = {weight:<5}{s['overall']['mrr']:>7.3f}{s['overall']['hit@5']:>7.3f}")
        return 0

    rows = [("bm25", bm25), ("dense", dense), (f"hybrid rrf (k={args.rrf_k})", rrf)]
    for normalize in ("minmax", "zscore", "none"):
        label = "raw sum" if normalize == "none" else f"weighted {normalize}"
        rows.append(
            (
                f"hybrid {label} (w={args.weight})",
                hybrid(method="weighted", weights=weights, normalize=normalize),
            )
        )
    reranked = None
    if not args.no_rerank:
        model = load_model(RERANKERS[args.reranker])
        shared = load_reranker(args.reranker, model=model)
        wrapped = [("bm25", bm25), ("dense", dense), ("hybrid rrf", rrf)]
        for name, base in wrapped:
            reranker = (
                shared if args.share_rerank_cache else load_reranker(args.reranker, model=model)
            )
            rows.append(
                (f"{name} + rerank", RerankingRetriever(base, reranker, depth=args.rerank_depth))
            )
        reranked = rows[-1][1]

    results = []
    for name, retriever in rows:
        results.append((name, summarize(run_retriever(retriever, questions, k), (1, 5, k))))
    print_table(results, k)
    if args.by_category:
        print_by_category(results)

    if args.expansion:
        best_name, best = ("hybrid rrf + rerank", reranked) if reranked else ("hybrid rrf", rrf)
        print(f"\nsmall-to-big on '{best_name}', top 5 contexts:")
        print(f"{'':<24}{'hit@5':>7}{'gold covered':>14}{'tokens':>9}")
        base_runs = run_retriever(best, questions, 5)

        def row(label, contexts):
            hit = mean(
                any(
                    h.path == g.path and h.start_line <= g.end_line and g.start_line <= h.end_line
                    for h in hs
                    for g in r.question.gold
                )
                for hs, r in zip(contexts, base_runs, strict=True)
            )
            covered = mean(
                gold_coverage(hs, r.question.gold)
                for hs, r in zip(contexts, base_runs, strict=True)
            )
            tokens = mean(sum(count_tokens(h.text) for h in hs) for hs in contexts)
            print(f"{label:<24}{hit:>7.3f}{covered:>14.3f}{tokens:>9.0f}")

        row("chunks as retrieved", [list(r.hits) for r in base_runs])
        for mode in ("section", "parent"):
            for budget in args.expand_tokens:
                expander = ContextExpander(chunks, max_tokens=budget, mode=mode)
                row(f"{mode}, up to {budget}", [expander.expand(r.hits) for r in base_runs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
