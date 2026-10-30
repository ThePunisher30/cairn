"""Where does each retriever put the gold passage? A reading aid for eval/failures/phase6.md.

    python scripts/list_failures.py                     questions the focus row misses at rank 1
    python scripts/list_failures.py --all               every question
    python scripts/list_failures.py --focus dense       judge another row (bm25, dense, rrf, minmax)
    python scripts/list_failures.py --rerank            add the 'rrf+rerank' row (loads the model)
    python scripts/list_failures.py --markdown          the failures as table rows to fill in

For every dev question it prints the rank of the first hit that overlaps a gold span, in each row
("-" = not in the top --k). It then lists what the focus row put first, so you can read the passage
and write the cause yourself. It does not guess causes. Dev split only: this is for reading, and
the test split must never be read question by question.
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import MODELS, load_embedder
from cairn.dense.retriever import DenseRetriever
from cairn.evaluation.harness import run_retriever
from cairn.evaluation.metrics import overlaps, relevance_flags
from cairn.evaluation.questions import QuestionError, load_questions
from cairn.ingest.pipeline import read_chunks
from cairn.jsonl import JsonlError
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.rerank import RerankingRetriever
from cairn.retrieval.rerank_models import RERANKERS, load_model, load_reranker

ROW_NAMES = ("bm25", "dense", "rrf", "minmax", "rrf+rerank")


def gold_rank(hits, gold):
    """The 1-based rank of the first hit that overlaps a gold span, or None."""
    for position, flag in enumerate(relevance_flags(hits, gold), start=1):
        if flag:
            return position
    return None


def where(hit, gold):
    """How far the hit is from the gold: it overlaps it, is in the same file, or elsewhere."""
    if hit is None:
        return "no hits"
    if any(overlaps(hit, g) for g in gold):
        return "overlaps the gold"
    if any(hit.path == g.path for g in gold):
        return "same file"
    return "other file"


def snippet(hit, width=110):
    text = " ".join(hit.text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def cell(rank):
    return "-" if rank is None else str(rank)


def span(item):
    return f"{item.path}:{item.start_line}-{item.end_line}"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Where each retriever ranks the gold passage.")
    parser.add_argument("--focus", choices=ROW_NAMES, default="minmax")
    parser.add_argument("--all", action="store_true", help="show every question, not just misses")
    parser.add_argument("--markdown", action="store_true", help="print table rows to fill in")
    parser.add_argument("--rerank", action="store_true", help="add the rrf+rerank row")
    parser.add_argument(
        "--questions", type=Path, default=config.PROJECT_ROOT / "eval" / "questions.jsonl"
    )
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--reranker", choices=list(RERANKERS), default="minilm-ce")
    parser.add_argument("--k", type=int, default=30, help="how deep to look for the gold")
    parser.add_argument("--depth", type=int, default=50, help="chunks taken from each retriever")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--weight", type=float, default=0.5, help="dense weight, BM25 gets 1 - it")
    args = parser.parse_args(argv)

    if args.k < 1 or args.depth < 1 or not 0 <= args.weight <= 1:
        print("error: --k and --depth must be >= 1, --weight in [0, 1]", file=sys.stderr)
        return 1
    use_rerank = args.rerank or args.focus == "rrf+rerank"

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
        questions = load_questions(args.questions, split="dev")
        chunks = read_chunks(args.chunks)
        bm25 = BM25Retriever(InvertedIndex.load(args.bm25_index), chunks)
        dense = DenseRetriever(load_embedder(args.model), FlatIndex.load(dense_path), chunks)
    except (QuestionError, JsonlError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    rrf = HybridRetriever([bm25, dense], depth=args.depth, method="rrf", rrf_k=args.rrf_k)
    minmax = HybridRetriever(
        [bm25, dense],
        depth=args.depth,
        method="weighted",
        weights=[1 - args.weight, args.weight],
    )
    rows = [("bm25", bm25), ("dense", dense), ("rrf", rrf), ("minmax", minmax)]
    if use_rerank:
        reranker = load_reranker(args.reranker, model=load_model(RERANKERS[args.reranker]))
        rows.append(("rrf+rerank", RerankingRetriever(rrf, reranker)))
    names = [name for name, _ in rows]

    runs = {name: run_retriever(retriever, questions, args.k) for name, retriever in rows}
    ranks = {
        name: [gold_rank(r.hits, r.question.gold) for r in results]
        for name, results in runs.items()
    }
    focus = args.focus

    def rank_of(name, i):
        return ranks[name][i]

    failures = [i for i in range(len(questions)) if rank_of(focus, i) != 1]
    print(f"{len(questions)} dev questions, rank of the first gold hit (top {args.k})")
    print(f"focus row: {focus}\n")

    header = f"{'id':<6}{'category':<13}" + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, question in enumerate(questions):
        if args.all or i in failures:
            cells = "".join(f"{cell(rank_of(n, i)):>12}" for n in names)
            print(f"{question.id:<6}{question.category:<13}{cells}")

    def ids(indexes):
        return ", ".join(questions[i].id for i in indexes) or "none"

    singles = [n for n in ("bm25", "dense") if n != focus]
    lost = [i for i in failures if any(rank_of(n, i) == 1 for n in singles)]
    shared = [i for i in failures if all(rank_of(n, i) != 1 for n in singles)]
    fixed = [
        i
        for i in range(len(questions))
        if rank_of(focus, i) == 1 and all(rank_of(n, i) != 1 for n in singles)
    ]
    missed = [i for i in failures if rank_of(focus, i) is None]
    either = " or ".join(singles)
    print(
        f"\n{focus} has the gold at rank 1 for {len(questions) - len(failures)} of {len(questions)}"
    )
    print(f"  not in the top {args.k} at all ({len(missed)}): {ids(missed)}")
    print(f"  not rank 1 in {focus}, nor in {either} ({len(shared)}): {ids(shared)}")
    print(f"  lost: rank 1 in {either}, but not in {focus} ({len(lost)}): {ids(lost)}")
    print(f"  gained: rank 1 in {focus}, but not in {either} ({len(fixed)}): {ids(fixed)}")

    print("\n--- what the focus row put first, for each miss ---")
    for i in failures:
        question, result = questions[i], runs[focus][i]
        top = result.hits[0] if result.hits else None
        print(f"\n{question.id} [{question.category}] {focus} rank {cell(rank_of(focus, i))}")
        print(f"  question: {question.text}")
        print(f"  gold:     {'; '.join(span(g) for g in question.gold)}")
        if top is None:
            print("  top hit:  (none)")
        else:
            print(f"  top hit:  {span(top)} ({where(top, question.gold)})")
            print(f"            {snippet(top)}")

    if args.markdown:
        print("\n| Q | Category | " + " | ".join(names) + " | Cause | What the text shows |")
        print("|---|---|" + "---|" * len(names) + "---|---|")
        for i in failures:
            q = questions[i]
            print(
                f"| {q.id} | {q.category} | "
                + " | ".join(cell(rank_of(n, i)) for n in names)
                + " | | |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
