"""Does the context given to the model contain the gold passage? (Phase 7, no LLM, dev split)

    python scripts/context_recall.py
    python scripts/context_recall.py --retrievers dense hybrid bm25 --ks 3 5 8 --budgets 1500 3000

For every answerable dev question the retriever's top-k hits are packed into the token budget,
exactly as the answering pipeline does, and two numbers are read from the packed chunks:
  hit rate   share of questions where at least one packed chunk overlaps a gold span
  coverage   mean share of a question's gold spans that some packed chunk overlaps
No judge is involved, so the numbers do not move with a prompt. A chunk counts when its line range
overlaps the gold span; a chunk the packer cut short still counts (the cut may have removed the
gold lines, so this is an upper bound). Dev split only.
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.generation import load_gen_questions
from cairn.evaluation.metrics import overlaps
from cairn.generation.factory import RETRIEVERS, build_retrievers
from cairn.generation.packer import ContextPacker


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Gold coverage of the packed context.")
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=["dense", "hybrid"])
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1500, 3000])
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument(
        "--dense-index", type=Path, default=settings.data_dir / "dense-bge-small.npz"
    )
    args = parser.parse_args(argv)

    questions = [
        q
        for q in load_gen_questions(
            config.PROJECT_ROOT / "eval" / "gen_questions.jsonl",
            config.PROJECT_ROOT / "eval" / "questions.jsonl",
            split="dev",
        )
        if q.answerable
    ]
    print(f"{len(questions)} answerable dev questions\n")
    print(f"{'retriever':10} {'k':>2} {'budget':>6} {'hit rate':>9} {'coverage':>9} {'tokens':>7}")
    for name in args.retrievers:
        retriever, _ = build_retrievers(
            name,
            chunks_path=args.chunks,
            bm25_path=args.bm25_index,
            dense_path=args.dense_index,
        )
        ranked = {q.id: retriever.search(q.text, max(args.ks)) for q in questions}
        for budget in args.budgets:
            packer = ContextPacker(budget)
            for k in args.ks:
                hit_rate, coverage, tokens = 0.0, 0.0, 0
                for q in questions:
                    context = packer.pack(ranked[q.id][:k])
                    packed = [c.hit for c in context.chunks]
                    covered = sum(any(overlaps(h, g) for h in packed) for g in q.gold)
                    hit_rate += covered > 0
                    coverage += covered / len(q.gold)
                    tokens += context.tokens
                n = len(questions)
                print(
                    f"{name:10} {k:>2} {budget:>6} {hit_rate / n:9.3f} {coverage / n:9.3f} "
                    f"{tokens / n:7.0f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
