"""Which retrieval score tells an answerable question from an unanswerable one? (Phase 7, no LLM)

    python scripts/abstain_scores.py
    python scripts/abstain_scores.py --rerank --run data/gen_runs/gemma4-hybrid.jsonl

For each of the 50 dev questions the top score of each retriever is read (no model is asked). A
score is useful for abstaining when answerable questions score high and unanswerable ones low.
Two numbers per score:
  AUC          the chance that a random answerable question scores above a random unanswerable one
               (1.0 = perfect, 0.5 = no better than a coin). It needs no threshold.
  best f1      the best abstain F1 over every threshold (`calibrate_threshold`). It is measured on
               the same 50 questions that picked the threshold, so it is optimistic.
With --run, the model's own abstentions from a gen_eval run file are summarised as well.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.generation import load_gen_questions
from cairn.generation.abstain import abstain_metrics, calibrate_threshold, top_score
from cairn.generation.factory import build_retrievers


def auc(answerable, unanswerable):
    """P(an answerable score is higher than an unanswerable one); ties count one half."""
    wins = 0.0
    for a in answerable:
        for u in unanswerable:
            wins += 1.0 if a > u else 0.5 if a == u else 0.0
    return wins / (len(answerable) * len(unanswerable))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Compare scores for abstaining (no LLM).")
    parser.add_argument("--rerank", action="store_true", help="also the cross-encoder score")
    parser.add_argument("--run", type=Path, default=None, help="a gen_eval run file")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument(
        "--dense-index", type=Path, default=settings.data_dir / "dense-bge-small.npz"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the scores as JSON")
    args = parser.parse_args(argv)

    questions = load_gen_questions(
        config.PROJECT_ROOT / "eval" / "gen_questions.jsonl",
        config.PROJECT_ROOT / "eval" / "questions.jsonl",
        split="dev",
    )
    names = ["bm25", "dense", "hybrid"] + (["rerank"] if args.rerank else [])
    scores = {}
    for name in names:
        retriever, _ = build_retrievers(
            name,
            chunks_path=args.chunks,
            bm25_path=args.bm25_index,
            dense_path=args.dense_index,
        )
        scores[name] = [top_score(retriever.search(q.text, 5)) for q in questions]
    should = [not q.answerable for q in questions]

    print(
        f"{len(questions)} questions: {should.count(False)} answerable, {should.count(True)} not\n"
    )
    print(f"{'score':8} {'AUC':>6} {'best f1':>8} {'threshold':>10}", end="")
    print("   answerable (min/median)   unanswerable (median/max)")
    for name in names:
        ok = [s for s, u in zip(scores[name], should, strict=True) if not u]
        no = [s for s, u in zip(scores[name], should, strict=True) if u]
        threshold, m = calibrate_threshold(ok, no)
        print(
            f"{name:8} {auc(ok, no):6.3f} {m['f1']:8.3f} {threshold:10.4f}   "
            f"{min(ok):8.4f} / {statistics.median(ok):8.4f}      "
            f"{statistics.median(no):8.4f} / {max(no):8.4f}"
        )
    print("\n(best f1 is in-sample: only 10 unanswerable questions picked the threshold)")
    # F1 at the lowest threshold that keeps every answerable question: what 'never refuse a
    # question that has an answer' would cost
    for name in names:
        ok = [s for s, u in zip(scores[name], should, strict=True) if not u]
        pred = [s < min(ok) for s in scores[name]]
        m = abstain_metrics(pred, should)
        print(f"  {name:8} threshold {min(ok):.4f} (no false abstention): recall {m['recall']:.2f}")

    if args.out:
        rows = {q.id: {n: scores[n][i] for n in names} for i, q in enumerate(questions)}
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.run:
        rows = [json.loads(line) for line in args.run.read_text("utf-8").splitlines() if line]
        print(f"\nthe model's own abstentions in {args.run.name}:")
        reasons = Counter(r["abstain_reason"] for r in rows if r["abstained"])
        print("  reasons:", dict(reasons) or "none")
        false = [r["id"] for r in rows if r["abstained"] and r["answerable"]]
        print("  refused although answerable:", false or "none")
        missed = [r["id"] for r in rows if not r["abstained"] and not r["answerable"]]
        print("  answered although unanswerable:", missed or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
