"""Generation evaluation (Phase 7): answer every dev question, judge the answers, summarise.

    python scripts/gen_eval.py --llm fake --retriever dense --limit 5      no network, no cost
    python scripts/gen_eval.py --llm anthropic --model claude-haiku-5-5 --name haiku-hybrid
    python scripts/gen_eval.py --retriever dense --threshold 0.62 --confidence dense

Questions come from eval/gen_questions.jsonl joined with eval/questions.jsonl (see
`load_gen_questions`). The DEV split only: there is no --split option, because the one test run that
is left is reserved for Phase 11. One line per question is written to data/gen_runs/<name>.jsonl as
soon as the question is done (the answer, the chunks the model saw, the verdict, the cost), so a
crash does not lose answers you paid for; `judge_agreement.py` reads that file. The cost shown is
that of the answering calls; the judge's cost is shown apart.
Provided, not part of the exercise; it uses your generation and evaluation modules.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.dense.models import MODELS
from cairn.evaluation.generation import (
    GenQuestionError,
    GenResult,
    Judge,
    JudgeParseError,
    load_gen_questions,
    summarize_generation,
)
from cairn.generation.citations import citation_precision_recall
from cairn.generation.errors import LLMError
from cairn.generation.factory import (
    CONFIDENCES,
    LLMS,
    RETRIEVERS,
    build_confidence,
    build_llm,
    build_retrievers,
    load_prices,
)
from cairn.generation.packer import ORDERS, ContextPacker
from cairn.generation.pipeline import Answerer, CostTracker
from cairn.generation.prompts import format_chunks, load_prompt
from cairn.generation.types import Usage
from cairn.retrieval.rerank_models import RERANKERS


def pct(value):
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(s, judge_cost, judge_errors, model):
    print(f"\nquestions: {s['n']} ({s['n_answerable']} answerable, {s['n_unanswerable']} not)")
    print(f"model: {model}")
    print(f"  correctness        {pct(s['correctness'])}   ({s['n_graded']} graded)")
    print(f"  faithfulness       {pct(s['faithfulness'])}")
    print(f"  citation precision {pct(s['citation_precision'])}", end="")
    print(f"   recall {pct(s['citation_recall'])}")
    print(f"  false abstentions  {pct(s['false_abstain_rate'])} of the answerable questions")
    a = s["abstain"]
    print(
        f"  abstain            precision {a['precision']:.3f}  recall {a['recall']:.3f}  "
        f"f1 {a['f1']:.3f}   (tp {a['tp']}, fp {a['fp']}, fn {a['fn']}, tn {a['tn']})"
    )
    cost = s["cost_per_question"] or 0.0
    print(
        f"  cost per question  ${cost:.4f}   "
        f"(total ${s['total_cost_usd']:.4f}; the judge cost ${judge_cost:.4f} more)"
    )
    print(f"  latency p50        {pct(s['latency_p50_ms'])} ms")
    if judge_errors:
        print(f"  judge replies that could not be read: {judge_errors}")


def run_row(q, answer, verdict, correct, faithful, precision, recall):
    """The line written to the run file for one question."""
    return {
        "id": q.id,
        "category": q.category,
        "answerable": q.answerable,
        "question": q.text,
        "reference": q.reference,
        "answer": answer.text,
        "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason,
        "citations": list(answer.citations),
        "invalid_citations": list(answer.invalid_citations),
        "sources": list(answer.sources),
        "correct": correct,
        "faithful": faithful,
        "judge_reason": verdict.reason if verdict else None,
        "citation_precision": precision,
        "citation_recall": recall,
        "cost_usd": answer.cost_usd,
        "latency_ms": answer.latency_ms,
        "usage": [answer.usage.input_tokens, answer.usage.output_tokens],
        "model": answer.model,
        "truncated": answer.truncated,
        "context": [
            {
                "number": c.number,
                "path": c.hit.path,
                "start_line": c.hit.start_line,
                "end_line": c.hit.end_line,
                "score": c.hit.score,
                "text": c.text,
            }
            for c in answer.context.chunks
        ],
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Generation evaluation on the dev questions.")
    parser.add_argument("--name", default=None, help="name of the run file (default: a timestamp)")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "gen_runs")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument(
        "--gen-questions", type=Path, default=config.PROJECT_ROOT / "eval" / "gen_questions.jsonl"
    )
    parser.add_argument(
        "--questions", type=Path, default=config.PROJECT_ROOT / "eval" / "questions.jsonl"
    )
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--llm", choices=LLMS, default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--judge-llm", choices=LLMS, default=None, help="default: same as --llm")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=3000)
    parser.add_argument("--order", choices=ORDERS, default="best_first")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--confidence", choices=CONFIDENCES, default="top-score")
    parser.add_argument("--prompt-version", type=int, default=None)
    parser.add_argument(
        "--prices", type=Path, default=config.PROJECT_ROOT / "eval" / "llm_prices.json"
    )
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--embedding-model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--reranker", choices=list(RERANKERS), default="minilm-ce")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        return 1
    dense_path = args.dense_index or settings.data_dir / f"dense-{args.embedding_model}.npz"
    try:
        questions = load_gen_questions(args.gen_questions, args.questions, split="dev")
        if args.limit:
            questions = questions[: args.limit]
        retriever, dense = build_retrievers(
            args.retriever,
            chunks_path=args.chunks,
            bm25_path=args.bm25_index,
            dense_path=dense_path,
            model=args.embedding_model,
            reranker=args.reranker,
        )
        llm, model = build_llm(args.llm, model=args.model, effort=args.effort)
        judge_kind = args.judge_llm or args.llm
        judge_model = args.judge_model
        if judge_model is None and judge_kind == args.llm:
            judge_model = args.model  # the same model judges, unless another one is named
        judge_llm, judge_model = build_llm(judge_kind, model=judge_model, effort=args.effort)
        prices = load_prices(args.prices) if args.prices.is_file() else None
        if args.llm == "fake":
            prices = {model: (3.0, 15.0)}  # so that the demo run shows a cost
        elif prices is not None and model not in prices:
            raise ValueError(f"no price for model {model!r} in {args.prices} (add it)")
        answerer = Answerer(
            retriever,
            llm,
            packer=ContextPacker(args.max_context_tokens, order=args.order),
            template=load_prompt("answer", args.prompt_version),
            k=args.k,
            max_tokens=args.max_tokens,
            confidence=build_confidence(args.confidence, dense),
            threshold=args.threshold,
            costs=prices,
        )
        judge = Judge(judge_llm, load_prompt("judge"))
    except (GenQuestionError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if prices is None:
        print("note: no prices known for this model, so every cost is shown as 0", file=sys.stderr)
    judge_costs = CostTracker(prices if prices and judge_model in prices else None)

    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    results, judge_errors, failed = [], 0, False
    with open(out, "w", encoding="utf-8") as run_file:
        for number, q in enumerate(questions, start=1):
            try:
                answer = answerer.ask(q.text)
                verdict, correct, faithful, precision, recall = None, None, None, None, None
                if q.answerable and answer.abstained:
                    correct = False
                elif q.answerable:
                    cited = [c.hit for c in answer.context.chunks if c.number in answer.citations]
                    precision, recall = citation_precision_recall(cited, q.gold)
                    try:
                        verdict = judge.grade(
                            q.text, q.reference, answer.text, format_chunks(answer.context)
                        )
                        correct, faithful = verdict.correct, verdict.faithful
                        judge_costs.add(judge_model, judge_llm.last_usage or Usage())
                    except JudgeParseError:
                        judge_errors += 1
            except LLMError as e:
                print(f"error on {q.id}: {e}", file=sys.stderr)
                failed = True
                break
            results.append(
                GenResult(
                    q.id,
                    q.category,
                    q.answerable,
                    answer.abstained,
                    correct,
                    faithful,
                    precision,
                    recall,
                    answer.cost_usd,
                    answer.latency_ms,
                )
            )
            row = run_row(q, answer, verdict, correct, faithful, precision, recall)
            run_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            run_file.flush()
            status = "abstained" if answer.abstained else "answered"
            print(f"[{number}/{len(questions)}] {q.id} {status}", flush=True)

    summary = summarize_generation(results)
    print_summary(summary, judge_costs.total_usd, judge_errors, model)
    print(f"\nwrote {out}" + (" (stopped early by an error)" if failed else ""))
    settings_used = {key: str(value) for key, value in vars(args).items()}
    out.with_suffix(".summary.json").write_text(
        json.dumps({"args": settings_used, "summary": summary}, indent=2), encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
