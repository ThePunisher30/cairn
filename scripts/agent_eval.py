"""Phase 8: run the agent or the single-shot pipeline on the same questions and judge the answers.

    python scripts/agent_eval.py --mode agent  --set multihop --llm fake --limit 3   no model
    python scripts/agent_eval.py --mode agent  --set multihop --llm ollama --model gemma4:e4b
    python scripts/agent_eval.py --mode single --set multihop --llm ollama --model gemma4:e4b
    python scripts/agent_eval.py --mode agent  --set dev      --llm ollama --model gemma4:e4b

`--set multihop` is the 10 dev questions of eval/multihop_questions.jsonl, `--set dev` the 40
answerable dev questions of Phase 7. The test split is not reachable from here: the one test run
that is left is reserved for Phase 11. Both modes use the same retriever, model and judge, so the
numbers can be put side by side. One line per question is written to data/agent_runs/<name>.jsonl
as soon as it is done (a slow local model makes a crash expensive; --resume skips the questions
already in the file). In agent mode every run's trace goes to <name>.traces.jsonl.

A run that ends without an answer (a budget used up, a failing model call) counts as incorrect.
Provided, not part of the exercise; it uses your agent and generation modules.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from cairn import config
from cairn.agent.budget import Budget
from cairn.agent.loop import cited_hits, run_agent
from cairn.agent.tools import build_corpus_tools
from cairn.config import load_settings
from cairn.dense.models import MODELS
from cairn.evaluation.agent import (
    AgentQuestionError,
    estimate_cost,
    hop_recall,
    judge_context,
    load_multihop_questions,
    load_single_hop_questions,
    summarize_agent_rows,
)
from cairn.evaluation.generation import GenQuestionError, Judge, JudgeParseError
from cairn.generation.abstain import says_cannot_answer
from cairn.generation.errors import LLMError
from cairn.generation.factory import LLMS, RETRIEVERS, build_llm, build_retrievers, load_prices
from cairn.generation.fake import FakeLLM, tool_reply
from cairn.generation.packer import ContextPacker
from cairn.generation.pipeline import Answerer
from cairn.generation.prompts import format_chunks, load_prompt
from cairn.retrieval.rerank_models import RERANKERS

SETS = ("multihop", "dev")
MODES = ("agent", "single")


def agent_demo_responder(system, messages):
    """What `--llm fake` does in agent mode: search once for the question, then answer."""
    if any(m["role"] == "tool" for m in messages):
        return "This is a demo answer [1]."
    return tool_reply(("search", {"query": messages[0]["content"]}))


def pct(value):
    return "n/a" if value is None else f"{value:.3f}"


def num(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(s, mode, model):
    print(f"\nmode: {mode}   model: {model}   questions: {s['n']}")
    print(f"  correctness         {pct(s['correctness'])}   ({s['n_graded']} graded)")
    print(f"  faithfulness        {pct(s['faithfulness'])}")
    print(f"  evidence seen       {pct(s['hop_recall'])}  (share of the gold spans in what it saw)")
    print(f"  abstained           {s['abstained']}   no answer at all: {s['no_answer']}")
    print(
        f"  per question        {num(s['retrievals_per_question'])} searches, "
        f"{num(s['tool_calls_per_question'])} tool calls, "
        f"{num(s['model_calls_per_question'])} model calls"
    )
    print(
        f"                      {num(s['tool_errors_per_question'], 2)} tool errors, "
        f"{num(s['duplicates_per_question'], 2)} repeated calls"
    )
    print(
        f"  tokens per question {num(s['input_tokens_per_question'], 0)} in, "
        f"{num(s['output_tokens_per_question'], 0)} out"
    )
    print(
        f"  latency             p50 {num(s['latency_p50_ms'], 0)} ms, "
        f"p95 {num(s['latency_p95_ms'], 0)} ms"
    )
    print(
        f"  est. cost / question ${s['est_cost_per_question'] or 0:.4f}  (priced as a paid model)"
    )
    print(f"  stop reasons        {s['stop_reasons']}")


def read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_single(q, answerer):
    """One retrieval, one model call (Phase 7). Returns (answer text or None, fields)."""
    started = time.perf_counter()
    answer = answerer.ask(q.text)
    latency_ms = (time.perf_counter() - started) * 1000
    seen = [c.hit for c in answer.context.chunks]
    return {
        "answer": answer.text,
        "abstained": answer.abstained,
        "stop_reason": "abstained" if answer.abstained else "answered",
        "seen": seen,
        "judge_context": answer.context,
        "retrievals": 1,
        "tool_calls": 0,
        "steps": 1,
        "tool_errors": 0,
        "duplicates": 0,
        "input_tokens": answer.usage.input_tokens,
        "output_tokens": answer.usage.output_tokens,
        "latency_ms": latency_ms,
        "trace": None,
    }


def run_agent_once(q, llm, tools, budget, max_output_tokens):
    tools.reset()
    started = time.perf_counter()
    result = run_agent(q.text, llm, tools.registry, budget, max_output_tokens=max_output_tokens)
    latency_ms = (time.perf_counter() - started) * 1000
    text = result.answer
    cited = cited_hits(text, tools.seen) if text else []
    ids = [tools.ids[(h.path, h.start_line, h.end_line)] for h in cited]
    return {
        "answer": text,
        "abstained": bool(text) and says_cannot_answer(text),
        "stop_reason": result.stop_reason,
        "seen": list(tools.seen),
        "judge_context": judge_context(list(zip(ids, cited, strict=True))),
        "retrievals": sum(r.name == "search" for s in result.trace.steps for r in s.tool_calls),
        "tool_calls": result.n_tool_calls,
        "steps": result.steps,
        "tool_errors": result.trace.n_errors,
        "duplicates": result.trace.n_duplicates,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "latency_ms": latency_ms,
        "trace": result.trace,
    }


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Agent vs single-shot RAG on the dev questions.")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--set", dest="qset", choices=SETS, default="multihop")
    parser.add_argument("--name", default=None, help="name of the run file (default: a timestamp)")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "agent_runs")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--resume", action="store_true", help="skip the questions already done")
    parser.add_argument(
        "--multihop", type=Path, default=config.PROJECT_ROOT / "eval" / "multihop_questions.jsonl"
    )
    parser.add_argument(
        "--gen-questions", type=Path, default=config.PROJECT_ROOT / "eval" / "gen_questions.jsonl"
    )
    parser.add_argument(
        "--questions", type=Path, default=config.PROJECT_ROOT / "eval" / "questions.jsonl"
    )
    parser.add_argument("--corpus", type=Path, default=settings.data_dir / "raw" / "fastapi")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--llm", choices=LLMS, default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-prompt-version", type=int, default=1)
    parser.add_argument("--k", type=int, default=5, help="chunks per search (and for single-shot)")
    parser.add_argument("--max-context-tokens", type=int, default=1500, help="single-shot only")
    parser.add_argument("--max-tokens", type=int, default=1024, help="the model's output limit")
    parser.add_argument("--max-steps", type=int, default=Budget().max_steps)
    parser.add_argument("--budget-tokens", type=int, default=Budget().max_tokens)
    parser.add_argument("--timeout", type=float, default=Budget().timeout_s)
    parser.add_argument(
        "--cost-model", default="claude-opus-5-5", help="the paid model to price the tokens as"
    )
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
        if args.qset == "multihop":
            questions = load_multihop_questions(args.multihop, split="dev")
        else:
            questions = load_single_hop_questions(args.gen_questions, args.questions, split="dev")
        if args.limit:
            questions = questions[: args.limit]
        budget = Budget(
            max_steps=args.max_steps, max_tokens=args.budget_tokens, timeout_s=args.timeout
        )
        retriever, _ = build_retrievers(
            args.retriever,
            chunks_path=args.chunks,
            bm25_path=args.bm25_index,
            dense_path=dense_path,
            model=args.embedding_model,
            reranker=args.reranker,
        )
        llm, model = build_llm(args.llm, model=args.model)
        judge_llm, _ = build_llm(args.llm, model=args.model)
        if args.llm == "fake" and args.mode == "agent":
            llm = FakeLLM(responder=agent_demo_responder)
        prices = load_prices(args.prices) if args.prices.is_file() else {}
        price = prices.get(args.cost_model, (0.0, 0.0))
        if prices and args.cost_model not in prices:
            raise ValueError(f"no price for {args.cost_model!r} in {args.prices}")
        answerer = tools = None
        if args.mode == "single":
            answerer = Answerer(
                retriever,
                llm,
                packer=ContextPacker(args.max_context_tokens),
                template=load_prompt("answer"),
                k=args.k,
                max_tokens=args.max_tokens,
            )
        else:
            if not args.corpus.is_dir():
                raise FileNotFoundError(f"{args.corpus} not found (run scripts/fetch_corpus.py)")
            tools = build_corpus_tools(retriever, args.corpus)
        judge = Judge(judge_llm, load_prompt("judge", args.judge_prompt_version))
    except (AgentQuestionError, GenQuestionError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"{name}.jsonl"
    traces_out = args.out_dir / f"{name}.traces.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out) if args.resume else []
    done = {r["id"] for r in rows}
    file_mode = "a" if args.resume else "w"
    judge_errors, failed = 0, False
    with (
        open(out, file_mode, encoding="utf-8") as run_file,
        open(traces_out, file_mode, encoding="utf-8") as trace_file,
    ):
        for number, q in enumerate(questions, start=1):
            if q.id in done:
                continue
            try:
                if args.mode == "single":
                    r = run_single(q, answerer)
                else:
                    r = run_agent_once(q, llm, tools, budget, args.max_tokens)
                correct = faithful = None
                reason = ""
                if r["answer"] is None or r["abstained"]:
                    correct = False  # no answer, or a refusal to a question that has one
                else:
                    try:
                        verdict = judge.grade(
                            q.text, q.reference, r["answer"], format_chunks(r["judge_context"])
                        )
                        correct, faithful, reason = (
                            verdict.correct,
                            verdict.faithful,
                            verdict.reason,
                        )
                    except JudgeParseError:
                        judge_errors += 1
            except LLMError as e:
                print(f"error on {q.id}: {e}", file=sys.stderr)
                failed = True
                break
            seen = r["seen"]
            row = {
                "id": q.id,
                "mode": args.mode,
                "question": q.text,
                "reference": q.reference,
                "answer": r["answer"],
                "abstained": r["abstained"],
                "stop_reason": r["stop_reason"],
                "correct": correct,
                "faithful": faithful,
                "judge_reason": reason,
                "hop_recall": hop_recall(seen, q.spans),
                "seen": [[h.path, h.start_line, h.end_line] for h in seen],
                "retrievals": r["retrievals"],
                "tool_calls": r["tool_calls"],
                "steps": r["steps"],
                "tool_errors": r["tool_errors"],
                "duplicates": r["duplicates"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "latency_ms": r["latency_ms"],
                "est_cost_usd": estimate_cost(r["input_tokens"], r["output_tokens"], price),
                "model": model,
            }
            rows.append(row)
            run_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            run_file.flush()
            if r["trace"] is not None:
                trace_file.write(json.dumps(r["trace"].to_dict(), ensure_ascii=False) + "\n")
                trace_file.flush()
            print(f"[{number}/{len(questions)}] {q.id} {r['stop_reason']}", flush=True)

    summary = summarize_agent_rows(rows)
    print_summary(summary, args.mode, model)
    if judge_errors:
        print(f"  judge replies that could not be read: {judge_errors}")
    print(f"\nwrote {out}" + (" (stopped early by an error)" if failed else ""))
    used = {key: str(value) for key, value in vars(args).items()}
    out.with_suffix(".summary.json").write_text(
        json.dumps({"args": used, "summary": summary}, indent=2), encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
