"""Ask one question and get a cited answer (Phase 7).

    python scripts/ask.py "How do I enable CORS?"
    python scripts/ask.py "..." --retriever dense --llm ollama --model llama3.2
    python scripts/ask.py "..." --stream --threshold 0.6 --confidence dense

`--llm anthropic` needs the ANTHROPIC_API_KEY environment variable and `pip install -e ".[llm]"`.
`--llm fake` answers with a fixed sentence and needs no network (for trying the plumbing).
Provided, not part of the exercise; it uses your generation modules.
"""

import argparse
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.dense.models import MODELS
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
from cairn.generation.pipeline import Answerer
from cairn.generation.prompts import load_prompt
from cairn.retrieval.rerank_models import RERANKERS


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Ask a question and get a cited answer.")
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--llm", choices=LLMS, default="anthropic")
    parser.add_argument("--model", default=None, help="the model name (a default per --llm)")
    parser.add_argument("--effort", default="low", help="Anthropic thinking effort")
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
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--chunks", type=Path, default=settings.data_dir / "chunks.jsonl")
    parser.add_argument("--bm25-index", type=Path, default=settings.data_dir / "bm25.pkl")
    parser.add_argument("--embedding-model", choices=list(MODELS), default="bge-small")
    parser.add_argument("--dense-index", type=Path, default=None)
    parser.add_argument("--reranker", choices=list(RERANKERS), default="minilm-ce")
    args = parser.parse_args(argv)

    dense_path = args.dense_index or settings.data_dir / f"dense-{args.embedding_model}.npz"
    try:
        retriever, dense = build_retrievers(
            args.retriever,
            chunks_path=args.chunks,
            bm25_path=args.bm25_index,
            dense_path=dense_path,
            model=args.embedding_model,
            reranker=args.reranker,
        )
        llm, model = build_llm(args.llm, model=args.model, effort=args.effort)
        prices = load_prices(args.prices) if args.prices.is_file() else None
        if prices is not None and model not in prices:
            print(f"note: no price for {model!r} in {args.prices}; the cost shows as 0",
                  file=sys.stderr)  # fmt: skip
        answerer = Answerer(
            retriever,
            llm,
            packer=ContextPacker(args.max_context_tokens, order=args.order),
            template=load_prompt("answer", args.prompt_version),
            k=args.k,
            max_tokens=args.max_tokens,
            confidence=build_confidence(args.confidence, dense),
            threshold=args.threshold,
            costs=prices if prices and model in prices else None,
        )
        if args.stream:
            for piece in answerer.stream(args.question):
                print(piece, end="", flush=True)
            print()
        else:
            print(answerer.ask(args.question).text)
    except (LLMError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    answer = answerer.last_answer
    if answer.abstained:
        print(f"\n(abstained: {answer.abstain_reason})")
    for line in answer.sources:
        print(line)
    if answer.invalid_citations:
        print(f"(the model cited chunks that do not exist: {list(answer.invalid_citations)})")
    if answer.uncited:
        print("(the answer cites nothing)")
    print(
        f"\n{answer.usage.input_tokens} tokens in, {answer.usage.output_tokens} out, "
        f"${answer.cost_usd:.4f}, {answer.latency_ms:.0f} ms, "
        f"{len(answer.context.chunks)} chunks ({answer.context.tokens} tokens, "
        f"{answer.context.dropped} dropped), model {answer.model or model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
