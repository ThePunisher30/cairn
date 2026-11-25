"""Helpers for the Phase 8 comparison of the agent with single-shot RAG (`scripts/agent_eval.py`).

Both are run on the same questions, so a question here is just an id, the text, a reference answer
and the places (spans) where the facts are: one span for a single-hop question, two or more for a
multi-hop one.
"""

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cairn.evaluation.generation import load_gen_questions
from cairn.generation.packer import PackedChunk, PackedContext
from cairn.tokens import count_tokens, truncate_tokens
from cairn.types import Span


class AgentQuestionError(ValueError):
    pass


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    text: str
    reference: str
    spans: tuple[Span, ...]


def load_multihop_questions(path, split="dev"):
    """The questions of `eval/multihop_questions.jsonl` of one split (dev unless asked)."""
    questions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            if row["split"] != split:
                continue
            spans = tuple(Span(h["path"], h["start_line"], h["end_line"]) for h in row["hops"])
            questions.append(EvalQuestion(row["id"], row["question"], row["reference"], spans))
        except (KeyError, TypeError, ValueError) as e:
            raise AgentQuestionError(f"{Path(path).name} line {number}: {e!r}") from e
    return questions


def load_single_hop_questions(gen_path, questions_path, split="dev"):
    """The answerable questions of the Phase 7 set (the unanswerable ones have no reference)."""
    return [
        EvalQuestion(q.id, q.text, q.reference, tuple(q.gold))
        for q in load_gen_questions(gen_path, questions_path, split=split)
        if q.answerable
    ]


def overlaps(hit, span):
    """True when the chunk `hit` touches the lines of `span` in the same file."""
    return (
        hit.path == span.path
        and hit.start_line <= span.end_line
        and span.start_line <= hit.end_line
    )


def hop_recall(hits, spans):
    """The share of `spans` that some hit overlaps: how much of the evidence the model saw."""
    if not spans:
        return None
    return sum(any(overlaps(h, s) for h in hits) for s in spans) / len(spans)


def judge_context(numbered_hits, max_tokens=1500):
    """The chunks the judge reads to check an answer, kept to `max_tokens` in all.

    `numbered_hits` is a list of (number, Hit); the numbers are the ones the answer cites. The
    last chunk is cut to fit; a chunk is skipped when almost nothing of the budget is left.
    """
    chunks, used = [], 0
    for number, hit in numbered_hits:
        room = max_tokens - used
        if room < 40:
            break
        text = truncate_tokens(hit.text, room)
        chunks.append(PackedChunk(number, hit, text, text != hit.text))
        used += count_tokens(text)
    return PackedContext(tuple(chunks), used, len(numbered_hits) - len(chunks))


def estimate_cost(input_tokens, output_tokens, price):
    """Dollars for these tokens at `price` = (dollars per million input, per million output)."""
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def percentile(values, q):
    """The q-th percentile (0-100) by linear interpolation; None for no values."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_agent_rows(rows):
    """The numbers of the comparison table, from the rows `agent_eval.py` writes."""
    rows = list(rows)
    latencies = [r["latency_ms"] for r in rows]
    graded = [r["correct"] for r in rows if r["correct"] is not None]
    return {
        "n": len(rows),
        "correctness": _mean(graded),
        "n_graded": len(graded),
        "faithfulness": _mean(r["faithful"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "no_answer": sum(r["answer"] is None for r in rows),
        "hop_recall": _mean(r["hop_recall"] for r in rows),
        "retrievals_per_question": _mean(r["retrievals"] for r in rows),
        "tool_calls_per_question": _mean(r["tool_calls"] for r in rows),
        "model_calls_per_question": _mean(r["steps"] for r in rows),
        "tool_errors_per_question": _mean(r["tool_errors"] for r in rows),
        "duplicates_per_question": _mean(r["duplicates"] for r in rows),
        "input_tokens_per_question": _mean(r["input_tokens"] for r in rows),
        "output_tokens_per_question": _mean(r["output_tokens"] for r in rows),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95),
        "est_cost_per_question": _mean(r["est_cost_usd"] for r in rows),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
    }
