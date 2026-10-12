"""Run a retriever over the questions and summarise how well it did."""

from dataclasses import dataclass
from statistics import fmean

from cairn.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from cairn.timing import Timer
from cairn.types import CATEGORIES, Hit, Question


@dataclass(frozen=True)
class QuestionResult:
    """What one question produced: the question, the ranked hits, and how long the search took."""

    question: Question
    hits: tuple[Hit, ...]
    latency_ms: float


def run_retriever(retriever, questions, k) -> list[QuestionResult]:
    """Ask `retriever` every question and record the hits and the time taken."""
    results = []
    for question in questions:
        with Timer() as timer:  # the Timer you wrote in Lesson 2
            hits = retriever.search(question.text, k)
        # Keep at most k hits: a retriever that returns extra results must not get extra credit.
        results.append(QuestionResult(question, tuple(hits[:k]), timer.seconds * 1000))
    return results


def _group(results, ks) -> dict:
    """The numbers for one group of results (all questions, or just one category)."""
    biggest = max(ks)
    row = {"n": len(results)}
    for k in ks:
        row[f"hit@{k}"] = fmean(hit_at_k(r.hits, r.question.gold, k) for r in results)
        row[f"recall@{k}"] = fmean(recall_at_k(r.hits, r.question.gold, k) for r in results)
        row[f"ndcg@{k}"] = fmean(ndcg_at_k(r.hits, r.question.gold, k) for r in results)
    row["mrr"] = fmean(reciprocal_rank(r.hits, r.question.gold, biggest) for r in results)
    latencies = [r.latency_ms for r in results]
    row["latency_p50_ms"] = percentile(latencies, 50)
    row["latency_p95_ms"] = percentile(latencies, 95)
    return row


def summarize(results, ks=(1, 5, 10)) -> dict:
    """Overall numbers, numbers per category, and the ids of questions that found nothing."""
    if not results:
        raise ValueError("no results to summarize")

    by_category = {}
    for category in CATEGORIES:
        group = [r for r in results if r.question.category == category]
        if group:  # only list categories that actually appear
            by_category[category] = _group(group, ks)

    biggest = max(ks)
    misses = [r.question.id for r in results if hit_at_k(r.hits, r.question.gold, biggest) == 0.0]
    return {"overall": _group(results, ks), "by_category": by_category, "misses": misses}


def format_report(summary: dict) -> str:
    """Turn the summary into a readable text table."""
    overall = summary["overall"]
    columns = [key for key in overall if key != "n" and not key.startswith("latency")]

    def row(name: str, numbers: dict) -> str:
        cells = "".join(f"{numbers[column]:>10.3f}" for column in columns)
        return f"{name:<13}{numbers['n']:>4}{cells}"

    header = f"{'':<13}{'n':>4}" + "".join(f"{column:>10}" for column in columns)
    lines = [header, row("overall", overall)]
    lines += [row(category, numbers) for category, numbers in summary["by_category"].items()]

    misses = summary["misses"]
    lines.append("")
    lines.append(
        f"latency: p50 {overall['latency_p50_ms']:.1f} ms, p95 {overall['latency_p95_ms']:.1f} ms"
    )
    lines.append(f"misses ({len(misses)}): {', '.join(misses) if misses else 'none'}")
    return "\n".join(lines)
