import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cairn.evaluation.questions import load_questions
from cairn.generation.abstain import abstain_metrics
from cairn.generation.prompts import load_prompt, neutralize
from cairn.jsonl import JsonlError, read_jsonl


class GenQuestionError(ValueError):
    pass


class JudgeParseError(ValueError):
    pass


@dataclass(frozen=True)
class GenQuestion:
    id: str
    category: str
    split: str
    text: str
    reference: str | None
    answerable: bool
    gold: tuple


def _blank(value):
    return not isinstance(value, str) or not value.strip()


def load_gen_questions(gen_path, questions_path, split="dev"):
    known = {q.id: q for q in load_questions(questions_path)}
    name = Path(gen_path).name
    try:
        rows = list(read_jsonl(gen_path))
    except JsonlError as e:
        raise GenQuestionError(str(e)) from e
    result, seen = [], set()
    for number, row in enumerate(rows, start=1):
        question_id = row.get("id")
        if not question_id:
            raise GenQuestionError(f"{name}: row {number} has no id")
        where = f"{name}: {question_id!r}"
        if question_id in seen:
            raise GenQuestionError(f"{where}: duplicate id")
        seen.add(question_id)
        if row.get("answerable") is False:
            if question_id in known:
                raise GenQuestionError(f"{where}: an unanswerable row cannot reuse a question id")
            if _blank(row.get("question")):
                raise GenQuestionError(f"{where}: the question is blank")
            item = GenQuestion(
                question_id,
                "unanswerable",
                row.get("split", "dev"),
                row["question"],
                None,
                False,
                (),
            )
        else:
            if question_id not in known:
                raise GenQuestionError(f"{where}: not a question of {Path(questions_path).name}")
            if _blank(row.get("reference")):
                raise GenQuestionError(f"{where}: the reference is blank")
            q = known[question_id]
            item = GenQuestion(
                question_id, q.category, q.split, q.text, row["reference"], True, q.gold
            )
        result.append(item)
    return [q for q in result if split is None or q.split == split]


@dataclass(frozen=True)
class Verdict:
    correct: bool | None
    faithful: bool | None
    reason: str = ""


def _flag(value):
    if value is None:
        return None
    if isinstance(value, (bool, int)) and value in (0, 1):
        return bool(value)
    raise JudgeParseError(f"a flag must be true, false, 0, 1 or null, not {value!r}")


def parse_verdict(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise JudgeParseError("no JSON object in the judge's reply")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"the judge's reply is not valid JSON: {e}") from e
    if not isinstance(data, dict) or ("correct" not in data and "faithful" not in data):
        raise JudgeParseError("the verdict has neither 'correct' nor 'faithful'")
    return Verdict(
        _flag(data.get("correct")), _flag(data.get("faithful")), str(data.get("reason") or "")
    )


class Judge:
    def __init__(self, llm, template=None, *, max_tokens=300):
        self.llm = llm
        self.template = template if template is not None else load_prompt("judge")
        self.max_tokens = max_tokens
        self.calls = 0

    def grade(self, question, reference, answer, context_text):
        system, user = self.template.render(
            question=neutralize(question),
            reference=neutralize(reference),
            answer=neutralize(answer),
            context=context_text,
        )
        reply = self.llm.complete(
            system, [{"role": "user", "content": user}], max_tokens=self.max_tokens
        )
        self.calls += 1
        return parse_verdict(reply.text)


@dataclass(frozen=True)
class GenResult:
    id: str
    category: str
    answerable: bool
    abstained: bool
    correct: bool | None
    faithful: bool | None
    citation_precision: float | None
    citation_recall: float | None
    cost_usd: float
    latency_ms: float


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_generation(results):
    results = list(results)
    answerable = [r for r in results if r.answerable]
    graded = [r.correct for r in answerable if r.correct is not None]
    return {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(results) - len(answerable),
        "correctness": _mean(graded),
        "n_graded": len(graded),
        "faithfulness": _mean(r.faithful for r in answerable if r.faithful is not None),
        "citation_precision": _mean(
            r.citation_precision for r in answerable if r.citation_precision is not None
        ),
        "citation_recall": _mean(
            r.citation_recall for r in answerable if r.citation_recall is not None
        ),
        "false_abstain_rate": _mean(r.abstained for r in answerable),
        "abstain": abstain_metrics(
            [r.abstained for r in results], [not r.answerable for r in results]
        ),
        "cost_per_question": _mean(r.cost_usd for r in results),
        "total_cost_usd": sum(r.cost_usd for r in results),
        "latency_p50_ms": statistics.median(r.latency_ms for r in results) if results else None,
    }


def _check_pair(a, b):
    if len(a) != len(b):
        raise ValueError(f"the lists have {len(a)} and {len(b)} items")
    if not a:
        raise ValueError("the lists are empty")


def agreement_rate(a, b):
    a, b = list(a), list(b)
    _check_pair(a, b)
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def cohens_kappa(a, b):
    a, b = list(a), list(b)
    _check_pair(a, b)
    observed = agreement_rate(a, b)
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in count_a) / len(a) ** 2
    if expected == 1:
        return 1.0  # both always say the same single thing: there is nothing left to disagree on
    return (observed - expected) / (1 - expected)
