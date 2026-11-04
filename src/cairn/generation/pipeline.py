import time
from dataclasses import dataclass

from cairn.generation.abstain import (
    ABSTAIN_MESSAGE,
    is_low_confidence,
    says_cannot_answer,
    top_score,
)
from cairn.generation.citations import (
    extract_citations,
    render_sources,
    strip_invalid,
    validate_citations,
)
from cairn.generation.packer import ContextPacker, PackedContext
from cairn.generation.prompts import build_messages, load_prompt
from cairn.generation.types import Usage


class CostTracker:
    def __init__(self, prices=None):
        self.prices = prices
        self.calls = 0
        self.usage = Usage()
        self.total_usd = 0.0

    def cost(self, model, usage):
        if self.prices is None:
            return 0.0
        input_price, output_price = self.prices[model]  # a missing price is a KeyError, not free
        return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000

    def add(self, model, usage):
        cost = self.cost(model, usage)
        self.calls += 1
        self.usage = self.usage + usage
        self.total_usd += cost
        return cost


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    abstained: bool
    abstain_reason: str | None
    citations: tuple
    invalid_citations: tuple
    sources: tuple
    context: PackedContext
    usage: Usage
    cost_usd: float
    latency_ms: float
    model: str
    truncated: bool = False

    @property
    def uncited(self):
        return not self.abstained and not self.citations


def _best_score(question, hits):
    return top_score(hits)


class Answerer:
    def __init__(
        self,
        retriever,
        llm,
        *,
        packer=None,
        template=None,
        k=8,
        max_tokens=1024,
        confidence=None,
        threshold=None,
        costs=None,
        expander=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.packer = packer if packer is not None else ContextPacker()
        self.template = template if template is not None else load_prompt("answer")
        self.k = k
        self.max_tokens = max_tokens
        self.confidence = confidence if confidence is not None else _best_score
        self.threshold = threshold
        self.costs = CostTracker(costs)
        self.expander = expander
        self.last_answer = None

    def _retrieve(self, question):
        hits = self.retriever.search(question, self.k)
        if self.expander is not None:
            hits = self.expander.expand(hits)
        return hits, self.packer.pack(hits)

    def _low_confidence(self, question, hits):
        if self.threshold is None:
            return False
        return is_low_confidence(self.confidence(question, hits), self.threshold)

    def _answer(
        self,
        question,
        context,
        started,
        text,
        *,
        abstain_reason=None,
        citations=(),
        invalid=(),
        sources=(),
        usage=None,
        cost=0.0,
        model="",
        truncated=False,
    ):
        return Answer(
            question=question,
            text=text,
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
            citations=tuple(citations),
            invalid_citations=tuple(invalid),
            sources=tuple(sources),
            context=context,
            usage=usage if usage is not None else Usage(),
            cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=model,
            truncated=truncated,
        )

    def _finish(self, question, context, started, text, stop_reason, usage, model):
        cost = self.costs.add(model, usage)
        paid = {"usage": usage, "cost": cost, "model": model}
        if stop_reason == "refusal":
            return self._answer(
                question, context, started, ABSTAIN_MESSAGE, abstain_reason="refused", **paid
            )
        if says_cannot_answer(text):
            return self._answer(
                question, context, started, ABSTAIN_MESSAGE, abstain_reason="model_declined", **paid
            )
        n_chunks = len(context.chunks)
        valid, invalid = validate_citations(extract_citations(text), n_chunks)
        return self._answer(
            question,
            context,
            started,
            strip_invalid(text, n_chunks),
            citations=valid,
            invalid=invalid,
            sources=render_sources(context, valid),
            truncated=stop_reason == "max_tokens",
            **paid,
        )

    def ask(self, question):
        started = time.perf_counter()
        hits, context = self._retrieve(question)
        if self._low_confidence(question, hits):
            answer = self._answer(
                question, context, started, ABSTAIN_MESSAGE, abstain_reason="low_confidence"
            )
        else:
            system, messages = build_messages(self.template, question, context)
            done = self.llm.complete(system, messages, max_tokens=self.max_tokens)
            answer = self._finish(
                question, context, started, done.text, done.stop_reason, done.usage, done.model
            )
        self.last_answer = answer
        return answer

    def stream(self, question):
        started = time.perf_counter()
        hits, context = self._retrieve(question)
        if self._low_confidence(question, hits):
            self.last_answer = self._answer(
                question, context, started, ABSTAIN_MESSAGE, abstain_reason="low_confidence"
            )
            yield ABSTAIN_MESSAGE
            return
        system, messages = build_messages(self.template, question, context)
        pieces = []
        for piece in self.llm.stream(system, messages, max_tokens=self.max_tokens):
            pieces.append(piece)
            yield piece
        usage = self.llm.last_usage if self.llm.last_usage is not None else Usage()
        model = getattr(self.llm, "model", "")
        self.last_answer = self._finish(
            question, context, started, "".join(pieces), "end_turn", usage, model
        )
