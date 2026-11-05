"""Phase 7, step 7: from a question to a cited, accounted answer (`generation/pipeline.py`).

CostTracker(prices=None)
    `prices` maps a model name to (dollars per million input tokens, dollars per million output
    tokens). cost(model, usage) -> float: 0.0 when prices is None; KeyError for a model that
    has no price (a missing price must not look like "free"). add(model, usage) -> the cost,
    and it counts `.calls`, sums `.usage` (a Usage) and `.total_usd`.

Answer    a frozen dataclass: question, text, abstained, abstain_reason, citations (tuple of the
    valid numbers), invalid_citations (tuple), sources (tuple of rendered lines), context (the
    PackedContext), usage, cost_usd, latency_ms, model, truncated=False; and the property
    `uncited`: not abstained and no valid citation.

Answerer(retriever, llm, *, packer=None, template=None, k=8, max_tokens=1024, confidence=None,
         threshold=None, costs=None, expander=None)
    packer defaults to ContextPacker(), template to load_prompt("answer"), `costs` is the
    price table of a CostTracker kept as `.costs`, `confidence(question, hits) -> float | None`
    defaults to the best hit score, `expander.expand(hits)` (optional) widens the hits before
    packing.
    ask(question) -> Answer:
      1. hits = retriever.search(question, k), expanded if there is an expander; pack them.
      2. Low confidence: when `threshold` is not None and the confidence is low (None counts as
         low), do NOT call the model: Answer(text=ABSTAIN_MESSAGE, abstained=True,
         abstain_reason="low_confidence", no citations, Usage(), cost 0.0, model "").
      3. Otherwise build the messages (build_messages) and call llm.complete(system, messages,
         max_tokens=max_tokens). The cost is added to `.costs` under completion.model.
      4. stop_reason "refusal" -> abstained, reason "refused"; an answer that says it cannot
         answer -> abstained, reason "model_declined". Both show ABSTAIN_MESSAGE, no citations.
      5. Otherwise the citations of the text are extracted and validated against the number of
         chunks; the text is the one with the invalid markers stripped; `sources` renders the
         valid ones; truncated is True when stop_reason == "max_tokens".
      latency_ms is the whole ask. `.last_answer` is set.
    stream(question) -> Iterator[str]:  the same steps, but the model is streamed. The pieces
      are yielded as they come (raw: the marker or an invented citation can still be in them).
      When the stream has been read to the end `.last_answer` is the finished Answer, with the
      usage from llm.last_usage (Usage() if None) and the model from `llm.model` ("" if the
      client has none). On low confidence it yields ABSTAIN_MESSAGE once and calls no model.
"""

import pytest

from cairn.generation.abstain import ABSTAIN_MARKER, ABSTAIN_MESSAGE
from cairn.generation.fake import FakeLLM
from cairn.generation.packer import ContextPacker, PackedContext
from cairn.generation.pipeline import Answer, Answerer, CostTracker
from cairn.generation.prompts import PromptTemplate
from cairn.generation.types import Completion, Usage
from cairn.types import Hit

HITS = [
    Hit("docs/a.md", 1, 4, 0.9, "Use Depends to declare a dependency."),
    Hit("fastapi/b.py", 10, 20, 0.5, "def Depends(): ..."),
]


class Scripted:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.hits[:k]


def answerer(llm=None, hits=HITS, **options):
    llm = llm if llm is not None else FakeLLM(["Declare it with Depends [1]."])
    return Answerer(Scripted(hits), llm, **options), llm


# --------------------------------------------------------------------------- CostTracker
def test_without_prices_everything_is_free_but_still_counted():
    tracker = CostTracker()
    assert tracker.cost("any-model", Usage(1000, 1000)) == 0.0
    assert tracker.add("any-model", Usage(100, 20)) == 0.0
    tracker.add("any-model", Usage(1, 2))
    assert (tracker.calls, tracker.usage, tracker.total_usd) == (2, Usage(101, 22), 0.0)


def test_the_cost_is_priced_per_million_tokens():
    tracker = CostTracker({"m": (3.0, 15.0)})
    assert tracker.cost("m", Usage(1_000_000, 0)) == pytest.approx(3.0)
    assert tracker.cost("m", Usage(1000, 200)) == pytest.approx(0.006)


def test_adding_accumulates_the_total():
    tracker = CostTracker({"m": (3.0, 15.0), "n": (1.0, 5.0)})
    assert tracker.add("m", Usage(1000, 200)) == pytest.approx(0.006)
    tracker.add("n", Usage(2000, 0))
    assert tracker.total_usd == pytest.approx(0.008)
    assert tracker.usage == Usage(3000, 200) and tracker.calls == 2


def test_a_model_without_a_price_is_an_error_not_free():
    with pytest.raises(KeyError):
        CostTracker({"m": (1.0, 1.0)}).add("unknown", Usage(1, 1))


def test_usage_adds_up():
    assert Usage(1, 2) + Usage(10, 20) == Usage(11, 22)


# ---------------------------------------------------------------------------- ask: basics
def test_the_retriever_is_asked_for_k_hits():
    a, _ = answerer(k=1)
    a.ask("How do I declare a dependency?")
    assert a.retriever.calls == [("How do I declare a dependency?", 1)]
    assert Answerer(Scripted(HITS), FakeLLM()).k == 8


def test_the_model_gets_the_question_and_the_numbered_chunks():
    a, llm = answerer(max_tokens=321)
    a.ask("How do I declare a dependency?")
    (call,) = llm.calls
    assert call["max_tokens"] == 321 and call["stream"] is False
    user = call["messages"][0]["content"]
    assert call["messages"][0]["role"] == "user"
    assert "How do I declare a dependency?" in user
    assert '<chunk id="1" source="docs/a.md:1-4">' in user
    assert '<chunk id="2" source="fastapi/b.py:10-20">' in user
    assert ABSTAIN_MARKER in call["system"]
    assert "Depends to declare" not in call["system"]


def test_a_cited_answer():
    a, _ = answerer(llm=FakeLLM([Completion("Declare it with Depends [1].", Usage(30, 8), "m1")]))
    answer = a.ask("q?")
    assert isinstance(answer, Answer)
    assert answer.text == "Declare it with Depends [1]."
    assert (answer.abstained, answer.abstain_reason) == (False, None)
    assert answer.citations == (1,) and answer.invalid_citations == ()
    assert answer.sources == (
        "[1] [docs/a.md:1-4](https://github.com/fastapi/fastapi/blob/0.143.0/docs/a.md#L1-L4)",
    )
    assert answer.uncited is False and answer.truncated is False
    assert answer.usage == Usage(30, 8) and answer.model == "m1"
    assert answer.latency_ms >= 0 and answer.question == "q?"
    assert isinstance(answer.context, PackedContext) and len(answer.context.chunks) == 2
    assert a.last_answer is answer


def test_an_invented_citation_is_reported_and_removed():
    a, _ = answerer(llm=FakeLLM(["Use it [1] and also this [7]."]))
    answer = a.ask("q?")
    assert answer.citations == (1,) and answer.invalid_citations == (7,)
    assert answer.text == "Use it [1] and also this."
    assert len(answer.sources) == 1


def test_an_answer_without_citations_is_flagged():
    a, _ = answerer(llm=FakeLLM(["Use it, trust me."]))
    answer = a.ask("q?")
    assert answer.citations == () and answer.uncited is True and not answer.abstained


def test_max_tokens_cut_is_reported():
    done = Completion("Cut off [1]", Usage(1, 1), "m", "max_tokens")
    assert answerer(llm=FakeLLM([done]))[0].ask("q?").truncated is True


def test_a_model_that_says_it_cannot_answer_makes_an_abstention():
    a, _ = answerer(llm=FakeLLM([Completion(ABSTAIN_MARKER, Usage(40, 3), "m")]))
    answer = a.ask("q?")
    assert answer.abstained is True and answer.abstain_reason == "model_declined"
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == () and answer.sources == () and answer.uncited is False
    assert answer.usage == Usage(40, 3)  # the call was made and is paid for


def test_a_refusal_by_the_provider_makes_an_abstention():
    a, _ = answerer(llm=FakeLLM([Completion("", Usage(5, 0), "m", "refusal")]))
    answer = a.ask("q?")
    assert (answer.abstained, answer.abstain_reason) == (True, "refused")
    assert answer.text == ABSTAIN_MESSAGE


def test_no_hits_still_asks_the_model_when_there_is_no_threshold():
    a, llm = answerer(llm=FakeLLM([ABSTAIN_MARKER]), hits=[])
    answer = a.ask("q?")
    assert len(llm.calls) == 1 and answer.abstain_reason == "model_declined"
    assert answer.context.chunks == ()


# ---------------------------------------------------------------------- ask: confidence
def test_low_confidence_skips_the_model_altogether():
    a, llm = answerer(threshold=0.95)  # the best score is 0.9
    answer = a.ask("q?")
    assert llm.calls == []
    assert (answer.abstained, answer.abstain_reason) == (True, "low_confidence")
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.usage == Usage() and answer.cost_usd == 0.0 and answer.model == ""
    assert len(answer.context.chunks) == 2  # what was found is still there to inspect
    assert a.last_answer is answer


def test_a_confidence_at_the_threshold_is_enough():
    a, llm = answerer(threshold=0.9)
    assert a.ask("q?").abstained is False and len(llm.calls) == 1


def test_no_hits_with_a_threshold_is_low_confidence():
    a, llm = answerer(threshold=0.1, hits=[])
    assert a.ask("q?").abstain_reason == "low_confidence" and llm.calls == []


def test_without_a_threshold_confidence_is_never_checked():
    seen = []
    a, llm = answerer(confidence=lambda q, hits: seen.append(q) or 0.0)
    assert a.ask("q?").abstained is False
    assert len(llm.calls) == 1


def test_a_custom_confidence_function_sees_the_question_and_the_hits():
    seen = []

    def confidence(question, hits):
        seen.append((question, list(hits)))
        return 0.99

    a, llm = answerer(confidence=confidence, threshold=0.95)
    assert a.ask("why?").abstained is False
    assert seen == [("why?", HITS)]


# --------------------------------------------------------------------------- ask: cost
def test_the_cost_of_every_call_is_priced_and_added_up():
    llm = FakeLLM(["A [1].", "B [1]."], model="m", usage=Usage(1000, 200))
    a, _ = answerer(llm=llm, costs={"m": (3.0, 15.0)})
    first = a.ask("q1?")
    a.ask("q2?")
    assert first.cost_usd == pytest.approx(0.006)
    assert a.costs.total_usd == pytest.approx(0.012) and a.costs.calls == 2


def test_an_unpriced_model_is_an_error_when_prices_are_given():
    a, _ = answerer(llm=FakeLLM(["A [1]."], model="other"), costs={"m": (1.0, 1.0)})
    with pytest.raises(KeyError):
        a.ask("q?")


def test_a_skipped_call_costs_nothing():
    a, _ = answerer(threshold=5.0, costs={"m": (1.0, 1.0)})
    a.ask("q?")
    assert a.costs.calls == 0 and a.costs.total_usd == 0.0


# --------------------------------------------------------------- ask: packer and expander
def test_the_packer_decides_what_the_model_sees():
    long_hit = Hit("a.md", 1, 9, 1.0, " ".join(["word"] * 200))
    a, llm = answerer(packer=ContextPacker(max_tokens=50), hits=[long_hit, HITS[1]])
    answer = a.ask("q?")
    assert len(answer.context.chunks) == 1 and answer.context.chunks[0].truncated
    assert answer.context.dropped == 1


def test_an_expander_widens_the_hits_before_packing():
    class Wider:
        def expand(self, hits):
            return [Hit(h.path, h.start_line, h.end_line, h.score, h.text + " plus context")
                    for h in hits]  # fmt: skip

    a, llm = answerer(expander=Wider())
    a.ask("q?")
    assert "plus context" in llm.calls[0]["messages"][0]["content"]


def test_a_custom_template_is_used():
    template = PromptTemplate("t", 1, "MY SYSTEM", "{{question}} // {{chunks}}")
    a, llm = answerer(template=template)
    a.ask("q?")
    assert llm.calls[0]["system"] == "MY SYSTEM"
    assert llm.calls[0]["messages"][0]["content"].startswith("q? // <chunk")


# ------------------------------------------------------------------------------ stream
def read(a, question="q?"):
    return list(a.stream(question))


def test_stream_yields_the_pieces_and_leaves_the_finished_answer_behind():
    llm = FakeLLM([Completion("Use Depends [1] now.", Usage(50, 6), "m")], model="m")
    a, _ = answerer(llm=llm, costs={"m": (1.0, 1.0)})
    pieces = read(a)
    assert "".join(pieces) == "Use Depends [1] now." and len(pieces) > 1
    answer = a.last_answer
    assert answer.text == "Use Depends [1] now." and answer.citations == (1,)
    assert answer.usage == Usage(50, 6) and answer.model == "m"
    assert answer.cost_usd == pytest.approx(0.000056)
    assert llm.calls[0]["stream"] is True


def test_stream_cleans_the_final_answer_but_not_the_pieces():
    a, _ = answerer(llm=FakeLLM(["Fact [1] and bogus [9]."]))
    assert "".join(read(a)) == "Fact [1] and bogus [9]."
    assert a.last_answer.text == "Fact [1] and bogus." and a.last_answer.invalid_citations == (9,)


def test_stream_recognises_a_model_that_cannot_answer():
    a, _ = answerer(llm=FakeLLM([ABSTAIN_MARKER]))
    assert "".join(read(a)) == ABSTAIN_MARKER
    assert a.last_answer.abstained and a.last_answer.abstain_reason == "model_declined"
    assert a.last_answer.text == ABSTAIN_MESSAGE


def test_stream_on_low_confidence_says_so_once_and_calls_nobody():
    a, llm = answerer(threshold=0.99)
    assert read(a) == [ABSTAIN_MESSAGE]
    assert llm.calls == []
    assert a.last_answer.abstain_reason == "low_confidence"


def test_stream_without_usage_or_model_still_finishes():
    class Bare:
        last_usage = None

        def stream(self, system, messages, *, max_tokens=1024):
            yield "Answer [1]."

    a = Answerer(Scripted(HITS), Bare(), costs=None)
    read(a)
    assert a.last_answer.usage == Usage() and a.last_answer.model == ""
