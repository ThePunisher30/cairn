"""Phase 1, step 7: running a retriever over the questions and summarising the results."""
import time

import pytest

from cairn.evaluation.harness import QuestionResult, format_report, run_retriever, summarize
from cairn.types import Hit, Question, Span


def H(path, a, b):
    return Hit(path=path, start_line=a, end_line=b)


Q1 = Question("q1", "lexical", "dev", "first?", (Span("a.md", 10, 20),))
Q2 = Question("q2", "code", "dev", "second?", (Span("c.py", 1, 10), Span("c.py", 50, 60)))
Q3 = Question("q3", "lexical", "dev", "third?", (Span("z.md", 1, 5),))


class Scripted:
    """A fake retriever: returns a fixed list of hits per question text and records its calls."""

    def __init__(self, answers, delay=0.0):
        self.answers, self.delay, self.calls = answers, delay, []

    def search(self, query, k):
        self.calls.append((query, k))
        time.sleep(self.delay)
        return list(self.answers.get(query, []))


ANSWERS = {
    "first?": [H("b.md", 1, 5), H("a.md", 15, 25)],      # gold found at rank 2
    "second?": [H("c.py", 5, 8)],                          # only one of two gold spans found
    "third?": [],                                          # nothing found
}


# ------------------------------------------------------------ run_retriever
def test_one_result_per_question_in_order_and_search_gets_text_and_k():
    r = Scripted(ANSWERS)
    results = run_retriever(r, [Q1, Q2, Q3], k=10)
    assert [res.question for res in results] == [Q1, Q2, Q3]
    assert r.calls == [("first?", 10), ("second?", 10), ("third?", 10)]
    assert isinstance(results[0], QuestionResult)
    assert results[0].hits == (H("b.md", 1, 5), H("a.md", 15, 25))      # a tuple of Hits


def test_hits_are_cut_to_k_even_if_the_retriever_returns_more():
    many = {"first?": [H(f"x{i}.md", 1, 1) for i in range(10)]}
    results = run_retriever(Scripted(many), [Q1], k=3)
    assert len(results[0].hits) == 3


def test_latency_is_measured_in_milliseconds():
    results = run_retriever(Scripted(ANSWERS, delay=0.03), [Q1], k=5)
    assert 25 <= results[0].latency_ms < 2000


# ---------------------------------------------------------------- summarize
@pytest.fixture
def results():
    return [
        QuestionResult(Q1, (H("b.md", 1, 5), H("a.md", 15, 25)), 10.0),
        QuestionResult(Q2, (H("c.py", 5, 8),), 20.0),
        QuestionResult(Q3, (), 30.0),
    ]


def test_overall_numbers_worked_out_by_hand(results):
    o = summarize(results)["overall"]
    assert o["n"] == 3
    assert o["hit@1"] == pytest.approx(1 / 3)          # only q2 has gold at rank 1
    assert o["hit@5"] == pytest.approx(2 / 3)
    assert o["hit@10"] == pytest.approx(2 / 3)
    assert o["recall@5"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert o["mrr"] == pytest.approx((0.5 + 1 + 0) / 3)
    ndcg_q1 = 1 / 1.5849625007211562                    # 1 / log2(3)
    ndcg_q2 = 1 / (1 + 0.6309297535714574)              # DCG 1, ideal 1 + 1/log2(3)
    assert o["ndcg@5"] == pytest.approx((ndcg_q1 + ndcg_q2 + 0) / 3)


def test_numbers_are_split_by_category(results):
    by = summarize(results)["by_category"]
    assert set(by) == {"lexical", "code"}
    assert by["lexical"]["n"] == 2
    assert by["lexical"]["hit@5"] == pytest.approx(0.5)
    assert by["lexical"]["mrr"] == pytest.approx(0.25)
    assert by["code"]["n"] == 1
    assert by["code"]["hit@5"] == 1.0
    assert by["code"]["recall@5"] == pytest.approx(0.5)


def test_misses_lists_questions_with_no_hit_at_the_largest_k(results):
    assert summarize(results)["misses"] == ["q3"]


def test_custom_cutoffs(results):
    o = summarize(results, ks=(2,))["overall"]
    assert {"hit@2", "recall@2", "ndcg@2", "mrr", "n"} <= set(o)
    assert "hit@5" not in o
    assert o["hit@2"] == pytest.approx(2 / 3)


def test_latency_percentiles_use_nearest_rank():
    four = [QuestionResult(Q1, (), ms) for ms in (10.0, 20.0, 30.0, 40.0)]
    o = summarize(four)["overall"]
    assert o["latency_p50_ms"] == 20.0
    assert o["latency_p95_ms"] == 40.0


def test_summarize_rejects_no_results():
    with pytest.raises(ValueError):
        summarize([])


# ------------------------------------------------------------ format_report
def test_report_mentions_the_essentials(results):
    text = format_report(summarize(results))
    for expected in ("overall", "lexical", "code", "hit@5", "0.667", "q3", "latency"):
        assert expected in text, f"{expected!r} missing from the report:\n{text}"
