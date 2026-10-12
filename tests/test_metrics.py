"""Phase 1, step 3: the evaluation metrics. Every expected number below was worked out by hand.

Vocabulary
- gold: the list of Spans that contain the answer to a question.
- hits: the ranked list of Hits a retriever returned (best first).
- A hit "overlaps" a gold span when they are in the same file and their line ranges share a line.
- Credit rule: walking down the ranking, a hit is RELEVANT only if it overlaps at least one gold
  span that no earlier hit had already covered. Repeats of an already-covered span earn nothing.
"""

import math

import pytest

from cairn.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    overlaps,
    percentile,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
)
from cairn.types import Hit, Span


def H(path, a, b):
    return Hit(path=path, start_line=a, end_line=b)


def S(path, a, b):
    return Span(path, a, b)


# ---------------------------------------------------------------- overlaps
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (("f.md", 1, 10), ("f.md", 5, 15), True),  # partial overlap
        (("f.md", 1, 10), ("f.md", 10, 20), True),  # share exactly one line (ranges are inclusive)
        (("f.md", 1, 10), ("f.md", 11, 20), False),  # adjacent, no shared line
        (("f.md", 1, 10), ("g.md", 1, 10), False),  # same lines, different file
        (("f.md", 1, 100), ("f.md", 40, 50), True),  # one contains the other
        (("f.md", 7, 7), ("f.md", 7, 7), True),  # single lines
        (("f.md", 7, 7), ("f.md", 8, 8), False),
    ],
)
def test_overlaps_is_symmetric_and_inclusive(a, b, expected):
    assert overlaps(H(*a), S(*b)) is expected
    assert overlaps(S(*b), H(*a)) is expected


# --------------------------------------------------------- relevance_flags
def test_flags_for_no_hits_is_empty():
    assert relevance_flags([], [S("f.md", 1, 5)]) == []


def test_flags_miss_then_hit():
    hits = [H("x.md", 1, 5), H("f.md", 3, 8)]
    assert relevance_flags(hits, [S("f.md", 1, 5)]) == [False, True]


def test_repeat_of_a_covered_span_earns_nothing():
    hits = [H("f.md", 1, 3), H("f.md", 4, 6)]  # both overlap the same gold span
    assert relevance_flags(hits, [S("f.md", 1, 10)]) == [True, False]


def test_one_hit_can_cover_two_gold_spans():
    hits = [H("f.md", 1, 100), H("f.md", 5, 6)]
    gold = [S("f.md", 2, 3), S("f.md", 50, 60)]
    assert relevance_flags(hits, gold) == [True, False]


def test_two_hits_covering_two_gold_spans_are_both_relevant():
    hits = [H("f.md", 2, 3), H("g.md", 9, 9)]
    gold = [S("f.md", 1, 5), S("g.md", 8, 10)]
    assert relevance_flags(hits, gold) == [True, True]


# --------------------------------------------------------------- hit_at_k
def test_hit_at_k_found_within_k():
    hits = [H("x.md", 1, 2), H("f.md", 1, 2)]
    assert hit_at_k(hits, [S("f.md", 1, 5)], k=2) == 1.0


def test_hit_at_k_gold_at_rank_k_plus_one_is_a_miss():
    hits = [H("x.md", 1, 2), H("y.md", 1, 2), H("f.md", 1, 2)]
    assert hit_at_k(hits, [S("f.md", 1, 5)], k=2) == 0.0
    assert hit_at_k(hits, [S("f.md", 1, 5)], k=3) == 1.0


def test_hit_at_k_no_hits_and_bad_k():
    assert hit_at_k([], [S("f.md", 1, 5)], k=5) == 0.0
    with pytest.raises(ValueError):
        hit_at_k([], [S("f.md", 1, 5)], k=0)


# ------------------------------------------------------------- recall_at_k
def test_recall_single_gold_found():
    assert recall_at_k([H("f.md", 2, 3)], [S("f.md", 1, 5)], k=1) == 1.0


def test_recall_half_the_gold_found():
    gold = [S("f.md", 1, 5), S("g.md", 1, 5)]
    assert recall_at_k([H("f.md", 2, 3)], gold, k=5) == 0.5


def test_recall_respects_the_cutoff():
    gold = [S("f.md", 1, 5), S("g.md", 1, 5)]
    hits = [H("f.md", 1, 2), H("x.md", 1, 2), H("g.md", 1, 2)]
    assert recall_at_k(hits, gold, k=2) == 0.5
    assert recall_at_k(hits, gold, k=3) == 1.0


def test_recall_one_hit_covering_both_gold_spans():
    gold = [S("f.md", 2, 3), S("f.md", 50, 60)]
    assert recall_at_k([H("f.md", 1, 100)], gold, k=1) == 1.0


def test_recall_rejects_empty_gold_and_bad_k():
    with pytest.raises(ValueError):
        recall_at_k([H("f.md", 1, 2)], [], k=1)
    with pytest.raises(ValueError):
        recall_at_k([H("f.md", 1, 2)], [S("f.md", 1, 2)], k=0)


# --------------------------------------------------------- reciprocal_rank
def test_reciprocal_rank_first_place():
    assert reciprocal_rank([H("f.md", 1, 2)], [S("f.md", 1, 5)]) == 1.0


def test_reciprocal_rank_fourth_place():
    hits = [H("a.md", 1, 1), H("b.md", 1, 1), H("c.md", 1, 1), H("f.md", 1, 2)]
    assert reciprocal_rank(hits, [S("f.md", 1, 5)]) == 0.25


def test_reciprocal_rank_none_found_is_zero():
    assert reciprocal_rank([H("a.md", 1, 1)], [S("f.md", 1, 5)]) == 0.0
    assert reciprocal_rank([], [S("f.md", 1, 5)]) == 0.0


def test_reciprocal_rank_cutoff():
    hits = [H("a.md", 1, 1), H("b.md", 1, 1), H("c.md", 1, 1), H("f.md", 1, 2)]
    assert reciprocal_rank(hits, [S("f.md", 1, 5)], k=3) == 0.0
    assert reciprocal_rank(hits, [S("f.md", 1, 5)], k=4) == 0.25


# ------------------------------------------------------------- ndcg_at_k
def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([H("f.md", 1, 2)], [S("f.md", 1, 5)], k=5) == pytest.approx(1.0)


def test_ndcg_single_gold_at_rank_three():
    hits = [H("a.md", 1, 1), H("b.md", 1, 1), H("f.md", 1, 2)]
    # DCG = 1/log2(3+1) = 0.5, ideal DCG = 1/log2(1+1) = 1
    assert ndcg_at_k(hits, [S("f.md", 1, 5)], k=5) == pytest.approx(0.5)


def test_ndcg_two_gold_spans_with_a_miss_between():
    gold = [S("f.md", 1, 5), S("g.md", 1, 5)]
    hits = [H("f.md", 1, 2), H("x.md", 1, 2), H("g.md", 1, 2)]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # = 1.5
    ideal = 1 / math.log2(2) + 1 / math.log2(3)  # both gold in ranks 1 and 2
    assert ndcg_at_k(hits, gold, k=3) == pytest.approx(dcg / ideal)


def test_ndcg_repeats_of_a_covered_span_never_push_it_above_one():
    hits = [H("f.md", 1, 2), H("f.md", 3, 4)]
    assert ndcg_at_k(hits, [S("f.md", 1, 10)], k=2) == pytest.approx(1.0)


def test_ndcg_ideal_is_capped_by_k():
    gold = [S("a.md", 1, 1), S("b.md", 1, 1), S("c.md", 1, 1)]
    hits = [H("a.md", 1, 1), H("b.md", 1, 1)]
    assert ndcg_at_k(hits, gold, k=2) == pytest.approx(1.0)  # best possible with only 2 slots


def test_ndcg_nothing_found_and_cutoff():
    assert ndcg_at_k([H("x.md", 1, 1)], [S("f.md", 1, 5)], k=5) == 0.0
    hits = [H("a.md", 1, 1), H("b.md", 1, 1), H("c.md", 1, 1), H("f.md", 1, 2)]
    assert ndcg_at_k(hits, [S("f.md", 1, 5)], k=3) == 0.0


def test_ndcg_rejects_empty_gold_and_bad_k():
    with pytest.raises(ValueError):
        ndcg_at_k([], [], k=5)
    with pytest.raises(ValueError):
        ndcg_at_k([], [S("f.md", 1, 2)], k=0)


# -------------------------------------------------------------- percentile
@pytest.mark.parametrize(
    "values,p,expected",
    [
        ([10, 20, 30, 40], 50, 20),
        ([10, 20, 30, 40], 95, 40),
        ([40, 10, 30, 20], 50, 20),  # input order does not matter
        (list(range(1, 101)), 95, 95),
        (list(range(1, 101)), 100, 100),
        ([7], 50, 7),
        ([5, 1], 25, 1),
    ],
)
def test_percentile_nearest_rank(values, p, expected):
    assert percentile(values, p) == expected


@pytest.mark.parametrize("values,p", [([], 50), ([1, 2], 0), ([1, 2], 101), ([1, 2], -5)])
def test_percentile_rejects_bad_input(values, p):
    with pytest.raises(ValueError):
        percentile(values, p)
