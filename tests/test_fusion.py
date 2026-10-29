"""Phase 6, step 1: fusing two ranked lists into one (`src/cairn/retrieval/fusion.py`).

BM25 scores are unbounded (10, 25, ...) and cosine scores sit between -1 and 1, so the two cannot be
added: BM25 would always win. Two ways round that:

    rrf(rankings, k=60) -> [(key, score), ...]
        Reciprocal Rank Fusion. `rankings` is a list of ranked lists of keys, best first. Only
        the RANK of a key counts: a key at (1-based) rank r in a list adds 1 / (k + r), and the
        scores of all the lists are summed. A key missing from a list adds nothing for it. A key
        repeated inside one list counts once, at its best rank. Result: every key once, best
        first.

    normalize(scores, method="minmax") -> [float, ...]
        Puts one list of scores on a common scale, keeping their order.
          "minmax": (s - lowest) / (highest - lowest), so the best is 1 and the worst 0.
                    All scores equal (or only one score): every score becomes 1.0.
          "zscore": (s - mean) / std, with the population std (divide by n). All equal: all 0.0.
          "none":   the scores unchanged (as floats): the broken baseline.
        An empty list gives an empty list. Any other method is a ValueError.

    weighted_fusion(scored, weights=None, method="minmax") -> [(key, score), ...]
        `scored` is a list of ranked lists of (key, score), best first. Each list's scores are
        normalised on their own, multiplied by the list's weight (default: 1 for every list) and
        summed per key. A key missing from a list adds 0 for it. A key repeated inside one list uses
        its first score. Result: every key once, best first.

Ties (in all three) go to the key met first when reading the lists top to bottom, one list after
the other. `weights` of the wrong length, or a negative weight, is a ValueError (as is k < 0 in
`rrf`).
"""

import math

import pytest

from cairn.retrieval.fusion import normalize, rrf, weighted_fusion


def keys(pairs):
    return [key for key, _ in pairs]


def scores(pairs):
    return [score for _, score in pairs]


# ------------------------------------------------------------------------------- rrf
def test_one_list_keeps_its_order_and_scores_one_over_k_plus_rank():
    result = rrf([["a", "b", "c"]], k=60)
    assert keys(result) == ["a", "b", "c"]
    assert scores(result) == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_the_default_k_is_60():
    assert scores(rrf([["a"]])) == pytest.approx([1 / 61])


def test_scores_of_the_lists_are_added():
    result = dict(rrf([["a", "b"], ["b", "a"]], k=60))
    assert result["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert result["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_agreement_beats_a_single_first_place():
    # x is first in one list only; y is second in one list and first in the other
    result = rrf([["x", "y", "z"], ["y", "w", "v"]], k=60)
    assert keys(result)[:2] == ["y", "x"]
    assert dict(result)["y"] == pytest.approx(1 / 62 + 1 / 61)


def test_a_key_missing_from_a_list_just_adds_nothing_for_it():
    result = dict(rrf([["a", "b"], ["c"]], k=10))
    assert result["a"] == pytest.approx(1 / 11)
    assert result["c"] == pytest.approx(1 / 11)
    assert result["b"] == pytest.approx(1 / 12)


def test_k_zero_is_plain_reciprocal_rank():
    assert scores(rrf([["a", "b", "c", "d"]], k=0)) == pytest.approx([1, 1 / 2, 1 / 3, 1 / 4])


def test_a_big_k_flattens_the_scores():
    low = scores(rrf([["a", "b"]], k=1))
    high = scores(rrf([["a", "b"]], k=1000))
    assert low[0] / low[1] == pytest.approx(3 / 2)
    assert high[0] / high[1] == pytest.approx(1002 / 1001)


def test_ties_go_to_the_key_met_first():
    # a and b have exactly the same score; a is met first, then b
    assert keys(rrf([["a", "b"], ["b", "a"]])) == ["a", "b"]
    # a and c tie at 1/61 + 0 and 0 + 1/61; the first list is read first
    assert keys(rrf([["c"], ["a"]]))[:2] == ["c", "a"]
    # c and d tie lower down
    assert keys(rrf([["a", "b", "c"], ["b", "a", "d"]])) == ["a", "b", "c", "d"]


def test_ties_are_decided_by_where_a_key_was_FIRST_met_not_last():
    # with k = 0 all three keys score exactly 1.0: b (rank 1), a (1/2 + 1/2) and x (rank 1)
    result = rrf([["b", "a"], ["x", "a"]], k=0)
    assert scores(result) == [1.0, 1.0, 1.0]
    assert keys(result) == ["b", "a", "x"]


def test_a_key_repeated_in_a_list_counts_once_at_its_best_rank():
    result = rrf([["a", "a", "b"]], k=60)
    assert keys(result) == ["a", "b"]
    assert scores(result) == pytest.approx([1 / 61, 1 / 62])  # b is the second KEY, rank 2


def test_every_key_appears_once():
    result = rrf([["a", "b", "c"], ["c", "d"], ["d", "a", "e"]])
    assert sorted(keys(result)) == ["a", "b", "c", "d", "e"]


def test_keys_can_be_tuples():
    result = rrf([[("a.md", 1, 5), ("b.md", 3, 9)], [("b.md", 3, 9)]])
    assert keys(result) == [("b.md", 3, 9), ("a.md", 1, 5)]


def test_nothing_in_nothing_out():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_the_scores_are_floats_and_the_input_is_not_changed():
    lists = [["a", "b"], ["b"]]
    result = rrf(lists)
    assert all(type(s) is float for s in scores(result))
    assert lists == [["a", "b"], ["b"]]


def test_a_negative_k_is_refused():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)


# ------------------------------------------------------------------------- normalize
def test_minmax_puts_the_worst_at_0_and_the_best_at_1():
    assert normalize([2, 4, 6]) == pytest.approx([0.0, 0.5, 1.0])
    assert normalize([6, 2, 4]) == pytest.approx([1.0, 0.0, 0.5])  # the order is kept


def test_minmax_works_with_negative_scores_like_a_cross_encoders():
    assert normalize([-10, 0, 10], "minmax") == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_of_equal_scores_or_one_score_is_all_ones():
    assert normalize([3, 3, 3]) == [1.0, 1.0, 1.0]
    assert normalize([5]) == [1.0]


def test_zscore_centres_on_zero_with_the_population_std():
    result = normalize([1, 2, 3], "zscore")  # mean 2, std sqrt(2/3)
    assert result == pytest.approx([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    assert sum(result) == pytest.approx(0.0)


def test_zscore_of_equal_scores_is_all_zeros():
    assert normalize([4, 4], "zscore") == [0.0, 0.0]
    assert normalize([4], "zscore") == [0.0]


def test_none_returns_the_scores_unchanged_as_floats():
    result = normalize([12, 3, -1], "none")
    assert result == [12.0, 3.0, -1.0]
    assert all(type(s) is float for s in result)


def test_an_empty_list_gives_an_empty_list_for_every_method():
    assert [normalize([], m) for m in ("minmax", "zscore", "none")] == [[], [], []]


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        normalize([1, 2], "softmax")


def test_normalize_does_not_change_its_input():
    original = [3, 1, 2]
    normalize(original)
    assert original == [3, 1, 2]


# ------------------------------------------------------------------- weighted_fusion
BM25 = [("a", 10.0), ("b", 5.0), ("c", 0.0)]  # minmax: 1, 0.5, 0
DENSE = [("b", 0.9), ("a", 0.5), ("d", 0.1)]  # minmax: 1, 0.5, 0


def test_equal_weights_add_the_normalised_scores():
    result = weighted_fusion([BM25, DENSE])
    assert dict(result) == pytest.approx({"a": 1.5, "b": 1.5, "c": 0.0, "d": 0.0})
    assert keys(result) == ["a", "b", "c", "d"]  # ties: met first


def test_weights_scale_each_list():
    assert dict(weighted_fusion([BM25, DENSE], [3, 1])) == pytest.approx(
        {"a": 3.5, "b": 2.5, "c": 0.0, "d": 0.0}
    )
    only_dense = weighted_fusion([BM25, DENSE], [0, 1])
    assert keys(only_dense)[:2] == ["b", "a"]
    only_bm25 = weighted_fusion([BM25, DENSE], [1, 0])
    assert keys(only_bm25)[:2] == ["a", "b"]


def test_a_key_missing_from_a_list_adds_zero_for_it():
    lists = [[("a", 1.0), ("b", 0.0)], [("c", 5.0)]]
    result = weighted_fusion(lists)
    assert dict(result) == pytest.approx({"a": 1.0, "b": 0.0, "c": 1.0})  # c alone: 1.0
    assert keys(result) == ["a", "c", "b"]


def test_zscore_fusion():
    result = weighted_fusion([[("a", 1.0), ("b", 2.0), ("c", 3.0)]], method="zscore")
    assert keys(result) == ["c", "b", "a"]
    assert scores(result) == pytest.approx([math.sqrt(1.5), 0.0, -math.sqrt(1.5)])


def test_the_raw_sum_lets_the_bigger_scale_win_and_normalising_does_not():
    bm25 = [("a", 12.0), ("b", 11.0), ("c", 3.0)]
    dense = [("b", 0.9), ("a", 0.1), ("c", 0.0)]
    assert keys(weighted_fusion([bm25, dense], method="none"))[0] == "a"  # 12.1 against 11.9
    assert keys(weighted_fusion([bm25, dense], method="minmax"))[0] == "b"  # 1.889 against 1.111


def test_a_key_repeated_in_a_list_uses_its_first_score():
    lists = [[("a", 4.0), ("b", 2.0), ("a", 0.0)]]
    result = weighted_fusion(lists, method="none")
    assert dict(result) == {"a": 4.0, "b": 2.0}


def test_weighted_keys_can_be_tuples_and_empty_input_gives_nothing():
    assert keys(weighted_fusion([[(("a.md", 1, 2), 3.0)]])) == [("a.md", 1, 2)]
    assert weighted_fusion([]) == []
    assert weighted_fusion([[], []]) == []


def test_weights_must_match_the_lists_and_not_be_negative():
    with pytest.raises(ValueError):
        weighted_fusion([BM25, DENSE], [1])
    with pytest.raises(ValueError):
        weighted_fusion([BM25, DENSE], [1, -1])


def test_weighted_fusion_refuses_an_unknown_method():
    with pytest.raises(ValueError):
        weighted_fusion([BM25], method="rank")


def test_weighted_fusion_does_not_change_its_input():
    lists = [list(BM25), list(DENSE)]
    weighted_fusion(lists)
    assert lists == [BM25, DENSE]
