"""Phase 6, step 3: the reranker (`src/cairn/retrieval/rerank.py`).

A cross-encoder reads the question and one passage TOGETHER and says how well the passage answers
it. It is far more accurate than comparing two separate vectors and far slower (one model run per
pair), so it only re-orders a short list that a cheaper retriever has already found.

    Reranker(model, name, *, batch_size=16, max_chars=2000)
        `model.score(pairs) -> list[float]` takes a list of (query, passage) pairs and returns one
        score per pair, higher = better (see `rerank_models.py`: provided).
    reranker.score(query, passages) -> list[float]       one score per passage, in the same order
        - a passage longer than `max_chars` is cut to its first `max_chars` characters BEFORE
          anything else (the model reads at most 512 tokens anyway, and a cut passage is what the
          cache is keyed on)
        - a (name, query, passage) pair that was scored before is not scored again: scores are
          kept in the dict `reranker.cache`. `reranker.hits` counts the passages answered from it
          and `reranker.misses` the pairs sent to the model, over the reranker's whole life
        - a passage repeated inside one call (after cutting) is sent to the model once
        - the pairs that still need a score are sorted by passage length (shortest first, like the
          embedder: padding is wasted work), sent to the model in batches of `batch_size`, and the
          scores are put back in the order of `passages`
        - the scores returned are Python floats; the model returning the wrong number of scores
          for a batch is a ValueError; `batch_size < 1` or `max_chars < 1` is a ValueError

    RerankingRetriever(base, reranker, *, depth=30).search(query, k) -> [Hit, ...]
        asks `base` for `depth` hits, scores each hit's `text` against the query, and returns the
        best `k` by reranker score (equal scores keep the order `base` gave), with the reranker's
        score in `Hit.score` and the place and text of the hit unchanged. At most `depth` hits come
        back. `k <= 0` returns [] without asking anything. `depth < 1` is a ValueError.
"""

import numpy as np
import pytest

from cairn.retrieval.rerank import Reranker, RerankingRetriever
from cairn.retrieval.rerank_models import OverlapModel, load_reranker
from cairn.types import Hit


class ByLength:
    """A model whose score is the passage length, so a result can be checked against its input."""

    def __init__(self):
        self.batches = []

    def score(self, pairs):
        self.batches.append(list(pairs))
        return [float(len(passage)) for _, passage in pairs]


def sent(model):
    return [pair for batch in model.batches for pair in batch]


# ------------------------------------------------------------------------- scores
def test_scores_come_back_in_the_order_of_the_passages():
    reranker = Reranker(ByLength(), "m")
    assert reranker.score("q", ["ccc", "a", "bb"]) == [3.0, 1.0, 2.0]


def test_the_model_is_given_query_passage_pairs():
    model = ByLength()
    Reranker(model, "m").score("my question", ["one", "three"])
    assert sorted(sent(model)) == [("my question", "one"), ("my question", "three")]


def test_scores_are_python_floats_even_if_the_model_returns_numpy_ones():
    class Numpy:
        def score(self, pairs):
            return np.array([0.5] * len(pairs), dtype=np.float32)

    (score,) = Reranker(Numpy(), "m").score("q", ["a"])
    assert type(score) is float and score == 0.5


def test_no_passages_means_no_model_call():
    model = ByLength()
    assert Reranker(model, "m").score("q", []) == []
    assert model.batches == []


# --------------------------------------------------------------- cutting long passages
def test_a_long_passage_is_cut_before_scoring():
    model = ByLength()
    scores = Reranker(model, "m", max_chars=5).score("q", ["abcdefghij", "xy"])
    assert scores == [5.0, 2.0]
    assert sorted(p for _, p in sent(model)) == ["abcde", "xy"]


def test_passages_that_are_equal_after_cutting_are_scored_once():
    model = ByLength()
    scores = Reranker(model, "m", max_chars=4).score("q", ["aaaaX", "aaaaY"])
    assert scores == [4.0, 4.0]
    assert len(sent(model)) == 1


# ---------------------------------------------------------------------------- batching
def test_pairs_are_sent_in_batches_of_batch_size():
    model = ByLength()
    Reranker(model, "m", batch_size=3).score("q", ["a", "b", "c", "d", "e", "f", "g"])
    assert [len(batch) for batch in model.batches] == [3, 3, 1]


def test_pairs_go_to_the_model_shortest_first_and_the_answer_is_still_in_input_order():
    model = ByLength()
    passages = ["aaaaaaaa", "a", "aaaa", "aa", "aaaaaa"]
    scores = Reranker(model, "m", batch_size=2).score("q", passages)
    assert scores == [8.0, 1.0, 4.0, 2.0, 6.0]
    lengths = [len(p) for _, p in sent(model)]
    assert lengths == sorted(lengths)


def test_a_repeated_passage_is_scored_once_and_scored_in_every_place():
    model = ByLength()
    assert Reranker(model, "m").score("q", ["ab", "c", "ab"]) == [2.0, 1.0, 2.0]
    assert len(sent(model)) == 2


def test_a_model_that_returns_the_wrong_number_of_scores_is_an_error():
    class Broken:
        def score(self, pairs):
            return [1.0]

    with pytest.raises(ValueError):
        Reranker(Broken(), "m").score("q", ["a", "b"])


# ---------------------------------------------------------------------------- the cache
def test_a_pair_scored_before_is_not_scored_again():
    model = ByLength()
    reranker = Reranker(model, "m")
    first = reranker.score("q", ["aa", "bbb"])
    second = reranker.score("q", ["bbb", "aa"])
    assert first == [2.0, 3.0] and second == [3.0, 2.0]
    assert len(model.batches) == 1  # the second call did not reach the model


def test_the_counters_say_what_was_cached_and_what_was_scored():
    reranker = Reranker(ByLength(), "m")
    reranker.score("q", ["a", "bb"])
    assert (reranker.hits, reranker.misses) == (0, 2)
    reranker.score("q", ["a", "bb", "ccc"])
    assert (reranker.hits, reranker.misses) == (2, 3)


def test_the_same_passage_for_another_query_is_scored_again():
    model = ByLength()
    reranker = Reranker(model, "m")
    reranker.score("one question", ["passage"])
    reranker.score("another question", ["passage"])
    assert len(sent(model)) == 2


def test_the_cache_is_a_dict_and_the_name_is_part_of_its_keys():
    first, second = Reranker(ByLength(), "model-a"), Reranker(ByLength(), "model-b")
    first.score("q", ["abc"])
    second.score("q", ["abc"])
    assert isinstance(first.cache, dict) and len(first.cache) == 1
    assert set(first.cache) != set(second.cache)  # a shared dict could not mix the two up


def test_only_new_pairs_reach_the_model_when_some_are_cached():
    model = ByLength()
    reranker = Reranker(model, "m")
    reranker.score("q", ["a", "bb"])
    reranker.score("q", ["a", "bb", "ccc"])
    assert [p for _, p in model.batches[1]] == ["ccc"]


@pytest.mark.parametrize("arguments", [{"batch_size": 0}, {"max_chars": 0}])
def test_bad_settings_are_refused(arguments):
    with pytest.raises(ValueError):
        Reranker(ByLength(), "m", **arguments)


# --------------------------------------------------------------- the offline model
def test_the_offline_overlap_model_scores_the_share_of_query_words_found():
    model = OverlapModel()
    scores = model.score(
        [("apple banana", "an apple a day"), ("apple banana", "APPLE, banana!"), ("x", "y")]
    )
    assert scores == [0.5, 1.0, 0.0]


def test_load_reranker_builds_a_reranker_around_the_offline_model():
    reranker = load_reranker("overlap", batch_size=4)
    assert isinstance(reranker, Reranker)
    assert (reranker.name, reranker.batch_size) == ("overlap", 4)
    assert reranker.score("apple", ["apple pie", "banana"]) == [1.0, 0.0]


# ----------------------------------------------------------------- RerankingRetriever
def hit(name, text, score=0.0):
    line = ord(name) - ord("a") + 1
    return Hit("f.md", line, line, score, text)


class Scripted:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.hits[:k]


def names(hits):
    return [chr(ord("a") + h.start_line - 1) for h in hits]


def test_the_reranker_has_the_last_word_on_the_order():
    base = Scripted([hit("a", "xx"), hit("b", "xxxx"), hit("c", "x")])
    result = RerankingRetriever(base, Reranker(ByLength(), "m")).search("q", 3)
    assert names(result) == ["b", "a", "c"]  # longest passage first


def test_scores_are_the_rerankers_and_place_and_text_are_kept():
    base = Scripted([hit("a", "xx", score=7.0), hit("b", "xxxx", score=3.0)])
    first, second = RerankingRetriever(base, Reranker(ByLength(), "m")).search("q", 2)
    assert (first.path, first.start_line, first.end_line, first.text) == ("f.md", 2, 2, "xxxx")
    assert (first.score, second.score) == (4.0, 2.0)


def test_it_asks_the_base_for_depth_hits_and_returns_k_of_them():
    base = Scripted([hit(c, "x" * i) for i, c in enumerate("abcdef", start=1)])
    result = RerankingRetriever(base, Reranker(ByLength(), "m"), depth=5).search("q", 2)
    assert base.calls == [("q", 5)]
    assert names(result) == ["e", "d"]  # f is beyond the depth and never seen


def test_equal_reranker_scores_keep_the_order_of_the_base():
    base = Scripted([hit("a", "same"), hit("b", "same"), hit("c", "same")])
    result = RerankingRetriever(base, Reranker(ByLength(), "m")).search("q", 3)
    assert names(result) == ["a", "b", "c"]


def test_k_larger_than_what_the_base_found_returns_all_of_it():
    base = Scripted([hit("a", "x"), hit("b", "xx")])
    assert len(RerankingRetriever(base, Reranker(ByLength(), "m")).search("q", 10)) == 2


def test_k_zero_or_less_asks_nobody():
    base, model = Scripted([hit("a", "x")]), ByLength()
    retriever = RerankingRetriever(base, Reranker(model, "m"))
    assert retriever.search("q", 0) == [] and retriever.search("q", -1) == []
    assert base.calls == [] and model.batches == []


def test_nothing_found_gives_nothing():
    assert RerankingRetriever(Scripted([]), Reranker(ByLength(), "m")).search("q", 3) == []


def test_depth_must_be_at_least_one():
    with pytest.raises(ValueError):
        RerankingRetriever(Scripted([]), Reranker(ByLength(), "m"), depth=0)


def test_repeating_a_search_does_not_score_again():
    model = ByLength()
    retriever = RerankingRetriever(Scripted([hit("a", "x"), hit("b", "xx")]), Reranker(model, "m"))
    retriever.search("q", 2)
    retriever.search("q", 2)
    assert len(model.batches) == 1


def test_it_moves_the_chunk_that_answers_the_question_to_the_top():
    base = Scripted(
        [
            hit("a", "ocean river mountain"),
            hit("b", "how to declare a query parameter in a path operation"),
            hit("c", "deploying with docker"),
        ]
    )
    reranker = load_reranker("overlap")
    result = RerankingRetriever(base, reranker).search("declare query parameter", 3)
    assert names(result)[0] == "b"
