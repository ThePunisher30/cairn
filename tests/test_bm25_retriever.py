"""Phase 3, step 4: BM25Retriever.search, the whole formula over the whole index.

Toy corpus (the same as in the index tests), with k1 = 1.2 and b = 0.75:
    chunk 0: "apple banana apple"        length 3
    chunk 1: "banana cherry"             length 2
    chunk 2: "apple apple apple dog"     length 4        N = 3, average length = 3.0
    idf: apple = banana = ln(1 + 1.5/2.5) = 0.470004      cherry = dog = ln(1 + 2.5/1.5) = 0.980829

Hand-worked scores (idf * tf_weight, summed over the distinct query terms):
    "apple"           chunk 2: 0.470004 * 1.466667 = 0.689339
                      chunk 0: 0.470004 * 1.375    = 0.646255
    "banana cherry"   chunk 1: (0.470004 + 0.980829) * 1.157895 = 1.679912
                      chunk 0: 0.470004 * 1.0 = 0.470004
    "dog"             chunk 2: 0.980829 * 0.88 = 0.863130
"""

import dataclasses
import time

import pytest

from cairn.evaluation.harness import run_retriever, summarize
from cairn.ingest.chunkers import make_chunk
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import Tokenizer
from cairn.types import Hit, Question, Span


def chunk(number, text, heading_path=(), path="a.md"):
    return make_chunk(path, number + 1, number + 1, text, "prose", heading_path)


@pytest.fixture
def toy():
    return [
        chunk(0, "apple banana apple"),
        chunk(1, "banana cherry"),
        chunk(2, "apple apple apple dog"),
    ]


def retriever(chunks, tokenizer=None, **options):
    return BM25Retriever(InvertedIndex.build(chunks, tokenizer), chunks, **options)


def scores(hits):
    return [round(hit.score, 4) for hit in hits]


def where(hits):
    return [hit.start_line for hit in hits]  # chunk number n sits on line n + 1


# ------------------------------------------------------------------ the numbers
def test_scores_match_the_hand_worked_values(toy):
    search = retriever(toy).search
    assert scores(search("apple", 10)) == [0.6893, 0.6463]
    assert where(search("apple", 10)) == [3, 1]
    assert scores(search("banana cherry", 10)) == [1.6799, 0.4700]
    assert where(search("banana cherry", 10)) == [2, 1]
    assert scores(search("dog", 10)) == [0.8631]
    assert scores(search("apple dog", 10)) == [1.5525, 0.6463]


def test_results_come_best_first(toy):
    hits = retriever(toy).search("apple banana cherry dog", 10)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_chunks_without_any_query_word_are_not_returned(toy):
    assert where(retriever(toy).search("dog", 10)) == [3]
    assert where(retriever(toy).search("cherry", 10)) == [2]


def test_a_query_with_no_known_words_finds_nothing(toy):
    search = retriever(toy).search
    assert search("zebra", 10) == []
    assert search("", 10) == []
    assert search("!!! ???", 10) == []


def test_k_limits_the_results(toy):
    search = retriever(toy).search
    assert len(search("apple banana", 1)) == 1
    assert len(search("apple banana", 2)) == 2
    assert len(search("apple banana", 3)) == 3
    assert len(search("apple banana", 100)) == 3  # only 3 chunks exist
    assert where(search("apple banana", 1)) == [where(search("apple banana", 10))[0]]


def test_k_zero_or_less_gives_nothing(toy):
    assert retriever(toy).search("apple", 0) == []
    assert retriever(toy).search("apple", -3) == []


def test_a_hit_carries_the_location_the_score_and_the_text_of_its_chunk():
    chunks = [
        make_chunk("docs/x.md", 10, 14, "alpha beta", "prose"),
        make_chunk("fastapi/y.py", 3, 9, "def gamma(): pass", "code"),
    ]
    (hit,) = retriever(chunks).search("gamma", 5)
    assert isinstance(hit, Hit)
    assert (hit.path, hit.start_line, hit.end_line) == ("fastapi/y.py", 3, 9)
    assert hit.text == "def gamma(): pass"
    assert hit.score > 0


# -------------------------------------------------------------------- behaviour
def test_equal_scores_are_ranked_by_chunk_number():
    chunks = [chunk(i, "same words here") for i in range(5)]
    assert where(retriever(chunks).search("words", 10)) == [1, 2, 3, 4, 5]
    assert where(retriever(chunks).search("words", 3)) == [1, 2, 3]


def test_repeating_a_word_in_the_query_changes_nothing(toy):
    search = retriever(toy).search
    assert search("apple apple apple", 10) == search("apple", 10)


def test_the_query_is_lowercased_and_tokenized_like_the_chunks(toy):
    search = retriever(toy).search
    assert search("APPLE!", 10) == search("apple", 10)


def test_a_rare_word_beats_a_common_word():
    chunks = [chunk(0, "common filler"), chunk(1, "common words"), chunk(2, "rare words")]
    # "common" is in 2 of 3 chunks, "rare" in 1; chunks 0 and 2 have the same length and tf
    hits = retriever(chunks).search("common rare", 10)
    assert hits[0].start_line == 3  # chunk 2 (the one with "rare")


def test_a_short_chunk_beats_a_long_one_with_the_same_word_count():
    chunks = [chunk(0, "needle " + "pad " * 30), chunk(1, "needle pad"), chunk(2, "other text")]
    hits = retriever(chunks).search("needle", 10)
    assert where(hits) == [2, 1]


def test_b_zero_ignores_chunk_length():
    chunks = [chunk(0, "needle " + "pad " * 30), chunk(1, "needle pad"), chunk(2, "other text")]
    hits = retriever(chunks, b=0).search("needle", 10)
    assert hits[0].score == pytest.approx(hits[1].score)
    assert where(hits) == [1, 2]  # equal scores: the lower chunk number first


def test_more_occurrences_rank_higher_but_with_diminishing_returns():
    chunks = [chunk(0, "t " + "x " * 9), chunk(1, "t t " + "x " * 8), chunk(2, "t " * 10)]
    hits = retriever(chunks, b=0).search("t", 10)  # b = 0: all three have length 10
    assert where(hits) == [3, 2, 1]
    assert hits[0].score / hits[2].score < 2.2  # 10 occurrences are not 10 times better


def test_k1_zero_ignores_how_often_a_word_occurs():
    chunks = [chunk(0, "t x x"), chunk(1, "t t x"), chunk(2, "t t t")]
    hits = retriever(chunks, k1=0).search("t", 10)
    assert hits[0].score == pytest.approx(hits[1].score) == pytest.approx(hits[2].score)


def test_the_query_uses_the_tokenizer_the_index_was_built_with():
    chunks = [chunk(0, "def get_request_handler(): pass"), chunk(1, "nothing to see")]
    assert where(retriever(chunks).search("request", 5)) == [1]  # parts are indexed
    plain = retriever(chunks, tokenizer=Tokenizer(split_identifiers=False))
    assert plain.search("request", 5) == []  # parts are not indexed, and the query is not split
    assert where(plain.search("get_request_handler", 5)) == [1]


def test_a_query_is_split_exactly_when_the_index_was_split():
    chunks = [chunk(0, "request handler"), chunk(1, "get_request_handler")]
    split = retriever(chunks)  # the index splits identifiers, so the query does too
    assert where(split.search("get_request_handler", 5)) == [2, 1]  # chunk 0 matches on the parts
    plain = retriever(chunks, tokenizer=Tokenizer(split_identifiers=False))
    assert where(plain.search("get_request_handler", 5)) == [2]  # no parts: only the whole name


def test_the_heading_path_makes_a_chunk_findable():
    chunks = [
        chunk(0, "unrelated words here", heading_path=("OAuth2 scopes", "Check it")),
        chunk(1, "other text"),
    ]
    assert where(retriever(chunks).search("scopes", 5)) == [1]


def test_hits_work_with_the_evaluation_harness(toy):
    questions = [
        Question("q1", "lexical", "dev", "dog", (Span("a.md", 3, 3),)),
        Question("q2", "lexical", "dev", "cherry", (Span("a.md", 2, 2),)),
        Question("q3", "lexical", "dev", "zebra", (Span("a.md", 1, 1),)),
    ]
    summary = summarize(run_retriever(retriever(toy), questions, k=5))
    assert summary["overall"]["hit@1"] == pytest.approx(2 / 3)
    assert summary["misses"] == ["q3"]


# ------------------------------------------------------------------ the knobs
def test_the_knobs_are_checked():
    with pytest.raises(ValueError):
        retriever([chunk(0, "x")], k1=-0.1)
    with pytest.raises(ValueError):
        retriever([chunk(0, "x")], b=-0.1)
    with pytest.raises(ValueError):
        retriever([chunk(0, "x")], b=1.1)
    retriever([chunk(0, "x")], k1=0, b=0)  # the edges are allowed
    retriever([chunk(0, "x")], k1=5, b=1)


def test_the_chunks_must_be_the_ones_the_index_was_built_from(toy):
    index = InvertedIndex.build(toy)
    BM25Retriever(index, list(toy))  # fine
    with pytest.raises(ValueError):
        BM25Retriever(index, toy[:2])  # fewer chunks
    with pytest.raises(ValueError):
        BM25Retriever(index, [toy[1], toy[0], toy[2]])  # same chunks, different order
    other = [dataclasses.replace(c, id="x" + c.id) for c in toy]
    with pytest.raises(ValueError):
        BM25Retriever(index, other)  # different chunks


# ------------------------------------------------------------------- speed
def test_a_search_does_not_rescan_the_corpus():
    import random

    rng = random.Random(7)
    vocabulary = [f"word{i}" for i in range(300)]
    chunks = [
        make_chunk("a.md", n + 1, n + 1, " ".join(rng.choices(vocabulary, k=40)), "prose")
        for n in range(20_000)
    ]
    search = retriever(chunks).search
    queries = [" ".join(rng.choices(vocabulary, k=3)) for _ in range(20)]
    start = time.perf_counter()
    for query in queries:
        assert len(search(query, 10)) == 10
        # fail fast instead of making you wait: 20 searches should take well under a second
        assert time.perf_counter() - start < 4.0, "too slow: is work repeated for every posting?"
