"""Phase 4, step 4: building the dense index from chunks, and DenseRetriever.search.

    build_dense_index(chunks, embedder, cache=None) -> (FlatIndex, computed)
        embeds every chunk's index_text (heading path + text) and remembers each chunk's id
    DenseRetriever(embedder, index, chunks).search(query, k) -> [Hit, ...]
        embeds the query (with the query prefix), asks the index, returns Hits best first

These tests use the offline HashingModel, which counts words: texts that share words are similar.
With it, cosine similarity is easy to work out by hand:
    "apple"  vs  "apple apple apple dog"   = 3 / sqrt(1 * 10) = 0.949
    "apple"  vs  "apple banana apple"      = 2 / sqrt(1 *  5) = 0.894
"""

import numpy as np
import pytest

from cairn.dense.cache import EmbeddingCache
from cairn.dense.embedder import Embedder
from cairn.dense.models import HashingModel
from cairn.dense.retriever import DenseRetriever, build_dense_index
from cairn.evaluation.harness import run_retriever, summarize
from cairn.ingest.chunkers import make_chunk
from cairn.types import Hit, Question, Span


def chunk(number, text, heading_path=(), path="a.md"):
    return make_chunk(path, number + 1, number + 1, text, "prose", heading_path)


@pytest.fixture
def toy():
    return [
        chunk(0, "apple banana apple"),
        chunk(1, "banana cherry"),
        chunk(2, "apple apple apple dog"),
        chunk(3, "ocean river mountain"),
    ]


def make_embedder(model=None, query_prefix=""):
    return Embedder(model or HashingModel(), "hashing", query_prefix=query_prefix)


def make_retriever(chunks, embedder=None, cache=None):
    embedder = embedder or make_embedder()
    index, _ = build_dense_index(chunks, embedder, cache)
    return DenseRetriever(embedder, index, chunks)


def where(hits):
    return [hit.start_line for hit in hits]  # chunk number n sits on line n + 1


# --------------------------------------------------------------- building the index
def test_the_index_has_one_unit_vector_and_one_id_per_chunk(toy):
    index, computed = build_dense_index(toy, make_embedder())
    assert (index.n_docs, index.dim, computed) == (4, 256, 4)
    assert index.doc_ids == [c.id for c in toy]
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)


def test_a_cache_makes_the_second_build_free(toy):
    cache = EmbeddingCache()
    model = HashingModel()
    first, _ = build_dense_index(toy, make_embedder(model), cache)
    calls = len(model.batch_sizes)
    second, computed = build_dense_index(toy, make_embedder(model), cache)
    assert computed == 0 and len(model.batch_sizes) == calls
    assert np.array_equal(first.vectors, second.vectors)


def test_a_new_chunk_is_the_only_one_embedded_when_chunks_change(toy):
    cache = EmbeddingCache()
    build_dense_index(toy, make_embedder(), cache)
    _, computed = build_dense_index([*toy, chunk(4, "a brand new passage")], make_embedder(), cache)
    assert computed == 1


def test_the_heading_path_is_part_of_what_is_embedded():
    plain = chunk(0, "unrelated words here")
    titled = chunk(0, "unrelated words here", heading_path=("OAuth2 scopes", "Check it"))
    cache = EmbeddingCache()
    index, computed = build_dense_index([plain, titled], make_embedder(), cache)
    assert computed == 2  # same text, different heading: two different vectors
    assert not np.allclose(index.vectors[0], index.vectors[1])


def test_the_heading_path_makes_a_chunk_findable():
    chunks = [
        chunk(0, "unrelated words here", heading_path=("OAuth2 scopes", "Check it")),
        chunk(1, "other text entirely"),
    ]
    assert where(make_retriever(chunks).search("scopes", 5)[:1]) == [1]


def test_building_from_no_chunks_gives_an_empty_index():
    index, computed = build_dense_index([], make_embedder())
    assert (index.n_docs, computed) == (0, 0)


# ------------------------------------------------------------------- searching
def test_search_ranks_by_cosine_similarity(toy):
    hits = make_retriever(toy).search("apple", 10)
    assert where(hits[:2]) == [3, 1]  # "apple apple apple dog", then "apple banana apple"
    assert hits[0].score == pytest.approx(0.949, abs=1e-3)
    assert hits[1].score == pytest.approx(0.894, abs=1e-3)


def test_results_come_best_first(toy):
    hits = make_retriever(toy).search("apple banana cherry", 10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_the_results_and_k_zero_gives_nothing(toy):
    search = make_retriever(toy).search
    assert len(search("apple banana", 2)) == 2
    assert len(search("apple banana", 100)) == 4  # every chunk is returned: dense search ranks all
    assert search("apple", 0) == []
    assert search("apple", -1) == []


def test_a_hit_carries_the_location_the_score_and_the_text():
    chunks = [
        make_chunk("docs/x.md", 10, 14, "alpha beta", "prose"),
        make_chunk("fastapi/y.py", 3, 9, "def gamma(): pass", "code"),
    ]
    (top,) = make_retriever(chunks).search("gamma", 1)
    assert isinstance(top, Hit)
    assert (top.path, top.start_line, top.end_line) == ("fastapi/y.py", 3, 9)
    assert top.text == "def gamma(): pass"
    assert 0 < top.score <= 1.0001


def test_equal_scores_are_ranked_by_chunk_number():
    chunks = [chunk(i, "same words here") for i in range(4)]
    assert where(make_retriever(chunks).search("words", 3)) == [1, 2, 3]


def test_the_query_is_embedded_with_the_query_prefix_and_chunks_without_it():
    class Spy(HashingModel):
        def __init__(self):
            super().__init__()
            self.seen = []

        def encode(self, texts):
            self.seen.extend(texts)
            return super().encode(texts)

    spy = Spy()
    embedder = Embedder(spy, "spy", query_prefix="QUERY: ", doc_prefix="")
    retriever = make_retriever([chunk(0, "alpha beta")], embedder)
    retriever.search("gamma", 1)
    assert spy.seen == ["alpha beta", "QUERY: gamma"]  # the chunk got no query prefix


def test_hits_work_with_the_evaluation_harness(toy):
    questions = [
        Question("q1", "lexical", "dev", "ocean", (Span("a.md", 4, 4),)),
        Question("q2", "lexical", "dev", "cherry", (Span("a.md", 2, 2),)),
        Question("q3", "lexical", "dev", "banana", (Span("a.md", 3, 3),)),  # not the best match
    ]
    summary = summarize(run_retriever(make_retriever(toy), questions, k=1))
    assert summary["overall"]["hit@1"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ consistency
def test_the_chunks_must_be_the_ones_the_index_was_built_from(toy):
    embedder = make_embedder()
    index, _ = build_dense_index(toy, embedder)
    DenseRetriever(embedder, index, list(toy))  # fine
    with pytest.raises(ValueError):
        DenseRetriever(embedder, index, toy[:3])
    with pytest.raises(ValueError):
        DenseRetriever(embedder, index, [toy[1], toy[0], toy[2], toy[3]])
