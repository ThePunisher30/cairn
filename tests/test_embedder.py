"""Phase 4, step 1: the Embedder, a careful wrapper around an embedding model.

A *model* is anything with `dim`, `max_tokens`, `encode(texts) -> (n, dim) array` and
`token_lengths(texts)`. The tests use the offline HashingModel (src/cairn/dense/models.py).

    l2_normalize(x)                       scale every row to length 1 (an all-zero row stays zero)
    Embedder(model, name, query_prefix="", doc_prefix="", batch_size=32)
        .embed_documents(texts) -> (n, dim) float32, unit rows, SAME ORDER as `texts`
        .embed_query(text)      -> (dim,) float32, unit length
        .n_truncated(texts)     -> how many documents are longer than the model can read
        .cache_key              -> a string naming everything that changes a document vector
        .dim, .max_tokens       -> taken from the model

Three details that decide quality and speed:
- prefixes: some models were trained with a fixed phrase in front of queries (and/or documents);
  leave it out and the vectors silently get worse. The doc prefix goes on every document, the
  query prefix only on queries.
- batching: texts go to the model in batches of `batch_size`, SORTED BY LENGTH (characters) so that
  short texts are not padded to the length of a long neighbour; the result is put back in the
  original order. (On the real corpus this alone made embedding about twice as fast.)
- truncation: a model silently ignores everything after `max_tokens`, so you must be able to ask
  how many documents it cannot fully read (the document prefix counts too).
"""

import warnings

import numpy as np
import pytest

from cairn.dense.embedder import Embedder, l2_normalize
from cairn.dense.models import HashingModel


def make(model=None, **options):
    return Embedder(model or HashingModel(dim=32), options.pop("name", "hashing"), **options)


# ---------------------------------------------------------------- l2_normalize
def test_a_row_is_scaled_to_length_one():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(result, [[0.6, 0.8], [0.0, 1.0]])
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_an_all_zero_row_stays_zero_and_makes_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 0 / 0 would raise a RuntimeWarning here
        result = l2_normalize(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert result.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert not np.isnan(result).any()


def test_a_single_vector_works_too_and_the_result_is_float32():
    result = l2_normalize(np.array([3.0, 4.0]))
    assert result.shape == (2,) and result.dtype == np.float32
    assert result.tolist() == pytest.approx([0.6, 0.8])


def test_normalizing_does_not_change_its_input():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    l2_normalize(x)
    assert x.tolist() == [[3.0, 4.0]]


def test_for_unit_vectors_the_dot_product_is_the_cosine():
    a, b = l2_normalize(np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert float(a @ b) == pytest.approx(np.cos(np.radians(45)), abs=1e-6)


# ----------------------------------------------------------------- embed_documents
def test_documents_give_one_unit_vector_each():
    vectors = make().embed_documents(["apple banana", "banana cherry", "dog"])
    assert vectors.shape == (3, 32) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_no_documents_give_an_empty_matrix_with_the_right_width():
    assert make().embed_documents([]).shape == (0, 32)


def test_a_text_with_no_words_gives_a_zero_vector_not_nan():
    vector = make().embed_documents([""])[0]
    assert not vector.any() and not np.isnan(vector).any()


def test_the_rows_come_back_in_the_order_of_the_texts():
    texts = ["a rather long text with many words in it", "short", "medium length text", "x"]
    e = make(batch_size=2)
    together = e.embed_documents(texts)
    for row, text in enumerate(texts):
        assert np.allclose(together[row], e.embed_documents([text])[0])


def test_texts_go_to_the_model_in_batches_of_batch_size():
    model = HashingModel(dim=32)
    make(model, batch_size=3).embed_documents([f"text number {i}" for i in range(7)])
    assert model.batch_sizes == [3, 3, 1]


def test_texts_go_to_the_model_shortest_first():
    model = HashingModel(dim=32)
    texts = ["a" * 50, "b" * 5, "c" * 30, "d" * 1, "e" * 40, "f" * 10]
    make(model, batch_size=2).embed_documents(texts)
    sent = [text for batch in model.batches for text in batch]
    assert [len(t) for t in sent] == [1, 5, 10, 30, 40, 50]


def test_texts_of_equal_length_keep_their_order():
    model = HashingModel(dim=32)
    make(model, batch_size=10).embed_documents(["bb", "aa", "cc", "a"])
    assert model.batches == [["a", "bb", "aa", "cc"]]


def test_embedding_is_deterministic_and_repeats_give_identical_rows():
    e = make()
    first = e.embed_documents(["same text", "other", "same text"])
    assert np.array_equal(first[0], first[2])
    assert np.array_equal(first, e.embed_documents(["same text", "other", "same text"]))


# ------------------------------------------------------------------- the query
def test_a_query_is_one_unit_vector():
    vector = make().embed_query("apple banana")
    assert vector.shape == (32,) and vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_without_prefixes_a_query_matches_the_same_text_as_a_document():
    e = make()
    assert np.allclose(e.embed_query("apple banana"), e.embed_documents(["apple banana"])[0])


# ------------------------------------------------------------------- prefixes
def test_the_doc_prefix_goes_on_every_document_and_the_query_prefix_does_not():
    model = HashingModel(dim=32)
    make(model, doc_prefix="passage: ", query_prefix="query: ").embed_documents(["one", "two"])
    assert sorted(model.batches[0]) == ["passage: one", "passage: two"]


def test_the_query_prefix_goes_on_the_query_and_the_doc_prefix_does_not():
    model = HashingModel(dim=32)
    make(model, doc_prefix="passage: ", query_prefix="query: ").embed_query("find this")
    assert model.batches == [["query: find this"]]


def test_a_prefix_changes_the_vector():
    plain = make().embed_query("find this")
    prefixed = make(query_prefix="query: ").embed_query("find this")
    assert not np.allclose(plain, prefixed)


# -------------------------------------------------------------------- the cache key
def test_the_cache_key_names_the_model_and_the_doc_prefix_only():
    base = make(name="m1").cache_key
    assert make(name="m1", batch_size=7).cache_key == base  # speed settings do not matter
    assert make(name="m1", query_prefix="q: ").cache_key == base  # queries are never cached
    assert make(name="m2").cache_key != base
    assert make(name="m1", doc_prefix="passage: ").cache_key != base
    assert (
        make(name="m1", doc_prefix="a: ").cache_key != make(name="m1", doc_prefix="b: ").cache_key
    )


# ---------------------------------------------------------------- model settings
def test_the_size_and_limit_come_from_the_model():
    e = make(HashingModel(dim=48, max_tokens=100))
    assert (e.dim, e.max_tokens) == (48, 100)


def test_a_batch_size_below_one_is_refused():
    with pytest.raises(ValueError):
        make(batch_size=0)
    with pytest.raises(ValueError):
        make(batch_size=-3)


# ------------------------------------------------------------------- truncation
def test_n_truncated_counts_the_documents_the_model_cannot_fully_read():
    e = make(HashingModel(dim=32, max_tokens=5))
    texts = ["one two three four", "a b c d e f g", "", "p q r s t", "u v w x y z"]
    assert e.n_truncated(texts) == 2  # 7 and 6 words are over the limit; exactly 5 is fine


def test_the_doc_prefix_counts_towards_the_limit():
    e = make(HashingModel(dim=32, max_tokens=5), doc_prefix="pa pb pc ")
    assert e.n_truncated(["wa wb", "wa wb wc"]) == 1  # 3 + 2 = 5 fits, 3 + 3 = 6 does not


def test_nothing_is_truncated_by_default_for_short_texts():
    assert make().n_truncated(["short text", "another short one"]) == 0
    assert make().n_truncated([]) == 0
