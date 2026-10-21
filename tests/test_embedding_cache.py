"""Phase 4, step 2: the embedding cache, so unchanged chunks are never embedded twice.

Embedding is the slow part of the pipeline (minutes on a CPU), so a vector is stored the first
time it is computed and looked up every time after. The cache key must name EVERYTHING that
changes the vector, and nothing else:
    - the exact text the model reads        (the chunk text AND its heading path, not just a hash
                                             of the chunk text: a new heading is a new vector)
    - which model                           (Embedder.name)
    - the document prefix                   (Embedder.doc_prefix: another prefix, another vector)
and NOT the query prefix, because queries are never cached.

    cache_key(model_key, text) -> str                      a hash of both
    cache = EmbeddingCache(path)                           loads the file if it exists
    embed_cached(embedder, texts, cache) -> (matrix, computed)
        matrix: one row per text, in order; computed: how many texts the model had to embed
"""

import numpy as np
import pytest

from cairn.dense.cache import EmbeddingCache, cache_key, embed_cached
from cairn.dense.embedder import Embedder
from cairn.dense.models import HashingModel

TEXTS = ["apple banana", "banana cherry", "cherry dog fox"]


def embedder(model=None, name="hashing", doc_prefix="", query_prefix="", batch_size=32):
    return Embedder(
        model or HashingModel(dim=32),
        name,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        batch_size=batch_size,
    )


# --------------------------------------------------------------------- the key
def test_a_key_is_a_stable_hash_of_the_model_and_the_text():
    key = cache_key("model-a", "some text")
    assert key == cache_key("model-a", "some text")
    assert isinstance(key, str) and len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_a_different_model_or_text_gives_a_different_key():
    keys = {
        cache_key("model-a", "text"),
        cache_key("model-b", "text"),
        cache_key("model-a", "other text"),
    }
    assert len(keys) == 3


# ------------------------------------------------------------- the basic cache
def test_a_new_cache_is_empty_and_get_returns_none():
    cache = EmbeddingCache()
    assert len(cache) == 0
    assert cache.get("missing") is None
    assert "missing" not in cache


def test_put_and_get_round_trip_and_the_cache_keeps_its_own_copy():
    cache = EmbeddingCache()
    vector = np.ones(3, dtype=np.float32)
    cache.put("k", vector)
    vector[0] = 99.0  # changing the original afterwards must not change the cached one
    assert "k" in cache and len(cache) == 1
    assert cache.get("k").tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------- embed_cached
def test_the_first_call_embeds_everything():
    cache = EmbeddingCache()
    e = embedder()
    matrix, computed = embed_cached(e, TEXTS, cache)
    assert computed == 3
    assert matrix.shape == (3, 32) and matrix.dtype == np.float32
    assert np.allclose(matrix, embedder().embed_documents(TEXTS))
    assert len(cache) == 3


def test_the_second_call_does_not_touch_the_model():
    cache = EmbeddingCache()
    model = HashingModel(dim=32)
    e = embedder(model)
    first, _ = embed_cached(e, TEXTS, cache)
    calls = len(model.batch_sizes)
    second, computed = embed_cached(e, TEXTS, cache)
    assert computed == 0
    assert len(model.batch_sizes) == calls  # the model was not called at all
    assert np.array_equal(first, second)


def test_only_the_new_texts_are_embedded_and_the_order_is_kept():
    cache = EmbeddingCache()
    e = embedder()
    embed_cached(e, ["apple banana", "banana cherry"], cache)
    matrix, computed = embed_cached(e, ["banana cherry", "dog fox", "apple banana"], cache)
    assert computed == 1  # only "dog fox"
    assert np.allclose(
        matrix, embedder().embed_documents(["banana cherry", "dog fox", "apple banana"])
    )


def test_the_same_text_twice_in_one_call_is_embedded_once():
    cache = EmbeddingCache()
    model = HashingModel(dim=32)
    matrix, computed = embed_cached(embedder(model), ["x y", "x y", "z"], cache)
    assert computed == 2
    assert sum(model.batch_sizes) == 2  # the model saw two texts, not three
    assert np.array_equal(matrix[0], matrix[1])
    assert len(cache) == 2


def test_the_missing_texts_are_embedded_in_batches():
    cache = EmbeddingCache()
    model = HashingModel(dim=32)
    texts = [f"word{i} filler" for i in range(70)]
    _, computed = embed_cached(embedder(model, batch_size=32), texts, cache)
    assert computed == 70
    assert model.batch_sizes == [32, 32, 6]


def test_no_texts_gives_an_empty_matrix():
    matrix, computed = embed_cached(embedder(), [], EmbeddingCache())
    assert matrix.shape == (0, 32) and computed == 0


def test_a_different_model_name_does_not_reuse_the_vectors():
    cache = EmbeddingCache()
    embed_cached(embedder(name="model-a"), TEXTS, cache)
    _, computed = embed_cached(embedder(name="model-b"), TEXTS, cache)
    assert computed == 3
    assert len(cache) == 6


def test_a_different_document_prefix_does_not_reuse_the_vectors():
    cache = EmbeddingCache()
    plain, _ = embed_cached(embedder(doc_prefix=""), TEXTS, cache)
    prefixed, computed = embed_cached(embedder(doc_prefix="passage: "), TEXTS, cache)
    assert computed == 3  # the cache key includes the prefix
    assert not np.allclose(plain, prefixed)


def test_the_query_prefix_does_not_matter_to_the_cache():
    cache = EmbeddingCache()
    embed_cached(embedder(query_prefix=""), TEXTS, cache)
    _, computed = embed_cached(embedder(query_prefix="query: "), TEXTS, cache)
    assert computed == 0


# ------------------------------------------------------------------ save / load
def test_a_saved_cache_is_used_by_the_next_run(tmp_path):
    path = tmp_path / "cache.npz"
    cache = EmbeddingCache(path)
    first, _ = embed_cached(embedder(), TEXTS, cache)
    cache.save()

    model = HashingModel(dim=32)  # a fresh run: nothing in memory
    reloaded = EmbeddingCache(path)
    assert len(reloaded) == 3
    second, computed = embed_cached(embedder(model), TEXTS, reloaded)
    assert computed == 0 and model.batch_sizes == []
    assert np.array_equal(first, second)


def test_save_creates_folders_and_the_file_needs_no_pickle(tmp_path):
    path = tmp_path / "deep" / "er" / "cache.npz"
    cache = EmbeddingCache(path)
    embed_cached(embedder(), TEXTS, cache)
    cache.save()
    with np.load(path, allow_pickle=False) as data:
        assert len(data["keys"]) == 3 and data["vectors"].shape == (3, 32)


def test_a_missing_file_is_just_an_empty_cache(tmp_path):
    assert len(EmbeddingCache(tmp_path / "nope.npz")) == 0


def test_an_empty_cache_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / "cache.npz"
    EmbeddingCache(path).save()
    assert len(EmbeddingCache(path)) == 0


def test_a_cache_without_a_path_cannot_be_saved():
    with pytest.raises(ValueError):
        EmbeddingCache().save()


def test_new_vectors_are_added_to_an_existing_file(tmp_path):
    path = tmp_path / "cache.npz"
    cache = EmbeddingCache(path)
    embed_cached(embedder(), TEXTS[:2], cache)
    cache.save()
    cache = EmbeddingCache(path)
    embed_cached(embedder(), TEXTS, cache)
    cache.save()
    assert len(EmbeddingCache(path)) == 3
