"""Phase 4: the provided model registry and the offline HashingModel (src/cairn/dense/models.py)."""

import os

import numpy as np
import pytest

from cairn.dense.embedder import Embedder
from cairn.dense.models import (
    BGE_QUERY_PREFIX,
    MODELS,
    HashingModel,
    SentenceTransformerModel,
    load_embedder,
)


def test_the_registry_knows_its_models():
    assert set(MODELS) == {"bge-small", "minilm", "hashing", "hashing-prefixed"}
    assert MODELS["bge-small"].hf_id == "BAAI/bge-small-en-v1.5"
    assert MODELS["bge-small"].query_prefix == BGE_QUERY_PREFIX
    assert MODELS["bge-small"].doc_prefix == ""
    assert MODELS["minilm"].query_prefix == "" and MODELS["minilm"].doc_prefix == ""


def test_the_hashing_model_makes_one_vector_per_text():
    model = HashingModel(dim=64)
    vectors = model.encode(["apple banana", "banana cherry", ""])
    assert vectors.shape == (3, 64) and vectors.dtype == np.float32
    assert not vectors[2].any()  # an empty text has no words, so an all-zero vector


def test_the_hashing_model_is_deterministic_across_instances():
    a = HashingModel().encode(["get_request_handler returns 404"])
    b = HashingModel().encode(["get_request_handler returns 404"])
    assert np.array_equal(a, b)


def test_texts_that_share_words_get_more_similar_vectors():
    model = HashingModel(dim=256)
    v = model.encode(["apple banana cherry", "apple banana dog", "ocean river mountain"])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert v[0] @ v[1] > 0.5
    assert abs(v[0] @ v[2]) < 0.2


def test_the_hashing_model_reports_token_lengths_and_records_its_calls():
    model = HashingModel()
    assert model.token_lengths(["one two three", ""]) == [3, 0]
    model.encode(["a", "b"])
    model.encode(["c"])
    assert model.batch_sizes == [2, 1]


def test_load_embedder_builds_an_embedder_from_a_registry_key():
    embedder = load_embedder("hashing", batch_size=8)
    assert isinstance(embedder, Embedder)
    assert embedder.name == "hashing" and embedder.batch_size == 8
    assert embedder.dim == 256 and embedder.max_tokens == 512


def test_the_query_prefix_can_be_switched_off_and_a_loaded_model_can_be_reused():
    model = HashingModel()
    with_prefix = load_embedder("bge-small", model=model)
    without = load_embedder("bge-small", use_query_prefix=False, model=model)
    assert with_prefix.query_prefix == BGE_QUERY_PREFIX
    assert without.query_prefix == ""
    assert with_prefix.doc_prefix == without.doc_prefix == ""
    assert with_prefix.model is model and without.model is model


@pytest.mark.skipif(
    os.environ.get("CAIRN_REAL_MODEL_TESTS") != "1",
    reason="loads a real model (slow); run with CAIRN_REAL_MODEL_TESTS=1",
)
def test_a_real_model_gives_unit_vectors_and_sensible_similarities():
    embedder = load_embedder("bge-small")
    assert isinstance(embedder.model, SentenceTransformerModel)
    assert (embedder.dim, embedder.max_tokens) == (384, 512)
    docs = embedder.embed_documents(
        ["How to run a task after the response is sent", "The weather in Paris is mild"]
    )
    query = embedder.embed_query("background work after returning a response")
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert query @ docs[0] > query @ docs[1]
