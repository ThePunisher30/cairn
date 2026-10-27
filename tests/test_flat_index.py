"""Phase 4, step 3: FlatIndex, brute-force nearest-neighbour search over unit vectors.

"Flat" means no tricks: one matrix-vector product scores every chunk, then the best k are picked.
For unit-length vectors the dot product IS the cosine similarity (cos = a . b / (|a| |b|), and
both lengths are 1), so the scores here lie between -1 and 1, higher = more similar.

    index = FlatIndex(vectors, doc_ids)       vectors: (n, d) array, one row per chunk
    index.search(query, k) -> [(row, score), ...]    best first; equal scores: lower row first
"""

import time

import numpy as np
import pytest

from cairn.dense.flat_index import FlatIndex


def angle(*degrees):
    """2-D unit vectors at the given angles, one per row."""
    radians = np.radians(degrees)
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


def index_of(vectors):
    vectors = np.asarray(vectors)
    return FlatIndex(vectors, [f"id{i}" for i in range(len(vectors))])


# ------------------------------------------------------------------- the scores
def test_scores_are_dot_products_best_first():
    index = index_of(angle(0, 60, 90, 180))
    result = index.search([1.0, 0.0], 4)  # the query points along 0 degrees
    assert [row for row, _ in result] == [0, 1, 2, 3]
    assert [score for _, score in result] == pytest.approx([1.0, 0.5, 0.0, -1.0], abs=1e-6)


def test_the_best_row_need_not_be_the_first():
    index = index_of(angle(180, 20, 90, 45))
    result = index.search([0.0, 1.0], 4)  # query at 90 degrees
    assert [row for row, _ in result] == [2, 3, 1, 0]
    assert result[0][1] == pytest.approx(1.0, abs=1e-6)


def test_k_limits_the_results_to_the_best_ones():
    index = index_of(angle(0, 60, 90, 180))
    assert [row for row, _ in index.search([1.0, 0.0], 2)] == [0, 1]
    assert [row for row, _ in index.search([1.0, 0.0], 1)] == [0]


def test_rows_and_scores_are_plain_python_numbers():
    ((row, score),) = index_of(angle(0)).search([1.0, 0.0], 1)
    assert type(row) is int and type(score) is float


def test_k_larger_than_the_index_returns_every_row():
    result = index_of(angle(0, 90)).search([1.0, 0.0], 50)
    assert [row for row, _ in result] == [0, 1]


def test_k_zero_or_less_gives_nothing():
    index = index_of(angle(0, 90))
    assert index.search([1.0, 0.0], 0) == []
    assert index.search([1.0, 0.0], -2) == []


def test_an_empty_index_finds_nothing():
    index = FlatIndex(np.zeros((0, 2), dtype=np.float32), [])
    assert index.n_docs == 0
    assert index.search([1.0, 0.0], 5) == []


# ---------------------------------------------------------------------- ties
def test_equal_scores_are_ranked_by_row_number():
    index = index_of(np.tile([[1.0, 0.0]], (10, 1)))
    assert [row for row, _ in index.search([1.0, 0.0], 3)] == [0, 1, 2]


def test_a_tie_at_the_edge_of_the_top_k_goes_to_the_lower_row():
    # scores 0.9, 0.5, 0.5, 0.5, 0.1: the cut falls inside the three-way tie
    index = index_of([[0.9, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.1, 0.0]])
    assert [row for row, _ in index.search([1.0, 0.0], 2)] == [0, 1]
    assert [row for row, _ in index.search([1.0, 0.0], 3)] == [0, 1, 2]
    assert [row for row, _ in index.search([1.0, 0.0], 4)] == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", range(20))
def test_it_always_matches_a_plain_full_sort_even_with_many_ties(seed):
    rng = np.random.default_rng(seed)
    vectors = rng.integers(0, 4, size=(300, 3)).astype(np.float32)  # lots of equal scores
    query = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    scores = vectors @ query
    expected = sorted(range(len(vectors)), key=lambda i: (-scores[i], i))[:12]
    result = FlatIndex(vectors, [str(i) for i in range(300)]).search(query, 12)
    assert [row for row, _ in result] == expected


# --------------------------------------------------------------- the arguments
def test_a_query_of_the_wrong_size_is_refused():
    index = index_of(angle(0, 90))
    with pytest.raises(ValueError):
        index.search([1.0, 0.0, 0.0], 1)
    with pytest.raises(ValueError):
        index.search([[1.0, 0.0]], 1)  # a 2-D array is not a single query


def test_the_vectors_and_ids_must_agree():
    with pytest.raises(ValueError):
        FlatIndex(angle(0, 90), ["only-one-id"])
    with pytest.raises(ValueError):
        FlatIndex(np.zeros(5), ["a"] * 5)  # vectors must be 2-D


def test_vectors_are_stored_as_float32_and_the_input_is_not_changed():
    original = angle(0, 90).astype(np.float64)
    snapshot = original.copy()
    index = FlatIndex(original, ["a", "b"])
    assert index.vectors.dtype == np.float32
    assert np.array_equal(original, snapshot)


def test_size_properties_and_memory_use():
    index = FlatIndex(np.zeros((1000, 384)), [str(i) for i in range(1000)])
    assert (index.n_docs, index.dim) == (1000, 384)
    assert index.nbytes == 1000 * 384 * 4  # 4 bytes per float32 number


# ------------------------------------------------------------------ save / load
def test_a_saved_index_loads_back_identical(tmp_path):
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, 8)).astype(np.float32)
    original = FlatIndex(vectors, [f"chunk-{i}" for i in range(50)])
    original.save(tmp_path / "dense.npz")
    loaded = FlatIndex.load(tmp_path / "dense.npz")
    assert np.array_equal(loaded.vectors, original.vectors)
    assert loaded.doc_ids == original.doc_ids
    query = rng.normal(size=8).astype(np.float32)
    assert loaded.search(query, 10) == original.search(query, 10)


def test_save_creates_missing_folders_and_overwrites(tmp_path):
    path = tmp_path / "a" / "b" / "dense.npz"
    index_of(angle(0, 90, 180)).save(path)
    index_of(angle(0)).save(path)
    assert FlatIndex.load(path).n_docs == 1


def test_the_file_is_plain_numpy_data_that_needs_no_pickle(tmp_path):
    index_of(angle(0, 90)).save(tmp_path / "dense.npz")
    with np.load(tmp_path / "dense.npz", allow_pickle=False) as data:
        assert "vectors" in data.files and "doc_ids" in data.files


def test_a_file_that_is_not_an_index_is_refused(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, something=np.arange(3))
    with pytest.raises(ValueError):
        FlatIndex.load(path)


def test_an_ann_index_file_is_refused_not_read_as_a_flat_index(tmp_path):
    from cairn.ann.hnsw import HNSWIndex
    from cairn.ann.ivf import IVFIndex

    vectors = np.random.default_rng(2).normal(size=(30, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"id{i}" for i in range(30)]
    IVFIndex(vectors, ids, 3).save(tmp_path / "ivf.npz")
    HNSWIndex(vectors, ids, M=4, ef_construction=10).save(tmp_path / "hnsw.npz")
    for name in ("ivf.npz", "hnsw.npz"):
        with pytest.raises(ValueError):
            FlatIndex.load(tmp_path / name)


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        FlatIndex.load(tmp_path / "nope.npz")


# ----------------------------------------------------------------------- speed
def test_a_search_is_one_matrix_product_not_a_python_loop():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(100_000, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = FlatIndex(vectors, [str(i) for i in range(100_000)])
    start = time.perf_counter()
    for _ in range(50):
        assert len(index.search(vectors[7], 10)) == 10
        assert time.perf_counter() - start < 3.0, "too slow: is there a Python loop over the rows?"
