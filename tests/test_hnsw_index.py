"""Phase 5, step 3: HNSWIndex, a graph that is walked towards the nearest vectors.

HNSW (Hierarchical Navigable Small World) links every vector to a few of its near neighbours. To
search, start somewhere and keep stepping to whichever linked vector is closer to the query; the
walk reaches the neighbourhood after a handful of steps instead of scanning everything. The
"hierarchical" part: a few vectors also appear in sparser upper layers (long-distance shortcuts),
so the walk starts at the top with big steps and refines on the way down to layer 0, where every
vector lives.

    index = HNSWIndex(vectors, doc_ids, M=16, ef_construction=100, ef_search=50, seed=0,
                      heuristic=True)         builds the graph by inserting vectors in row order
    index.search(query, k, ef=None) -> [(row, score), ...]   best first, equal scores: lower row
    index.neighbors(row, level) -> list of rows               the links of one node in one layer
    index.levels                the top layer of every row (layer 0 = every vector)
    index.entry_point, index.max_level                        where every search starts
    index.n_links               directed links over all layers; index.extra_nbytes = their memory
    index.last_n_distances      dot products computed by the last search
    index.save(path) / HNSWIndex.load(path)

M = links kept per node (layer 0 keeps up to 2M); ef = how many candidates the walk keeps in mind
(a bigger ef = a more careful, slower, more accurate search; ef is never below k).
A node's top layer is drawn as floor(-ln(U) / ln(M)), U uniform in (0, 1]: each layer holds about
1/M of the one below.
"""

import time
from collections import deque

import numpy as np
import pytest

from cairn.ann.hnsw import HNSWIndex
from cairn.dense.flat_index import FlatIndex


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def clustered(n, d, n_clusters, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    centers = unit(rng.normal(size=(n_clusters, d)))
    pick = rng.integers(n_clusters, size=n)
    return unit(centers[pick] + noise * rng.normal(size=(n, d)) / np.sqrt(d)), centers


def queries_near(centers, count, seed=1, noise=0.5):
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(centers), size=count)
    return unit(
        centers[pick]
        + noise * rng.normal(size=(count, centers.shape[1])) / np.sqrt(centers.shape[1])
    )


def ids(n):
    return [f"id{i}" for i in range(n)]


N, D, M = 1500, 32, 8


@pytest.fixture(scope="module")
def data():
    x, centers = clustered(N, D, 25, seed=0)
    return x, queries_near(centers, 60, seed=50)


@pytest.fixture(scope="module")
def index(data):
    return HNSWIndex(data[0], ids(N), M=M, ef_construction=60, ef_search=50, seed=0)


@pytest.fixture(scope="module")
def flat(data):
    return FlatIndex(data[0], ids(N))


def mean_recall(idx, flat, queries, **kwargs):
    total = 0.0
    for q in queries:
        want = {r for r, _ in flat.search(q, 10)}
        total += len({r for r, _ in idx.search(q, 10, **kwargs)} & want) / 10
    return total / len(queries)


# ------------------------------------------------------------------ building and checking
def test_basic_properties(index, data):
    assert index.n_docs == N and index.dim == D
    assert index.doc_ids == ids(N)
    assert index.M == M and index.ef_search == 50 and index.ef_construction == 60


def test_rejects_bad_input():
    x = unit(np.random.default_rng(0).normal(size=(20, 4)))
    with pytest.raises(ValueError):
        HNSWIndex(x, ids(19))
    with pytest.raises(ValueError):
        HNSWIndex(x[0], ids(4))
    with pytest.raises(ValueError):
        HNSWIndex(x, ids(20), M=1)
    with pytest.raises(ValueError):
        HNSWIndex(x, ids(20), ef_construction=0)
    with pytest.raises(ValueError):
        HNSWIndex(x, ids(20), ef_search=0)


def test_an_empty_index_and_tiny_indexes_work():
    empty = HNSWIndex(np.zeros((0, 4), dtype=np.float32), [])
    assert empty.n_docs == 0 and empty.search(np.ones(4, dtype=np.float32), 5) == []
    one = HNSWIndex(unit([[1.0, 2.0, 3.0]]), ["only"])
    assert one.search(unit([[1.0, 2.0, 3.0]])[0], 5) == [(0, pytest.approx(1.0, abs=1e-5))]
    two = HNSWIndex(unit([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"])
    assert [r for r, _ in two.search(unit([[0.1, 1.0]])[0], 5)] == [1, 0]


def test_the_input_vectors_are_not_changed_and_float64_is_accepted():
    x, _ = clustered(60, 8, 4)
    before = x.copy()
    HNSWIndex(x, ids(60), M=4, ef_construction=20)
    assert np.array_equal(x, before)
    HNSWIndex(x.astype(np.float64), ids(60), M=4, ef_construction=20)


# ------------------------------------------------------------------------- the layers
def test_top_layers_are_drawn_geometrically(index):
    levels = np.asarray(index.levels)
    assert levels.shape == (N,) and np.issubdtype(levels.dtype, np.integer) and levels.min() >= 0
    assert 0.09 <= np.mean(levels >= 1) <= 0.16  # about 1 / M = 0.125
    assert 0.09 <= levels.mean() <= 0.20  # about 1 / (M - 1) = 0.143
    assert np.mean(levels >= 2) < np.mean(levels >= 1)


def test_the_entry_point_is_on_the_top_layer(index):
    assert index.max_level == int(np.max(index.levels)) >= 1
    assert index.levels[index.entry_point] == index.max_level


def test_a_node_has_links_exactly_in_the_layers_it_belongs_to(index):
    for node in range(0, N, 7):
        for level in range(int(index.levels[node]) + 1):
            for other in index.neighbors(node, level):
                assert index.levels[other] >= level


def test_links_are_valid_and_degrees_are_bounded(index):
    for node in range(N):
        for level in range(int(index.levels[node]) + 1):
            links = index.neighbors(node, level)
            assert node not in links
            assert len(set(links)) == len(links)
            assert all(0 <= other < N for other in links)
            assert len(links) <= (2 * M if level == 0 else M)


def test_the_upper_layers_are_real_graphs_with_links_of_their_own(index):
    layer_1 = [node for node in range(N) if index.levels[node] >= 1]
    assert len(layer_1) > 50
    mean_degree = np.mean([len(index.neighbors(node, 1)) for node in layer_1])
    assert mean_degree >= M / 2


def test_every_node_is_linked_to_something_on_layer_0(index):
    assert all(len(index.neighbors(node, 0)) >= 1 for node in range(N))


def test_layer_0_can_be_walked_from_the_entry_point_to_nearly_every_node(index):
    seen = {index.entry_point}
    todo = deque([index.entry_point])
    while todo:
        for other in index.neighbors(todo.popleft(), 0):
            if other not in seen:
                seen.add(other)
                todo.append(other)
    assert len(seen) >= 0.99 * N


def test_links_join_similar_vectors(index, data):
    x = data[0]
    rng = np.random.default_rng(0)
    linked, random_pairs = [], []
    for node in range(0, N, 5):
        for other in index.neighbors(node, 0):
            linked.append(float(x[node] @ x[other]))
        random_pairs.append(float(x[node] @ x[int(rng.integers(N))]))
    assert np.mean(linked) > np.mean(random_pairs) + 0.2


def test_the_same_seed_builds_the_same_graph_and_ef_search_does_not_matter():
    x, _ = clustered(300, 16, 8)
    a = HNSWIndex(x, ids(300), M=6, ef_construction=30, ef_search=10, seed=4)
    b = HNSWIndex(x, ids(300), M=6, ef_construction=30, ef_search=99, seed=4)
    assert np.array_equal(a.levels, b.levels) and a.entry_point == b.entry_point
    for node in range(300):
        for level in range(int(a.levels[node]) + 1):
            assert a.neighbors(node, level) == b.neighbors(node, level)
    c = HNSWIndex(x, ids(300), M=6, ef_construction=30, seed=5)
    assert not np.array_equal(a.levels, c.levels)


# ---------------------------------------------------------------------------- searching
def test_scores_are_the_true_dot_products_best_first_without_repeats(index, data):
    x, queries = data
    result = index.search(queries[0], 15)
    rows = [r for r, _ in result]
    assert len(rows) == 15 and len(set(rows)) == 15
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert scores == pytest.approx([float(x[r] @ queries[0]) for r in rows], abs=1e-5)


def test_a_careful_search_finds_nearly_the_exact_answer(index, flat, data):
    queries = data[1]
    assert mean_recall(index, flat, queries, ef=10) >= 0.9
    assert mean_recall(index, flat, queries, ef=30) >= 0.97
    assert mean_recall(index, flat, queries, ef=N) >= 0.99


def test_a_bigger_ef_does_not_lower_recall(index, flat, data):
    recalls = [mean_recall(index, flat, data[1], ef=ef) for ef in (10, 30, 100, N)]
    assert all(b >= a - 0.005 for a, b in zip(recalls, recalls[1:], strict=False))


def test_ef_below_k_is_raised_to_k(index, data):
    assert len(index.search(data[1][0], 10, ef=1)) == 10


def test_the_default_ef_is_the_ef_search_of_the_index_and_a_call_does_not_change_it(index, data):
    q = data[1][0]
    assert index.search(q, 10) == index.search(q, 10, ef=50)
    index.search(q, 10, ef=500)
    assert index.ef_search == 50


def test_a_small_ef_looks_at_a_small_part_of_the_data(index, data):
    q = data[1][0]
    index.search(q, 10, ef=10)
    small = index.last_n_distances
    index.search(q, 10, ef=200)
    large = index.last_n_distances
    assert 0 < small < N / 5
    assert small < large <= N


def test_searching_everything_counts_about_one_dot_product_per_vector(index, data):
    index.search(data[1][0], 10, ef=N)
    assert 0.99 * N <= index.last_n_distances <= 1.2 * N


def test_equal_scores_come_out_lower_row_first():
    base = unit(np.random.default_rng(8).normal(size=(8, 6)))
    x = np.repeat(base, 5, axis=0)  # 8 directions, each 5 times: scores tie 5 ways
    index = HNSWIndex(x, ids(40), M=6, ef_construction=40, seed=0)
    for q in unit(np.random.default_rng(9).normal(size=(6, 6))):
        result = index.search(q, 40, ef=40)
        for (r1, s1), (r2, s2) in zip(result, result[1:], strict=False):
            assert s1 > s2 or r1 < r2


def test_the_work_count_is_reset_by_every_search(index, data):
    q = data[1][0]
    index.search(q, 10, ef=10)
    first = index.last_n_distances
    index.search(q, 10, ef=10)
    assert index.last_n_distances == first


def test_k_larger_than_the_index_returns_everything_and_k_zero_nothing():
    x, _ = clustered(40, 8, 3)
    index = HNSWIndex(x, ids(40), M=4, ef_construction=30)
    q = x[0]
    assert sorted(r for r, _ in index.search(q, 100)) == list(range(40))
    assert index.search(q, 0) == [] and index.search(q, -1) == []


def test_bad_queries_are_rejected(index, data):
    with pytest.raises(ValueError):
        index.search(np.zeros(5, dtype=np.float32), 3)
    with pytest.raises(ValueError):
        index.search(data[1][0], 3, ef=0)


def test_many_identical_vectors_do_not_crash_the_build_or_the_search():
    # Identical vectors are a known weak spot of HNSW (nothing separates their links), so this
    # only asks for a clean, valid answer, not for finding all of the copies.
    rng = np.random.default_rng(3)
    x = np.vstack([np.tile(unit([[1.0, 2.0, 3.0, 4.0]]), (60, 1)), unit(rng.normal(size=(10, 4)))])
    index = HNSWIndex(x, ids(70), M=4, ef_construction=20, seed=0)
    result = index.search(unit([[1.0, 2.0, 3.0, 4.0]])[0], 10, ef=70)
    rows = [r for r, _ in result]
    assert 1 <= len(rows) <= 10 and len(set(rows)) == len(rows)
    assert result[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in result] == sorted((s for _, s in result), reverse=True)


# --------------------------------------------------------- the neighbour-choosing heuristic
def test_the_simple_choice_still_builds_a_connected_graph_and_finds_everything_at_large_ef(
    data, flat
):
    simple = HNSWIndex(data[0], ids(N), M=M, ef_construction=60, seed=0, heuristic=False)
    assert mean_recall(simple, flat, data[1], ef=N) >= 0.97


def test_the_heuristic_matters_on_clustered_data(index, data, flat):
    # Keeping only the M closest neighbours links the vectors of one cluster to each other and to
    # nothing else. The heuristic keeps a neighbour only if it is closer to the new vector than to
    # the neighbours already kept, which leaves links that lead out to other clusters.
    simple = HNSWIndex(data[0], ids(N), M=M, ef_construction=60, seed=0, heuristic=False)
    better = mean_recall(index, flat, data[1], ef=30)
    worse = mean_recall(simple, flat, data[1], ef=30)
    assert better >= worse + 0.05


# -------------------------------------------------------------------------- persistence
def test_a_saved_graph_answers_exactly_like_the_original(index, data, tmp_path):
    index.save(tmp_path / "sub" / "hnsw.npz")  # the folder does not exist yet
    loaded = HNSWIndex.load(tmp_path / "sub" / "hnsw.npz")
    assert loaded.doc_ids == index.doc_ids
    assert (loaded.M, loaded.ef_construction, loaded.ef_search) == (M, 60, 50)
    assert loaded.entry_point == index.entry_point and loaded.max_level == index.max_level
    assert np.array_equal(loaded.levels, index.levels)
    assert loaded.n_links == index.n_links
    for node in range(0, N, 11):
        for level in range(int(index.levels[node]) + 1):
            assert loaded.neighbors(node, level) == index.neighbors(node, level)
    for q in data[1][:30]:
        for ef in (10, 50):
            assert loaded.search(q, 10, ef=ef) == index.search(q, 10, ef=ef)
            assert loaded.last_n_distances == index.last_n_distances


def test_the_file_is_plain_arrays_with_no_pickle(index, tmp_path):
    index.save(tmp_path / "hnsw.npz")
    with np.load(tmp_path / "hnsw.npz", allow_pickle=False) as data:
        assert len(data.files) >= 4


def test_an_empty_index_can_be_saved_and_loaded(tmp_path):
    HNSWIndex(np.zeros((0, 4), dtype=np.float32), []).save(tmp_path / "e.npz")
    assert HNSWIndex.load(tmp_path / "e.npz").n_docs == 0


def test_loading_something_else_is_an_error(tmp_path):
    from cairn.ann.ivf import IVFIndex

    x, _ = clustered(50, 8, 3)
    FlatIndex(x, ids(50)).save(tmp_path / "flat.npz")
    IVFIndex(x, ids(50), 3).save(tmp_path / "ivf.npz")
    np.savez(tmp_path / "other.npz", whatever=np.arange(3))
    for name in ("flat.npz", "ivf.npz", "other.npz"):
        with pytest.raises(ValueError):
            HNSWIndex.load(tmp_path / name)


def test_a_file_with_another_version_number_is_refused(tmp_path):
    index = HNSWIndex(*clustered(60, 8, 3)[:1], ids(60), M=4, ef_construction=20)
    index.save(tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["version"] = np.array(99)
    np.savez(tmp_path / "future.npz", **arrays)
    with pytest.raises(ValueError):
        HNSWIndex.load(tmp_path / "future.npz")


def test_extra_memory_is_four_bytes_per_link_plus_a_little_per_node(index, data):
    total = sum(
        len(index.neighbors(node, level))
        for node in range(N)
        for level in range(int(index.levels[node]) + 1)
    )
    assert index.n_links == total
    assert 4 * total <= index.extra_nbytes < data[0].nbytes


# ------------------------------------------------------------------------------- speed
def test_building_1000_vectors_and_searching_200_times_is_fast_enough():
    x, centers = clustered(1000, 32, 20, seed=9)
    start = time.perf_counter()
    index = HNSWIndex(x, ids(1000), M=8, ef_construction=40, seed=0)
    assert time.perf_counter() - start < 30.0, "building is too slow"
    start = time.perf_counter()
    for q in queries_near(centers, 200, seed=10):
        index.search(q, 10, ef=40)
    assert time.perf_counter() - start < 3.0, "200 searches are too slow"


# ----------------------------------------------------------- it plugs into the retriever
def test_it_can_replace_the_flat_index_in_the_dense_retriever():
    from cairn.dense.embedder import Embedder
    from cairn.dense.models import HashingModel
    from cairn.dense.retriever import DenseRetriever, build_dense_index
    from cairn.ingest.chunkers import make_chunk

    words = "apple banana cherry dog eel fig goat hat ink jam kiwi lamp".split()
    chunks = [
        make_chunk("a.md", i + 1, i + 1, f"{words[i % 12]} {words[(i * 5) % 12]} note{i}", "prose")
        for i in range(60)
    ]
    embedder = Embedder(HashingModel(), "hashing")
    exact, _ = build_dense_index(chunks, embedder)
    graph = HNSWIndex(exact.vectors, exact.doc_ids, M=4, ef_construction=30, ef_search=60, seed=0)
    want = DenseRetriever(embedder, exact, chunks).search("apple note3", 5)
    got = DenseRetriever(embedder, graph, chunks).search("apple note3", 5)
    assert [h.text for h in got] == [h.text for h in want]
