"""Phase 5, step 2: IVFIndex, an approximate index that searches only a few groups.

IVF ("inverted file") cuts the vectors into `nlist` groups with k-means (step 1). To answer a
query it ranks the groups by their centroid, then scans only the `nprobe` best groups instead of
every vector. Fewer vectors scanned = faster, but the true nearest vector may sit in a group that
was skipped, so the answer is only *approximately* right.

    index = IVFIndex(vectors, doc_ids, nlist, nprobe=8, seed=0)      trains k-means, fills the lists
    index.search(query, k, nprobe=None) -> [(row, score), ...]   best first; ties: lower row
    index.list_rows(j)                    rows in list j, in increasing order
    index.last_n_distances                dot products computed by the last search
                                          (nlist centroid scores + the vectors scanned)
    index.extra_nbytes                    memory of centroids + lists (not counting the vectors)
    index.save(path) / IVFIndex.load(path)

Same score and tie rules as FlatIndex. With nprobe = nlist every vector is scanned, so the answer
must be exactly the flat index's.
"""

import time

import numpy as np
import pytest

from cairn.ann.ivf import IVFIndex
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
    d = centers.shape[1]
    pick = rng.integers(len(centers), size=count)
    return unit(centers[pick] + noise * rng.normal(size=(count, d)) / np.sqrt(d))


def ids(n):
    return [f"id{i}" for i in range(n)]


def make(n=600, d=16, nlist=12, nprobe=4, seed=0, n_clusters=12):
    x, centers = clustered(n, d, n_clusters, seed=seed)
    return IVFIndex(x, ids(n), nlist, nprobe=nprobe, seed=seed), x, centers


def recall(found, truth):
    return len({r for r, _ in found} & {r for r, _ in truth}) / len(truth)


# ------------------------------------------------------------------ building and checking
def test_basic_properties():
    index, x, _ = make(n=300, d=16, nlist=9, nprobe=3)
    assert index.n_docs == 300
    assert index.dim == 16
    assert index.nlist == 9
    assert index.nprobe == 3
    assert index.doc_ids == ids(300)
    assert index.centroids.shape == (9, 16)


def test_rejects_bad_input():
    x = unit(np.random.default_rng(0).normal(size=(20, 4)))
    with pytest.raises(ValueError):
        IVFIndex(x, ids(19), 3)  # one id too few
    with pytest.raises(ValueError):
        IVFIndex(x[0], ids(4), 1)  # not 2-D
    for bad in (0, -2, 21):
        with pytest.raises(ValueError):
            IVFIndex(x, ids(20), bad)
    with pytest.raises(ValueError):
        IVFIndex(x, ids(20), 3, nprobe=0)


def test_the_lists_hold_every_row_exactly_once_in_increasing_order():
    index, x, _ = make()
    seen = []
    for j in range(index.nlist):
        rows = np.asarray(index.list_rows(j))
        assert np.array_equal(rows, np.sort(rows))
        seen.extend(rows.tolist())
    assert sorted(seen) == list(range(index.n_docs))


def test_each_row_is_in_the_list_of_its_nearest_centroid():
    index, x, _ = make()
    nearest = (x @ index.centroids.T).argmax(axis=1)
    for j in range(index.nlist):
        assert all(nearest[r] == j for r in index.list_rows(j))


def test_the_same_seed_builds_the_same_lists():
    a, _, _ = make(seed=3)
    b, _, _ = make(seed=3)
    assert np.array_equal(a.centroids, b.centroids)
    for j in range(a.nlist):
        assert np.array_equal(a.list_rows(j), b.list_rows(j))


def test_the_input_vectors_are_not_changed():
    x, _ = clustered(100, 8, 5)
    before = x.copy()
    IVFIndex(x, ids(100), 5)
    assert np.array_equal(x, before)


# -------------------------------------------------------------------- searching: exactness
def test_scanning_every_list_gives_exactly_the_flat_answer():
    index, x, centers = make(n=500, nlist=10, nprobe=1)
    flat = FlatIndex(x, ids(500))
    for q in queries_near(centers, 25):
        got = index.search(q, 10, nprobe=10)
        want = flat.search(q, 10)
        assert [r for r, _ in got] == [r for r, _ in want]
        assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-6)


def test_scanning_every_list_breaks_ties_like_the_flat_index():
    # 12 distinct directions, each repeated 25 times: every score is tied 25 ways
    base = unit(np.random.default_rng(5).normal(size=(12, 6)))
    x = np.repeat(base, 25, axis=0)
    index = IVFIndex(x, ids(300), 6, seed=0)
    flat = FlatIndex(x, ids(300))
    for q in unit(np.random.default_rng(6).normal(size=(10, 6))):
        assert [r for r, _ in index.search(q, 40, nprobe=6)] == [r for r, _ in flat.search(q, 40)]


def test_one_list_is_the_flat_index():
    x, centers = clustered(200, 8, 4)
    index = IVFIndex(x, ids(200), 1)
    flat = FlatIndex(x, ids(200))
    q = queries_near(centers, 1)[0]
    assert [r for r, _ in index.search(q, 10)] == [r for r, _ in flat.search(q, 10)]


def test_scores_are_the_true_dot_products_best_first_without_repeats():
    index, x, centers = make()
    q = queries_near(centers, 1)[0]
    result = index.search(q, 15, nprobe=3)
    rows = [r for r, _ in result]
    assert len(set(rows)) == len(rows)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert scores == pytest.approx([float(x[r] @ q) for r in rows], abs=1e-6)


# ---------------------------------------------------------------- searching: approximate
def test_only_the_chosen_lists_are_searched():
    index, x, centers = make(n=600, nlist=12)
    q = queries_near(centers, 1, seed=9)[0]
    best_list = int(np.argmax(index.centroids @ q))
    allowed = set(index.list_rows(best_list).tolist())
    result = index.search(q, 5, nprobe=1)
    assert result and all(r in allowed for r, _ in result)


def test_the_lists_are_chosen_by_how_close_their_centroid_is():
    index, x, centers = make(n=600, nlist=12)
    q = queries_near(centers, 1, seed=10)[0]
    top2 = np.argsort(-(index.centroids @ q))[:2]
    allowed = set(index.list_rows(int(top2[0])).tolist()) | set(
        index.list_rows(int(top2[1])).tolist()
    )
    result = index.search(q, 10_000, nprobe=2)
    assert {r for r, _ in result} == allowed


def test_the_work_done_is_the_centroids_plus_the_vectors_scanned():
    index, x, centers = make(n=600, nlist=12)
    q = queries_near(centers, 1, seed=11)[0]
    index.search(q, 10, nprobe=12)
    assert index.last_n_distances == 12 + 600
    top = int(np.argmax(index.centroids @ q))
    index.search(q, 10, nprobe=1)
    assert index.last_n_distances == 12 + len(index.list_rows(top))


def test_scanning_few_lists_is_much_less_work_than_scanning_everything():
    x, centers = clustered(4000, 16, 40, seed=2)
    index = IVFIndex(x, ids(4000), 40, nprobe=2, seed=0)
    index.search(queries_near(centers, 1)[0], 10)
    assert index.last_n_distances < 4000 / 4


def test_more_lists_scanned_never_lowers_recall_on_any_query():
    index, x, centers = make(n=800, nlist=16, n_clusters=16)
    flat = FlatIndex(x, ids(800))
    for q in queries_near(centers, 20, seed=12):
        truth = flat.search(q, 10)
        recalls = [recall(index.search(q, 10, nprobe=p), truth) for p in (1, 2, 4, 8, 16)]
        assert recalls == sorted(recalls)
        assert recalls[-1] == 1.0


def test_on_clustered_data_a_few_lists_find_most_neighbours():
    x, centers = clustered(3000, 24, 30, seed=3)
    index = IVFIndex(x, ids(3000), 30, seed=0)
    flat = FlatIndex(x, ids(3000))
    qs = queries_near(centers, 40, seed=4)

    def mean(p):
        return np.mean([recall(index.search(q, 10, nprobe=p), flat.search(q, 10)) for q in qs])

    assert mean(4) >= 0.9
    assert mean(1) <= mean(4)


def test_a_list_with_fewer_than_k_vectors_returns_fewer_than_k_results():
    index, x, centers = make(n=600, nlist=12)
    q = queries_near(centers, 1, seed=13)[0]
    top = int(np.argmax(index.centroids @ q))
    size = len(index.list_rows(top))
    assert len(index.search(q, size + 50, nprobe=1)) == size
    assert len(index.search(q, 10_000, nprobe=12)) == 600


def test_nprobe_above_nlist_means_all_lists_and_the_default_nprobe_is_kept():
    x, centers = clustered(100, 8, 3)
    index = IVFIndex(x, ids(100), 3, nprobe=8)
    q = queries_near(centers, 1)[0]
    assert len(index.search(q, 100)) == 100
    assert len(index.search(q, 100, nprobe=50)) == 100
    index.search(q, 10, nprobe=1)
    assert index.nprobe == 8  # a per-call nprobe does not change the index


def test_k_zero_or_negative_gives_nothing_and_bad_queries_are_rejected():
    index, x, centers = make(n=200, d=16, nlist=5)
    q = queries_near(centers, 1)[0]
    assert index.search(q, 0) == [] and index.search(q, -3) == []
    with pytest.raises(ValueError):
        index.search(np.zeros(5, dtype=np.float32), 3)
    with pytest.raises(ValueError):
        index.search(q, 3, nprobe=0)


def test_works_with_a_single_vector():
    x = unit([[1.0, 2.0, 3.0]])
    index = IVFIndex(x, ["only"], 1)
    assert index.search(x[0], 5) == [(0, pytest.approx(1.0, abs=1e-6))]


# -------------------------------------------------------------------------- persistence
def test_a_saved_index_answers_exactly_like_the_original(tmp_path):
    index, x, centers = make(n=500, nlist=10, nprobe=3, seed=4)
    index.save(tmp_path / "sub" / "ivf.npz")  # the folder does not exist yet
    loaded = IVFIndex.load(tmp_path / "sub" / "ivf.npz")
    assert loaded.doc_ids == index.doc_ids
    assert loaded.nlist == 10 and loaded.nprobe == 3
    for q in queries_near(centers, 20, seed=14):
        for p in (1, 3, 10):
            assert loaded.search(q, 10, nprobe=p) == index.search(q, 10, nprobe=p)
        assert loaded.search(q, 10) == index.search(q, 10)


def test_the_file_is_plain_arrays_with_no_pickle(tmp_path):
    index, _, _ = make(n=200, nlist=5)
    index.save(tmp_path / "ivf.npz")
    with np.load(tmp_path / "ivf.npz", allow_pickle=False) as data:
        assert len(data.files) >= 4


def test_loading_something_else_is_an_error(tmp_path):
    x, _ = clustered(50, 8, 3)
    FlatIndex(x, ids(50)).save(tmp_path / "flat.npz")
    with pytest.raises(ValueError):
        IVFIndex.load(tmp_path / "flat.npz")
    np.savez(tmp_path / "other.npz", whatever=np.arange(3))
    with pytest.raises(ValueError):
        IVFIndex.load(tmp_path / "other.npz")


def test_a_file_with_another_version_number_is_refused(tmp_path):
    index = make(n=200, nlist=5)[0]
    index.save(tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["version"] = np.array(99)
    np.savez(tmp_path / "future.npz", **arrays)
    with pytest.raises(ValueError):
        IVFIndex.load(tmp_path / "future.npz")


def test_extra_memory_counts_the_centroids_and_the_lists_but_not_the_vectors():
    index, x, _ = make(n=1000, d=16, nlist=10)
    assert index.centroids.nbytes <= index.extra_nbytes < x.nbytes
    assert index.extra_nbytes >= 10 * 16 * 4 + 1000 * 4


# ------------------------------------------------------------------------------- speed
def test_building_20000_vectors_takes_seconds_and_searching_is_fast():
    x, centers = clustered(20_000, 64, 100, seed=7)
    start = time.perf_counter()
    index = IVFIndex(x, ids(20_000), 141, nprobe=4, seed=0)
    assert time.perf_counter() - start < 25.0, "building is too slow"
    qs = queries_near(centers, 200, seed=8)
    start = time.perf_counter()
    for q in qs:
        index.search(q, 10)
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
    flat, _ = build_dense_index(chunks, embedder)
    approx = IVFIndex(flat.vectors, flat.doc_ids, 5, nprobe=5, seed=0)
    want = DenseRetriever(embedder, flat, chunks).search("apple note3", 5)
    got = DenseRetriever(embedder, approx, chunks).search("apple note3", 5)
    assert [h.text for h in got] == [h.text for h in want]
