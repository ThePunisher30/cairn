"""Phase 5, step 1: k-means, the clustering that IVF is built on.

k-means cuts n vectors into k groups of nearby vectors. Here the vectors have length 1, so
"nearby" means the dot product (cosine) is large, and each group is described by a centroid (the
group's average direction, scaled to length 1: "spherical" k-means).

    centroids, labels = kmeans(vectors, k, seed=0, max_iter=25)
        centroids: (k, d) array, every row of length 1
        labels:    (n,) integer array, labels[i] = the centroid that vector i is closest to

The steps: pick k starting centroids with k-means++ (the first at random, each next one at random
but likely to be far from the ones already chosen), then repeat "give every vector to its nearest
centroid, move every centroid to the average of its vectors" until nothing changes or max_iter.
"""

import time

import numpy as np
import pytest

from cairn.ann.kmeans import kmeans


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def blobs(centers, per_blob, noise, seed=0):
    """Tight groups of unit vectors around the given directions; also the true group of each."""
    rng = np.random.default_rng(seed)
    centers = unit(centers)
    points = [unit(c + noise * rng.normal(size=(per_blob, len(c)))) for c in centers]
    truth = np.repeat(np.arange(len(centers)), per_blob)
    return np.vstack(points), truth


def random_unit(n, d, seed=0):
    return unit(np.random.default_rng(seed).normal(size=(n, d)))


def best_similarity(x, centroids):
    return float((x @ centroids.T).max(axis=1).sum())


# --------------------------------------------------------------- the shape of the answer
def test_returns_k_unit_centroids_and_one_label_per_vector():
    x = random_unit(200, 16)
    centroids, labels = kmeans(x, 7, seed=1)
    assert centroids.shape == (7, 16)
    assert centroids.dtype == np.float32
    assert labels.shape == (200,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0 and labels.max() < 7
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)


def test_every_vector_is_labelled_with_its_nearest_centroid():
    x = random_unit(300, 12, seed=2)
    for max_iter in (1, 2, 25):  # also when it stops before converging
        centroids, labels = kmeans(x, 9, seed=2, max_iter=max_iter)
        assert np.array_equal(labels, (x @ centroids.T).argmax(axis=1)), max_iter


def test_centroids_are_the_normalised_average_of_their_vectors():
    x, _ = blobs(np.eye(5, 8) + 0.1, 40, 0.15, seed=3)
    centroids, labels = kmeans(x, 5, seed=3)
    for j in range(5):
        mean = x[labels == j].mean(axis=0)
        assert np.allclose(centroids[j], mean / np.linalg.norm(mean), atol=1e-4)


# ------------------------------------------------------------------ finding real groups
def test_well_separated_groups_are_recovered():
    x, truth = blobs(np.eye(4, 10) * 3 + 0.05, 60, 0.03, seed=4)
    for seed in range(5):
        _, labels = kmeans(x, 4, seed=seed)
        # each true group gets exactly one label, and no two groups share one
        group_labels = [set(labels[truth == g]) for g in range(4)]
        assert all(len(s) == 1 for s in group_labels), f"seed {seed}"
        assert len({next(iter(s)) for s in group_labels}) == 4, f"seed {seed}"


def test_k_means_plus_plus_finds_every_group_from_the_start():
    # With max_iter=1 only the starting centroids matter. Picking 4 random vectors would hit all
    # 4 groups only about 9% of the time; k-means++ spreads them out and almost always does.
    x, truth = blobs(np.eye(4, 10) * 3 + 0.05, 50, 0.03, seed=5)
    found = 0
    for seed in range(30):
        _, labels = kmeans(x, 4, seed=seed, max_iter=1)
        per_group = [set(labels[truth == g].tolist()) for g in range(4)]
        found += all(len(s) == 1 for s in per_group) and len({min(s) for s in per_group}) == 4
    assert found >= 27


def test_more_iterations_never_make_the_fit_worse_and_usually_improve_it():
    improved = 0
    for seed in range(5):
        x = random_unit(400, 8, seed=seed)
        one = best_similarity(x, kmeans(x, 12, seed=seed, max_iter=1)[0])
        many = best_similarity(x, kmeans(x, 12, seed=seed, max_iter=25)[0])
        assert many >= one - 1e-3
        improved += many > one + 1e-3
    assert improved >= 4


# ------------------------------------------------------------------------ determinism
def test_same_seed_same_result():
    x = random_unit(250, 10, seed=6)
    a, la = kmeans(x, 8, seed=11)
    b, lb = kmeans(x, 8, seed=11)
    assert np.array_equal(a, b) and np.array_equal(la, lb)


def test_different_seeds_both_give_valid_clusterings():
    x = random_unit(250, 10, seed=7)
    for seed in (1, 2):
        centroids, labels = kmeans(x, 8, seed=seed)
        assert np.array_equal(labels, (x @ centroids.T).argmax(axis=1))


# --------------------------------------------------------------------------- edge cases
def test_one_cluster_is_the_normalised_mean():
    x = random_unit(50, 6, seed=8)
    centroids, labels = kmeans(x, 1)
    mean = x.mean(axis=0)
    assert np.allclose(centroids[0], mean / np.linalg.norm(mean), atol=1e-5)
    assert set(labels) == {0}


def test_as_many_clusters_as_vectors_gives_every_vector_its_own():
    x = random_unit(12, 20, seed=9)
    centroids, labels = kmeans(x, 12, seed=9)
    assert sorted(labels.tolist()) == list(range(12))
    assert np.allclose(np.sort(centroids, axis=0), np.sort(x, axis=0), atol=1e-5)


def test_no_cluster_is_left_empty():
    for seed in range(5):
        x = random_unit(150, 6, seed=seed)
        _, labels = kmeans(x, 25, seed=seed)
        assert len(set(labels.tolist())) == 25, f"seed {seed}"


def test_fewer_distinct_vectors_than_clusters_still_gives_k_unit_centroids():
    # 5 distinct directions, 6 copies of each, k = 8: three clusters must stay empty. They must
    # not turn into all-zero centroids (a zero centroid is not a direction and matches nothing).
    base = random_unit(5, 6, seed=10)
    x = np.repeat(base, 6, axis=0)
    for seed in range(5):
        centroids, labels = kmeans(x, 8, seed=seed)
        assert centroids.shape == (8, 6) and np.isfinite(centroids).all()
        assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5), f"seed {seed}"
        assert labels.min() >= 0 and labels.max() < 8


def test_identical_vectors_do_not_crash():
    x = np.tile(unit([[1.0, 2.0, 3.0, 4.0]]), (20, 1))
    centroids, labels = kmeans(x, 3, seed=0)
    assert centroids.shape == (3, 4)
    assert np.isfinite(centroids).all()
    assert labels.shape == (20,) and labels.min() >= 0 and labels.max() < 3


def test_rejects_a_bad_k_and_bad_input():
    x = random_unit(10, 4)
    for bad_k in (0, -1, 11):
        with pytest.raises(ValueError):
            kmeans(x, bad_k)
    with pytest.raises(ValueError):
        kmeans(np.zeros(5, dtype=np.float32), 2)
    with pytest.raises(ValueError):
        kmeans(np.zeros((0, 4), dtype=np.float32), 1)


def test_the_input_is_not_changed():
    x = random_unit(100, 8, seed=12)
    before = x.copy()
    kmeans(x, 6, seed=0)
    assert np.array_equal(x, before)


def test_accepts_float64_input():
    x = random_unit(100, 8, seed=13).astype(np.float64)
    centroids, labels = kmeans(x, 6, seed=0)
    assert centroids.dtype == np.float32 and labels.shape == (100,)


# ----------------------------------------------------------------------------- speed
def test_clustering_20000_vectors_takes_seconds_not_minutes():
    x = random_unit(20_000, 64, seed=14)
    start = time.perf_counter()
    kmeans(x, 100, seed=0, max_iter=10)
    assert time.perf_counter() - start < 12.0, "k-means is too slow: use matrix products"
