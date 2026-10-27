import numpy as np


def _assign(x, centroids, block=8192):
    """For every vector: the centroid it is most similar to, and that similarity."""
    labels = np.empty(len(x), dtype=np.int64)
    best = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), block):
        sims = x[start : start + block] @ centroids.T  # (block, k) dot products
        labels[start : start + block] = sims.argmax(axis=1)
        best[start : start + block] = sims.max(axis=1)
    return labels, best


def _normalise(c):
    """Scale every row to length 1; a row of zeros stays zeros."""
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    return np.divide(c, norms, out=np.zeros_like(c), where=norms > 0)


def _seed_centroids(x, k, rng):
    """k-means++: the first at random, each next one likely to be far from those chosen."""
    n = len(x)
    chosen = [int(rng.integers(n))]
    closest = x @ x[chosen[0]]  # similarity of every vector to its nearest chosen centroid
    for _ in range(1, k):
        d2 = np.clip(2.0 - 2.0 * closest, 0.0, None).astype(np.float64)  # squared distance
        total = d2.sum()
        if total <= 0:  # every vector is identical to a chosen one
            nxt = int(rng.integers(n))
        else:
            nxt = int(rng.choice(n, p=d2 / total))
        chosen.append(nxt)
        closest = np.maximum(closest, x @ x[nxt])
    return x[chosen].copy()


def kmeans(vectors, k, *, seed=0, max_iter=25):
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("vectors must be non empty 2D array")
    if not 1 <= k <= len(x):
        raise ValueError(f"k must be between 1 and {len(x)}, got {k}")
    rng = np.random.default_rng(seed)
    centroids = _seed_centroids(x, k, rng)
    previous = None
    for _ in range(max_iter):
        labels, best = _assign(x, centroids)
        if previous is not None and np.array_equal(labels, previous):
            break  # nothing moved: converged
        previous = labels
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, x)  # add every vector into its cluster's row
        counts = np.bincount(labels, minlength=k)
        centroids = _normalise(sums)
        empty = np.flatnonzero((counts == 0) | (np.linalg.norm(sums, axis=1) == 0))
        if len(empty):  # a cluster lost all its vectors: restart it at the worst-fitting vectors
            worst = np.argsort(best, kind="stable")[: len(empty)]
            centroids[empty] = x[worst]
        labels, _ = _assign(x, centroids)
    return centroids, labels
