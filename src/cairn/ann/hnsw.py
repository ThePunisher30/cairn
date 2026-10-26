import math

import numpy as np


class HNSWIndex:
    def __init__(
        self,
        vectors,
        doc_ids,
        *,
        M=16,
        ef_construction=100,
        ef_search=50,
        seed=0,
        heuristic=True,
    ):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2-D array (one row per chunk)")
        if len(doc_ids) != len(vectors):
            raise ValueError("There must be exactly one doc id per vector")
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be at least 1")
        self.vectors = vectors
        self.doc_ids = list(doc_ids)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.heuristic = heuristic

        n = len(vectors)
        rng = np.random.default_rng(seed)
        ml = 1.0 / math.log(M)
        self.levels = np.floor(-np.log(1.0 - rng.random(n)) * ml).astype(np.int64)
        self._links = [[[] for _ in range(int(top) + 1)] for top in self.levels]
        self.entry_point = int(np.argmax(self.levels)) if n else None
        self.max_level = int(self.levels[self.entry_point]) if n else -1

    @property
    def n_docs(self):
        return len(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[1]

    def neighbors(self, node, level):
        return list(self._links[node][level])
