import heapq
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
        self.last_n_distances = 0

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

    def _sims(self, nodes, q):
        """Similarity of the query to several vectors at once; every one is counted."""
        self.last_n_distances += len(nodes)
        return self.vectors[nodes] @ q

    def _search_layer(self, q, entries, ef, level):
        """Walk one layer from the entry nodes; return the best ef found as (similarity, row)."""
        visited = set(entries)
        sims = self._sims(list(entries), q)
        # to expand: best first. heapq pops the smallest, so we store the negated similarity
        candidates = [(-float(s), e) for s, e in zip(sims, entries, strict=True)]
        heapq.heapify(candidates)
        # the best ef so far: the worst on top, so it is the one thrown out when a better one comes
        results = [(float(s), -e) for s, e in zip(sims, entries, strict=True)]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        while candidates:
            neg, c = heapq.heappop(candidates)
            if len(results) >= ef and -neg < results[0][0]:
                break  # the best unexpanded node is worse than all we keep: nothing can improve
            new = [e for e in self._links[c][level] if e not in visited]
            if not new:
                continue
            visited.update(new)
            for s, e in zip(self._sims(new, q), new, strict=True):
                s = float(s)
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, e))
                    heapq.heappush(results, (s, -e))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted(((s, -e) for s, e in results), key=lambda t: (-t[0], t[1]))
