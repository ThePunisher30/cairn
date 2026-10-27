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
        self.entry_point = 0 if n else None
        for i in range(1, n):
            self._insert(i)
        self.max_level = int(self.levels[self.entry_point]) if n else -1
        self.last_n_distances = 0  # the work of building is not the work of a search

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

    def _select(self, scored, limit):
        """Pick which neighbours to keep from (similarity, row) pairs, best first."""
        return [e for _, e in scored[:limit]]

    def _insert(self, i):
        """Link the new node i into the graph that holds rows 0 .. i-1."""
        q = self.vectors[i]
        level = int(self.levels[i])
        top = int(self.levels[self.entry_point])
        entry = [self.entry_point]
        # above the new node's own top layer: only travel down, one best node per layer
        for lv in range(top, level, -1):
            entry = [self._search_layer(q, entry, 1, lv)[0][1]]
        # from there down to layer 0: find neighbours, link both ways
        for lv in range(min(level, top), -1, -1):
            found = self._search_layer(q, entry, self.ef_construction, lv)
            chosen = self._select(found, self.M)
            self._links[i][lv] = list(chosen)
            cap = 2 * self.M if lv == 0 else self.M
            for e in chosen:
                links = self._links[e][lv]
                links.append(i)
                if len(links) > cap:  # too many friends: keep only the closest ones
                    sims = self.vectors[links] @ self.vectors[e]
                    ranked = sorted(
                        zip(sims.tolist(), links, strict=True), key=lambda t: (-t[0], t[1])
                    )
                    self._links[e][lv] = self._select(ranked, cap)
            entry = [e for _, e in found]
        if level > top:
            self.entry_point = i
