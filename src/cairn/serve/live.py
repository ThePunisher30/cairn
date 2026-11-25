"""An index that can change while it is being searched (Phase 9).

`LiveIndex` holds the chunks, the BM25 postings and the dense vectors together, in "slots". A slot
is a row number that is never reused. Removing a chunk does not shift the others: it leaves a
tombstone (the slot is marked dead and skipped by every search). Adding appends a new slot.

Why tombstones: deleting from the middle of an inverted index or a vector matrix would mean
renumbering everything after it, and every posting that mentions those numbers. A tombstone is
one write. The price is paid later: dead postings and dead matrix rows are still scanned, and
still use memory, until `compact()` rebuilds the index without them. `compact_threshold` is how
many dead slots (as a fraction of all slots) are tolerated before that rebuild happens by itself.

Scores stay exact: the BM25 statistics (number of documents, document frequencies, average
length) count only living slots, so a live index scores like one built fresh from the same chunks
(the tests check this against `InvertedIndex.build` and `FlatIndex`).
"""

import heapq
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cairn.dense.flat_index import FlatIndex
from cairn.jsonl import write_jsonl
from cairn.retrieval.bm25 import idf, tf_weight
from cairn.retrieval.tokenizer import Tokenizer
from cairn.types import Hit


class RWLock:
    """Many readers at once, or one writer. A waiting writer holds back new readers, so a steady
    stream of searches cannot starve an update."""

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writing or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writing or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()


class LiveIndex:
    def __init__(self, dim, tokenizer=None, compact_threshold=0.25, min_dead_to_compact=50):
        if not 0 < compact_threshold <= 1:
            raise ValueError("compact_threshold must be in (0, 1]")
        self.dim = dim
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.compact_threshold = compact_threshold
        self.min_dead_to_compact = min_dead_to_compact
        self.lock = RWLock()
        self.generation = 0  # +1 for every change; caches key on it
        self.compactions = 0
        self._reset()

    def _reset(self):
        self._chunks = []  # slot -> Chunk, or None for a tombstone
        self._terms = []  # slot -> Counter of the terms in it (kept to undo it on removal)
        self._lengths = []  # slot -> number of tokens
        self._postings = {}  # term -> [(slot, tf)], may mention dead slots
        self._df = defaultdict(int)  # term -> number of LIVING slots that contain it
        self._by_path = defaultdict(list)  # path -> living slots
        self._matrix = np.zeros((16, self.dim), dtype=np.float32)
        self._alive = np.zeros(16, dtype=bool)
        self._n_alive = 0
        self._total_length = 0  # tokens in living slots

    # -- reading ----------------------------------------------------------------------------

    @property
    def n_slots(self):
        return len(self._chunks)

    @property
    def n_alive(self):
        return self._n_alive

    @property
    def n_dead(self):
        return self.n_slots - self._n_alive

    @property
    def dead_fraction(self):
        return self.n_dead / self.n_slots if self.n_slots else 0.0

    def paths(self):
        return {path for path, slots in self._by_path.items() if slots}

    def chunks_of(self, path):
        return [self._chunks[slot] for slot in self._by_path.get(path, ())]

    def vectors_of(self, path):
        return {self._chunks[s].id: self._matrix[s].copy() for s in self._by_path.get(path, ())}

    def stats(self):
        return {
            "chunks": self.n_alive,
            "slots": self.n_slots,
            "tombstones": self.n_dead,
            "dead_fraction": round(self.dead_fraction, 4),
            "generation": self.generation,
            "compactions": self.compactions,
            "terms": len(self._postings),
        }

    def bm25_search(self, query, k, k1=1.2, b=0.75):
        """Same scoring as `BM25Retriever` over the living chunks. Call with the read lock held."""
        if k <= 0 or self._n_alive == 0:
            return []
        avg_length = self._total_length / self._n_alive
        scores = defaultdict(float)
        for term in set(self.tokenizer.tokenize(query)):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            weight = idf(self._n_alive, df)
            for slot, tf in self._postings[term]:
                if self._chunks[slot] is None:
                    continue  # a tombstone: still in the postings, never in a result
                scores[slot] += weight * tf_weight(tf, self._lengths[slot], avg_length, k1, b)
        best = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        return [(slot, score) for slot, score in best]

    def dense_search(self, vector, k):
        """Best `k` living slots by dot product. Call with the read lock held."""
        if k <= 0 or self._n_alive == 0:
            return []
        n = self.n_slots
        scores = self._matrix[:n] @ np.asarray(vector, dtype=np.float32)
        scores[~self._alive[:n]] = -np.inf  # a tombstone must never win
        k = min(k, self._n_alive)
        best = np.argpartition(-scores, k - 1)[:k]
        candidates = np.flatnonzero(scores >= scores[best].min())
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]

    def hit(self, slot, score):
        chunk = self._chunks[slot]
        return Hit(chunk.path, chunk.start_line, chunk.end_line, score, chunk.text)

    # -- changing (all of these need the write lock, `apply` takes it) -----------------------

    def _grow(self, needed):
        capacity = len(self._alive)
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        matrix = np.zeros((capacity, self.dim), dtype=np.float32)
        matrix[: self.n_slots] = self._matrix[: self.n_slots]
        alive = np.zeros(capacity, dtype=bool)
        alive[: self.n_slots] = self._alive[: self.n_slots]
        self._matrix, self._alive = matrix, alive

    def _add(self, chunk, vector):
        slot = self.n_slots
        self._grow(slot + 1)
        terms = Counter(self.tokenizer.tokenize(chunk.index_text))
        length = sum(terms.values())
        self._chunks.append(chunk)
        self._terms.append(terms)
        self._lengths.append(length)
        for term, tf in terms.items():
            self._postings.setdefault(term, []).append((slot, tf))
            self._df[term] += 1
        self._matrix[slot] = vector
        self._alive[slot] = True
        self._by_path[chunk.path].append(slot)
        self._n_alive += 1
        self._total_length += length

    def _tombstone(self, slot):
        for term in self._terms[slot]:
            self._df[term] -= 1
        self._total_length -= self._lengths[slot]
        self._n_alive -= 1
        self._alive[slot] = False
        self._chunks[slot] = None
        self._terms[slot] = None  # free the memory; the postings keep the dead slot until compact

    def _remove_path(self, path):
        removed = 0
        for slot in self._by_path.pop(path, ()):
            self._tombstone(slot)
            removed += 1
        return removed

    def apply(self, remove_paths=(), add=()):
        """One atomic change: drop every chunk of `remove_paths`, then add `add`, a list of
        (chunks, vectors) per file. Searches see the index before it or after it, never half.
        Returns what happened."""
        with self.lock.write():
            removed = sum(self._remove_path(path) for path in remove_paths)
            added = 0
            for chunks, vectors in add:
                if len(chunks) != len(vectors):
                    raise ValueError("one vector per chunk")
                for chunk, vector in zip(chunks, vectors, strict=True):
                    self._add(chunk, vector)
                    added += 1
            if removed or added:
                self.generation += 1
            compacted = self._compact_if_needed()
            return {"removed": removed, "added": added, "compacted": compacted}

    def _compact_if_needed(self):
        if self.n_dead >= self.min_dead_to_compact and self.dead_fraction > self.compact_threshold:
            self._compact()
            return True
        return False

    def compact(self):
        with self.lock.write():
            self._compact()

    def _compact(self):
        """Rebuilds every structure without the tombstones. Cost: proportional to the whole
        index, paid under the write lock, so searches wait for it."""
        live = [slot for slot in range(self.n_slots) if self._chunks[slot] is not None]
        remap = {old: new for new, old in enumerate(live)}
        matrix = np.zeros((max(16, len(live)), self.dim), dtype=np.float32)
        matrix[: len(live)] = self._matrix[live]
        postings = {}
        for term, entries in self._postings.items():
            kept = [(remap[slot], tf) for slot, tf in entries if slot in remap]
            if kept:
                postings[term] = kept
        by_path = defaultdict(list)
        for old in live:
            by_path[self._chunks[old].path].append(remap[old])
        self._chunks = [self._chunks[s] for s in live]
        self._terms = [self._terms[s] for s in live]
        self._lengths = [self._lengths[s] for s in live]
        self._postings = postings
        self._df = defaultdict(int, {t: len(e) for t, e in postings.items()})
        self._by_path = by_path
        self._matrix = matrix
        self._alive = np.zeros(len(matrix), dtype=bool)
        self._alive[: len(live)] = True
        self.compactions += 1

    # -- building and saving -----------------------------------------------------------------

    @classmethod
    def build(cls, chunks, vectors, **kwargs):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        dim = vectors.shape[1] if vectors.ndim == 2 else kwargs.pop("dim")
        index = cls(dim, **kwargs)
        for chunk, vector in zip(chunks, vectors, strict=True):
            index._add(chunk, vector)
        return index

    def snapshot(self):
        """(chunks, vectors) of the living slots, in (path, start_line) order like chunks.jsonl."""
        with self.lock.read():
            live = [s for s in range(self.n_slots) if self._chunks[s] is not None]
            live.sort(key=lambda s: (self._chunks[s].path, self._chunks[s].start_line))
            return [self._chunks[s] for s in live], self._matrix[live].copy()

    def save(self, chunks_path, dense_path):
        """Writes chunks.jsonl and the dense index, each to a temporary file that then replaces
        the old one, so a crash never leaves half a file."""
        import dataclasses
        import os

        chunks, vectors = self.snapshot()
        chunks_path, dense_path = Path(chunks_path), Path(dense_path)
        tmp_chunks = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
        tmp_dense = dense_path.with_suffix(".tmp.npz")
        write_jsonl(tmp_chunks, (dataclasses.asdict(c) for c in chunks))
        FlatIndex(vectors, [c.id for c in chunks]).save(tmp_dense)
        os.replace(tmp_chunks, chunks_path)
        os.replace(tmp_dense, dense_path)


class LiveBM25:
    """A retriever over a LiveIndex: the same `search(query, k)` as `BM25Retriever`."""

    def __init__(self, index):
        self.index = index

    def search(self, query, k):
        with self.index.lock.read():
            return [self.index.hit(slot, score) for slot, score in self.index.bm25_search(query, k)]


class LiveDense:
    """A retriever over a LiveIndex: the same `search(query, k)` as `DenseRetriever`."""

    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index

    def search(self, query, k):
        if k <= 0:
            return []
        vector = self.embedder.embed_query(query)  # the slow part: done before taking the lock
        with self.index.lock.read():
            return [
                self.index.hit(slot, score) for slot, score in self.index.dense_search(vector, k)
            ]
