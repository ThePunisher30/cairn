"""Answer caches (Phase 9): an exact one, and one that matches questions by meaning.

Both key every entry on a *scope*, a tuple such as (index generation, k, model). The index
generation is what keeps them honest: when the corpus changes, the generation changes, and an
answer written from the old chunks can no longer be found. Without it a cache would keep serving
text that cites files that were since edited or deleted (the classic cache invalidation bug).
"""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np


def normalize_question(question):
    """Same question, different spacing, case or final punctuation: one key."""
    return " ".join(question.lower().split()).rstrip("?!. ")


class LRUCache:
    """Least recently used out when full. `ttl_s` (optional) expires entries by age."""

    def __init__(self, max_entries=1000, ttl_s=None, clock=time.monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items = OrderedDict()  # key -> (value, stored_at)
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            entry = self._items.get(key)
            if entry is not None and self.ttl_s is not None:
                if self._clock() - entry[1] > self.ttl_s:
                    del self._items[key]
                    entry = None
            if entry is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key, value):
        with self._lock:
            self._items[key] = (value, self._clock())
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self):
        return len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()


@dataclass(frozen=True)
class SemanticHit:
    value: object
    similarity: float
    matched_question: str


class SemanticCache:
    """Finds a stored answer for a question that *means* the same thing: embed the question, take
    the stored question with the highest cosine similarity, accept it if it reaches `threshold`.

    The failure mode is built in: two questions can be close in embedding space and still want
    different answers ("how do I enable X" / "how do I disable X"). A higher threshold means
    fewer false hits and fewer hits. `recent_hits` keeps the last matches so they can be read."""

    def __init__(self, embedder, threshold=0.92, max_entries=500, recent=50):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries = []  # (scope, question, vector, value), oldest first
        self.hits = 0
        self.misses = 0
        self.recent_hits = deque(maxlen=recent)  # (question, matched question, similarity)

    def embed(self, question):
        return self.embedder.embed_query(question)

    def lookup(self, question, scope, vector=None):
        """Returns a `SemanticHit` or None. Pass `vector` if you already embedded the question."""
        if vector is None:
            vector = self.embed(question)
        with self._lock:
            best, best_similarity = None, -1.0
            for entry in self._entries:
                if entry[0] != scope:
                    continue
                similarity = float(np.dot(entry[2], vector))
                if similarity > best_similarity:
                    best, best_similarity = entry, similarity
            if best is None or best_similarity < self.threshold:
                self.misses += 1
                return None
            self.hits += 1
            self.recent_hits.append((question, best[1], round(best_similarity, 4)))
            return SemanticHit(best[3], best_similarity, best[1])

    def store(self, question, scope, value, vector=None):
        if vector is None:
            vector = self.embed(question)
        with self._lock:
            self._entries.append((scope, question, np.asarray(vector, dtype=np.float32), value))
            del self._entries[: max(0, len(self._entries) - self.max_entries)]

    def purge_except(self, generation):
        """Drops entries from other index generations (scope[0] is the generation)."""
        with self._lock:
            self._entries = [e for e in self._entries if e[0][0] == generation]

    def __len__(self):
        return len(self._entries)
