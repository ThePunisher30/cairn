import hashlib
from pathlib import Path

import numpy as np


def cache_key(model_key, text):
    return hashlib.sha256(f"{model_key}\n{text}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._vectors = {}
        if self.path is not None and self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                self._vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def __len__(self):
        return len(self._vectors)

    def __contains__(self, key):
        return key in self._vectors

    def get(self, key):
        return self._vectors.get(key)

    def put(self, key, vector):
        self._vectors[key] = np.array(vector, dtype=np.float32)

    def save(self):
        if self.path is None:
            raise ValueError("this cache has no path to save to")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors), dtype=str)
        vectors = (
            np.vstack(list(self._vectors.values()))
            if self._vectors
            else np.zeros((0, 0), dtype=np.float32)
        )
        with open(self.path, "wb") as f:
            np.savez(f, keys=keys, vectors=vectors)


def embed_cached(embedder, texts, cache):
    texts = list(texts)
    keys = [cache_key(embedder.cache_key, text) for text in texts]
    missing = {}
    for key, text in zip(keys, texts, strict=True):
        if key not in cache and key not in missing:
            missing[key] = text
    if missing:
        vectors = embedder.embed_documents(list(missing.values()))
        for key, vector in zip(missing, vectors, strict=True):
            cache.put(key, vector)
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32), 0
    return np.vstack([cache.get(key) for key in keys]), len(missing)
