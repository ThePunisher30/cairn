from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


class FlatIndex:
    def __init__(self, vectors, doc_ids):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2-D array (one row per chunk)")
        if len(doc_ids) != len(vectors):
            raise ValueError("There must be exactly one doc id per vector")
        self.vectors = vectors
        self.doc_ids = list(doc_ids)

    @property
    def n_docs(self):
        return len(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[1]

    @property
    def nbytes(self):
        return self.vectors.nbytes

    def search(self, query, k):
        if k <= 0 or self.n_docs == 0:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query must have shape ({(self.dim,)}), got {query.shape}")
        scores = self.vectors @ query
        k = min(k, self.n_docs)
        if k < self.n_docs:
            best = np.argpartition(-scores, k - 1)[:k]
            candidates = np.flatnonzero(scores >= scores[best].min())
        else:
            candidates = np.arange(self.n_docs)
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(
                f,
                version=np.array(FORMAT_VERSION),
                vectors=self.vectors,
                doc_ids=np.array(self.doc_ids, dtype=str),
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if "version" not in data or int(data["version"]) != FORMAT_VERSION:
                raise ValueError(f"{path} is not a version-{FORMAT_VERSION} dense index file")
            return cls(data["vectors"], data["doc_ids"].tolist())
