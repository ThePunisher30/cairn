from pathlib import Path

import numpy as np

from cairn.ann.kmeans import kmeans

FORMAT_VERSION = 1


class IVFIndex:
    def __init__(self, vectors, doc_ids, nlist, *, nprobe=8, seed=0, max_iter=25, _parts=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2-D array (one row per vector)")
        if len(doc_ids) != len(vectors):
            raise ValueError("There must be exactly one doc_id per vector")
        if not 1 <= nlist <= len(vectors):
            raise ValueError(f"nlist must be between 1 and {len(vectors)}, got {nlist}")
        self.vectors = vectors
        self.doc_ids = list(doc_ids)
        self.nlist = nlist
        self.nprobe = nprobe
        self.last_n_distances = 0
        if _parts is not None:  # loading from disk: reuse the saved groups
            self.centroids, self.order, self.offsets = _parts
        else:
            self.centroids, labels = kmeans(vectors, nlist, seed=seed, max_iter=max_iter)
            self.order = np.argsort(labels, kind="stable").astype(np.int64)
            counts = np.bincount(labels, minlength=nlist)
            self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._check_nprobe(nprobe)

    def _check_nprobe(self, nprobe):
        if nprobe < 1:
            raise ValueError("nprobe must be atleast 1")

    @property
    def n_docs(self):
        return len(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[1]

    @property
    def extra_nbytes(self):
        return self.centroids.nbytes + self.order.nbytes + self.offsets.nbytes

    def list_rows(self, j):
        return self.order[self.offsets[j] : self.offsets[j + 1]]

    def search(self, query, k, nprobe=None):
        if k <= 0 or self.n_docs == 0:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query must have shape ({(self.dim,)}), got {query.shape}")
        nprobe = self.nprobe if nprobe is None else nprobe
        self._check_nprobe(nprobe)
        nprobe = min(nprobe, self.nlist)
        sims = self.centroids @ query
        lists = np.lexsort((np.arange(self.nlist), -sims))[:nprobe]
        rows = np.concatenate([self.list_rows(j) for j in lists])
        self.last_n_distances = self.nlist + len(rows)
        if len(rows) == 0:
            return []
        scores = self.vectors[rows] @ query
        order = np.lexsort((rows, -scores))[:k]
        return [(int(rows[i]), float(scores[i])) for i in order]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(
                f,
                version=np.array(FORMAT_VERSION),
                kind=np.array("ivf"),
                vectors=self.vectors,
                doc_ids=np.array(self.doc_ids, dtype=str),
                centroids=self.centroids,
                order=self.order,
                offsets=self.offsets,
                nprobe=np.array(self.nprobe),
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            ok = "version" in data and int(data["version"]) == FORMAT_VERSION
            if not ok or "kind" not in data or str(data["kind"]) != "ivf":
                raise ValueError(f"{path} is not a version-{FORMAT_VERSION} IVF index file")
            return cls(
                data["vectors"],
                data["doc_ids"].tolist(),
                len(data["centroids"]),
                nprobe=int(data["nprobe"]),
                _parts=(data["centroids"], data["order"], data["offsets"]),
            )
