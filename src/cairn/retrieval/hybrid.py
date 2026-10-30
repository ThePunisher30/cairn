from cairn.retrieval.fusion import METHODS, rrf, weighted_fusion
from cairn.types import Hit

FUSIONS = ("rrf", "weighted")


def key_of(hit):
    return (hit.path, hit.start_line, hit.end_line)


class HybridRetriever:
    def __init__(
        self, retrievers, *, method="rrf", depth=50, rrf_k=60, weights=None, normalize="minmax"
    ):
        retrievers = list(retrievers)
        if not retrievers:
            raise ValueError("a hybrid retriever needs atleast one retriever")
        if method not in FUSIONS:
            raise ValueError(f"method must be one of {FUSIONS}, it is currently {method}")
        if depth < 1:
            raise ValueError(f"depth must be >=1, it is currently {depth}")
        if rrf_k < 0:
            raise ValueError(f"rrf_k must be >=0, it is currently {rrf_k}")
        if normalize not in METHODS:
            raise ValueError(f"normalize must be one of {METHODS}, it is currently {normalize}")
        if weights is not None:
            weights = list(weights)
            if len(weights) != len(retrievers):
                raise ValueError(f"{len(retrievers)} retrievers but {len(weights)} weights")
            if any(w < 0 for w in weights):
                raise ValueError("weights must not be negative")
        self.retrievers = retrievers
        self.method = method
        self.depth = depth
        self.rrf_k = rrf_k
        self.weights = weights
        self.normalize = normalize

    def search(self, query, k):
        if k <= 0:
            return []
        depth = max(self.depth, k)
        lists = [retriever.search(query, depth) for retriever in self.retrievers]
        first = {}
        for hits in lists:
            for hit in hits:
                first.setdefault(key_of(hit), hit)
        if self.method == "rrf":
            fused = fused = rrf([[key_of(h) for h in hits] for hits in lists], self.rrf_k)
        else:
            scored = [[(key_of(h), h.score) for h in hits] for hits in lists]
            fused = weighted_fusion(scored, self.weights, self.normalize)
        result = []
        for key, score in fused[:k]:
            hit = first[key]
            result.append(Hit(hit.path, hit.start_line, hit.end_line, score, hit.text))
        return result
