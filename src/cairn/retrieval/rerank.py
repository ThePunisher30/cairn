from cairn.types import Hit


class Reranker:
    def __init__(self, model, name, *, batch_size=16, max_chars=2000):
        if batch_size < 1:
            raise ValueError(f"batch size must be >= 1, currently it is {batch_size}")
        if max_chars < 1:
            raise ValueError(f"max chars must be >=1 , currently it is {max_chars}")
        self.model = model
        self.name = name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def score(self, query, passages):
        passages = [p[: self.max_chars] for p in passages]
        todo = []
        for passage in passages:
            key = (self.name, query, passage)
            if key in self.cache:
                self.hits += 1
            elif passage not in todo:
                todo.append(passage)
        todo.sort(key=len)
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            scores = self.model.score([(query, passage) for passage in batch])
            if len(scores) != len(batch):
                raise ValueError(f"the model returned {len(scores)} scores for {len(batch)} pairs")
            for passage, value in zip(batch, scores, strict=True):
                self.cache[(self.name, query, passage)] = float(value)
            self.misses += len(batch)
        return [self.cache[(self.name, query, passage)] for passage in passages]


class RerankingRetriever:
    def __init__(self, base, reranker, *, depth=30):
        if depth < 1:
            raise ValueError(f"depth must be >=1, currently it is {depth}")
        self.base = base
        self.reranker = reranker
        self.depth = depth

    def search(self, query, k):
        if k <= 0:
            return []
        candidates = self.base.search(query, self.depth)
        scores = self.reranker.score(query, [hit.text for hit in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        result = []
        for i in order[:k]:
            hit = candidates[i]
            result.append(Hit(hit.path, hit.start_line, hit.end_line, scores[i], hit.text))
        return result
