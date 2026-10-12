import random
from pathlib import Path

from cairn.types import Hit, Question


def line_windows(root, window=40, suffixes=(".md", ".py")) -> list[Hit]:
    if window < 1:
        raise ValueError(f"window must be >=1, got {window}")
    hits = []
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)
    for path in files:
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        relative = path.relative_to(root).as_posix()
        for start in range(1, n + 1, window):
            end = min(start + window - 1, n)
            hits.append(Hit(relative, start, end))
    return hits


class RandomRetriever:
    def __init__(self, candidates, seed=0):
        self.candidates = list(candidates)
        self.rng = random.Random(seed)

    def search(self, query, k):
        if k <= 0:
            return []
        return self.rng.sample(self.candidates, min(k, len(self.candidates)))


class FirstNRetriever:
    def __init__(self, candidates):
        self.candidates = list(candidates)

    def search(self, query, k):
        return self.candidates[: max(k, 0)]


class OracleRetriever:
    def __init__(self, questions):
        self.by_text = {q.text: q.gold for q in questions}

    def search(self, query, k):
        gold = self.by_text.get(query, ())
        hits = [Hit(g.path, g.start_line, g.end_line, score=1.0) for g in gold]
        return hits[: max(k, 0)]
