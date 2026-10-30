import math

METHODS = ("minmax", "zscore", "none")


def _first_seen(rankings):
    """Every key once, in the order it is first met reading the lists top to bottom, one list
    after the other."""
    order = {}
    for ranking in rankings:
        for key in ranking:
            order.setdefault(key, len(order))  # setdefault keeps the FIRST sighting
    return order


def rrf(rankings, k=60):
    if k < 0:
        raise ValueError(f"k must be >= 0, it is currently {k}")
    scores = {}
    for ranking in rankings:
        seen = set()  # a key repeated inside one list counts once, at its best rank
        rank = 0
        for key in ranking:
            if key in seen:
                continue
            seen.add(key)
            rank += 1  # the rank counts distinct keys, so a skipped repeat does not use one up
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = _first_seen(rankings)
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


def normalize(scores, method="minmax"):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, it is currently {method!r}")
    scores = [float(s) for s in scores]
    if not scores or method == "none":
        return scores
    if method == "minmax":
        low, high = min(scores), max(scores)
        if high == low:
            return [1.0] * len(scores)  # nothing to separate: every score is "the best"
        return [(s - low) / (high - low) for s in scores]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))  # population std
    if std == 0:
        return [0.0] * len(scores)
    return [(s - mean) / std for s in scores]


def weighted_fusion(scored, weights=None, method="minmax"):
    if weights is None:
        weights = [1.0] * len(scored)
    if len(weights) != len(scored):
        raise ValueError(f"{len(scored)} lists but {len(weights)} weights")
    if any(w < 0 for w in weights):
        raise ValueError("weights must not be negative")
    fused = {}
    for pairs, weight in zip(scored, weights, strict=True):
        first = {}
        for key, score in pairs:
            first.setdefault(key, score)  # a repeated key uses its first score
        keys = list(first)
        for key, value in zip(keys, normalize([first[key] for key in keys], method), strict=True):
            fused[key] = fused.get(key, 0.0) + weight * value  # a missing key adds nothing
    order = _first_seen([[key for key, _ in pairs] for pairs in scored])
    return sorted(fused.items(), key=lambda item: (-item[1], order[item[0]]))
