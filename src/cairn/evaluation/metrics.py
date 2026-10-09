import math

from cairn.types import Hit, Span


def overlaps(span1: Hit | Span, span2: Hit | Span) -> bool:
    """Check if two spans overlap."""
    if span1.path != span2.path:
        return False
    return not (span1.end_line < span2.start_line or span2.end_line < span1.start_line)


def _walk(hits, gold):
    """Walk down the ranking and apply the credit rule.

    Returns (flags, found): one True/False flag per hit, and the set of gold indices
    covered so far. A hit is relevant only if it overlaps a gold span that no
    earlier hit had already covered.
    """
    found = set()  # indices of gold spans already covered
    flags = []  # one True/False per hit
    for hit in hits:
        new = {i for i, g in enumerate(gold) if i not in found and overlaps(hit, g)}
        flags.append(bool(new))
        found |= new
    return flags, found


def relevance_flags(hits, gold) -> list[bool]:
    """One flag per hit: True if it overlaps a gold span not covered by an earlier hit."""
    flags, _ = _walk(hits, gold)
    return flags


def _check_k(k):
    if k < 1:
        raise ValueError(f"k must be >= 1, it is currently {k}")


def hit_at_k(hits, gold, k) -> float:
    _check_k(k)
    flags, _ = _walk(hits[:k], gold)
    return float(any(flags))


def recall_at_k(hits, gold, k) -> float:
    _check_k(k)
    if not gold:
        raise ValueError("gold must be non-empty")
    _, found = _walk(hits[:k], gold)
    return len(found) / len(gold)


def reciprocal_rank(hits, gold, k=None) -> float:
    if k is not None:
        _check_k(k)
        hits = hits[:k]
    flags = relevance_flags(hits, gold)
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / position
    return 0.0


def ndcg_at_k(hits, gold, k) -> float:
    _check_k(k)
    if not gold:
        raise ValueError("gold must be non-empty")
    flags = relevance_flags(hits[:k], gold)
    dcg = sum(1.0 / (math.log2(i + 2)) for i, flag in enumerate(flags) if flag)
    ideal_dcg = sum(1.0 / (math.log2(i + 2)) for i in range(min(k, len(gold))))
    return dcg / ideal_dcg


def percentile(values, p):
    if not values:
        raise ValueError("values must be non-empty")
    if not (0 < p <= 100):
        raise ValueError(f"p must be in (0, 100], it is currently {p}")
    ordered = sorted(values)
    rank = math.ceil(p * len(ordered) / 100)
    return ordered[rank - 1]
