"""Phase 6, step 2: HybridRetriever (`src/cairn/retrieval/hybrid.py`).

    HybridRetriever(retrievers, *, method="rrf", depth=50, rrf_k=60, weights=None,
                    normalize="minmax")
    .search(query, k) -> [Hit, ...]

It asks EVERY retriever in `retrievers` for the top `max(depth, k)` hits of the same query, fuses
the lists with your `fusion.py`, and returns the best `k` fused hits, best first.

- A chunk is identified by its place: (path, start_line, end_line). The same place in two lists is
  one chunk.
- `method="rrf"` fuses the RANKS (`rrf_k` is the constant). `method="weighted"` fuses the scores
  of the hits (`Hit.score`) with `weights` (one per retriever, default equal) and `normalize`.
- The returned Hit has the chunk's path, lines and text (taken from the first retriever, in the
  order of `retrievers`, that returned it) and the FUSED score in `score`.
- `k <= 0` returns `[]` without asking anything. Fewer than `k` distinct chunks give fewer hits.
- ValueError for: no retrievers, a method not in ("rrf", "weighted"), `depth < 1`, `rrf_k < 0`,
  `normalize` not in ("minmax", "zscore", "none"), `weights` of the wrong length or negative.

The retrievers here are scripted fakes; the last test uses real BM25 and dense retrievers.
"""

import pytest

from cairn.dense.embedder import Embedder
from cairn.dense.models import HashingModel
from cairn.dense.retriever import DenseRetriever, build_dense_index
from cairn.evaluation.harness import run_retriever, summarize
from cairn.ingest.chunkers import make_chunk
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex
from cairn.types import Hit, Question, Span


def hit(name, score=0.0, text=None):
    """A hit whose place is decided by its name: 'a' is lines 1-1, 'b' lines 2-2, and so on."""
    line = ord(name) - ord("a") + 1
    return Hit("f.md", line, line, score, text if text is not None else f"text of {name}")


class Scripted:
    """A retriever that always answers with the same list and remembers how it was asked."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.hits[:k]


def names(hits):
    return [chr(ord("a") + h.start_line - 1) for h in hits]


# ------------------------------------------------------------------------------ rrf
def test_rrf_merges_two_lists_and_the_chunk_found_by_both_comes_first():
    first = Scripted([hit("a"), hit("b"), hit("c")])
    second = Scripted([hit("d"), hit("c"), hit("e")])
    result = HybridRetriever([first, second]).search("q", 5)
    assert names(result) == ["c", "a", "d", "b", "e"]


def test_the_score_is_the_fused_score_and_the_text_is_kept():
    first = Scripted([hit("a", 99.0, "from first")])
    second = Scripted([hit("a", 0.3, "from second")])
    (only,) = HybridRetriever([first, second], rrf_k=10).search("q", 5)
    assert only.score == pytest.approx(2 / 11)
    assert (only.path, only.start_line, only.end_line) == ("f.md", 1, 1)
    assert only.text == "from first"  # the first retriever that returned it


def test_the_same_place_in_both_lists_is_one_chunk():
    result = HybridRetriever([Scripted([hit("a"), hit("b")]), Scripted([hit("b")])]).search("q", 9)
    assert names(result) == ["b", "a"]


def test_the_same_start_line_with_another_end_line_is_another_chunk():
    longer = Hit("f.md", 1, 2, 0.0, "a longer chunk")
    result = HybridRetriever([Scripted([hit("a")]), Scripted([longer])]).search("q", 9)
    assert sorted((h.start_line, h.end_line) for h in result) == [(1, 1), (1, 2)]


def test_the_same_lines_in_another_file_are_another_chunk():
    other = Hit("g.md", 1, 1, 0.0, "elsewhere")
    result = HybridRetriever([Scripted([hit("a")]), Scripted([other])]).search("q", 9)
    assert sorted(h.path for h in result) == ["f.md", "g.md"]


def test_rrf_k_is_passed_on():
    retrievers = [Scripted([hit("a"), hit("b")])]
    flat = HybridRetriever(retrievers, rrf_k=1000).search("q", 2)
    steep = HybridRetriever(retrievers, rrf_k=0).search("q", 2)
    assert flat[0].score / flat[1].score == pytest.approx(1002 / 1001)
    assert steep[0].score / steep[1].score == pytest.approx(2.0)


def test_a_single_retriever_keeps_its_order():
    result = HybridRetriever([Scripted([hit("c"), hit("a"), hit("b")])]).search("q", 3)
    assert names(result) == ["c", "a", "b"]


# --------------------------------------------------------------------- weighted
BM25_LIKE = [hit("a", 12.0), hit("b", 11.0), hit("c", 3.0)]
DENSE_LIKE = [hit("b", 0.9), hit("a", 0.1), hit("c", 0.0)]


def test_weighted_fusion_uses_the_hit_scores_and_normalises_them():
    bm25, dense = Scripted(BM25_LIKE), Scripted(DENSE_LIKE)
    result = HybridRetriever([bm25, dense], method="weighted").search("q", 3)
    assert names(result) == ["b", "a", "c"]
    assert [h.score for h in result] == pytest.approx([1 + 8 / 9, 1 + 1 / 9, 0.0])


def test_without_normalising_the_larger_scale_wins():
    bm25, dense = Scripted(BM25_LIKE), Scripted(DENSE_LIKE)
    result = HybridRetriever([bm25, dense], method="weighted", normalize="none").search("q", 3)
    assert names(result) == ["a", "b", "c"]
    assert result[0].score == pytest.approx(12.1)


def test_weights_decide_whose_opinion_counts_more():
    bm25, dense = Scripted(BM25_LIKE), Scripted(DENSE_LIKE)
    for weights, best in (([1, 0], "a"), ([0, 1], "b")):
        retriever = HybridRetriever([bm25, dense], method="weighted", weights=weights)
        assert names(retriever.search("q", 1)) == [best]


def test_zscore_normalisation_is_available():
    bm25, dense = Scripted(BM25_LIKE), Scripted(DENSE_LIKE)
    retriever = HybridRetriever([bm25, dense], method="weighted", normalize="zscore")
    assert len(retriever.search("q", 3)) == 3


# ------------------------------------------------------------- how it asks, and k
def test_every_retriever_is_asked_the_same_question_for_depth_hits():
    first, second = Scripted([hit("a")]), Scripted([hit("b")])
    HybridRetriever([first, second], depth=7).search("how do I?", 3)
    assert first.calls == [("how do I?", 7)]
    assert second.calls == [("how do I?", 7)]


def test_asking_for_more_than_depth_asks_for_k():
    first = Scripted([hit("a")])
    HybridRetriever([first], depth=3).search("q", 10)
    assert first.calls == [("q", 10)]


def test_k_limits_the_result_and_fewer_chunks_give_fewer_hits():
    retriever = HybridRetriever([Scripted([hit("a"), hit("b"), hit("c")])])
    assert len(retriever.search("q", 2)) == 2
    assert len(retriever.search("q", 50)) == 3


def test_k_zero_or_less_gives_nothing_and_asks_nobody():
    first = Scripted([hit("a")])
    retriever = HybridRetriever([first])
    assert retriever.search("q", 0) == []
    assert retriever.search("q", -3) == []
    assert first.calls == []


def test_retrievers_with_no_hits_give_no_hits():
    assert HybridRetriever([Scripted([]), Scripted([])]).search("q", 5) == []


# ------------------------------------------------------------------- arguments
@pytest.mark.parametrize(
    "arguments",
    [
        {"method": "borda"},
        {"depth": 0},
        {"rrf_k": -1},
        {"normalize": "rank"},
        {"weights": [1.0]},
        {"weights": [1.0, -1.0]},
    ],
)
def test_bad_arguments_are_refused(arguments):
    with pytest.raises(ValueError):
        HybridRetriever([Scripted([]), Scripted([])], **arguments)


def test_no_retrievers_is_refused():
    with pytest.raises(ValueError):
        HybridRetriever([])


# --------------------------------------------------------------- the real thing
def test_it_fuses_real_bm25_and_dense_retrievers_and_the_harness_can_score_it():
    chunks = [
        make_chunk("a.md", 1, 1, "apple banana apple", "prose"),
        make_chunk("a.md", 2, 2, "banana cherry", "prose"),
        make_chunk("a.md", 3, 3, "zzz yyy xxx", "prose"),
        make_chunk("a.md", 4, 4, "ocean river mountain", "prose"),
    ]
    embedder = Embedder(HashingModel(), "hashing")
    index, _ = build_dense_index(chunks, embedder)
    dense = DenseRetriever(embedder, index, chunks)
    bm25 = BM25Retriever(InvertedIndex.build(chunks), chunks)
    hybrid = HybridRetriever([bm25, dense], depth=4)

    hits = hybrid.search("apple banana", 3)
    assert hits[0].start_line == 1 and hits[0].text == "apple banana apple"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    question = Question("q1", "lexical", "dev", "apple banana", (Span("a.md", 1, 1),))
    summary = summarize(run_retriever(hybrid, [question], 3))
    assert summary["overall"]["mrr"] == 1.0
