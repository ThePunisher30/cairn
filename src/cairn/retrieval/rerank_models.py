"""The reranking models Cairn can use.

Provided, not part of the exercise: it wraps the `sentence-transformers` cross-encoder (the one
thing Phase 6 does not build from scratch) and adds a tiny offline model for tests and scripts.
A model here is any object with `score(pairs) -> list[float]`, where `pairs` is a list of
`(query, passage)` and a higher score means "this passage answers the query better".
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerSpec:
    key: str  # the short name used on the command line
    hf_id: str  # the name on the Hugging Face hub ("overlap" needs no download)


RERANKERS = {
    "minilm-ce": RerankerSpec("minilm-ce", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "overlap": RerankerSpec("overlap", "overlap"),
}


class OverlapModel:
    """The share of the query's words that also appear in the passage. No weights, no download.

    Deterministic and instant, so tests and scripts can run offline.
    """

    def __init__(self):
        self.batches = []  # the pairs of every score() call, so tests can see what was sent

    @property
    def batch_sizes(self):
        return [len(batch) for batch in self.batches]

    def score(self, pairs):
        self.batches.append(list(pairs))
        scores = []
        for query, passage in pairs:
            wanted = set(re.findall(r"\w+", query.lower()))
            present = set(re.findall(r"\w+", passage.lower()))
            scores.append(len(wanted & present) / len(wanted) if wanted else 0.0)
        return scores


class CrossEncoderModel:
    """A real cross-encoder from the `sentence-transformers` library, run on the CPU.

    It reads the query and the passage together (at most 512 tokens, the rest is cut off) and
    returns one relevance score per pair. The scores are not bounded: expect values like -11 to 8.
    """

    def __init__(self, hf_id):
        from sentence_transformers import CrossEncoder  # slow to import: do it lazily

        self._model = CrossEncoder(hf_id, device="cpu")

    def score(self, pairs):
        scores = self._model.predict(
            [list(pair) for pair in pairs], batch_size=len(pairs), show_progress_bar=False
        )
        return [float(s) for s in scores]


def load_model(spec):
    if spec.hf_id == "overlap":
        return OverlapModel()
    return CrossEncoderModel(spec.hf_id)


def load_reranker(key, batch_size=16, max_chars=2000, model=None):
    """Build the Reranker for one of RERANKERS. Pass `model` to reuse an already loaded model."""
    from cairn.retrieval.rerank import Reranker

    spec = RERANKERS[key]
    return Reranker(
        model if model is not None else load_model(spec),
        spec.key,
        batch_size=batch_size,
        max_chars=max_chars,
    )
