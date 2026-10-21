"""The embedding models Cairn can use, and how each one wants to be called.

Provided, not part of the exercise: it wraps the `sentence-transformers` library (the one thing
Phase 4 does not build from scratch) and adds a tiny offline model for tests and experiments.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from cairn.retrieval.tokenizer import Tokenizer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    key: str  # the short name used on the command line and in cache keys
    hf_id: str  # the name on the Hugging Face hub ("hashing" needs no download)
    query_prefix: str = ""  # text put in front of every query before embedding it
    doc_prefix: str = ""  # text put in front of every chunk before embedding it


MODELS = {
    "bge-small": ModelSpec("bge-small", "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX),
    "minilm": ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    "hashing": ModelSpec("hashing", "hashing"),
    # the offline model with a query prefix, so scripts and tests can exercise prefix on/off
    "hashing-prefixed": ModelSpec("hashing-prefixed", "hashing", query_prefix="query: "),
}


class HashingModel:
    """A word-counting "embedding model" with no weights and no download.

    Every word is hashed to one of `dim` slots (with a random-looking sign) and the slots are
    counted. Texts that share words get similar vectors. It is deterministic and instant, so tests
    and scripts can run offline, and it is a useful floor: embeddings that only count words.
    """

    def __init__(self, dim=256, max_tokens=512):
        self.dim = dim
        self.max_tokens = max_tokens
        self.batches = []  # the texts of every encode() call, so tests can see what was sent
        self._tokenizer = Tokenizer()

    @property
    def batch_sizes(self):
        return [len(batch) for batch in self.batches]

    def encode(self, texts):
        self.batches.append(list(texts))
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenizer.tokenize(text):
                digest = hashlib.md5(token.encode()).digest()
                slot = int.from_bytes(digest[:4], "little") % self.dim
                vectors[row, slot] += 1.0 if digest[4] % 2 == 0 else -1.0
        return vectors

    def token_lengths(self, texts):
        return [len(self._tokenizer.tokenize(text)) for text in texts]


class SentenceTransformerModel:
    """A real model from the `sentence-transformers` library, run on the CPU."""

    def __init__(self, hf_id):
        from sentence_transformers import SentenceTransformer  # slow to import: do it lazily

        self._model = SentenceTransformer(hf_id, device="cpu")
        self.dim = self._model.get_embedding_dimension()
        self.max_tokens = self._model.max_seq_length

    def encode(self, texts):
        return self._model.encode(
            texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False
        )

    def token_lengths(self, texts):
        encoded = self._model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]


def load_model(spec):
    if spec.hf_id == "hashing":
        return HashingModel()
    return SentenceTransformerModel(spec.hf_id)


def load_embedder(key, batch_size=32, use_query_prefix=True, model=None):
    """Build the Embedder for one of MODELS. Pass `model` to reuse an already loaded model."""
    from cairn.dense.embedder import Embedder

    spec = MODELS[key]
    return Embedder(
        model if model is not None else load_model(spec),
        name=spec.key,
        query_prefix=spec.query_prefix if use_query_prefix else "",
        doc_prefix=spec.doc_prefix,
        batch_size=batch_size,
    )
