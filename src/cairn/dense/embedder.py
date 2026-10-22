import numpy as np


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


class Embedder:
    def __init__(self, model, name, query_prefix="", doc_prefix="", batch_size=32):
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        self.model = model
        self.name = name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

    @property
    def dim(self):
        return self.model.dim

    @property
    def max_tokens(self):
        return self.model.max_tokens

    @property
    def cache_key(self):
        return f"{self.name}|doc_prefix={self.doc_prefix}"

    def embed_query(self, text):
        return l2_normalize(self.model.encode([self.query_prefix + text]))[0]

    def n_truncated(self, texts):
        lengths = self.model.token_lengths([self.doc_prefix + text for text in texts])
        return sum(length > self.max_tokens for length in lengths)

    def embed_documents(self, texts):
        texts = [self.doc_prefix + text for text in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        parts = []
        for start in range(0, len(order), self.batch_size):
            batch = [texts[i] for i in order[start : start + self.batch_size]]
            parts.append(self.model.encode(batch))
        vectors = l2_normalize(np.vstack(parts))
        result = np.empty_like(vectors)
        result[order] = vectors
        return result
