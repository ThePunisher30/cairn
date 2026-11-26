"""The Cairn rows of the comparison, built the way the Cairn scripts build them (Phase 11)."""

from common import Stopwatch

from cairn.dense.cache import EmbeddingCache
from cairn.dense.models import load_embedder
from cairn.dense.retriever import DenseRetriever, build_dense_index
from cairn.ingest.documents import load_documents
from cairn.ingest.pipeline import default_chunkers
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex


def build_cairn(kind, root, limit_files=None):
    """kind is bm25, dense or hybrid. Returns (retriever, info). The embedding cache is empty, so
    every chunk is embedded: the build time is a cold build."""
    info = {}
    with Stopwatch() as total:
        documents = load_documents(root)
        if limit_files:
            documents = documents[:limit_files]
        chunkers = default_chunkers()
        chunks = []
        for doc in documents:
            chunks.extend(chunkers[doc.kind].split(doc))
        chunks.sort(key=lambda c: (c.path, c.start_line))
        info["files"] = len(documents)
        info["chunks"] = len(chunks)
        info["embed_texts"] = [c.index_text for c in chunks]
        bm25 = dense = None
        if kind in ("bm25", "hybrid"):
            bm25 = BM25Retriever(InvertedIndex.build(chunks), chunks)
        if kind in ("dense", "hybrid"):
            with Stopwatch() as embedding:
                embedder = load_embedder("bge-small", batch_size=16)
                index, computed = build_dense_index(chunks, embedder, EmbeddingCache())
            dense = DenseRetriever(embedder, index, chunks)
            info["embedded_chunks"] = computed
            info["embed_seconds"] = embedding.seconds
            info["embedder"] = embedder
    if kind == "bm25":
        retriever = bm25
    elif kind == "dense":
        retriever = dense
    else:
        retriever = HybridRetriever([bm25, dense], depth=50, method="weighted", normalize="minmax")
    info["build_seconds"] = total.seconds
    return retriever, info
