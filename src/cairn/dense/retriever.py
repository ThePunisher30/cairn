from cairn.dense.cache import EmbeddingCache, embed_cached
from cairn.dense.flat_index import FlatIndex
from cairn.types import Hit


def build_dense_index(chunks, embedder, cache=None):
    chunks = list(chunks)
    if cache is None:
        cache = EmbeddingCache()
    vectors, computed = embed_cached(embedder, [chunk.index_text for chunk in chunks], cache)
    return FlatIndex(vectors, [chunk.id for chunk in chunks]), computed


class DenseRetriever:
    def __init__(self, embedder, index, chunks):
        if [chunk.id for chunk in chunks] != index.doc_ids:
            raise ValueError("these are not the chunks the index was built from. Rebuild it!")
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    def search(self, query, k):
        if k <= 0:
            return []
        vector = self.embedder.embed_query(query)
        hits = []
        for row, score in self.index.search(vector, k):
            chunk = self.chunks[row]
            hits.append(Hit(chunk.path, chunk.start_line, chunk.end_line, score, chunk.text))
        return hits
