import heapq
import math
from collections import defaultdict

from cairn.types import Hit


def idf(n_docs, doc_freq):
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def tf_weight(tf, doc_length, avg_length, k1=1.2, b=0.75):
    length_factor = 1 - b + b * doc_length / avg_length
    return tf * (k1 + 1) / (tf + k1 * length_factor)


class BM25Retriever:
    def __init__(self, index, chunks, k1=1.2, b=0.75):
        if k1 < 0:
            raise ValueError("k1 must be atleast 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if [chunk.id for chunk in chunks] != index.doc_ids:
            raise ValueError("These are not the chunks from which index was built. Rebuild it.")
        self.index = index
        self.chunks = chunks
        self.k1 = k1
        self.b = b

    def search(self, query, k):
        if k <= 0:
            return []
        index = self.index
        avg_length = index.avg_length
        scores = defaultdict(float)
        for term in set(index.tokenizer.tokenize(query)):
            postings = index.postings_for(term)
            weight = idf(index.n_docs, len(postings))
            for number, tf in postings:
                length = index.doc_lengths[number]
                scores[number] += weight * tf_weight(tf, length, avg_length, self.k1, self.b)
        best = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        hits = []
        for number, score in best:
            chunk = self.chunks[number]
            hits.append(Hit(chunk.path, chunk.start_line, chunk.end_line, score, chunk.text))
        return hits
