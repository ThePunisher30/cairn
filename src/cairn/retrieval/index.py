from collections import Counter

from cairn.retrieval.tokenizer import Tokenizer


class InvertedIndex:
    def __init__(self, postings, doc_lengths, doc_ids, tokenizer):
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer

    @classmethod
    def build(cls, chunks, tokenizer=None):
        if tokenizer is None:
            tokenizer = Tokenizer()
        postings = {}
        doc_lengths = []
        doc_ids = []
        for number, chunk in enumerate(chunks):
            tokens = tokenizer.tokenize(chunk.index_text)
            doc_lengths.append(len(tokens))
            doc_ids.append(chunk.id)
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((number, tf))
        return cls(postings, doc_lengths, doc_ids, tokenizer)

    @property
    def n_docs(self):
        return len(self.doc_lengths)

    @property
    def avg_length(self):
        return sum(self.doc_lengths) / len(self.doc_lengths) if len(self.doc_lengths) else 0.0

    def doc_freq(self, term):
        return len(self.postings.get(term, ()))

    def postings_for(self, term):
        return self.postings.get(term, ())
