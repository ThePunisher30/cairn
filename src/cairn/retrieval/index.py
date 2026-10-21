import pickle
from collections import Counter
from pathlib import Path

from cairn.retrieval.tokenizer import Tokenizer

FORMAT_VERSION = 1


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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "doc_ids": self.doc_ids,
            "tokenizer": {
                "split_identifiers": self.tokenizer.split_identifiers,
                "stopwords": sorted(self.tokenizer.stopwords),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"{Path(path).name} is not a version-{FORMAT_VERSION} index file")
        settings = payload["tokenizer"]
        tokenizer = Tokenizer(settings["split_identifiers"], frozenset(settings["stopwords"]))
        return cls(payload["postings"], payload["doc_lengths"], payload["doc_ids"], tokenizer)
