"""Phase 3, step 5: saving the index to disk and loading it back.

The file is a pickle of a plain dict (numbers, strings, lists, tuples and dicts only), never of your
own classes: a file that mentions `cairn.something` breaks the moment you rename or move that
class. The dict has the key "version" with the value 1, so a future format can be told apart.

Security note: unpickling a file can run arbitrary code. Only load index files you made yourself.
"""

import pickle

import pytest

from cairn.ingest.chunkers import make_chunk
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import Tokenizer


def chunk(number, text, heading_path=()):
    return make_chunk("a.md", number + 1, number + 1, text, "prose", heading_path)


@pytest.fixture
def chunks():
    return [
        chunk(0, "apple banana apple"),
        chunk(1, "get_request_handler cherry", heading_path=("Routing", "Handlers")),
        chunk(2, "apple apple apple dog"),
    ]


def test_a_saved_index_loads_back_identical(chunks, tmp_path):
    tokenizer = Tokenizer(split_identifiers=False, stopwords=frozenset({"dog", "the"}))
    original = InvertedIndex.build(chunks, tokenizer)
    original.save(tmp_path / "index.pkl")
    loaded = InvertedIndex.load(tmp_path / "index.pkl")
    assert loaded.postings == original.postings
    assert list(loaded.doc_lengths) == list(original.doc_lengths)
    assert list(loaded.doc_ids) == list(original.doc_ids)
    assert loaded.tokenizer == tokenizer  # the settings travel with the index
    assert (loaded.n_docs, loaded.avg_length) == (original.n_docs, original.avg_length)


def test_the_default_tokenizer_survives_the_round_trip(chunks, tmp_path):
    InvertedIndex.build(chunks).save(tmp_path / "index.pkl")
    assert InvertedIndex.load(tmp_path / "index.pkl").tokenizer == Tokenizer()


def test_a_loaded_index_gives_the_same_search_results(chunks, tmp_path):
    original = InvertedIndex.build(chunks)
    original.save(tmp_path / "index.pkl")
    loaded = InvertedIndex.load(tmp_path / "index.pkl")
    for query in ("apple", "request", "apple cherry dog", "handlers", "zebra"):
        before = BM25Retriever(original, chunks).search(query, 10)
        after = BM25Retriever(loaded, chunks).search(query, 10)
        assert before == after, query


def test_save_creates_missing_folders_and_overwrites_old_files(chunks, tmp_path):
    path = tmp_path / "deep" / "er" / "index.pkl"
    InvertedIndex.build(chunks).save(path)
    InvertedIndex.build(chunks[:1]).save(path)
    assert InvertedIndex.load(path).n_docs == 1


def test_the_file_is_a_plain_dict_with_a_version(chunks, tmp_path):
    InvertedIndex.build(chunks).save(tmp_path / "index.pkl")
    raw = (tmp_path / "index.pkl").read_bytes()
    payload = pickle.loads(raw)
    assert isinstance(payload, dict)
    assert payload["version"] == 1
    assert b"cairn" not in raw  # none of our own classes were pickled


def test_a_file_with_another_version_is_refused(tmp_path):
    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps({"version": 99, "postings": {}}))
    with pytest.raises(ValueError):
        InvertedIndex.load(path)


def test_a_pickle_that_is_not_an_index_is_refused(tmp_path):
    for value in ([1, 2, 3], {"hello": "world"}, "text"):
        path = tmp_path / "other.pkl"
        path.write_bytes(pickle.dumps(value))
        with pytest.raises(ValueError):
            InvertedIndex.load(path)


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        InvertedIndex.load(tmp_path / "nope.pkl")
