"""Phase 3, step 2: the inverted index, term -> which chunks contain it and how often.

Toy corpus used below (chunk numbers are positions in the list you pass to `build`):
    chunk 0: "apple banana apple"        3 tokens
    chunk 1: "banana cherry"             2 tokens
    chunk 2: "apple apple apple dog"     4 tokens
so  apple -> [(0, 2), (2, 3)]   banana -> [(0, 1), (1, 1)]   cherry -> [(1, 1)]   dog -> [(2, 1)]
"""

import pytest

from cairn.ingest.chunkers import make_chunk
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.tokenizer import Tokenizer


def chunk(number, text, heading_path=()):
    return make_chunk("a.md", number + 1, number + 1, text, "prose", heading_path)


@pytest.fixture
def toy():
    return [
        chunk(0, "apple banana apple"),
        chunk(1, "banana cherry"),
        chunk(2, "apple apple apple dog"),
    ]


def test_postings_hold_the_chunk_number_and_the_term_frequency(toy):
    index = InvertedIndex.build(toy)
    assert list(index.postings_for("apple")) == [(0, 2), (2, 3)]
    assert list(index.postings_for("banana")) == [(0, 1), (1, 1)]
    assert list(index.postings_for("cherry")) == [(1, 1)]
    assert list(index.postings_for("dog")) == [(2, 1)]


def test_an_unknown_term_has_no_postings(toy):
    index = InvertedIndex.build(toy)
    assert list(index.postings_for("zebra")) == []
    assert index.doc_freq("zebra") == 0


def test_document_frequency_counts_chunks_not_occurrences(toy):
    index = InvertedIndex.build(toy)
    assert index.doc_freq("apple") == 2  # 5 occurrences, but only 2 chunks
    assert index.doc_freq("banana") == 2
    assert index.doc_freq("dog") == 1


def test_chunk_lengths_count_every_token_including_repeats(toy):
    index = InvertedIndex.build(toy)
    assert list(index.doc_lengths) == [3, 2, 4]


def test_counts_and_average_length(toy):
    index = InvertedIndex.build(toy)
    assert index.n_docs == 3
    assert index.avg_length == pytest.approx(3.0)


def test_postings_are_sorted_by_chunk_number():
    chunks = [chunk(i, "common " + ("rare " if i % 3 == 0 else "")) for i in range(30)]
    index = InvertedIndex.build(chunks)
    for term in ("common", "rare"):
        numbers = [number for number, _ in index.postings_for(term)]
        assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)
    assert index.doc_freq("common") == 30
    assert index.doc_freq("rare") == 10


def test_the_index_remembers_which_chunk_each_number_means(toy):
    index = InvertedIndex.build(toy)
    assert list(index.doc_ids) == [c.id for c in toy]


def test_an_empty_corpus_gives_an_empty_index():
    index = InvertedIndex.build([])
    assert index.n_docs == 0
    assert index.avg_length == 0.0
    assert list(index.postings_for("anything")) == []


def test_a_chunk_with_no_words_still_counts_as_a_chunk():
    index = InvertedIndex.build([chunk(0, "!!! ???"), chunk(1, "real words")])
    assert index.n_docs == 2
    assert list(index.doc_lengths) == [0, 2]
    assert index.avg_length == pytest.approx(1.0)
    assert list(index.postings_for("real")) == [(1, 1)]  # numbering still counts the empty chunk


def test_the_heading_path_is_indexed_too():
    chunks = [chunk(0, "unrelated words here", heading_path=("OAuth2 scopes", "Check it"))]
    index = InvertedIndex.build(chunks)
    assert index.doc_freq("scopes") == 1  # it only appears in the heading path
    assert index.doc_freq("check") == 1
    assert index.doc_freq("unrelated") == 1


def test_build_accepts_any_iterable_of_chunks(toy):
    assert InvertedIndex.build(c for c in toy).n_docs == 3


def test_building_twice_gives_the_same_index(toy):
    a, b = InvertedIndex.build(toy), InvertedIndex.build(toy)
    assert a.postings == b.postings
    assert list(a.doc_lengths) == list(b.doc_lengths)


# ---------------------------------------------------------- the tokenizer decides
def test_the_default_tokenizer_splits_identifiers():
    index = InvertedIndex.build([chunk(0, "get_request_handler")])
    assert index.doc_freq("get_request_handler") == 1
    assert index.doc_freq("request") == 1
    assert list(index.doc_lengths) == [4]  # whole token + 3 parts


def test_a_tokenizer_without_splitting_gives_a_smaller_vocabulary():
    index = InvertedIndex.build(
        [chunk(0, "get_request_handler")], tokenizer=Tokenizer(split_identifiers=False)
    )
    assert index.doc_freq("get_request_handler") == 1
    assert index.doc_freq("request") == 0
    assert list(index.doc_lengths) == [1]


def test_the_index_keeps_the_tokenizer_it_was_built_with():
    tokenizer = Tokenizer(split_identifiers=False, stopwords=frozenset({"the"}))
    assert InvertedIndex.build([], tokenizer=tokenizer).tokenizer == tokenizer
    assert InvertedIndex.build([]).tokenizer == Tokenizer()


def test_stopwords_do_not_reach_the_index_or_the_lengths():
    tokenizer = Tokenizer(stopwords=frozenset({"the"}))
    index = InvertedIndex.build([chunk(0, "the cat the hat")], tokenizer=tokenizer)
    assert index.doc_freq("the") == 0
    assert list(index.doc_lengths) == [2]
