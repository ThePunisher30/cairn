import numpy as np
import pytest

from cairn.serve.cache import LRUCache, SemanticCache, normalize_question


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_questions_that_differ_only_in_spacing_case_or_final_punctuation_share_a_key():
    keys = {
        normalize_question(q)
        for q in ("How do I enable CORS?", "  how do i   enable cors ", "how do I enable CORS?!")
    }
    assert keys == {"how do i enable cors"}
    assert normalize_question("enable CORS") != normalize_question("disable CORS")


def test_lru_returns_what_was_stored_and_counts_hits_and_misses():
    cache = LRUCache(max_entries=3)
    assert cache.get("a") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_lru_drops_the_least_recently_used_entry_when_full():
    cache = LRUCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # a is now the most recently used
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2


def test_lru_entries_expire_after_the_ttl():
    clock = FakeClock()
    cache = LRUCache(ttl_s=10, clock=clock)
    cache.put("a", 1)
    clock.now = 9
    assert cache.get("a") == 1
    clock.now = 11
    assert cache.get("a") is None
    assert len(cache) == 0


def test_lru_rejects_a_size_below_one():
    with pytest.raises(ValueError):
        LRUCache(max_entries=0)


class VectorEmbedder:
    """Maps a question to a vector chosen by the test, by its first word group."""

    def __init__(self, table):
        self.table = table

    def embed_query(self, text):
        v = np.array(self.table[text], dtype=np.float32)
        return v / np.linalg.norm(v)


TABLE = {
    "how do i enable cors": [1, 0, 0],
    "how can i turn on cors": [0.97, 0.2, 0],  # a paraphrase: close
    "how do i disable cors": [0.9, 0.0, 0.43],  # a different question: also close
    "what is a path parameter": [0, 1, 0],
}


def make_cache(threshold=0.92):
    return SemanticCache(VectorEmbedder(TABLE), threshold=threshold)


def test_a_paraphrase_finds_the_stored_answer():
    cache = make_cache()
    cache.store("how do i enable cors", (1, 5, "m"), "answer A")
    hit = cache.lookup("how can i turn on cors", (1, 5, "m"))
    assert hit is not None and hit.value == "answer A"
    assert hit.matched_question == "how do i enable cors" and hit.similarity >= 0.92
    assert cache.hits == 1 and cache.recent_hits[-1][:2] == (
        "how can i turn on cors",
        "how do i enable cors",
    )


def test_an_unrelated_question_misses():
    cache = make_cache()
    cache.store("how do i enable cors", (1, 5, "m"), "answer A")
    assert cache.lookup("what is a path parameter", (1, 5, "m")) is None
    assert cache.misses == 1


def test_the_failure_mode_a_different_question_can_be_close_enough_to_hit():
    """This is why the threshold matters: 'disable' is not 'enable', but the vectors are near."""
    cache = make_cache(threshold=0.85)
    cache.store("how do i enable cors", (1, 5, "m"), "answer for enabling")
    wrong = cache.lookup("how do i disable cors", (1, 5, "m"))
    assert wrong is not None and wrong.value == "answer for enabling"  # a false hit
    strict = make_cache(threshold=0.95)
    strict.store("how do i enable cors", (1, 5, "m"), "answer for enabling")
    assert strict.lookup("how do i disable cors", (1, 5, "m")) is None  # a stricter bar avoids it


def test_an_entry_from_another_scope_is_never_returned():
    cache = make_cache()
    cache.store("how do i enable cors", (1, 5, "m"), "answer A")
    assert cache.lookup("how do i enable cors", (2, 5, "m")) is None  # the index changed
    assert cache.lookup("how do i enable cors", (1, 8, "m")) is None  # another k
    assert cache.lookup("how do i enable cors", (1, 5, "other")) is None  # another model


def test_purge_drops_the_entries_of_old_generations():
    cache = make_cache()
    cache.store("how do i enable cors", (1, 5, "m"), "old")
    cache.store("what is a path parameter", (2, 5, "m"), "new")
    cache.purge_except(2)
    assert len(cache) == 1
    assert cache.lookup("what is a path parameter", (2, 5, "m")).value == "new"


def test_the_oldest_entries_go_when_the_cache_is_full():
    cache = SemanticCache(VectorEmbedder(TABLE), max_entries=2)
    cache.store("how do i enable cors", (1,), "a")
    cache.store("what is a path parameter", (1,), "b")
    cache.store("how do i disable cors", (1,), "c")
    assert len(cache) == 2
    assert cache.lookup("how do i enable cors", (1,)) is None


def test_a_vector_computed_once_can_be_reused_for_lookup_and_store():
    calls = []

    class Counting(VectorEmbedder):
        def embed_query(self, text):
            calls.append(text)
            return super().embed_query(text)

    cache = SemanticCache(Counting(TABLE))
    vector = cache.embed("how do i enable cors")
    assert cache.lookup("how do i enable cors", (1,), vector) is None
    cache.store("how do i enable cors", (1,), "a", vector)
    assert calls == ["how do i enable cors"]


def test_threshold_must_be_a_similarity():
    with pytest.raises(ValueError):
        SemanticCache(VectorEmbedder(TABLE), threshold=1.5)
