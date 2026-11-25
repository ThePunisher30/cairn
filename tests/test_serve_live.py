import random
import threading
import time

import numpy as np
import pytest

from cairn.dense.flat_index import FlatIndex
from cairn.ingest.chunkers import make_chunk
from cairn.ingest.pipeline import read_chunks
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.index import InvertedIndex
from cairn.serve.live import LiveBM25, LiveDense, LiveIndex, RWLock

WORDS = [f"word{i}" for i in range(60)] + ["cors", "middleware", "oauth", "token", "depends"]
DIM = 8


def unit(rng):
    v = np.array([rng.gauss(0, 1) for _ in range(DIM)], dtype=np.float32)
    return v / np.linalg.norm(v)


def random_chunk(rng, path, line):
    text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 40)))
    return make_chunk(path, line, line + 3, text, "prose")


def random_file(rng, path, n):
    chunks = [random_chunk(rng, path, 1 + 10 * i) for i in range(n)]
    return chunks, [unit(rng) for _ in chunks]


def fresh_results(chunks, vectors, query, query_vector, k):
    """What an index built from scratch over `chunks` answers: the reference."""
    index = InvertedIndex.build(chunks)
    bm25 = BM25Retriever(index, chunks).search(query, k)
    flat = FlatIndex(np.array(vectors), [c.id for c in chunks])
    dense = [
        (chunks[row].path, chunks[row].start_line, score)
        for row, score in flat.search(query_vector, k)
    ]
    return bm25, dense


def keyed(hits):
    return {(h.path, h.start_line): h.score for h in hits}


def assert_same_scores(live_hits, fresh_hits):
    """Same scores in the same order; the chunk behind a tied score may differ, so compare the
    scores, and the chunk for every score that is not tied."""
    live_scores = [round(h.score, 5) for h in live_hits]
    fresh_scores = [round(h.score, 5) for h in fresh_hits]
    assert live_scores == fresh_scores
    for score in set(live_scores):
        if live_scores.count(score) == 1:
            a = next(h for h in live_hits if round(h.score, 5) == score)
            b = next(h for h in fresh_hits if round(h.score, 5) == score)
            assert (a.path, a.start_line) == (b.path, b.start_line)


def test_a_live_index_scores_like_one_built_fresh_after_random_changes():
    rng = random.Random(1)
    files = {
        f"docs/f{i}.md": random_file(rng, f"docs/f{i}.md", rng.randint(2, 6)) for i in range(12)
    }
    live = LiveIndex.build(
        [c for chunks, _ in files.values() for c in chunks],
        [v for _, vectors in files.values() for v in vectors],
        compact_threshold=0.4,
        min_dead_to_compact=5,
    )
    bm25 = LiveBM25(live)
    for step in range(60):
        path = rng.choice([f"docs/f{i}.md" for i in range(16)])
        action = rng.choice(["remove", "replace", "add"])
        if action == "remove":
            live.apply(remove_paths=[path])
            files.pop(path, None)
        else:
            new = random_file(rng, path, rng.randint(1, 5))
            live.apply(remove_paths=[path], add=[new])
            files[path] = new
        if step % 5 == 0:
            chunks = [c for cs, _ in files.values() for c in cs]
            vectors = [v for _, vs in files.values() for v in vs]
            for query in ("cors middleware", "oauth token depends", "word3 word9 word41"):
                query_vector = unit(rng)
                want_bm25, want_dense = fresh_results(chunks, vectors, query, query_vector, 5)
                assert_same_scores(bm25.search(query, 5), want_bm25)
                with live.lock.read():
                    got = [
                        (live.hit(s, sc).path, live.hit(s, sc).start_line, sc)
                        for s, sc in live.dense_search(query_vector, 5)
                    ]
                assert [round(g[2], 4) for g in got] == [round(w[2], 4) for w in want_dense]
    assert live.n_alive == sum(len(cs) for cs, _ in files.values())
    assert live.compactions >= 1  # the threshold was crossed at least once along the way


def test_a_removed_chunk_never_comes_back_in_a_search():
    rng = random.Random(2)
    chunks, vectors = random_file(rng, "docs/a.md", 5)
    other, other_vectors = random_file(rng, "docs/b.md", 5)
    live = LiveIndex.build(chunks + other, vectors + other_vectors, min_dead_to_compact=10**6)
    live.apply(remove_paths=["docs/a.md"])
    assert live.n_dead == 5 and live.n_alive == 5
    for query in ("cors middleware oauth", "word1 word2 word3"):
        assert all(h.path == "docs/b.md" for h in LiveBM25(live).search(query, 10))
    with live.lock.read():
        slots = [s for s, _ in live.dense_search(vectors[0], 10)]
    assert len(slots) == 5 and all(live._chunks[s].path == "docs/b.md" for s in slots)


def test_document_frequencies_and_lengths_count_only_living_chunks():
    rng = random.Random(3)
    chunks, vectors = random_file(rng, "docs/a.md", 6)
    live = LiveIndex.build(chunks, vectors, min_dead_to_compact=10**6)
    live.apply(remove_paths=["docs/a.md"])
    assert live.n_alive == 0 and live._total_length == 0
    assert all(df == 0 for df in live._df.values())
    assert LiveBM25(live).search("cors", 5) == []
    assert live.dense_search(vectors[0], 5) == []


def test_compaction_drops_tombstones_and_changes_no_result():
    rng = random.Random(4)
    files = [random_file(rng, f"docs/f{i}.md", 5) for i in range(8)]
    live = LiveIndex.build(
        [c for cs, _ in files for c in cs],
        [v for _, vs in files for v in vs],
        min_dead_to_compact=10**6,
    )
    live.apply(remove_paths=[f"docs/f{i}.md" for i in range(0, 8, 2)])
    assert live.n_dead == 20 and live.compactions == 0
    before = [LiveBM25(live).search(q, 5) for q in ("cors", "word5 word7", "oauth token")]
    live.compact()
    assert live.n_dead == 0 and live.n_slots == 20 and live.compactions == 1
    after = [LiveBM25(live).search(q, 5) for q in ("cors", "word5 word7", "oauth token")]
    assert [keyed(a) for a in after] == [keyed(b) for b in before]
    assert all(len(p) > 0 for p in live._postings.values())  # no empty posting lists left


def test_compaction_starts_by_itself_past_the_threshold_and_not_before():
    rng = random.Random(5)
    files = [random_file(rng, f"docs/f{i}.md", 10) for i in range(10)]
    live = LiveIndex.build(
        [c for cs, _ in files for c in cs],
        [v for _, vs in files for v in vs],
        compact_threshold=0.25,
        min_dead_to_compact=1,
    )
    result = live.apply(remove_paths=["docs/f0.md", "docs/f1.md"])  # 20 of 100 dead: 0.20
    assert not result["compacted"] and live.n_dead == 20
    result = live.apply(remove_paths=["docs/f2.md"])  # 30 of 100: past 0.25
    assert result["compacted"] and live.n_dead == 0 and live.n_slots == 70


def test_the_generation_changes_only_when_the_index_does():
    rng = random.Random(6)
    chunks, vectors = random_file(rng, "docs/a.md", 3)
    live = LiveIndex.build(chunks, vectors)
    assert live.generation == 0
    live.apply(remove_paths=["docs/nothing.md"])
    assert live.generation == 0
    live.apply(remove_paths=["docs/a.md"])
    assert live.generation == 1


def test_a_reader_sees_a_file_entirely_old_or_entirely_new_never_a_mix():
    rng = random.Random(7)
    old = [
        make_chunk("docs/a.md", 1 + 10 * i, 5 + 10 * i, f"cors old{i}", "prose") for i in range(4)
    ]
    new = [
        make_chunk("docs/a.md", 1 + 10 * i, 5 + 10 * i, f"cors new{i}", "prose") for i in range(4)
    ]
    live = LiveIndex.build(old, [unit(rng) for _ in old])
    stop = threading.Event()
    mixed = []

    def reader():
        while not stop.is_set():
            texts = {h.text.split()[1][:3] for h in LiveBM25(live).search("cors", 10)}
            if len(texts) > 1:
                mixed.append(texts)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(200):
        chunks = new if i % 2 == 0 else old
        live.apply(remove_paths=["docs/a.md"], add=[(chunks, [unit(rng) for _ in chunks])])
    stop.set()
    for t in threads:
        t.join()
    assert mixed == []


def test_save_then_load_gives_the_same_chunks_and_vectors(tmp_path):
    rng = random.Random(8)
    files = [random_file(rng, f"docs/f{i}.md", 4) for i in range(5)]
    live = LiveIndex.build([c for cs, _ in files for c in cs], [v for _, vs in files for v in vs])
    live.apply(remove_paths=["docs/f1.md"])
    live.save(tmp_path / "chunks.jsonl", tmp_path / "dense.npz")
    chunks = read_chunks(tmp_path / "chunks.jsonl")
    flat = FlatIndex.load(tmp_path / "dense.npz")
    want, vectors = live.snapshot()
    assert [c.id for c in chunks] == [c.id for c in want] == flat.doc_ids
    assert np.allclose(flat.vectors, vectors)
    assert all(c.path != "docs/f1.md" for c in chunks)
    assert [c.path for c in chunks] == sorted(c.path for c in chunks)
    assert not list(tmp_path.glob("*.tmp*"))  # the temporary files were renamed away


def test_wrong_number_of_vectors_is_refused():
    live = LiveIndex(DIM)
    rng = random.Random(9)
    chunks, vectors = random_file(rng, "docs/a.md", 3)
    with pytest.raises(ValueError):
        live.apply(add=[(chunks, vectors[:2])])


def test_dense_retriever_embeds_outside_the_lock():
    rng = random.Random(10)
    chunks, vectors = random_file(rng, "docs/a.md", 3)
    live = LiveIndex.build(chunks, vectors)
    inside = []

    class Embedder:
        def embed_query(self, text):
            # if the read lock were held here, a writer could not start while this runs
            inside.append(live.lock._readers)
            return vectors[0]

    hits = LiveDense(Embedder(), live).search("anything", 2)
    assert len(hits) == 2 and inside == [0]


# --- the lock ------------------------------------------------------------------------------


def test_many_readers_can_hold_the_lock_together():
    lock = RWLock()
    inside = threading.Barrier(3, timeout=5)

    def reader():
        with lock.read():
            inside.wait()  # returns only if all three are inside at once

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not any(t.is_alive() for t in threads)


def test_a_writer_waits_for_readers_and_blocks_new_ones():
    lock = RWLock()
    events = []
    release_reader = threading.Event()
    reader_in = threading.Event()

    def reader():
        with lock.read():
            reader_in.set()
            release_reader.wait(5)
        events.append("reader done")

    def writer():
        with lock.write():
            events.append("writer in")

    def late_reader():
        with lock.read():
            events.append("late reader in")

    t1 = threading.Thread(target=reader)
    t1.start()
    assert reader_in.wait(5)
    t2 = threading.Thread(target=writer)
    t2.start()
    time.sleep(0.1)
    t3 = threading.Thread(target=late_reader)
    t3.start()
    time.sleep(0.1)
    assert events == []  # the writer waits for the first reader; the late reader waits behind it
    release_reader.set()
    for t in (t1, t2, t3):
        t.join(5)
    assert events == ["reader done", "writer in", "late reader in"]
