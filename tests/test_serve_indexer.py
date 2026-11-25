import hashlib
import logging
import sqlite3
import threading

import numpy as np
import pytest

from cairn.dense.flat_index import FlatIndex
from cairn.ingest.documents import load_documents
from cairn.ingest.pipeline import default_chunkers, read_chunks
from cairn.serve.indexer import Indexer
from cairn.serve.jobs import DONE, FAILED, JobRunner
from cairn.serve.live import LiveBM25, LiveIndex
from cairn.serve.manifest import Manifest, diff, hash_text
from cairn.types import Document

DIM = 16


class CountingEmbedder:
    """A deterministic stand-in that remembers every text it was asked to embed."""

    dim = DIM

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
        return v / np.linalg.norm(v)


def section(title, word, n=60):
    return f"## {title}\n\n" + " ".join(f"{word}{i}" for i in range(n)) + "\n\n"


def write_corpus(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def make_files(n=10):
    return {
        f"docs/en/docs/page{i}.md": f"# Page {i}\n\n"
        + section("Intro", f"intro{i}x")
        + section("Details", f"detail{i}x")
        + section("Notes", f"note{i}x")
        for i in range(n)
    }


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "corpus"
    write_corpus(root, make_files())
    embedder = CountingEmbedder()
    live = LiveIndex(DIM, min_dead_to_compact=10**6)
    manifest = Manifest(tmp_path / "manifest.sqlite")
    indexer = Indexer(
        live,
        manifest,
        embedder,
        root,
        chunks_path=tmp_path / "chunks.jsonl",
        dense_path=tmp_path / "dense.npz",
    )
    return indexer, live, manifest, embedder, root, tmp_path


# --- the diff ------------------------------------------------------------------------------


def doc(path, text):
    return Document(path, text, "prose")


def test_diff_names_added_changed_removed_and_counts_unchanged():
    indexed = {"a.md": hash_text("a"), "b.md": hash_text("b"), "c.md": hash_text("c")}
    documents = [doc("a.md", "a"), doc("b.md", "B changed"), doc("d.md", "d")]
    result = diff(indexed, documents)
    assert result.added == ("d.md",)
    assert result.changed == ("b.md",)
    assert result.removed == ("c.md",)
    assert result.unchanged == 1
    assert not result.empty


def test_diff_of_identical_state_is_empty():
    assert diff({"a.md": hash_text("a")}, [doc("a.md", "a")]).empty


# --- the manifest --------------------------------------------------------------------------


def test_manifest_stores_hashes_and_updates_them_in_one_transaction(tmp_path):
    manifest = Manifest(tmp_path / "m.sqlite")
    manifest.apply(upserts=[("a.md", "h1", 3), ("b.md", "h2", 1)])
    assert manifest.hashes() == {"a.md": "h1", "b.md": "h2"}
    manifest.apply(upserts=[("a.md", "h3", 2)], deletes=["b.md"])
    assert manifest.hashes() == {"a.md": "h3"}
    # a bad row anywhere in the batch undoes the whole batch
    with pytest.raises(sqlite3.IntegrityError):
        manifest.apply(upserts=[("c.md", "h", 1), ("d.md", None, 1)], deletes=["a.md"])
    assert manifest.hashes() == {"a.md": "h3"}
    manifest.close()
    assert Manifest(tmp_path / "m.sqlite").hashes() == {"a.md": "h3"}  # it is on disk


# --- re-indexing ---------------------------------------------------------------------------


def test_the_first_reindex_adds_every_file(setup):
    indexer, live, manifest, embedder, _, _ = setup
    report = indexer.reindex()
    assert report.added == 10 and report.changed == 0 and report.removed == 0
    assert live.n_alive == report.chunks_added == len(embedder.texts) > 10
    assert len(manifest) == 10


def test_a_second_reindex_with_nothing_changed_does_nothing(setup):
    indexer, live, _, embedder, _, _ = setup
    indexer.reindex()
    before, generation = len(embedder.texts), live.generation
    report = indexer.reindex()
    assert report.unchanged == 10 and report.embedded == 0 and report.chunks_added == 0
    assert len(embedder.texts) == before and live.generation == generation


def test_changing_three_files_embeds_only_chunks_of_those_files(setup, caplog):
    indexer, live, manifest, embedder, root, _ = setup
    indexer.reindex()
    embedder.texts.clear()
    paths = [f"docs/en/docs/page{i}.md" for i in (2, 5, 7)]
    old = sum(len(live.chunks_of(path)) for path in paths)  # whatever the chunk size is
    before = live.n_alive
    changed = {f"docs/en/docs/page{i}.md": f"# Page {i}\n\nrewritten{i} " * 40 for i in (2, 5, 7)}
    write_corpus(root, changed)
    with caplog.at_level(logging.INFO, logger="cairn.serve.indexer"):
        report = indexer.reindex()
    assert report.changed == 3 and report.added == 0 and report.removed == 0
    assert report.unchanged == 7
    assert report.embedded == len(embedder.texts) > 0
    assert all("rewritten" in text for text in embedder.texts)  # nothing from the other 7 files
    assert report.chunks_removed == old and live.n_dead == old > 0
    assert any("3 changed" in r.message and "7 unchanged" in r.message for r in caplog.records)
    assert live.n_alive == before - old + report.chunks_added


def test_searches_see_the_new_text_and_not_the_old(setup):
    indexer, live, _, _, root, _ = setup
    indexer.reindex()
    assert LiveBM25(live).search("intro3x5", 3)[0].path == "docs/en/docs/page3.md"
    write_corpus(root, {"docs/en/docs/page3.md": "# Page 3\n\nbrandnewword " * 30})
    indexer.reindex()
    assert LiveBM25(live).search("brandnewword", 3)[0].path == "docs/en/docs/page3.md"
    assert all(h.path != "docs/en/docs/page3.md" for h in LiveBM25(live).search("intro3x5", 5))


def test_an_edit_to_one_section_reuses_the_vectors_of_the_untouched_sections(setup):
    indexer, _, _, embedder, root, _ = setup
    indexer.reindex()
    embedder.texts.clear()
    page = make_files()["docs/en/docs/page4.md"].replace("detail4x1 ", "EDITED ")
    write_corpus(root, {"docs/en/docs/page4.md": page})
    report = indexer.reindex()
    assert report.changed == 1
    assert 1 <= report.embedded < report.chunks_added  # the other sections were not re-embedded
    assert report.reused == report.chunks_added - report.embedded


def test_a_deleted_file_leaves_tombstones_and_leaves_the_manifest(setup):
    indexer, live, manifest, _, root, _ = setup
    indexer.reindex()
    (root / "docs/en/docs/page1.md").unlink()
    report = indexer.reindex()
    assert report.removed == 1 and report.embedded == 0
    assert live.n_dead == report.chunks_removed > 0
    assert "docs/en/docs/page1.md" not in manifest.hashes()
    assert all(h.path != "docs/en/docs/page1.md" for h in LiveBM25(live).search("intro1x5", 10))


def test_a_new_file_is_added(setup):
    indexer, live, manifest, _, root, _ = setup
    indexer.reindex()
    write_corpus(root, {"docs/en/docs/new.md": "# New\n\nzebra zebra zebra " * 20})
    report = indexer.reindex()
    assert report.added == 1 and report.changed == 0
    assert LiveBM25(live).search("zebra", 1)[0].path == "docs/en/docs/new.md"
    assert len(manifest) == 11


def test_the_index_files_on_disk_match_the_live_index_after_a_reindex(setup):
    indexer, live, _, _, root, tmp_path = setup
    indexer.reindex()
    write_corpus(root, {"docs/en/docs/page0.md": "# Page 0\n\nchanged " * 30})
    indexer.reindex()
    chunks = read_chunks(tmp_path / "chunks.jsonl")
    flat = FlatIndex.load(tmp_path / "dense.npz")
    want, vectors = live.snapshot()
    assert [c.id for c in chunks] == [c.id for c in want] == flat.doc_ids
    assert np.allclose(flat.vectors, vectors)


def test_a_crash_before_the_manifest_is_written_is_repaired_by_the_next_reindex(setup):
    indexer, live, manifest, embedder, root, _ = setup
    indexer.reindex()
    write_corpus(root, {"docs/en/docs/page6.md": "# Page 6\n\nafter the crash " * 20})
    real_save = live.save

    def crash(*args):
        raise OSError("disk full")

    live.save = crash
    with pytest.raises(OSError):
        indexer.reindex()
    assert manifest.hashes()["docs/en/docs/page6.md"] != hash_text(
        (root / "docs/en/docs/page6.md").read_text(encoding="utf-8")
    )  # the manifest still describes the old file
    live.save = real_save
    report = indexer.reindex()  # the memory index already has the file, the manifest does not
    assert report.changed == 1
    hits = LiveBM25(live).search("crash", 5)
    assert len({(h.path, h.start_line) for h in hits}) == len(hits)  # no duplicated chunks
    assert indexer.reindex().unchanged == 10  # and now it is consistent


def test_bootstrap_records_the_files_that_match_an_existing_index(setup):
    indexer, live, manifest, embedder, root, tmp_path = setup
    indexer.reindex()
    fresh_manifest = Manifest(tmp_path / "other.sqlite")
    other = Indexer(live, fresh_manifest, embedder, root)
    write_corpus(root, {"docs/en/docs/page8.md": "# Page 8\n\nnot what was indexed " * 20})
    verified, mismatched = other.bootstrap()
    assert (verified, mismatched) == (9, 1)
    assert "docs/en/docs/page8.md" not in fresh_manifest.hashes()
    report = other.reindex()  # the mismatching file is redone, the other nine are not
    assert report.changed == 0 and report.added == 1 and report.unchanged == 9
    assert LiveBM25(live).search("indexed", 1)[0].path == "docs/en/docs/page8.md"


def test_two_reindexes_at_once_run_one_after_the_other(setup):
    indexer, live, _, _, root, _ = setup
    reports = []
    threads = [threading.Thread(target=lambda: reports.append(indexer.reindex())) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(r.added for r in reports) == [0, 10]  # the second found nothing left to do
    assert live.n_alive == len(read_chunks(setup[5] / "chunks.jsonl"))


def test_the_default_corpus_scope_is_what_gets_indexed(setup):
    _, _, _, _, root, _ = setup
    assert len(load_documents(root)) == 10
    assert set(default_chunkers()) == {"prose", "code"}


# --- jobs ----------------------------------------------------------------------------------


def test_a_job_runs_in_the_background_and_keeps_its_result():
    gate = threading.Event()
    runner = JobRunner(lambda: gate.wait(5) and {"ok": 1})
    job, created = runner.submit()
    assert created and job.status in ("queued", "running")
    gate.set()
    done = runner.wait(job.id)
    assert done.status == DONE and done.result == {"ok": 1} and done.finished_at >= done.started_at


def test_asking_for_a_job_while_one_runs_returns_the_running_one():
    gate = threading.Event()
    runner = JobRunner(lambda: gate.wait(5) and {})
    first, created = runner.submit()
    second, created_again = runner.submit()
    assert created and not created_again and second is first
    gate.set()
    runner.wait(first.id)
    third, created_third = runner.submit()
    assert created_third and third.id != first.id
    runner.wait(third.id)


def test_a_failing_job_records_the_error_and_does_not_block_the_next_one():
    calls = []

    def work():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("disk on fire")
        return {"second": True}

    runner = JobRunner(work)
    first, _ = runner.submit()
    failed = runner.wait(first.id)
    assert failed.status == FAILED and "disk on fire" in failed.error
    second, created = runner.submit()
    assert created and runner.wait(second.id).result == {"second": True}


def test_only_the_last_jobs_are_remembered():
    runner = JobRunner(lambda: {}, keep=3)
    ids = []
    for _ in range(6):
        job, _ = runner.submit()
        runner.wait(job.id)
        ids.append(job.id)
    assert runner.get(ids[0]) is None and runner.get(ids[-1]) is not None


def test_without_a_manifest_the_index_is_rebuilt_from_the_disk_reusing_vectors(setup):
    indexer, live, _, embedder, root, tmp_path = setup
    indexer.reindex()
    (root / "docs/en/docs/page9.md").unlink()  # gone from disk, still in the index
    lost = Indexer(live, Manifest(tmp_path / "lost.sqlite"), embedder, root)
    embedder.texts.clear()
    report = lost.reindex()
    assert report.added == 9 and report.removed == 1  # the orphan file is dropped too
    assert report.embedded == 0 and report.reused == report.chunks_added  # no model call at all
    assert all(h.path != "docs/en/docs/page9.md" for h in LiveBM25(live).search("intro9x5", 10))
    assert len(Manifest(tmp_path / "lost.sqlite")) == 9
