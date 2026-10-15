"""Phase 2, step 5: the ingest pipeline (documents -> chunks -> chunks.jsonl) and its stats."""

import dataclasses

import pytest

from cairn import config
from cairn.ingest.chunkers import FixedChunker, MarkdownChunker, PythonChunker, make_chunk
from cairn.ingest.documents import load_documents
from cairn.ingest.pipeline import (
    IngestStats,
    default_chunkers,
    format_stats,
    ingest,
    read_chunks,
    token_histogram,
)

MD = "# Title\nintro words here\n\n## Part\nmore words\n"
PY = "import os\n\n\ndef f(x):\n    return x + 1\n\n\nclass C:\n    def m(self):\n        pass\n"


@pytest.fixture
def repo(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": MD,
        "docs/en/docs/b.md": "# B\ntext\n",
        "fastapi/m.py": PY,
        "docs/de/docs/skip.md": "# skipped\n",  # not in the corpus scope
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def expected_chunks(repo, chunkers):
    chunks = []
    for doc in load_documents(repo):
        chunks.extend(chunkers[doc.kind].split(doc))
    return sorted(chunks, key=lambda c: (c.path, c.start_line))


# ------------------------------------------------------------------- ingest
def test_default_chunkers_cover_both_kinds():
    chunkers = default_chunkers()
    assert isinstance(chunkers["prose"], MarkdownChunker)
    assert isinstance(chunkers["code"], PythonChunker)


def test_ingest_writes_every_chunk_in_path_and_line_order(repo, tmp_path):
    out = tmp_path / "out" / "chunks.jsonl"  # the folder does not exist yet
    stats = ingest(repo, out)
    chunks = read_chunks(out)
    assert chunks == expected_chunks(repo, default_chunkers())
    assert stats.chunks == len(chunks) > 0
    keys = [(c.path, c.start_line) for c in chunks]
    assert keys == sorted(keys)
    assert all(c.path != "docs/de/docs/skip.md" for c in chunks)


def test_running_ingest_twice_gives_a_byte_identical_file(repo, tmp_path):
    out = tmp_path / "chunks.jsonl"
    ingest(repo, out)
    first = out.read_bytes()
    ingest(repo, out)  # must overwrite, not append, and must not change a single byte
    assert out.read_bytes() == first


def test_chunks_survive_the_round_trip_including_the_heading_path(repo, tmp_path):
    out = tmp_path / "chunks.jsonl"
    ingest(repo, out)
    chunks = read_chunks(out)
    assert any(c.heading_path == ("Title", "Part") for c in chunks)
    assert all(isinstance(c.heading_path, tuple) for c in chunks)
    assert chunks[0] == make_chunk(
        chunks[0].path,
        chunks[0].start_line,
        chunks[0].end_line,
        chunks[0].text,
        chunks[0].kind,
        chunks[0].heading_path,
    )


def test_custom_chunkers_can_be_passed_in(repo, tmp_path):
    default = ingest(repo, tmp_path / "a.jsonl")
    tiny = ingest(
        repo, tmp_path / "b.jsonl", chunkers={"prose": FixedChunker(2), "code": FixedChunker(2)}
    )
    assert tiny.chunks > default.chunks


def test_an_empty_corpus_gives_an_empty_file_and_zero_stats(tmp_path):
    out = tmp_path / "chunks.jsonl"
    stats = ingest(tmp_path / "nothing-here", out)
    assert out.read_text(encoding="utf-8") == ""
    assert (stats.documents, stats.chunks, stats.mean_tokens, stats.max_tokens) == (0, 0, 0.0, 0)
    assert stats.over_limit == 0


# -------------------------------------------------------------------- stats
def test_stats_describe_the_corpus(repo, tmp_path):
    stats = ingest(repo, tmp_path / "c.jsonl")
    chunks = expected_chunks(repo, default_chunkers())
    assert isinstance(stats, IngestStats)
    assert stats.documents == 3
    assert stats.chunks == len(chunks)
    assert stats.by_kind == {
        "prose": sum(c.kind == "prose" for c in chunks),
        "code": sum(c.kind == "code" for c in chunks),
    }
    assert stats.max_tokens == max(c.token_count for c in chunks)
    assert stats.mean_tokens == pytest.approx(sum(c.token_count for c in chunks) / len(chunks))


def test_over_limit_counts_the_chunks_above_the_token_limit(repo, tmp_path):
    chunks = expected_chunks(repo, default_chunkers())
    limit = sorted(c.token_count for c in chunks)[len(chunks) // 2]
    stats = ingest(repo, tmp_path / "c.jsonl", token_limit=limit)
    assert stats.over_limit == sum(c.token_count > limit for c in chunks)
    assert 0 < stats.over_limit < stats.chunks


def test_the_default_limit_is_the_512_token_limit_of_the_embedding_model(repo, tmp_path):
    big = "word " * 600  # one line of 600 tokens, which no chunker splits
    (repo / "docs" / "en" / "docs" / "big.md").write_text("# Big\n" + big + "\n", encoding="utf-8")
    assert ingest(repo, tmp_path / "c.jsonl").over_limit >= 1


def test_token_histogram_buckets():
    base = make_chunk("a.md", 1, 1, "x", "prose")
    sizes = (10, 50, 51, 100, 512, 513, 2000)  # chunks with exactly these token counts
    chunks = [dataclasses.replace(base, token_count=n) for n in sizes]
    assert token_histogram(chunks) == {
        "<=50": 2,
        "51-100": 2,
        "101-200": 0,
        "201-300": 0,
        "301-400": 0,
        "401-512": 1,
        ">512": 2,
    }


def test_token_histogram_of_nothing_is_all_zeros():
    assert set(token_histogram([]).values()) == {0}


def test_format_stats_is_readable(repo, tmp_path):
    text = format_stats(ingest(repo, tmp_path / "c.jsonl"))
    for expected in ("documents", "chunks", "mean", "max", "512", ">512", "<=50"):
        assert expected in text, f"{expected!r} missing from:\n{text}"


# -------------------------------------------------------------- repository
def test_the_generated_chunks_file_is_not_tracked_by_git():
    lines = {
        line.strip()
        for line in (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "data/chunks.jsonl" in lines
