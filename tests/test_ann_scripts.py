"""Phase 5: the scripts scripts/make_scale_set.py and bench/ann.py.

Both are provided, not part of the exercise. They need your ann package to work. The tests use tiny
data so they run in seconds.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from cairn import config
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.pipeline import ingest, read_chunks

MAKE = config.PROJECT_ROOT / "scripts" / "make_scale_set.py"
BENCH = config.PROJECT_ROOT / "bench" / "ann.py"


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


@pytest.fixture
def source(tmp_path):
    vectors = unit(np.random.default_rng(0).normal(size=(40, 8)))
    FlatIndex(vectors, [f"c{i}" for i in range(40)]).save(tmp_path / "source.npz")
    return tmp_path / "source.npz", vectors


@pytest.fixture
def scale_file(tmp_path):
    rng = np.random.default_rng(1)
    centers = unit(rng.normal(size=(8, 16)))
    pick = rng.integers(8, size=330)
    both = unit(centers[pick] + 0.4 * rng.normal(size=(330, 16)) / 4)
    np.savez(tmp_path / "scale.npz", vectors=both[:300], queries=both[300:])
    return tmp_path / "scale.npz"


# ------------------------------------------------------------------- make_scale_set.py
def test_the_scale_set_has_the_requested_shape_and_unit_vectors(source, tmp_path):
    path, _ = source
    out = tmp_path / "scale.npz"
    result = run(MAKE, "--source", str(path), "--n", "200", "--queries", "15", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert "wrote" in result.stdout
    with np.load(out, allow_pickle=False) as data:
        assert data["vectors"].shape == (200, 8) and data["queries"].shape == (15, 8)
        assert data["vectors"].dtype == np.float32 and data["queries"].dtype == np.float32
        assert np.allclose(np.linalg.norm(data["vectors"], axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(data["queries"], axis=1), 1.0, atol=1e-5)


def test_the_new_vectors_are_made_from_real_ones_but_are_not_copies(source, tmp_path):
    path, real = source
    out = tmp_path / "scale.npz"
    run(MAKE, "--source", str(path), "--n", "300", "--queries", "10", "--out", str(out))
    with np.load(out) as data:
        closest = (data["vectors"] @ real.T).max(axis=1)
    assert closest.min() > 0.5  # each is close to some real vector
    assert closest.max() < 0.9999  # and none is an exact copy


def test_the_same_seed_gives_the_same_file_and_another_seed_a_different_one(source, tmp_path):
    path, _ = source
    outs = []
    for name, seed in (("a", "1"), ("b", "1"), ("c", "2")):
        out = tmp_path / f"{name}.npz"
        run(
            MAKE,
            "--source",
            str(path),
            "--n",
            "50",
            "--queries",
            "5",
            "--seed",
            seed,
            "--out",
            str(out),
        )
        outs.append(np.load(out)["vectors"])
    assert np.array_equal(outs[0], outs[1])
    assert not np.array_equal(outs[0], outs[2])


def test_the_queries_are_not_rows_of_the_index(source, tmp_path):
    path, _ = source
    out = tmp_path / "scale.npz"
    run(MAKE, "--source", str(path), "--n", "100", "--queries", "20", "--out", str(out))
    with np.load(out) as data:
        assert (data["queries"] @ data["vectors"].T).max() < 0.9999


def test_the_scale_set_script_reports_problems_in_one_line(source, tmp_path):
    path, _ = source
    missing = run(MAKE, "--source", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "x.npz"))
    assert missing.returncode == 1 and "embed_chunks" in missing.stderr
    zero = run(MAKE, "--source", str(path), "--n", "0", "--out", str(tmp_path / "x.npz"))
    assert zero.returncode == 1 and zero.stderr.startswith("error:")
    np.savez(tmp_path / "other.npz", whatever=np.arange(3))
    other = run(MAKE, "--source", str(tmp_path / "other.npz"), "--out", str(tmp_path / "x.npz"))
    assert other.returncode == 1 and other.stderr.startswith("error:")
    assert not (tmp_path / "x.npz").exists()


# ------------------------------------------------------------------------- bench/ann.py
def bench(scale_file, tmp_path, *extra):
    return run(
        BENCH,
        "--scale", str(scale_file),
        "--index-dir", str(tmp_path / "ann"),
        "--nprobes", "1", "4", "17",
        "--ef-searches", "10", "300",
        "--M", "6",
        "--ef-construction", "30",
        *extra,
    )  # fmt: skip


def table_rows(out, header_start):
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(header_start))
    rows = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) != 4:
            break
        rows.append([float(p) for p in parts])
    return rows


def test_the_benchmark_prints_flat_ivf_and_hnsw_results(scale_file, tmp_path):
    result = bench(scale_file, tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "300 vectors of 16 numbers, 30 queries, k = 10" in out
    assert "flat (exact): recall 1.000" in out
    ivf = table_rows(out, "nprobe")
    assert [r[0] for r in ivf] == [1, 4, 17]
    assert ivf[-1][1] == 1.0  # all 17 lists scanned = exact
    assert [r[1] for r in ivf] == sorted(r[1] for r in ivf)  # more lists, no lower recall
    assert ivf[0][3] < ivf[-1][3]  # and more distance computations
    hnsw = table_rows(out, "ef")
    assert [r[0] for r in hnsw] == [10, 300]
    assert hnsw[-1][1] >= 0.95
    assert hnsw[0][3] < hnsw[-1][3]


def test_built_indexes_are_saved_and_reused(scale_file, tmp_path):
    first = bench(scale_file, tmp_path)
    saved = sorted(p.name for p in (tmp_path / "ann").glob("*.npz"))
    assert len(saved) == 2 and any(n.startswith("ivf") for n in saved)
    second = bench(scale_file, tmp_path)
    assert second.returncode == 0, second.stderr
    assert [r[:2] for r in table_rows(second.stdout, "nprobe")] == [
        r[:2] for r in table_rows(first.stdout, "nprobe")
    ]
    assert sorted(p.name for p in (tmp_path / "ann").glob("*.npz")) == saved


def test_skipping_an_index_leaves_it_out(scale_file, tmp_path):
    result = bench(scale_file, tmp_path, "--skip", "hnsw")
    assert result.returncode == 0, result.stderr
    assert "IVF nlist=17" in result.stdout and "HNSW" not in result.stdout
    both = bench(scale_file, tmp_path, "--skip", "hnsw", "--skip", "ivf")
    assert both.returncode == 0 and "flat (exact)" in both.stdout


def test_n_queries_limits_the_queries(scale_file, tmp_path):
    result = bench(scale_file, tmp_path, "--n-queries", "7", "--skip", "hnsw")
    assert "7 queries" in result.stdout


def test_the_benchmark_reports_problems_in_one_line(scale_file, tmp_path):
    missing = run(BENCH, "--scale", str(tmp_path / "nope.npz"))
    assert missing.returncode == 1 and missing.stderr.startswith("error:")
    np.savez(tmp_path / "other.npz", whatever=np.arange(3))
    other = run(BENCH, "--scale", str(tmp_path / "other.npz"))
    assert other.returncode == 1 and other.stderr.startswith("error:")
    for bad in (["--nlist", "5000"], ["--k", "0"], ["--nprobes", "0"]):
        result = bench(scale_file, tmp_path, *bad)
        assert result.returncode == 1 and result.stderr.startswith("error:"), bad


def test_plot_writes_a_picture_or_says_matplotlib_is_missing(scale_file, tmp_path):
    out = tmp_path / "curves.png"
    result = bench(scale_file, tmp_path, "--plot", str(out))
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert result.returncode == 1 and "matplotlib" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert out.stat().st_size > 1000


def test_real_mode_uses_the_chunk_vectors_and_the_dev_questions(tmp_path):
    corpus = tmp_path / "corpus" / "docs" / "en" / "docs"
    corpus.mkdir(parents=True)
    body = "".join(f"## Part {i}\nwords about topic number {i} and thing{i}\n\n" for i in range(12))
    (corpus / "a.md").write_text("# Title\n\n" + body, encoding="utf-8")
    ingest(tmp_path / "corpus", tmp_path / "chunks.jsonl")
    chunks = read_chunks(tmp_path / "chunks.jsonl")
    build_dense_index(chunks, load_embedder("hashing"))[0].save(tmp_path / "dense.npz")
    rows = [
        {
            "id": f"q{i}",
            "category": "lexical",
            "split": split,
            "question": f"words about topic number {i}",
            "gold": [{"path": "docs/en/docs/a.md", "start_line": 3, "end_line": 4}],
        }
        for i, split in enumerate(["dev", "dev", "dev", "test"])
    ]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    result = run(
        BENCH,
        "--real", "--model", "hashing",
        "--dense-index", str(tmp_path / "dense.npz"),
        "--questions", str(tmp_path / "questions.jsonl"),
        "--index-dir", str(tmp_path / "ann"),
        "--nprobes", "1", "--ef-searches", "20", "--M", "4", "--ef-construction", "20",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert (
        f"{len(chunks)} vectors of 256 numbers, 3 queries" in result.stdout
    )  # the test question is not used
    missing = run(BENCH, "--real", "--dense-index", str(tmp_path / "nope.npz"))
    assert missing.returncode == 1 and "embed_chunks" in missing.stderr
