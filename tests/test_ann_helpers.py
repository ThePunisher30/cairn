"""Phase 5: the helper scripts bench/ann_ablate.py and scripts/eval_ann.py.

Provided, not part of the exercise. They use tiny data so they run in seconds.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from cairn import config
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.pipeline import ingest, read_chunks

ABLATE = config.PROJECT_ROOT / "bench" / "ann_ablate.py"
EVAL = config.PROJECT_ROOT / "scripts" / "eval_ann.py"


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


@pytest.fixture
def scale_file(tmp_path):
    rng = np.random.default_rng(1)
    centers = unit(rng.normal(size=(8, 16)))
    pick = rng.integers(8, size=330)
    both = unit(centers[pick] + 0.4 * rng.normal(size=(330, 16)) / 4)
    np.savez(tmp_path / "scale.npz", vectors=both[:300], queries=both[300:])
    return tmp_path / "scale.npz"


def ablate(scale_file, tmp_path, experiment, *extra):
    return run(
        ABLATE,
        "--experiment", experiment,
        "--scale", str(scale_file),
        "--index-dir", str(tmp_path / "ann"),
        "--ef-searches", "10", "300",
        "--M", "6",
        "--ef-construction", "30",
        *extra,
    )  # fmt: skip


def table(out):
    """The rows of the ef table: [ef, recall, distances, recall, distances]."""
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            rows.append([float(p) for p in parts])
    return rows


# ------------------------------------------------------------------ bench/ann_ablate.py
def test_the_heuristic_ablation_compares_two_graphs(scale_file, tmp_path):
    result = ablate(scale_file, tmp_path, "heuristic")
    assert result.returncode == 0, result.stderr
    assert "heuristic on :" in result.stdout and "heuristic off:" in result.stdout
    rows = table(result.stdout)
    assert [r[0] for r in rows] == [10, 300]
    assert rows[-1][1] >= 0.95 and rows[-1][3] >= 0.95  # a huge ef finds nearly everything
    assert rows[0][2] < rows[-1][2] and rows[0][4] < rows[-1][4]  # and costs more
    names = sorted(p.name for p in (tmp_path / "ann").glob("*.npz"))
    assert len(names) == 2 and any(n.endswith("-simple.npz") for n in names)


def test_the_layers_ablation_searches_one_graph_both_ways(scale_file, tmp_path):
    result = ablate(scale_file, tmp_path, "layers")
    assert result.returncode == 0, result.stderr
    assert "nodes on layers 0, 1, 2, ...: [300," in result.stdout
    rows = table(result.stdout)
    assert [r[0] for r in rows] == [10, 300]
    assert rows[-1][1] >= 0.95 and rows[-1][3] >= 0.95
    assert len(list((tmp_path / "ann").glob("*.npz"))) == 1


def test_a_second_ablation_run_reuses_the_saved_graphs(scale_file, tmp_path):
    first = ablate(scale_file, tmp_path, "heuristic")
    saved = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "ann").glob("*.npz")}
    second = ablate(scale_file, tmp_path, "heuristic")
    assert second.returncode == 0, second.stderr
    assert table(second.stdout) == table(first.stdout)
    assert saved == {p.name: p.stat().st_mtime_ns for p in (tmp_path / "ann").glob("*.npz")}


def test_the_ablation_script_reports_problems_in_one_line(scale_file, tmp_path):
    missing = run(ABLATE, "--experiment", "layers", "--scale", str(tmp_path / "nope.npz"))
    assert missing.returncode == 1 and missing.stderr.startswith("error:")
    for bad in (["--k", "0"], ["--M", "1"], ["--ef-searches", "0"]):
        result = ablate(scale_file, tmp_path, "layers", *bad)
        assert result.returncode == 1 and result.stderr.startswith("error:"), bad


# --------------------------------------------------------------------- scripts/eval_ann.py
@pytest.fixture
def real_files(tmp_path):
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
    return tmp_path, len(chunks)


def evaluate(tmp_path, *extra):
    return run(
        EVAL,
        "--model", "hashing",
        "--dense-index", str(tmp_path / "dense.npz"),
        "--chunks", str(tmp_path / "chunks.jsonl"),
        "--questions", str(tmp_path / "questions.jsonl"),
        "--nprobes", "1", "1000",
        "--ef-searches", "50",
        "--M", "4",
        "--ef-construction", "20",
        *extra,
    )  # fmt: skip


def test_eval_ann_scores_flat_ivf_and_hnsw_on_the_dev_questions(real_files):
    tmp_path, n_chunks = real_files
    result = evaluate(tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert f"{n_chunks} chunks, 3 dev questions, k = 10" in out  # the test question is not used
    lines = [line for line in out.splitlines() if line.startswith(("flat", "IVF", "HNSW"))]
    assert len(lines) == 1 + 2 + 1
    assert lines[0].endswith("0 of 3")  # flat is not different from itself
    ivf_all = next(line for line in lines if "nprobe=1000" in line)
    assert ivf_all.endswith("0 of 3")  # every list scanned = the flat answer


def test_eval_ann_reports_problems_in_one_line(real_files):
    tmp_path, _ = real_files
    missing = run(EVAL, "--dense-index", str(tmp_path / "nope.npz"))
    assert missing.returncode == 1 and "embed_chunks" in missing.stderr
    bad = evaluate(tmp_path, "--k", "0")
    assert bad.returncode == 1 and bad.stderr.startswith("error:")
