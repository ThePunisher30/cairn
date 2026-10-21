"""Phase 4, step 5a: the command-line script scripts/embed_chunks.py, and the repository settings.

The tests use `--model hashing`, the offline word-counting model, so they run in a moment and
need no download.
"""

import os
import subprocess
import sys
import tomllib

import pytest

from cairn import config
from cairn.dense.flat_index import FlatIndex
from cairn.ingest.pipeline import ingest, read_chunks

SCRIPT = config.PROJECT_ROOT / "scripts" / "embed_chunks.py"


@pytest.fixture
def chunks_file(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": "# Title\nsome words here\n\n## Part\nmore words about the part\n",
        "fastapi/m.py": "def get_item(x):\n    return x + 1\n",
    }.items():
        path = tmp_path / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out = tmp_path / "chunks.jsonl"
    ingest(tmp_path / "corpus", out)
    return out


def run(*args, data_dir=None):
    env = dict(os.environ)
    if data_dir is not None:
        env["CAIRN_DATA_DIR"] = str(data_dir)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env
    )


def test_it_embeds_the_chunks_and_saves_the_index_and_the_cache(chunks_file, tmp_path):
    out, cache = tmp_path / "out" / "dense.npz", tmp_path / "cache.npz"
    result = run(
        "--chunks", str(chunks_file), "--model", "hashing", "--out", str(out), "--cache", str(cache)
    )
    assert result.returncode == 0, result.stderr
    chunks = read_chunks(chunks_file)
    index = FlatIndex.load(out)
    assert index.n_docs == len(chunks) > 0
    assert index.doc_ids == [c.id for c in chunks]
    assert cache.exists()
    assert f"embedded {len(chunks)} chunks" in result.stdout
    assert f"{len(chunks)} computed, 0 from cache" in result.stdout


def test_a_second_run_takes_every_vector_from_the_cache(chunks_file, tmp_path):
    args = [
        "--chunks", str(chunks_file), "--model", "hashing",
        "--out", str(tmp_path / "dense.npz"), "--cache", str(tmp_path / "cache.npz"),
    ]  # fmt: skip
    first = run(*args)
    second = run(*args)
    assert first.returncode == 0 and second.returncode == 0
    n = len(read_chunks(chunks_file))
    assert f"0 computed, {n} from cache" in second.stdout
    assert FlatIndex.load(tmp_path / "dense.npz").n_docs == n


def test_the_defaults_come_from_the_data_folder_and_name_the_model(chunks_file, tmp_path):
    data = tmp_path / "data"
    result = run("--chunks", str(chunks_file), "--model", "hashing", data_dir=data)
    assert result.returncode == 0, result.stderr
    assert (data / "dense-hashing.npz").exists()
    assert (data / "embedding_cache.npz").exists()


def test_it_reports_how_many_chunks_are_too_long_for_the_model(chunks_file, tmp_path):
    result = run("--chunks", str(chunks_file), "--model", "hashing", data_dir=tmp_path / "data")
    assert "0 chunks are longer than the 512-token limit" in result.stdout


def test_a_missing_chunks_file_is_a_clean_one_line_error(tmp_path):
    out = tmp_path / "dense.npz"
    result = run(
        "--chunks", str(tmp_path / "missing.jsonl"), "--model", "hashing", "--out", str(out)
    )
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "ingest" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_an_empty_chunks_file_is_a_clean_error(tmp_path):
    empty = tmp_path / "chunks.jsonl"
    empty.write_text("", encoding="utf-8")
    result = run("--chunks", str(empty), "--model", "hashing", "--out", str(tmp_path / "d.npz"))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_a_batch_size_below_one_is_a_clean_error(chunks_file, tmp_path):
    result = run("--chunks", str(chunks_file), "--model", "hashing", "--batch-size", "0",
                 "--out", str(tmp_path / "d.npz"))  # fmt: skip
    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert not (tmp_path / "d.npz").exists()


def test_an_unknown_model_is_refused(chunks_file):
    result = run("--chunks", str(chunks_file), "--model", "no-such-model")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


# ------------------------------------------------------------ repository settings
def test_generated_vector_files_are_not_tracked_by_git():
    lines = {
        line.strip()
        for line in (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "data/*.npz" in lines


def test_the_model_library_is_an_optional_dependency():
    data = tomllib.loads((config.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dense = " ".join(data["project"]["optional-dependencies"]["dense"]).lower()
    assert "sentence-transformers" in dense
    assert "sentence-transformers" not in " ".join(data["project"]["dependencies"]).lower()
